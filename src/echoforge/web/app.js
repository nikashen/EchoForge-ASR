(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const STATIC_BASE = new URL('.', document.currentScript.src).href;
  const SAMPLE_RATE = 16000;
  const CHUNK_SAMPLES = 640;
  const sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

  const state = {
    view: 'live',
    source: 'demo',
    running: false,
    sessionId: null,
    revisions: [],
    events: [],
    audio: [],
    startedAt: 0,
    timer: null,
    toastTimer: null,
    challengeFresh: false,
    demoRun: 0,
    demoTransport: null,
    transport: null,
    capture: null,
    selectedFile: null,
    hotwords: [],
    pendingSamples: new Float32Array(0),
    runtime: { localApi: false, backend: 'deterministic-demo' },
  };

  const demoSteps = [
    { delay: 260, stage: 'partial', text: '我们今天讨论图神经网路', lane: 'streaming', audio_end_ms: 480 },
    { delay: 520, stage: 'partial', text: '我们今天讨论图神经网络', lane: 'streaming', audio_end_ms: 900 },
    { delay: 760, stage: 'stream_final', text: '我们今天讨论图神经网路', lane: 'streaming', audio_end_ms: 1480 },
    { delay: 430, stage: 'dual_pass_final', text: '我们今天讨论图神经网络', lane: 'verifier', audio_end_ms: 1480 },
  ];

  class DemoTransport {
    constructor(onRevision, onFinish) {
      this.onRevision = onRevision;
      this.onFinish = onFinish;
      this.timers = [];
    }

    start() {
      let elapsed = 0;
      demoSteps.forEach((step) => {
        elapsed += step.delay;
        this.timers.push(setTimeout(() => {
          this.onRevision(step);
          if (step.stage === 'dual_pass_final') this.onFinish();
        }, elapsed));
      });
    }

    stop() {
      this.timers.forEach((timer) => clearTimeout(timer));
      this.timers = [];
    }
  }

  class WebSocketTransport {
    constructor(onEvent, onFailure) {
      this.onEvent = onEvent;
      this.onFailure = onFailure;
      this.socket = null;
      this.sequence = 0;
      this.started = false;
      this.requestSequence = 0;
    }

    requestId(prefix) {
      this.requestSequence += 1;
      return `${prefix}-${String(this.requestSequence).padStart(4, '0')}`;
    }

    async connect(hotwords) {
      const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const url = `${scheme}//${location.host}/api/v1/stream`;
      return new Promise((resolve, reject) => {
        const socket = new WebSocket(url, ['echoforge.v1']);
        this.socket = socket;
        const failStart = (message) => {
          if (!this.started) reject(new Error(message));
        };
        socket.addEventListener('open', () => {
          socket.send(JSON.stringify({
            type: 'session.start',
            request_id: this.requestId('start'),
            language: 'zh-CN',
            mode: 'dual_pass',
            sample_rate: SAMPLE_RATE,
            channels: 1,
            encoding: 'pcm_s16le',
            chunk_duration_ms: 40,
            hotwords,
          }));
        });
        socket.addEventListener('message', (event) => {
          let message;
          try {
            message = JSON.parse(event.data);
          } catch (error) {
            this.onFailure(`Invalid server event: ${error}`);
            return;
          }
          if (message.type === 'session.started') {
            this.started = true;
            resolve(message);
          }
          this.onEvent(message);
        });
        socket.addEventListener('error', () => failStart('WebSocket connection failed'));
        socket.addEventListener('close', (event) => {
          failStart(`WebSocket closed before start (${event.code})`);
          if (event.code !== 1000 && this.started) {
            this.onFailure(`Stream closed (${event.code} ${event.reason || 'without reason'})`);
          }
        });
      });
    }

    sendControl(payload) {
      if (!this.socket || this.socket.readyState !== WebSocket.OPEN) return false;
      this.socket.send(JSON.stringify(payload));
      return true;
    }

    sendAudio(samples) {
      if (!this.socket || this.socket.readyState !== WebSocket.OPEN || samples.length === 0) return false;
      this.socket.send(packAudioFrame(this.sequence, samples));
      this.sequence += 1;
      return true;
    }

    async waitForWritable() {
      while (this.socket && this.socket.bufferedAmount > 256 * 1024) await sleep(8);
    }

    flush() {
      return this.sendControl({
        type: 'stream.flush',
        request_id: this.requestId('flush'),
        expected_generation: 0,
      });
    }

    stopSession() {
      return this.sendControl({ type: 'session.stop', request_id: this.requestId('stop') });
    }

    updateHotwords(hotwords) {
      return this.sendControl({
        type: 'hotwords.update',
        request_id: this.requestId('hotwords'),
        hotwords,
      });
    }

    close() {
      if (this.socket && this.socket.readyState < WebSocket.CLOSING) this.socket.close(1000, 'client reset');
      this.socket = null;
    }
  }

  class MicrophoneCapture {
    constructor(onSamples) {
      this.onSamples = onSamples;
      this.stream = null;
      this.context = null;
      this.source = null;
      this.processor = null;
      this.sink = null;
    }

    async start() {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('Microphone capture is unavailable');
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: false },
      });
      this.context = new AudioContext({ sampleRate: SAMPLE_RATE });
      this.source = this.context.createMediaStreamSource(this.stream);
      this.sink = this.context.createGain();
      this.sink.gain.value = 0;
      if (this.context.audioWorklet && typeof AudioWorkletNode !== 'undefined') {
        await this.context.audioWorklet.addModule(`${STATIC_BASE}pcm-worklet.js`);
        this.processor = new AudioWorkletNode(this.context, 'echoforge-pcm');
        this.processor.port.onmessage = (event) => {
          this.onSamples(resampleTo16k(new Float32Array(event.data), this.context.sampleRate));
        };
        this.source.connect(this.processor).connect(this.sink).connect(this.context.destination);
      } else {
        this.processor = this.context.createScriptProcessor(2048, 1, 1);
        this.processor.onaudioprocess = (event) => {
          const input = new Float32Array(event.inputBuffer.getChannelData(0));
          this.onSamples(resampleTo16k(input, this.context.sampleRate));
        };
        this.source.connect(this.processor).connect(this.sink).connect(this.context.destination);
      }
      await this.context.resume();
    }

    async stop() {
      if (this.processor) this.processor.disconnect();
      if (this.source) this.source.disconnect();
      if (this.sink) this.sink.disconnect();
      if (this.stream) this.stream.getTracks().forEach((track) => track.stop());
      if (this.context && this.context.state !== 'closed') await this.context.close();
      this.stream = null;
      this.context = null;
      this.source = null;
      this.processor = null;
      this.sink = null;
    }
  }

  function resampleTo16k(input, sourceRate) {
    if (sourceRate === SAMPLE_RATE) return new Float32Array(input);
    const ratio = sourceRate / SAMPLE_RATE;
    const length = Math.max(1, Math.floor(input.length / ratio));
    const output = new Float32Array(length);
    for (let index = 0; index < length; index += 1) {
      const position = index * ratio;
      const left = Math.floor(position);
      const right = Math.min(input.length - 1, left + 1);
      const fraction = position - left;
      output[index] = input[left] * (1 - fraction) + input[right] * fraction;
    }
    return output;
  }

  function packAudioFrame(sequence, samples) {
    const frame = new ArrayBuffer(12 + samples.length * 2);
    const view = new DataView(frame);
    view.setUint8(0, 0x45);
    view.setUint8(1, 0x46);
    view.setUint8(2, 0x41);
    view.setUint8(3, 0x31);
    view.setUint32(4, sequence, false);
    view.setUint32(8, samples.length, false);
    for (let index = 0; index < samples.length; index += 1) {
      const value = Math.max(-1, Math.min(1, samples[index]));
      const pcm = value < 0 ? Math.round(value * 32768) : Math.round(value * 32767);
      view.setInt16(12 + index * 2, pcm, true);
    }
    return frame;
  }

  function toast(message) {
    const node = $('toast');
    node.textContent = message;
    node.hidden = false;
    clearTimeout(state.toastTimer);
    state.toastTimer = setTimeout(() => { node.hidden = true; }, 2800);
  }

  function setStatus(text, offline = false) {
    $('side-status').textContent = text;
    $('side-status-dot').classList.toggle('is-offline', offline);
  }

  function switchView(view) {
    state.view = view;
    document.body.dataset.view = view;
    document.querySelectorAll('.view').forEach((node) => node.classList.toggle('is-active', node.dataset.view === view));
    document.querySelectorAll('[data-view-target]').forEach((node) => node.classList.toggle('is-active', node.dataset.viewTarget === view));
    $('view-title').textContent = ({ live: 'Live Lab', arena: 'Robustness Arena', timeline: 'Session Timeline', evidence: 'Evidence' })[view];
    if (view === 'timeline') renderEventList();
  }

  function resetVisuals() {
    state.revisions = [];
    state.events = [];
    state.audio = [];
    state.sessionId = null;
    state.challengeFresh = false;
    $('partial-transcript').textContent = '—';
    $('fast-text').textContent = 'Waiting for speech…';
    $('verify-text').textContent = 'Awaiting endpoint…';
    $('fast-revision').textContent = 'revision —';
    $('verify-state').textContent = 'not started';
    $('verify-latency').textContent = '—';
    $('final-transcript').replaceChildren();
    $('repair-box').hidden = true;
    $('revision-count').textContent = '0 revisions';
    $('timeline-track').replaceChildren(Object.assign(document.createElement('span'), {
      className: 'timeline-empty', textContent: 'Start a session to populate the timeline.',
    }));
    ['kpi-first-partial', 'kpi-final-latency', 'kpi-rtf', 'kpi-stability'].forEach((id) => { $(id).textContent = '—'; });
    $('vad-badge').textContent = 'SILENCE';
    $('vad-meta').textContent = 'RMS — · peak —';
    $('wave-alt').textContent = 'No audio in the current session.';
    $('capture-timer').textContent = '00:00.0';
    drawWave();
    renderEventList();
  }

  function drawWave() {
    const canvas = $('waveform');
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const width = Math.max(320, Math.round(rect.width * dpr));
    const height = Math.max(180, Math.round(rect.height * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const context = canvas.getContext('2d');
    context.clearRect(0, 0, width, height);
    context.fillStyle = '#f5f9f9';
    context.fillRect(0, 0, width, height);
    context.strokeStyle = '#dce9ea';
    context.lineWidth = dpr;
    for (let y = height * 0.25; y < height; y += height * 0.25) {
      context.beginPath();
      context.moveTo(0, y);
      context.lineTo(width, y);
      context.stroke();
    }
    const values = state.audio.length ? state.audio : Array.from({ length: 180 }, () => 0);
    context.strokeStyle = '#087b78';
    context.lineWidth = 1.6 * dpr;
    context.beginPath();
    values.forEach((value, index) => {
      const x = (index / Math.max(1, values.length - 1)) * width;
      const y = height / 2 - Math.max(-1, Math.min(1, value)) * height * 0.38;
      if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
    });
    context.stroke();
  }

  function appendWaveSamples(samples) {
    const stride = Math.max(1, Math.floor(samples.length / 24));
    for (let index = 0; index < samples.length; index += stride) state.audio.push(samples[index]);
    if (state.audio.length > 600) state.audio.splice(0, state.audio.length - 600);
    let squareSum = 0;
    let peak = 0;
    samples.forEach((value) => {
      squareSum += value * value;
      peak = Math.max(peak, Math.abs(value));
    });
    const rms = Math.sqrt(squareSum / Math.max(1, samples.length));
    const rmsDb = rms > 0 ? 20 * Math.log10(rms) : -120;
    const peakDb = peak > 0 ? 20 * Math.log10(peak) : -120;
    $('vad-meta').textContent = `RMS ${rmsDb.toFixed(1)} dBFS · peak ${peakDb.toFixed(1)} dBFS`;
    $('wave-alt').textContent = `${state.audio.length} display points retained in the bounded waveform buffer.`;
    drawWave();
  }

  function addEvent(type, payload = {}) {
    state.events.push({ time: Date.now(), type, payload });
  }

  function addRevision(revision) {
    state.revisions.push(revision);
    addEvent('transcript.revision', revision);
    $('revision-count').textContent = `${state.revisions.length} revisions`;
    if (revision.stage === 'partial') {
      $('partial-transcript').textContent = revision.text;
      $('fast-text').textContent = revision.text;
      $('fast-revision').textContent = `revision ${revision.revision ?? state.revisions.length - 1}`;
      if ($('kpi-first-partial').textContent === '—') {
        const latency = state.source === 'demo' ? 260 : Math.round(performance.now() - state.startedAt);
        $('kpi-first-partial').textContent = `${latency} ms`;
      }
    }
    if (revision.stage === 'stream_final') {
      $('fast-text').textContent = revision.text;
      $('verify-state').textContent = 'verifying at endpoint';
      renderFinal(revision, 'STREAM FINAL');
    }
    if (revision.stage === 'dual_pass_final') {
      $('verify-text').textContent = revision.text;
      $('verify-state').textContent = 'verified final';
      const latency = revision.endpoint_to_final_ms ?? revision.server_compute_ms;
      $('verify-latency').textContent = latency == null ? 'not reported' : `${Math.round(latency)} ms`;
      $('kpi-final-latency').textContent = latency == null ? 'N/A' : `${Math.round(latency)} ms`;
      // RTF requires measured model compute and audio duration. The demo has
      // neither, so keep the metric explicitly unevaluated.
      const rtf = revision.metadata?.rtf;
      $('kpi-rtf').textContent = Number.isFinite(rtf) ? rtf.toFixed(2) : 'N/A';
      $('kpi-stability').textContent = `${partialStability()}%`;
      renderFinal(revision, 'VERIFIED FINAL');
      showRepair();
      $('final-announcer').textContent = `Verified final: ${revision.text}`;
    }
    if (revision.stage === 'stream_only') {
      $('verify-text').textContent = revision.text;
      $('verify-state').textContent = 'stream-only fallback';
      $('verify-latency').textContent = 'degraded';
      renderFinal(revision, 'STREAM-ONLY');
      $('final-announcer').textContent = `Stream-only final: ${revision.text}`;
    }
    if (state.source === 'demo' && revision.audio_end_ms) {
      const samples = Float32Array.from({ length: 80 }, (_, index) => {
        const position = state.audio.length + index;
        return Math.sin(position * 0.7) * (0.18 + (position % 5) * 0.04);
      });
      appendWaveSamples(samples);
    }
    renderTimeline();
  }

  function partialStability() {
    const partials = state.revisions.filter((revision) => revision.stage === 'partial');
    if (partials.length < 2) return state.source === 'demo' ? 86 : 100;
    let stable = 0;
    let total = 0;
    for (let index = 1; index < partials.length; index += 1) {
      const previous = partials[index - 1].text;
      const current = partials[index].text;
      let prefix = 0;
      while (prefix < previous.length && prefix < current.length && previous[prefix] === current[prefix]) prefix += 1;
      stable += prefix;
      total += Math.max(previous.length, current.length);
    }
    return Math.round((stable / Math.max(1, total)) * 100);
  }

  function renderFinal(revision, label) {
    const item = document.createElement('li');
    item.innerHTML = `<span class="revision-meta">${String(revision.audio_end_ms || 0).padStart(4, '0')} ms</span><span class="revision-text"></span><span class="revision-stage">${label}</span>`;
    item.querySelector('.revision-text').textContent = revision.text;
    $('final-transcript').appendChild(item);
  }

  function showRepair() {
    const fast = state.revisions.find((revision) => revision.stage === 'stream_final');
    const verified = state.revisions.find((revision) => revision.stage === 'dual_pass_final');
    if (!fast || !verified) return;
    let prefix = 0;
    while (prefix < fast.text.length && prefix < verified.text.length && fast.text[prefix] === verified.text[prefix]) prefix += 1;
    let suffix = 0;
    while (
      suffix < fast.text.length - prefix
      && suffix < verified.text.length - prefix
      && fast.text[fast.text.length - 1 - suffix] === verified.text[verified.text.length - 1 - suffix]
    ) suffix += 1;
    const removed = fast.text.slice(prefix, fast.text.length - suffix || undefined);
    const inserted = verified.text.slice(prefix, verified.text.length - suffix || undefined);
    $('repair-box').hidden = false;
    $('repair-fast').textContent = fast.text;
    $('repair-verify').textContent = verified.text;
    $('repair-diff').textContent = removed || inserted ? `-${removed || '∅'}  +${inserted || '∅'}` : 'No character change';
    $('repair-count').textContent = removed || inserted ? `${Math.max(removed.length, inserted.length)} character revised` : '0 characters revised';
  }

  function renderTimeline() {
    const track = $('timeline-track');
    if (!state.revisions.length) return;
    track.replaceChildren();
    state.revisions.forEach((revision, index) => {
      const segment = document.createElement('button');
      segment.type = 'button';
      segment.className = `timeline-segment ${revision.lane === 'verifier' ? 'verify' : ''}`;
      segment.style.flex = `${Math.max(1, revision.audio_end_ms || (index + 1) * 200)}`;
      segment.title = `${revision.stage} · ${revision.audio_end_ms || 0} ms`;
      segment.addEventListener('click', () => toast(`Selected ${revision.stage} at ${revision.audio_end_ms || 0} ms`));
      track.appendChild(segment);
    });
  }

  function renderEventList() {
    const list = $('event-list');
    list.replaceChildren();
    if (!state.events.length) {
      $('timeline-title').textContent = 'No session selected';
      return;
    }
    $('timeline-title').textContent = state.sessionId || 'Demo session';
    state.events.forEach((entry) => {
      const item = document.createElement('li');
      const date = new Date(entry.time);
      item.innerHTML = `<span class="event-time">${date.toLocaleTimeString([], { hour12: false })}</span><span class="event-code"></span><span class="event-text"></span><span class="revision-stage">${entry.payload.stage || 'event'}</span>`;
      item.querySelector('.event-code').textContent = entry.type;
      item.querySelector('.event-text').textContent = entry.payload.text || entry.payload.condition || entry.payload.code || 'protocol event';
      list.appendChild(item);
    });
  }

  function exportFile(kind) {
    const revisions = state.revisions.filter((revision) => revision.stage !== 'partial');
    let content = '';
    const filename = `echoforge-session.${kind}`;
    if (kind === 'json') {
      content = JSON.stringify({
        schema_version: 'echoforge.export/v1',
        session_id: state.sessionId,
        revisions,
        evidence_scope: 'interactive_sample',
        backend: state.runtime.backend,
      }, null, 2);
    }
    if (kind === 'srt') {
      content = revisions.map((revision, index) => `${index + 1}\n00:00:${String(Math.floor((revision.audio_start_ms || 0) / 1000)).padStart(2, '0')},000 --> 00:00:${String(Math.ceil((revision.audio_end_ms || 1) / 1000)).padStart(2, '0')},000\n${revision.text}\n`).join('\n');
    }
    if (kind === 'vtt') {
      content = `WEBVTT\n\n${revisions.map((revision) => `00:00:${String(Math.floor((revision.audio_start_ms || 0) / 1000)).padStart(2, '0')}.000 --> 00:00:${String(Math.ceil((revision.audio_end_ms || 1) / 1000)).padStart(2, '0')}.000\n${revision.text}`).join('\n\n')}`;
    }
    const url = URL.createObjectURL(new Blob([content], { type: 'text/plain;charset=utf-8' }));
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 0);
    toast(`Exported ${filename}`);
  }

  function startTimer() {
    state.startedAt = performance.now();
    clearInterval(state.timer);
    state.timer = setInterval(() => {
      const elapsed = (performance.now() - state.startedAt) / 1000;
      const minutes = Math.floor(elapsed / 60);
      const seconds = elapsed - minutes * 60;
      $('capture-timer').textContent = `${String(minutes).padStart(2, '0')}:${seconds.toFixed(1).padStart(4, '0')}`;
    }, 100);
  }

  function setCaptureState(label, running) {
    state.running = running;
    $('capture-state').textContent = label;
    $('start-button').disabled = running;
    $('stop-button').disabled = !running;
  }

  function runDemo() {
    resetVisuals();
    state.demoRun += 1;
    state.sessionId = `demo-${String(state.demoRun).padStart(3, '0')}`;
    setCaptureState('RECORDING', true);
    startTimer();
    setStatus('Deterministic replay running');
    addEvent('session.started', { stage: 'demo' });
    state.demoTransport = new DemoTransport((step) => {
      if (!state.running) return;
      addRevision({
        ...step,
        revision: state.revisions.length,
        model_id: step.lane === 'verifier' ? 'deterministic-verifier-fixture' : 'deterministic-protocol-fixture',
        server_compute_ms: step.lane === 'verifier' ? 84 : null,
        endpoint_to_final_ms: step.lane === 'verifier' ? 84 : null,
      });
    }, finishDemo);
    state.demoTransport.start();
  }

  function finishDemo() {
    clearInterval(state.timer);
    setCaptureState('COMPLETE', false);
    setStatus('Demo transport ready');
    renderEventList();
    toast('Verified final received; repair diff preserved.');
  }

  function stopDemo() {
    if (state.demoTransport) state.demoTransport.stop();
    clearInterval(state.timer);
    setCaptureState('STOPPED', false);
    addEvent('session.stopped');
    renderEventList();
    setStatus('Demo transport ready');
  }

  async function createLiveTransport() {
    const transport = new WebSocketTransport(handleServerEvent, handleTransportFailure);
    state.transport = transport;
    const started = await transport.connect(state.hotwords);
    state.sessionId = started.session_id;
    $('fast-model').textContent = started.payload.model_id || 'unknown';
    $('verify-model').textContent = started.payload.verifier_model_id || 'disabled';
    return transport;
  }

  function handleServerEvent(message) {
    addEvent(message.type, message.payload || {});
    if (message.type === 'transcript.revision') addRevision(message.payload.revision);
    if (message.type === 'session.started' && message.payload.hotwords?.length) {
      $('hotword-status').textContent = message.payload.hotwords_applied
        ? `${message.payload.hotwords.length} active`
        : `Not applied: ${message.payload.hotwords_reason}`;
    }
    if (message.type === 'vad.event') {
      const event = message.payload.event;
      $('vad-badge').textContent = event.state.toUpperCase();
      $('vad-badge').className = `badge ${event.state === 'speech' ? 'badge-teal' : 'badge-muted'}`;
    }
    if (message.type === 'audio.ack') $('audio-meta').textContent = `16 kHz · PCM16LE · seq ${message.payload.highest_contiguous_sequence}`;
    if (message.type === 'hotwords.updated') {
      $('hotword-status').textContent = message.payload.applied ? `${message.payload.hotwords.length} active` : 'Backend does not support live update';
    }
    if (message.type === 'stream.flushed' && state.transport) state.transport.stopSession();
    if (message.type === 'session.stopped') finishLiveSession();
    if (message.type === 'error') handleTransportFailure(`${message.payload.code}: ${message.payload.message}`);
  }

  function handleTransportFailure(message) {
    setStatus('Local API stream failed', true);
    setCaptureState('ERROR', false);
    toast(message);
  }

  async function startMicrophone() {
    resetVisuals();
    setCaptureState('CONNECTING', true);
    setStatus('Connecting to local API');
    try {
      const transport = await createLiveTransport();
      state.pendingSamples = new Float32Array(0);
      state.capture = new MicrophoneCapture((samples) => {
        appendWaveSamples(samples);
        const merged = new Float32Array(state.pendingSamples.length + samples.length);
        merged.set(state.pendingSamples);
        merged.set(samples, state.pendingSamples.length);
        let offset = 0;
        while (merged.length - offset >= CHUNK_SAMPLES) {
          transport.sendAudio(merged.slice(offset, offset + CHUNK_SAMPLES));
          offset += CHUNK_SAMPLES;
        }
        state.pendingSamples = merged.slice(offset);
      });
      await state.capture.start();
      setCaptureState('RECORDING', true);
      startTimer();
      setStatus(`Local API · ${state.runtime.backend}`);
    } catch (error) {
      await cleanupLiveResources();
      setCaptureState('ERROR', false);
      setStatus('Microphone unavailable', true);
      toast(error.message || String(error));
    }
  }

  async function startFile() {
    if (!state.selectedFile) {
      $('file-input').click();
      toast('Select an audio file, then press Start.');
      return;
    }
    resetVisuals();
    setCaptureState('DECODING', true);
    setStatus('Decoding local file');
    try {
      const context = new AudioContext();
      const decoded = await context.decodeAudioData(await state.selectedFile.arrayBuffer());
      const samples = resampleTo16k(decoded.getChannelData(0), decoded.sampleRate);
      await context.close();
      const transport = await createLiveTransport();
      setCaptureState('STREAMING', true);
      startTimer();
      for (let offset = 0; offset < samples.length; offset += CHUNK_SAMPLES) {
        const chunk = samples.slice(offset, Math.min(samples.length, offset + CHUNK_SAMPLES));
        appendWaveSamples(chunk);
        transport.sendAudio(chunk);
        await transport.waitForWritable();
        await sleep(2);
      }
      setCaptureState('FINALIZING', true);
      transport.flush();
    } catch (error) {
      await cleanupLiveResources();
      setCaptureState('ERROR', false);
      setStatus('File stream failed', true);
      toast(error.message || String(error));
    }
  }

  async function stopLive() {
    if (state.capture) await state.capture.stop();
    state.capture = null;
    clearInterval(state.timer);
    if (!state.transport) {
      setCaptureState('STOPPED', false);
      return;
    }
    setCaptureState('FINALIZING', true);
    if (state.pendingSamples.length) {
      state.transport.sendAudio(state.pendingSamples);
      state.pendingSamples = new Float32Array(0);
    }
    if (state.transport.sequence > 0) state.transport.flush(); else state.transport.stopSession();
  }

  function finishLiveSession() {
    clearInterval(state.timer);
    setCaptureState('COMPLETE', false);
    setStatus(`Local API · ${state.runtime.backend}`);
    if (state.transport) state.transport.close();
    state.transport = null;
    state.pendingSamples = new Float32Array(0);
    renderEventList();
  }

  async function cleanupLiveResources() {
    if (state.capture) await state.capture.stop();
    state.capture = null;
    if (state.transport) state.transport.close();
    state.transport = null;
    clearInterval(state.timer);
  }

  async function resetAll() {
    if (state.demoTransport) state.demoTransport.stop();
    await cleanupLiveResources();
    setCaptureState('IDLE', false);
    resetVisuals();
    setStatus(state.runtime.localApi ? `Local API · ${state.runtime.backend}` : 'Demo transport ready');
    toast('Lab reset.');
  }

  function parseHotwords() {
    return [...new Set($('hotword-input').value.split(/[,，\n]/).map((word) => word.trim()).filter(Boolean))].slice(0, 32);
  }

  function applyHotwords() {
    state.hotwords = parseHotwords();
    if (state.transport?.started) {
      state.transport.updateHotwords(state.hotwords);
      $('hotword-status').textContent = 'Update pending';
    } else {
      $('hotword-status').textContent = state.hotwords.length ? `${state.hotwords.length} staged for next session` : 'No active override';
    }
  }

  function runChallenge() {
    state.challengeFresh = false;
    $('challenge-status').textContent = 'RENDERING';
    $('result-freshness').textContent = 'STALE';
    const profile = $('noise-profile').value;
    const snr = $('snr-slider').value;
    const channel = $('channel-select').value;
    setTimeout(() => {
      state.challengeFresh = true;
      const condition = `${profile} · ${snr} dB · ${channel}`;
      $('challenge-status').textContent = 'RESULT READY';
      $('result-freshness').textContent = 'FIXTURE ONLY';
      $('result-condition').textContent = condition;
      $('result-stream').textContent = '我们今天讨论图神经网路';
      $('result-verify').textContent = '我们今天讨论图神经网络';
      $('result-cer').textContent = 'N/A · reference absent';
      $('result-artifact').textContent = `fixture-preview-${profile}-${snr}-${channel}`;
      $('challenge-note').textContent = 'Fixture preview only: the recipe is recorded, but no transformed audio is sent to ASR and no robustness metric is measured.';
      addEvent('challenge.fixture_preview', {
        condition,
        artifact: $('result-artifact').textContent,
        evaluated: false,
      });
      renderEventList();
      toast('Fixture preview recorded; no ASR metric was measured.');
    }, 380);
  }

  function chooseSource(source) {
    if (source !== 'demo' && !state.runtime.localApi) {
      toast('Mic and File require the local EchoForge API. Pages remains deterministic replay.');
      return;
    }
    state.source = source;
    document.querySelectorAll('[data-source]').forEach((node) => node.classList.toggle('is-active', node.dataset.source === source));
    if (source === 'file') $('file-input').click();
  }

  async function startSelectedSource() {
    if (state.source === 'demo') runDemo();
    else if (state.source === 'mic') await startMicrophone();
    else await startFile();
  }

  async function stopSelectedSource() {
    if (state.source === 'demo') stopDemo(); else await stopLive();
  }

  async function detectRuntime() {
    const isPages = location.hostname.endsWith('github.io') || location.protocol === 'file:';
    if (isPages) return;
    try {
      const response = await fetch('/api/v1/readiness', { cache: 'no-store' });
      if (!response.ok) return;
      const payload = await response.json();
      state.runtime.localApi = true;
      state.runtime.backend = payload.backend;
      document.querySelectorAll('[data-source="mic"], [data-source="file"]').forEach((button) => { button.disabled = false; });
      $('runtime-chip').textContent = payload.backend === 'deterministic-fake' ? 'LOCAL API · FIXTURE' : 'LOCAL API · LIVE';
      $('boundary-runtime').textContent = `Local API / ${payload.backend}`;
      $('evidence-backend').textContent = payload.backend;
      setStatus(`Local API · ${payload.backend}`);
    } catch (_error) {
      setStatus('Demo transport ready');
    }
  }

  document.querySelectorAll('[data-view-target]').forEach((node) => node.addEventListener('click', () => switchView(node.dataset.viewTarget)));
  document.querySelectorAll('[data-source]').forEach((node) => node.addEventListener('click', () => chooseSource(node.dataset.source)));
  $('start-button').addEventListener('click', startSelectedSource);
  $('stop-button').addEventListener('click', stopSelectedSource);
  $('reset-all').addEventListener('click', resetAll);
  $('apply-hotwords').addEventListener('click', applyHotwords);
  $('file-input').addEventListener('change', (event) => {
    state.selectedFile = event.target.files?.[0] || null;
    if (state.selectedFile) {
      state.source = 'file';
      document.querySelectorAll('[data-source]').forEach((node) => node.classList.toggle('is-active', node.dataset.source === 'file'));
      $('audio-meta').textContent = `${state.selectedFile.name} · awaiting decode`;
    }
  });
  $('run-challenge').addEventListener('click', runChallenge);
  $('mark-clean').addEventListener('click', () => {
    $('noise-profile').value = 'clean';
    $('snr-slider').value = 30;
    $('snr-output').textContent = 'clean';
    runChallenge();
  });
  $('snr-slider').addEventListener('input', (event) => {
    $('snr-output').textContent = `${event.target.value} dB`;
    state.challengeFresh = false;
    $('result-freshness').textContent = 'STALE';
  });
  ['json', 'srt', 'vtt'].forEach((kind) => $(`export-${kind}`).addEventListener('click', () => exportFile(kind)));
  window.addEventListener('resize', drawWave);
  window.addEventListener('pagehide', () => { cleanupLiveResources(); });

  resetVisuals();
  detectRuntime();
})();
