class EchoForgePCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.pending = [];
    this.pendingSamples = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel || channel.length === 0) return true;
    const copy = new Float32Array(channel);
    this.pending.push(copy);
    this.pendingSamples += copy.length;
    if (this.pendingSamples >= 2048) {
      const merged = new Float32Array(this.pendingSamples);
      let offset = 0;
      for (const block of this.pending) {
        merged.set(block, offset);
        offset += block.length;
      }
      this.pending = [];
      this.pendingSamples = 0;
      this.port.postMessage(merged.buffer, [merged.buffer]);
    }
    return true;
  }
}

registerProcessor('echoforge-pcm', EchoForgePCMProcessor);
