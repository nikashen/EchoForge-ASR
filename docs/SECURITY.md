# Security and Resource Limits

- Bind local development to `127.0.0.1` by default.
- Require an allowed Origin and the `echoforge.v1` WebSocket subprotocol.
- Limit frame size, sample count, session duration, pending audio, concurrent
  sessions, and outbound event queues.
- Do not accept arbitrary model paths, URLs, checkpoints, pickle files, or
  shell commands from a request.
- Reject malformed and oversized frames before decoding audio.
- Close failed stateful sessions rather than continuing with an unknown decoder
  state. A verifier timeout is reported as response degradation, not hard
  cancellation.
- Release MediaStream tracks, AudioContext objects, workers, and object URLs on
  reset or disconnect.
- Keep GitHub Pages on `DemoTransport`; it has no API credential, microphone
  upload, or hidden hosted inference endpoint.
- Treat `asyncio.to_thread` as isolation from the event loop, not hard model
  cancellation. A timed-out native call may continue until its runtime returns.
