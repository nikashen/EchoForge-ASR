# WebSocket Protocol v1

Endpoint: `/api/v1/stream`
Subprotocol: `echoforge.v1`
Audio: mono, 16 kHz, signed 16-bit little-endian PCM.

Every JSON event uses:

```json
{
  "schema_version": "echoforge.ws/v1",
  "type": "transcript.revision",
  "event_id": "server-generated-id",
  "session_id": "server-session-id",
  "generation": 0,
  "server_sequence": 4,
  "payload": {}
}
```

Binary audio frames use a 12-byte header:

```text
bytes 0..3   ASCII EFA1
bytes 4..7   uint32 big-endian audio sequence
bytes 8..11  uint32 big-endian sample count
bytes 12..   exactly sample_count * 2 PCM16LE bytes
```

The server rejects frames before `session.start`, sequence gaps, duplicate
sequences with a different payload, oversized frames, and non-finite session
accounting. Duplicate frames with the same hash are acknowledged idempotently.

Revision stages are monotonic:

```text
partial -> stream_final -> dual_pass_final
```

The final stage may be `stream_only` with an explicit degradation code when the
verifier is unavailable. Decoder scores are not calibrated probabilities.

The server also emits deterministic VAD transitions:

```json
{
  "type": "vad.event",
  "payload": {
    "event": {
      "event_type": "speech_started",
      "state": "speech",
      "sample_offset": 0,
      "observed_sample_offset": 1280,
      "state_version": 1,
      "reason": "energy"
    }
  }
}
```

Control commands are `session.start`, `stream.flush`, `session.stop`,
`session.reset`, `hotwords.update`, and `ping`. Unknown fields are rejected.
Unsupported live hotword updates return `applied: false`; the server never
pretends that a decoder accepted them.
