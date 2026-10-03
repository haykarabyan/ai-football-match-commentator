// Mic capture -> 16 kHz 16-bit mono LE PCM, and 24 kHz PCM playback.
// Kept deliberately small; no build step, no dependencies.

const CAPTURE_RATE = 16000;   // what the Live API wants
const PLAYBACK_RATE = 24000;  // what Live audio comes back as
const CHUNK_SAMPLES = 1600;   // 100 ms at 16 kHz

// AudioWorklet source, inlined as a blob so this stays a single file.
const WORKLET = `
class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(${CHUNK_SAMPLES});
    this.n = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.buf[this.n++] = ch[i];
      if (this.n === this.buf.length) {
        // float32 [-1,1] -> int16 LE
        const pcm = new Int16Array(this.buf.length);
        for (let j = 0; j < this.buf.length; j++) {
          const s = Math.max(-1, Math.min(1, this.buf[j]));
          pcm[j] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        this.port.postMessage(pcm.buffer, [pcm.buffer]);
        this.n = 0;
      }
    }
    return true;
  }
}
registerProcessor('capture', Capture);
`;

export class Mic {
  constructor(onChunk, onLevel) {
    this.onChunk = onChunk;
    this.onLevel = onLevel;
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    // Asking for 16 kHz lets the browser do the resampling for us.
    this.ctx = new AudioContext({ sampleRate: CAPTURE_RATE });
    if (this.ctx.sampleRate !== CAPTURE_RATE) {
      console.warn(`AudioContext gave ${this.ctx.sampleRate} Hz, wanted ${CAPTURE_RATE}`);
    }
    const url = URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' }));
    await this.ctx.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);

    this.node = new AudioWorkletNode(this.ctx, 'capture');
    this.node.port.onmessage = (e) => {
      this.onChunk(e.data);
      if (this.onLevel) {
        const pcm = new Int16Array(e.data);
        let peak = 0;
        for (let i = 0; i < pcm.length; i += 8) peak = Math.max(peak, Math.abs(pcm[i]));
        this.onLevel(peak / 32768);
      }
    };
    this.src = this.ctx.createMediaStreamSource(this.stream);
    this.src.connect(this.node);
    // Worklets need a sink to be pulled; a muted gain node is enough.
    const sink = this.ctx.createGain();
    sink.gain.value = 0;
    this.node.connect(sink).connect(this.ctx.destination);
    return this.ctx.sampleRate;
  }

  stop() {
    try { this.src && this.src.disconnect(); } catch (_) {}
    try { this.node && this.node.disconnect(); } catch (_) {}
    try { this.stream && this.stream.getTracks().forEach((t) => t.stop()); } catch (_) {}
    try { this.ctx && this.ctx.close(); } catch (_) {}
  }
}

export class Player {
  constructor() {
    this.ctx = null;
    this.playAt = 0;
    this.live = new Set();
  }

  _ensure() {
    if (!this.ctx || this.ctx.state === 'closed') {
      this.ctx = new AudioContext({ sampleRate: PLAYBACK_RATE });
      this.playAt = 0;
    }
    if (this.ctx.state === 'suspended') this.ctx.resume();
    return this.ctx;
  }

  // arrayBuffer of 16-bit mono LE PCM at 24 kHz
  push(arrayBuffer) {
    const ctx = this._ensure();
    const pcm = new Int16Array(arrayBuffer);
    if (!pcm.length) return;
    const buf = ctx.createBuffer(1, pcm.length, PLAYBACK_RATE);
    const ch = buf.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;

    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    // Schedule back-to-back; a little lead time absorbs jitter.
    const now = ctx.currentTime;
    this.playAt = Math.max(this.playAt, now + 0.06);
    src.start(this.playAt);
    this.playAt += buf.duration;
    this.live.add(src);
    src.onended = () => this.live.delete(src);
  }

  // On an interruption, drop everything already queued.
  flush() {
    for (const src of this.live) {
      try { src.stop(); } catch (_) {}
    }
    this.live.clear();
    this.playAt = this.ctx ? this.ctx.currentTime : 0;
  }

  get queuedMs() {
    if (!this.ctx) return 0;
    return Math.max(0, (this.playAt - this.ctx.currentTime) * 1000);
  }
}
