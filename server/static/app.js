import { Mic, Player } from '/static/audio.js';

const $ = (id) => document.getElementById(id);

// -------------------------------------------------------------- tabs
document.querySelectorAll('.tab').forEach((tab) => {
  tab.onclick = () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
    tab.classList.add('active');
    $(`panel-${tab.dataset.tab}`).classList.add('active');
  };
});

// -------------------------------------------------------------- a session per tab
class Session {
  // vad: 'manual'  -> activity_start / activity_end frame the turn (live, translate)
  // vad: 'stream'  -> audio_stream_end finalizes it (captions)
  constructor(mode, { buildOpts, onEvent, vad }) {
    this.mode = mode;
    this.vad = vad || 'manual';
    this.buildOpts = buildOpts;
    this.onEvent = onEvent;
    this.ws = null;
    this.mic = null;
    this.player = new Player();
    this.active = false;
    this.btn = $(`mic-${mode}`);
    this.statusEl = $(`status-${mode}`);
    this.levelEl = $(`level-${mode}`);
    this.logEl = $(`log-${mode}`);
    this.wire();
  }

  status(text) { this.statusEl.textContent = text; }

  log(cls, who, text, ms, id) {
    const row = document.createElement('div');
    row.className = `row ${cls}`;
    if (id) row.id = id;
    const label = document.createElement('span');
    label.className = 'who';
    label.textContent = who + (ms !== undefined ? ` · ${ms} ms` : '');
    row.appendChild(label);
    const body = document.createElement('span');
    body.className = 'body';
    body.textContent = text;
    row.appendChild(body);
    this.logEl.appendChild(row);
    this.logEl.scrollIntoView({ block: 'end', behavior: 'smooth' });
    return row;
  }

  // Append to the last row of this class instead of making a new one, so
  // streamed transcript fragments read as sentences.
  append(cls, who, text, ms) {
    const last = this.logEl.lastElementChild;
    if (last && last.classList.contains(cls) && last.dataset.open === '1') {
      last.querySelector('.body').textContent += text;
      return last;
    }
    const row = this.log(cls, who, text, ms);
    row.dataset.open = '1';
    return row;
  }

  closeOpenRows() {
    this.logEl.querySelectorAll('[data-open="1"]').forEach((r) => (r.dataset.open = '0'));
  }

  wire() {
    const down = (e) => { e.preventDefault(); this.start(); };
    const up = (e) => { e.preventDefault(); this.stop(); };
    this.btn.addEventListener('mousedown', down);
    this.btn.addEventListener('touchstart', down, { passive: false });
    window.addEventListener('mouseup', up);
    this.btn.addEventListener('touchend', up);
  }

  async start() {
    if (this.active) return;
    this.active = true;
    this.btn.classList.add('on');
    this.btn.textContent = 'Listening…';
    this.closeOpenRows();

    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    this.ws = new WebSocket(`${proto}://${location.host}/ws/${this.mode}`);
    this.ws.binaryType = 'arraybuffer';

    this.ws.onopen = async () => {
      this.ws.send(JSON.stringify({ type: 'start', ...this.buildOpts() }));
      if (this.vad === 'manual') this.ws.send(JSON.stringify({ type: 'activity_start' }));
      try {
        this.mic = new Mic(
          (buf) => { if (this.ws && this.ws.readyState === 1) this.ws.send(buf); },
          (level) => { this.levelEl.style.width = `${Math.min(100, level * 180)}%`; },
        );
        const rate = await this.mic.start();
        this.status(`mic open at ${rate} Hz, streaming 100 ms chunks`);
      } catch (err) {
        this.log('err', 'mic error', String(err));
        this.stop();
      }
    };

    this.ws.onmessage = (e) => {
      if (e.data instanceof ArrayBuffer) {
        this.player.push(e.data);
        return;
      }
      const msg = JSON.parse(e.data);
      this.handle(msg);
    };

    this.ws.onerror = () => this.log('err', 'socket error', 'WebSocket failed — is the server running?');
    this.ws.onclose = () => { if (this.active) this.stop(); };
  }

  handle(msg) {
    switch (msg.type) {
      case 'status':
        if (msg.state === 'connected') {
          this.log('sys', 'connected', msg.model, msg.ms);
        } else if (msg.state === 'turn_complete') {
          this.closeOpenRows();
          this.status(`turn complete at ${msg.ms} ms · ${Math.round(this.player.queuedMs)} ms audio queued`);
        } else if (msg.state === 'go_away') {
          this.log('sys', 'go_away', 'server is about to close this session — ' + msg.detail, msg.ms);
        }
        break;
      case 'you':
        this.append('you', 'you', msg.text, msg.ms);
        break;
      case 'gemini':
        this.append('gemini', 'gemini', msg.text, msg.ms);
        break;
      case 'tool':
        if (msg.state === 'called') {
          this.log('tool', 'tool call', `${msg.name}(${JSON.stringify(msg.args)})`, msg.ms);
        } else if (msg.state === 'running') {
          this.log('tool', 'tool running', msg.detail, msg.ms);
        } else {
          this.log('tool', 'tool returned', JSON.stringify(msg.result), msg.ms);
        }
        break;
      case 'interrupted':
        this.player.flush();
        this.log('sys', 'interrupted', 'you spoke over it — playback queue dropped', msg.ms);
        break;
      case 'thinking':
        this.status(`interaction_status = ${msg.status} at ${msg.ms} ms`);
        break;
      case 'error':
        this.log('err', 'server error', msg.detail, msg.ms);
        break;
      case 'closed':
        this.status(`session closed after ${(msg.ms / 1000).toFixed(1)} s`);
        break;
    }
    // caption / lang and anything else mode-specific
    if (this.onEvent) this.onEvent(msg, this);
  }

  stop() {
    if (!this.active) return;
    this.active = false;
    this.btn.classList.remove('on');
    this.btn.textContent = 'Hold to talk';
    this.levelEl.style.width = '0';
    if (this.mic) { this.mic.stop(); this.mic = null; }
    if (this.ws && this.ws.readyState === 1) {
      // Releasing the button cuts the audio stream dead, so automatic VAD
      // would never hear the trailing silence it needs to end the turn.
      // Manual VAD closes the turn explicitly instead.
      this.ws.send(JSON.stringify({
        type: this.vad === 'manual' ? 'activity_end' : 'audio_stream_end',
      }));
      this.status('turn closed — waiting for the reply, keep this tab open');
      // Stay connected long enough to hear the whole answer (and any slow tool).
      clearTimeout(this.closeTimer);
      this.closeTimer = setTimeout(() => { try { this.ws.close(); } catch (_) {} }, 45000);
    }
  }
}

// -------------------------------------------------------------- the three tabs
new Session('live', {
  vad: 'manual',
  buildOpts: () => ({ extended_thinking: $('ext-thinking').checked }),
});

new Session('translate', {
  vad: 'manual',
  buildOpts: () => ({ target: $('tgt-lang').value }),
});

new Session('captions', {
  vad: 'stream',
  buildOpts: () => ({
    smart: $('smart-mode').checked,
    vocabulary: $('vocabulary').value,
  }),
  onEvent: (msg, s) => {
    if (msg.type === 'caption') {
      if (!msg.final) {
        $('interim-captions').textContent = msg.text;
        return;
      }
      $('interim-captions').textContent = '';
      const row = s.log('you', 'caption', msg.text, msg.ms, msg.id);
      const badge = document.createElement('span');
      badge.className = 'badge pending';
      badge.textContent = msg.lang || 'detecting…';
      row.insertBefore(badge, row.firstChild);
    }
    if (msg.type === 'lang') {
      const row = document.getElementById(msg.id);
      const badge = row && row.querySelector('.badge');
      if (badge) { badge.textContent = msg.lang; badge.classList.remove('pending'); }
    }
  },
});
