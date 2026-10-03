const $ = (id) => document.getElementById(id);
const video = $('video');
const PLAYBACK_RATE = 24000;   // TTS audio comes back at 24 kHz

// Chunks arrive faster than real time; schedule each to start where the last
// one ends so playback is gapless and never overlaps.
class Player {
  constructor(){ this.ctx=null; this.playAt=0; this.live=new Set(); this.total=0; }
  _ensure(){
    if(!this.ctx||this.ctx.state==='closed'){
      this.ctx=new AudioContext({sampleRate:PLAYBACK_RATE}); this.playAt=0;
    }
    if(this.ctx.state==='suspended') this.ctx.resume();
    return this.ctx;
  }
  push(ab){
    const ctx=this._ensure(); const pcm=new Int16Array(ab);
    if(!pcm.length) return;
    this.total+=pcm.length/PLAYBACK_RATE;
    const buf=ctx.createBuffer(1,pcm.length,PLAYBACK_RATE);
    const ch=buf.getChannelData(0);
    for(let i=0;i<pcm.length;i++) ch[i]=pcm[i]/32768;
    const src=ctx.createBufferSource(); src.buffer=buf; src.connect(ctx.destination);
    this.playAt=Math.max(this.playAt,ctx.currentTime+0.08);
    src.start(this.playAt); this.playAt+=buf.duration;
    this.live.add(src); src.onended=()=>this.live.delete(src);
  }
  get queuedMs(){ return this.ctx?Math.max(0,(this.playAt-this.ctx.currentTime)*1000):0; }
}

const player = new Player();
let ws=null, statsTimer=null, chunks=0;

// Clip picker. The pipeline analyses server-side, so the chosen key goes in the
// start message as well as into the <video> src.
async function loadClips(){
  const { videos, default: def } = await (await fetch('/videos')).json();
  const sel = $('clip');
  for(const v of videos){
    if(!v.available) continue;
    const o=document.createElement('option');
    o.value=v.key; o.textContent=v.name;
    if(v.key===def) o.selected=true;
    sel.appendChild(o);
  }
  setClip(sel.value);
  sel.onchange=()=>setClip(sel.value);
}
function setClip(key){ video.src=`/video/${key}`; video.load(); }
loadClips();

// ---------------------------------------------------------------- commentators
// Every commentator is a Voice Design voice, spoken by gemini-3.8-flash-tts.
async function loadVoices(selectId = 'voice'){
  const sel = document.getElementById(selectId);
  const { designed, default: def } = await (await fetch('/voices')).json();
  sel.innerHTML = '';
  for(const d of designed){
    const o = document.createElement('option');
    o.value = d.id; o.textContent = d.name; o.title = d.description;
    if(d.id === def) o.selected = true;
    sel.appendChild(o);
  }
}

loadVoices();


let openLine=null;
function say(text, videoT){
  if(!openLine){
    openLine=document.createElement('div');
    openLine.className='line';
    const t=document.createElement('span'); t.className='t';
    t.textContent=`${videoT.toFixed(1)}s`;
    openLine.appendChild(t); openLine.appendChild(document.createElement('span'));
    $('transcript').appendChild(openLine);
  }
  openLine.lastChild.textContent+=text;
  $('transcript').scrollTop=$('transcript').scrollHeight;
}
function note(text, cls='sys'){
  openLine=null;
  const d=document.createElement('div'); d.className=`line ${cls}`; d.textContent=text;
  $('transcript').appendChild(d);
}

function addChunk(m){
  chunks++;
  const d=document.createElement('div'); d.className='chunk';
  const hd=document.createElement('div'); hd.className='hd';
  hd.innerHTML=`<span>${m.t0.toFixed(0)}&ndash;${m.t1.toFixed(0)}s</span>`+
               `<span>${m.took_ms} ms</span>`;
  d.appendChild(hd);
  const s=document.createElement('div'); s.className='sum'; s.textContent=m.summary;
  d.appendChild(s);
  for(const e of m.events){
    const r=document.createElement('div');
    r.className='ev'+(e.kind==='goal'?' goal':'');
    const k=document.createElement('span'); k.className='k'; k.textContent=e.kind;
    const t=document.createElement('span'); t.textContent=`${e.team} — ${e.description}`;
    r.appendChild(k); r.appendChild(t); d.appendChild(r);
  }
  $('log').appendChild(d);
  $('log').scrollTop=$('log').scrollHeight;
  $('score').textContent=`${m.teams[0]} ${m.score[0]} – ${m.score[1]} ${m.teams[1]}`;
}

function start(){
  $('start').disabled=true;
  $('transcript').innerHTML=''; $('log').innerHTML=''; chunks=0;
  $('status').textContent='connecting…';
  video.pause(); video.currentTime=0;

  ws=new WebSocket(`ws://${location.host}/ws/commentary`);
  ws.binaryType='arraybuffer';

  ws.onopen=()=>{
    ws.send(JSON.stringify({type:'start', clip:$('clip').value, voice:$('voice').value}));
    statsTimer=setInterval(()=>{
      // The server times each line off the real playhead, not its own clock.
      if(!video.paused && ws.readyState===1)
        ws.send(JSON.stringify({type:'playhead', t: video.currentTime}));
      $('stats').textContent=
        `video t=${video.currentTime.toFixed(1)}s · chunks analysed ${chunks} · `+
        `audio queued ${Math.round(player.queuedMs)} ms · spoken ${player.total.toFixed(1)}s`;
    },200);
  };

  ws.onmessage=async (e)=>{
    if(e.data instanceof ArrayBuffer){ player.push(e.data); return; }
    const m=JSON.parse(e.data);
    if(m.type==='says') say(m.text, m.video_t);
    else if(m.type==='log'){ addChunk(m); openLine=null; }
    else if(m.type==='play'){
      // Server decides the kickoff moment: the eyes have a head start by now.
      try{ await video.play(); }catch(err){ note(`video.play() failed: ${err}`,'err'); }
      $('status').textContent='live';
      note('playback started — eyes are ahead of the picture');
    }
    else if(m.type==='status'&&m.state==='connected'){
      $('status').textContent=`analysing… playback in ${m.pre_roll}s`;
      note(`${m.clip} (${m.duration}s) · eyes ${m.eyes} · voice ${m.voice} on ${m.engine} · pre-roll ${m.pre_roll}s`);
    }
    else if(m.type==='error') note(m.detail,'err');
    else if(m.type==='closed'){ $('status').textContent='finished'; note(m.summary); }
  };

  ws.onerror=()=>note('WebSocket failed — is server.py running?','err');
  ws.onclose=()=>{ clearInterval(statsTimer); $('start').disabled=false; };
}

$('start').onclick=start;
