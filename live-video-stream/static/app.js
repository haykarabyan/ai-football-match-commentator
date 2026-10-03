const $ = (id) => document.getElementById(id);
const video = $('video');      // what the viewer watches
const feed = $('feed');        // what the model watches: the same clip, a few seconds ahead
const RATE = 24000;            // Live and TTS both return 24 kHz PCM
const MAX_SIDE = 768;          // recommended frame size for Live
const FPS = 2;                 // 1 fps misses goals; 2 catches them
const MIC = 0x01;              // first byte of a mic message (frames are JPEGs: 0xFF)

// ------------------------------------------------------------------ audio out
// Chunks arrive faster than real time; each is scheduled where the last ends.
class Player {
  constructor(){ this.ctx=null; this.playAt=0; this.live=new Set(); }
  ensure(){
    if(!this.ctx||this.ctx.state==='closed'){ this.ctx=new AudioContext({sampleRate:RATE}); this.playAt=0; }
    if(this.ctx.state==='suspended') this.ctx.resume();
    return this.ctx;
  }
  push(ab){                    // returns seconds until this chunk is heard
    const ctx=this.ensure(), pcm=new Int16Array(ab);
    if(!pcm.length) return 0;
    const buf=ctx.createBuffer(1,pcm.length,RATE), ch=buf.getChannelData(0);
    for(let i=0;i<pcm.length;i++) ch[i]=pcm[i]/32768;
    const src=ctx.createBufferSource(); src.buffer=buf; src.connect(ctx.destination);
    this.playAt=Math.max(this.playAt,ctx.currentTime+0.05);
    src.start(this.playAt);
    this.live.add(src); src.onended=()=>this.live.delete(src);
    const heardIn=this.playAt-ctx.currentTime+(ctx.outputLatency||ctx.baseLatency||0);
    this.playAt+=buf.duration;
    return heardIn;
  }
  flush(){ for(const s of this.live){ try{ s.stop(); }catch{} } this.live.clear(); this.playAt=0; }
}
const player = new Player();

// ------------------------------------------------------------------ state
let voices=null, clips=[];
let ws=null, running=false, screenStream=null, startedAt=0;
let captureTimer=null, clockTimer=null, syncTimer=null;
let lookahead=0, videoStarted=false, holding=false;
let measured=[], pendingCue=null;          // auto look-ahead
let ignoreAudio=false, qEl=null, aEl=null; // viewer questions
let mic=null;

const isScreen = () => $('source').value==='screen';
const chosen = () => { const [mode,...v]=$('voice').value.split(':'); return {mode, voice:v.join(':')}; };

function setStatus(text, state){ const s=$('status'); s.textContent=text; s.dataset.state=state; }

// ------------------------------------------------------------------ setup
async function init(){
  const c = await (await fetch('/clips')).json();
  clips = c.clips.filter(x=>x.available);
  const sel=$('source');
  const group=(label)=>{ const g=document.createElement('optgroup'); g.label=label; sel.appendChild(g); return g; };
  const final=group('2020 Champions League final · PSG v Bayern');
  const other=group('More');
  for(const x of clips){
    const o=new Option(x.name, x.key, false, x.key===c.default);
    (x.context.includes('Champions League') ? final : other).appendChild(o);
  }
  other.appendChild(new Option('Share a screen or tab (e.g. a match on YouTube)', 'screen'));
  if(!final.children.length) final.remove();   // no clips downloaded yet
  sel.onchange=onSource;

  voices = await (await fetch('/voices')).json();
  fillVoices(voices.default_designed);
  onSource();
}

function fillVoices(selected){
  const sel=$('voice'); sel.innerHTML='';
  const d=document.createElement('optgroup'); d.label='Designed voices';
  for(const v of voices.designed){
    const o=new Option(v.name, `designed:${v.id}`, false, v.id===selected); o.title=v.description; d.appendChild(o);
  }
  const n=document.createElement('optgroup'); n.label='Stock voices · fastest';
  for(const v of voices.prebuilt) n.appendChild(new Option(v, `native:${v}`));
  sel.append(d, n);
  sel.onchange=()=>{ $('preview').hidden = chosen().mode!=='designed'; };
  sel.onchange();
}

function onSource(){
  const key=$('source').value;
  $('teamsBox').hidden = key!=='screen';
  if(key==='screen'){ video.removeAttribute('src'); video.load(); return; }
  video.srcObject=null; video.src=feed.src=`/video/${key}`;
  video.load(); feed.load();
}

// ------------------------------------------------------------------ transcript
// Commentary flows into paragraphs; a new one starts after a pause or a Q&A.
const FLOW_BREAK_MS = 6000;
let flowEl=null, flowAt=0, openSeg=null, captionTimer=null;

function scrollFeed(){ const t=$('transcript'); t.scrollTop=t.scrollHeight; }
function clearEmpty(){ $('empty')?.remove(); }
function clock(){
  const s = isScreen() ? (performance.now()-startedAt)/1000 : video.currentTime;
  return `${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}`;
}
function seg(){
  clearEmpty();
  const now=performance.now();
  if(!flowEl || now-flowAt>FLOW_BREAK_MS){
    flowEl=document.createElement('div'); flowEl.className='flow';
    const t=document.createElement('span'); t.className='t'; t.textContent=clock();
    flowEl.append(t, document.createElement('div'));
    $('transcript').appendChild(flowEl);
  }
  flowAt=now;
  const body=flowEl.lastChild;
  if(body.childNodes.length) body.appendChild(document.createTextNode(' '));
  const s=document.createElement('span'); s.className='seg'; body.appendChild(s);
  scrollFeed();
  return s;
}
function bubble(kind, text){
  clearEmpty(); flowEl=null;
  const d=document.createElement('div'); d.className=`bubble ${kind}`;
  d.innerHTML=`<small>${kind==='q'?'You':'Commentator'}</small><span></span>`;
  d.lastChild.textContent=text;
  $('transcript').appendChild(d); scrollFeed();
  return d;
}
function note(text, err=false){
  clearEmpty(); flowEl=null;
  const d=document.createElement('div'); d.className='note'+(err?' err':''); d.textContent=text;
  $('transcript').appendChild(d); scrollFeed();
}
function caption(text, answer=false){
  const c=$('caption'); c.textContent=text; c.classList.toggle('answer',answer); c.classList.add('on');
  clearTimeout(captionTimer);
  captionTimer=setTimeout(()=>c.classList.remove('on'), 4000);
}

// ------------------------------------------------------------------ look-ahead
// The model watches a copy of the clip `lookahead` seconds ahead of yours, so
// a line lands on the moment it describes. It is tuned from how long each
// line actually takes to reach your ears.
const sourceTime = () => isScreen() ? (performance.now()-startedAt)/1000 : feed.currentTime;
const median = (a) => { const b=[...a].sort((x,y)=>x-y); return b[Math.floor(b.length/2)]; };
function measure(cueT, heardIn){
  if(isScreen()) return;
  measured.push(sourceTime()+heardIn-cueT); if(measured.length>3) measured.shift();
  lookahead=Math.min(12, Math.max(0.5, median(measured)));
}

// ------------------------------------------------------------------ frames out
const canvas=document.createElement('canvas'), ctx2d=canvas.getContext('2d');
function sendFrame(src){
  if(!ws||ws.readyState!==1||!src.videoWidth||ws.bufferedAmount>2_000_000) return;
  const k=Math.min(1, MAX_SIDE/Math.max(src.videoWidth,src.videoHeight));
  canvas.width=Math.round(src.videoWidth*k); canvas.height=Math.round(src.videoHeight*k);
  ctx2d.drawImage(src,0,0,canvas.width,canvas.height);
  canvas.toBlob(b=>{ if(b&&ws&&ws.readyState===1) ws.send(b); },'image/jpeg',0.7);
}

// ------------------------------------------------------------------ mic in
// Always listening. Live's voice activity detection decides when someone is
// talking to the commentator.
const WORKLET = `
class Mic extends AudioWorkletProcessor {
  constructor(){ super(); this.buf=new Int16Array(1600); this.n=0; }
  process(inputs){
    const ch=inputs[0][0];
    if(ch){ for(let i=0;i<ch.length;i++){
      const v=Math.max(-1,Math.min(1,ch[i]));
      this.buf[this.n++]=v<0?v*32768:v*32767;
      if(this.n===this.buf.length){ this.port.postMessage(this.buf.slice().buffer); this.n=0; }
    }}
    return true;
  }
}
registerProcessor('mic', Mic);`;

async function startMic(){
  try{
    const stream=await navigator.mediaDevices.getUserMedia({audio:{
      echoCancellation:true, noiseSuppression:true, autoGainControl:true, channelCount:1}});
    const ctx=new AudioContext({sampleRate:16000});
    await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET],{type:'text/javascript'})));
    const node=new AudioWorkletNode(ctx,'mic');
    let level=0;
    node.port.onmessage=(e)=>{
      const pcm=new Int16Array(e.data);
      let peak=0; for(let i=0;i<pcm.length;i+=8) peak=Math.max(peak,Math.abs(pcm[i]));
      level=Math.max(peak/32768, level*0.85);
      $('mic').style.setProperty('--lvl', Math.min(1, level*3).toFixed(2));
      if(!ws||ws.readyState!==1) return;
      const out=new Uint8Array(pcm.byteLength+1); out[0]=MIC; out.set(new Uint8Array(e.data),1);
      ws.send(out);
    };
    ctx.createMediaStreamSource(stream).connect(node);
    mic={ctx, stream};
  }catch{
    note('Microphone unavailable. Commentary works, but you can’t ask questions.');
  }
}
function stopMic(){
  if(!mic) return;
  mic.stream.getTracks().forEach(t=>t.stop()); mic.ctx.close(); mic=null;
}

// ------------------------------------------------------------------ run
async function start(){
  player.ensure();
  const screen=isScreen();
  if(screen){
    try{ screenStream=await navigator.mediaDevices.getDisplayMedia({video:{frameRate:10}, audio:false}); }
    catch{ return; }
    video.srcObject=screenStream; await video.play();
    screenStream.getVideoTracks()[0].onended=stop;
  }
  await startMic();

  running=true; measured=[]; pendingCue=null; videoStarted=false; holding=false; ignoreAudio=false;
  $('transcript').querySelectorAll('.flow,.bubble,.note').forEach(n=>n.remove());
  flowEl=null; openSeg=null;
  $('start').textContent='Stop'; $('start').classList.add('stop'); $('start').onclick=stop;
  for(const id of ['source','voice','preview']) $(id).disabled=true;
  setStatus('Warming up…','warming'); $('veil').hidden=screen;

  const {mode, voice}=chosen();
  ws=new WebSocket(`ws://${location.host}/ws/commentary`);
  ws.binaryType='arraybuffer';
  ws.onopen=()=>ws.send(JSON.stringify({
    type:'start', source: screen?'screen':'clip', clip: screen?null:$('source').value,
    mode, voice, context: screen ? ($('context').value.trim()||null) : null,
  }));
  ws.onmessage=(e)=>onMessage(e, mode);
  ws.onerror=()=>note('Can’t reach the server. Is it running?', true);
  ws.onclose=()=>{ if(running) stop(); };
}

function onMessage(e, mode){
  if(e.data instanceof ArrayBuffer){
    if(ignoreAudio) return;                // commentary from before a question
    const heardIn=player.push(e.data);
    if(pendingCue!==null){ measure(pendingCue, heardIn); pendingCue=null; }
    return;
  }
  const m=JSON.parse(e.data);
  switch(m.type){
    case 'status':
      startedAt=performance.now();
      lookahead=isScreen()?0:m.lookahead;
      begin();
      break;
    case 'text':                           // stock voice: words stream in as spoken
      if(mode!=='native') break;
      if(m.kind==='answer'){ if(aEl){ aEl.lastChild.textContent+=m.text; caption(aEl.lastChild.textContent,true); } }
      else{
        if(!openSeg) openSeg=seg();
        openSeg.textContent=(openSeg.textContent+m.text).replace(/^\s+/,'');
        flowAt=performance.now(); caption(openSeg.textContent); scrollFeed();
      }
      break;
    case 'line': openSeg=null; break;
    case 'spoken':                         // designed voice: a line as its audio starts
      if(m.text && m.kind==='answer'){ bubble('a', m.text); caption(m.text,true); }
      else if(m.text){ seg().textContent=m.text; caption(m.text); }
      if(m.cue_t!=null) pendingCue=m.cue_t;
      break;
    case 'interrupt':                      // a viewer started talking
      player.flush(); ignoreAudio=true; pendingCue=null; openSeg=null; aEl=null;
      qEl=bubble('q','…'); $('mic').classList.add('hearing');
      setStatus('Listening…','asking'); $('caption').classList.remove('on');
      break;
    case 'question': if(qEl) qEl.lastChild.textContent=m.text; break;
    case 'answer_start':
      ignoreAudio=false; openSeg=null; $('mic').classList.remove('hearing');
      setStatus('Answering','asking');
      if(mode==='native') aEl=bubble('a','');
      break;
    case 'answer':
      if(qEl && m.question) qEl.lastChild.textContent=m.question;
      qEl=null; setStatus('Live','live');
      break;
    case 'resume':                         // it wasn't a question after all
      ignoreAudio=false; $('mic').classList.remove('hearing');
      if(qEl && qEl.lastChild.textContent==='…') qEl.remove();
      qEl=null; setStatus('Live','live');
      break;
    case 'error': note(m.detail, true); break;
    case 'closed': break;
  }
}

async function begin(){
  clockTimer=setInterval(()=>{             // the server stamps each request with this
    if(ws&&ws.readyState===1) ws.send(JSON.stringify({type:'clock', t: sourceTime()}));
  },200);
  if(mic) $('mic').hidden=false;
  if(isScreen()){
    captureTimer=setInterval(()=>sendFrame(video),1000/FPS);
    goLive();
    return;
  }
  feed.currentTime=0; video.pause(); video.currentTime=0;
  try{ await feed.play(); }catch{ note('Couldn’t start the video.', true); return; }
  captureTimer=setInterval(()=>{ if(!feed.paused&&!feed.ended) sendFrame(feed); },1000/FPS);
  feed.onended=()=>clearInterval(captureTimer);
  setTimeout(async ()=>{
    if(!running) return;
    try{ await video.play(); }catch{}
    videoStarted=true; goLive();
    // Keep your copy `lookahead` behind the model's. If the look-ahead grew,
    // hold the picture for the difference; if it shrank, skip forward.
    syncTimer=setInterval(()=>{
      if(feed.ended||video.ended||holding) return;
      const diff=video.currentTime-(feed.currentTime-lookahead);
      if(diff>0.3){ holding=true; video.pause(); setTimeout(()=>{ holding=false; if(running) video.play(); }, diff*1000); }
      else if(diff<-0.3) video.currentTime=feed.currentTime-lookahead;
    },500);
  }, lookahead*1000);
  video.onended=()=>setTimeout(()=>{ stop(); setStatus('Full time','idle'); },1500);
}

function goLive(){ $('veil').hidden=true; $('liveBadge').hidden=false; setStatus('Live','live'); }

function stop(){
  if(!running) return;
  running=false;
  if(ws&&ws.readyState===1) ws.send(JSON.stringify({type:'stop'}));
  ws=null;
  clearInterval(captureTimer); clearInterval(clockTimer); clearInterval(syncTimer);
  feed.pause(); video.pause(); video.onended=null; feed.onended=null;
  if(screenStream){ screenStream.getTracks().forEach(t=>t.stop()); screenStream=null; }
  stopMic(); player.flush();
  $('veil').hidden=true; $('liveBadge').hidden=true; $('mic').hidden=true;
  $('caption').classList.remove('on');
  $('start').textContent='Start'; $('start').classList.remove('stop'); $('start').onclick=start;
  for(const id of ['source','voice','preview']) $(id).disabled=false;
  setStatus('Ready','idle');
}

$('start').onclick=start;

// ------------------------------------------------------------------ voices
function playWav(b64){ new Audio(`data:audio/wav;base64,${b64}`).play(); }

$('preview').onclick=async ()=>{
  const b=$('preview'); b.disabled=true;
  try{
    const r=await (await fetch(`/voices/${chosen().voice}/preview`)).json();
    if(!r.error) playWav(r.preview_b64);
  } finally { b.disabled=false; }
};

$('design').onclick=async ()=>{
  const description=$('dDesc').value.trim();
  if(!description){ $('dStatus').textContent='Describe the voice first.'; return; }
  $('design').disabled=true; $('dStatus').textContent='Designing… this takes about 20 seconds.';
  try{
    const r=await (await fetch('/voices',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({description, name:$('dName').value.trim(), gender:$('dGender').value})})).json();
    if(r.error){ $('dStatus').textContent='That didn’t work. Try rewording the description.'; return; }
    voices=await (await fetch('/voices')).json();
    fillVoices(r.id);
    $('dStatus').textContent=`“${r.name}” is ready and selected.`;
    playWav(r.preview_b64);
  } finally { $('design').disabled=false; }
};

init();
