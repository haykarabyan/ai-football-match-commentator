"""Live AI football commentator.

  browser   plays the clip (or a shared screen/tab) and streams JPEG frames,
            like AI Studio's "Share Screen"
  live      ONE gemini-3.8-live session watches the frames and writes the lines
  voice     native: Live speaks itself in a stock voice
            designed: each line -> gemini-3.8-flash-tts in a Voice Design voice

The browser feeds the model from a hidden copy of the clip that runs LOOKAHEAD
seconds ahead of the one on screen, so the commentary lands on the action you
see instead of trailing it. A shared screen is live, so it gets no look-ahead.

Run:  ../.venv/bin/python server.py   ->  http://localhost:8002
"""
import asyncio
import json
import re
import time

from aiohttp import web, WSMsgType

from common import (CLIPS, DEFAULT_CLIP, DEFAULT_DESIGNED, DEFAULT_PREBUILT,
                    HERE, PREBUILT_VOICES, clip,
                    commentary_preview, design_voice, designed_voices,
                    make_client, persona_of)
from live import CUE, KICKOFF, LiveCommentator, build_system_prompt
from speaker import DesignedSpeaker, NativeSpeaker

STATIC = HERE / "static"
PORT = 8002

# When to ask for the next line: once the audio still to be heard (queued in
# the browser, plus any line still in TTS) drops below this.
#   native    Live starts speaking ~1.3 s after a cue
#   designed  the line's text is complete ~1.4-2.1 s after a cue and TTS
#             adds ~2.8 s, so the next line is asked for ~4.5 s before the
#             audio runs out, while the current one is still in TTS. Measured
#             medians replace the starting value.
PREFETCH_S = {"native": 1.0, "designed": 4.3}
GAP_S = 0.3             # designed: aim for this much air between lines
# Starting look-ahead: how far the model's copy of the clip runs ahead of the
# visible one, i.e. cue -> heard. The page then tunes it from what it measures.
#   native ~1.4 s to first audio; designed ~1.8 s (Live) + ~2.8 s (TTS).
LOOKAHEAD_S = {"native": 1.5, "designed": 5.0}
MIN_GAP_S = 0.6         # never cue more often than this
CUE_TIMEOUT_S = 6.0     # the model may stay silent on a cue; re-cue after this
KICKOFF_AFTER_FRAMES = 3

REPEAT_BACKOFF_S = 2.5  # after a repeated line is dropped, wait this much longer
ANSWER_TIMEOUT_S = 8.0  # no answer this long after the viewer stops: carry on
FALSE_ALARM_S = 2.0     # VAD heard "speech" but no words by this long after it
                        # ended (a cough, a door): carry on
MIC = 0x01              # first byte of a binary message carrying mic audio
                        # (frames are JPEGs, which start with 0xFF)

client = make_client()


def _words(text):
    """Content words of a line, for spotting repeats."""
    return {w.rstrip("s") for w in re.findall(r"[^\W\d_]+", text.lower().replace("'s", ""))
            if len(w) > 3}


def is_repeat(text, recent):
    """True if a line mostly restates one of the recent ones.

    "The referee brandishes a yellow card." repeats "Now, the referee flashes
    a yellow card." (3 of 4 words); "Thomas Müller goes into the referee's
    notebook." does not.
    """
    w = _words(text)
    for prev in recent:
        p = _words(prev)
        if w and p and len(w & p) / min(len(w), len(p)) >= 0.6:
            return True
    return False


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024)
    await ws.prepare(request)
    lock = asyncio.Lock()
    t0 = time.perf_counter()

    def log(text):
        t = time.perf_counter() - t0
        print(f"  {int(t // 60):02d}:{t % 60:04.1f}  {text}", flush=True)

    async def send(msg: dict | bytes):
        """A JSON message, or raw PCM audio."""
        if not ws.closed:
            async with lock:
                if isinstance(msg, bytes):
                    await ws.send_bytes(msg)
                else:
                    await ws.send_str(json.dumps(msg))

    # ---- wait for the start message --------------------------------------
    start = None
    async for msg in ws:
        if msg.type == WSMsgType.TEXT:
            m = json.loads(msg.data)
            if m.get("type") == "start":
                start = m
                break
    if start is None:
        return ws

    mode = "designed" if start.get("mode") == "designed" else "native"
    source = start.get("source", "clip")
    clip_key = start.get("clip") or DEFAULT_CLIP
    voice = (start.get("voice") or "").strip() or (
        DEFAULT_DESIGNED if mode == "designed" else DEFAULT_PREBUILT)
    if mode == "native" and voice not in PREBUILT_VOICES:
        voice = DEFAULT_PREBUILT
    path, _, clip_context = clip(clip_key)
    context = start.get("context") or (clip_context if source == "clip" else None)
    persona = persona_of(voice) if mode == "designed" else None

    stats = {"lines": 0, "questions": 0}
    where = "shared screen" if source == "screen" else path.name
    print(f"\n▶ {where} · {'designed voice' if mode == 'designed' else 'stock voice'} {voice}")

    # ---- speaker ----------------------------------------------------------
    pending = {"cue_t": None, "hold_until": 0.0}

    # The page reports where its frame source is (clip: the model's copy's
    # currentTime; screen: seconds since start). Each cue is stamped with that
    # time, and the page compares it with when the line is actually heard.
    clock = {"t": None, "at": 0.0}

    def source_time():
        if clock["t"] is None:
            return None
        return round(clock["t"] + time.perf_counter() - clock["at"], 2)

    async def on_spoken(text, cue_t, kind):
        await send({"type": "spoken", "text": text, "cue_t": cue_t, "kind": kind})

    if mode == "designed":
        speaker = DesignedSpeaker(client, voice, on_audio=send,
                                  on_spoken=on_spoken, log=log)
    else:
        speaker = NativeSpeaker(on_audio=send)

    # ---- live -------------------------------------------------------------
    first_audio = {"seen": False}    # native: mark where each line's audio begins

    # A viewer talking to the commentator. Live's VAD hears them and stops
    # generating; the server stops everything queued, waits for the answer,
    # plays it, then the director carries on.
    q = {"active": False, "talking": False, "heard_at": 0.0, "ended_at": None,
         "final": "", "interim": "", "answered_at": 0.0}

    async def begin_question():
        q.update(active=True, heard_at=time.perf_counter(), ended_at=None,
                 final="", interim="")
        stats["questions"] += 1
        speaker.flush()
        live.expect_answer()
        await send({"type": "interrupt"})

    async def on_voice(kind):
        now = time.perf_counter()
        if kind == "start":
            q["talking"], q["heard_at"] = True, now
            if not q["active"]:
                await begin_question()
        else:
            q["talking"], q["heard_at"] = False, now
            if q["active"]:
                q["ended_at"] = now

    async def on_user(text, final):
        if not q["active"]:
            if time.perf_counter() - q["answered_at"] < 3.0:
                return                  # late transcript of the question just answered
            await begin_question()      # no voice activity signal came first
        q["heard_at"] = time.perf_counter()
        if final:
            q["final"] = (q["final"] + " " + text).strip()
        else:
            q["interim"] = text                        # cumulative
        await send({"type": "question", "text": q["final"] or q["interim"]})

    async def on_answer_start():
        await send({"type": "answer_start"})

    async def on_answer(text):
        heard = q["final"] or q["interim"]
        log(f"Q  {heard}")
        log(f"A  {text}")
        said.append(f"(to a viewer: {text})")
        await send({"type": "answer", "text": text, "question": heard})
        if mode == "designed":
            await speaker.line(text, None, kind="answer")
        q["active"], q["answered_at"] = False, time.perf_counter()

    async def on_audio(pcm, kind):
        if kind == "answer":
            if mode == "native":
                await speaker.audio(pcm)
            return
        if q["active"]:
            return                     # commentary from before the question
        if mode == "native" and not first_audio["seen"] and live.cued_at:
            first_audio["seen"] = True
            await send({"type": "spoken", "text": None, "cue_t": pending["cue_t"]})
        await speaker.audio(pcm)

    async def on_text(frag, kind):
        # only the stock voice shows words as they're spoken; designed lines
        # arrive whole with their audio ("spoken")
        if mode != "native" or (kind == "line" and q["active"]):
            return
        await send({"type": "text", "text": frag, "kind": kind})

    said = []            # lines actually passed on to be spoken

    async def on_line(text):
        if q["active"]:
            return                     # written before the viewer cut in
        if mode == "designed" and is_repeat(text, said[-3:]):
            pending["hold_until"] = time.perf_counter() + REPEAT_BACKOFF_S
            return
        stats["lines"] += 1
        log(text)
        said.append(text)
        await send({"type": "line", "text": text})
        await speaker.line(text, pending["cue_t"])

    live = LiveCommentator(
        client, system_prompt=build_system_prompt(context, persona),
        on_line=on_line, on_text=on_text, on_audio=on_audio,
        on_user=on_user, on_answer=on_answer, on_answer_start=on_answer_start,
        on_voice=on_voice,
        # Live always produces audio; in designed mode the speaker drops it
        # and speaks the transcript instead.
        voice_name=voice if mode == "native" else DEFAULT_PREBUILT,
        early_lines=(mode == "designed"), log=log)

    live_task = asyncio.create_task(live.run())

    async def cue(text=None):
        pending["cue_t"] = source_time()
        first_audio["seen"] = False
        if text is None:
            # Quote everything said so far: in designed mode Live's turns get
            # cut off by the next cue, so don't rely on it remembering them.
            # A 3 min clip is ~40 lines, ~400 words per cue.
            # As one running transcript, not a list: a list invites a list
            # of separate statements back.
            text = CUE
            if said:
                text += "\nYour commentary so far: " + " ".join(said)
        await live.cue(text)

    # ---- director: keep the commentary flowing ----------------------------
    async def direct():
        def prefetch():
            # designed: a cue is heard after Live writes the line plus TTS, so
            # ask that long before the audio runs out (measured medians).
            if mode == "designed" and live.line_ms:
                recent = sorted(live.line_ms[-5:])
                return recent[len(recent) // 2] / 1000 + speaker.expected_tts_s - GAP_S
            return PREFETCH_S[mode]

        while live.frames < KICKOFF_AFTER_FRAMES:
            await asyncio.sleep(0.05)
        await live.connected.wait()
        await cue(KICKOFF)
        while True:
            await asyncio.sleep(0.05)
            now = time.perf_counter()
            if not live.connected.is_set():
                continue
            if live.last_frame_at is None or now - live.last_frame_at > 2.0:
                continue                # paused / ended: don't talk over nothing
            if q["active"]:
                false_alarm = (not (q["final"] or q["interim"]) and q["ended_at"] is not None
                               and now - q["ended_at"] > FALSE_ALARM_S)
                no_answer = not q["talking"] and now - q["heard_at"] > ANSWER_TIMEOUT_S
                if false_alarm or no_answer:
                    q["active"], live.answer_pending = False, False
                    await send({"type": "resume"})
                continue
            if q["talking"] or now - q["heard_at"] < 1.0:
                continue                # someone is talking: don't cue over them
            since = now - (live.cued_at or 0)
            if live.generating:
                if since > CUE_TIMEOUT_S:
                    live.generating = False
                continue
            if since < MIN_GAP_S or speaker.busy() or now < pending["hold_until"]:
                continue
            if speaker.backlog() <= prefetch():
                await cue()

    director = asyncio.create_task(direct())

    await send({"type": "status", "state": "connected", "lookahead": LOOKAHEAD_S[mode]})

    # ---- frames in --------------------------------------------------------
    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                if msg.data[:1] == bytes([MIC]):
                    await live.send_audio(msg.data[1:])
                else:
                    await live.send_frame(msg.data)
            elif msg.type == WSMsgType.TEXT:
                m = json.loads(msg.data)
                if m.get("type") == "stop":
                    break
                if m.get("type") == "clock":
                    clock["t"], clock["at"] = float(m["t"]), time.perf_counter()
            elif msg.type == WSMsgType.ERROR:
                break
    except Exception as exc:
        log(f"error: {type(exc).__name__}: {exc}")
        await send({"type": "error", "detail": "Something went wrong. Press Start to try again."})
    finally:
        director.cancel()
        await live.close()
        live_task.cancel()
        speaker.stop()

    print(f"■ {stats['lines']} lines, {stats['questions']} questions\n")
    if not ws.closed:
        await ws.close()
    return ws


# ---- plain HTTP -------------------------------------------------------------
async def index(request):
    return web.FileResponse(STATIC / "index.html")


async def video(request):
    path = clip(request.match_info["key"])[0]
    if not path.exists():
        raise web.HTTPNotFound(text=f"{path.name} not found in the project root")
    return web.FileResponse(path)


async def clips(request):
    return web.json_response({
        "default": DEFAULT_CLIP,
        "clips": [{"key": k, "name": label, "context": ctx, "available": p.exists()}
                  for k, (p, label, ctx) in CLIPS.items()],
    })


async def voices(request):
    return web.json_response({
        "prebuilt": PREBUILT_VOICES,
        "designed": designed_voices(), "default_designed": DEFAULT_DESIGNED,
    })


def _error(exc: Exception) -> web.Response:
    return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=400)


async def create_voice(request):
    """Voice Design: prose description -> stored voice id (+ a spoken preview)."""
    body = await request.json()
    try:
        made = await asyncio.to_thread(
            design_voice, client, body.get("description", ""),
            body.get("name", "").strip(), body.get("gender", "male"))
        made["preview_b64"] = await commentary_preview(client, made["id"])
    except Exception as exc:
        return _error(exc)
    return web.json_response(made)


async def preview_voice(request):
    try:
        preview = await commentary_preview(client, request.match_info["id"])
    except Exception as exc:
        return _error(exc)
    return web.json_response({"preview_b64": preview})


def main():
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/video/{key}", video)
    app.router.add_get("/clips", clips)
    app.router.add_get("/voices", voices)
    app.router.add_post("/voices", create_voice)
    app.router.add_get("/voices/{id}/preview", preview_voice)
    app.router.add_get("/ws/commentary", ws_handler)
    app.router.add_static("/static/", STATIC)
    print(f"\n  Live AI commentary  →  http://localhost:{PORT}\n")
    web.run_app(app, host="127.0.0.1", port=PORT, print=None)


if __name__ == "__main__":
    main()
