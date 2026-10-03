"""Hybrid football commentator.

  eyes   gemini-3.8-flash on real 4s video chunks -> structured events
  log    running match state (teams, score, shots, recent play)
  voice  a writer model + gemini-3.8-flash-tts speak from the log in a designed
         voice; the voice never sees the video

The browser only plays the video and plays audio; it sends no frames. The server
reads demo_video.mp4 directly, so it can analyse AHEAD of playback. That head
start (PRE_ROLL) is what makes the commentary land on the action instead of
trailing it.

Run:  ../.venv/bin/python server.py
"""
import asyncio
import json
import pathlib
import time
import traceback

from aiohttp import web, WSMsgType

from common import (CHUNK_SECONDS, CLIPS, DEFAULT_VIDEO, EYES_MODEL, ROOT,
                    TTS_MODEL, VIDEOS, VOICE, _cache, make_client,
                    video_duration, video_path)
from pipeline import Eyes
from voice import TtsVoice

HERE = pathlib.Path(__file__).resolve().parent
STATIC = HERE / "static"

# Playback starts this many seconds after Start. The first chunk takes ~5 s to
# analyse, its line ~2.5 s to write and ~2.8 s to first audio; later lines are
# prepared while earlier chunks play, so only the first one needs this head
# start.
PRE_ROLL_S = 10.5
CONCURRENCY = 3
LEAD_S = 0.1              # start a line this long before the event it calls
MAX_BACKLOG_S = 0.3       # let the previous line (nearly) finish first
STALE_S = 1.5             # drop a routine line this far past its chunk's end

client = make_client()


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
    await ws.prepare(request)
    lock = asyncio.Lock()

    async def send(obj):
        if not ws.closed:
            async with lock:
                await ws.send_str(json.dumps(obj))

    async def send_audio(data: bytes):
        if not ws.closed:
            async with lock:
                await ws.send_bytes(data)

    clip = DEFAULT_VIDEO
    chosen_voice = VOICE
    async for msg in ws:
        if msg.type == WSMsgType.TEXT:
            payload = json.loads(msg.data)
            if payload.get("type") == "start":
                clip = payload.get("clip", DEFAULT_VIDEO)
                chosen_voice = (payload.get("voice") or VOICE).strip()
                break
    else:
        return ws

    src = video_path(clip)
    duration = video_duration(src)
    pre_roll = PRE_ROLL_S
    t0 = time.perf_counter()

    def ms():
        return round((time.perf_counter() - t0) * 1000)

    # The browser reports its real playhead a few times a second. Until the
    # first report (and for clients that never send one), assume playback
    # started exactly at the end of the pre-roll.
    playhead = {"t": None, "at": 0.0}

    def video_time():
        """Where the browser's playhead is, in video seconds."""
        if playhead["t"] is not None:
            return playhead["t"] + (time.perf_counter() - playhead["at"])
        return (time.perf_counter() - t0) - pre_roll

    # The server owns the score. Chunks are analysed in parallel, so a chunk
    # cannot see that its neighbour already reported a goal: the celebration or
    # replay after a goal comes back as a second goal, sometimes for the other
    # team. A goal in a chunk adjacent to the last goal is the same goal.
    # The starting score comes from the first chunk (clips cut from a real
    # broadcast start mid-match, with the score on the scoreboard).
    tally = {"goals": None, "last_goal_end": None, "scorer": None}
    stats = {"chunks": 0, "lines": 0}

    print(f"\n=== session start: {src.name} ({duration:.1f}s) "
          f"eyes={EYES_MODEL} voice={TTS_MODEL}/{chosen_voice} "
          f"pre-roll={pre_roll}s concurrency={CONCURRENCY}")

    async def on_text(text):
        await send({"type": "says", "text": text, "ms": ms(),
                    "video_t": round(max(0.0, video_time()), 1)})

    # The description that designed this voice also shapes what it says.
    persona = next((v["description"] for v in _cache().values()
                    if v["id"] == chosen_voice), None)
    voice = TtsVoice(client, on_audio=send_audio, on_text=on_text,
                     voice_name=chosen_voice, persona=persona)

    try:
        voice.start()
        await send({"type": "status", "state": "connected",
                    "eyes": EYES_MODEL, "engine": TTS_MODEL,
                    "pre_roll": pre_roll, "clip": src.name,
                    "duration": round(duration, 1),
                    "voice": chosen_voice, "ms": ms()})
        print(f"[{ms():6d} ms] voice ready")

        # ---- eyes ---------------------------------------------------
        def settle_goals(c0, c1, analysis):
            """Dedupe goals across chunks and overwrite the per-chunk score."""
            goals = [e for e in analysis.events if e.kind == "goal"]
            teams = (analysis.team_a, analysis.team_b)

            def scorer_of(goal):
                if goal.team in teams:
                    return goal.team
                cel = [e.team for e in analysis.events
                       if e.kind == "celebration" and e.team in teams]
                return cel[0] if cel else None

            if tally["goals"] is None:
                # First chunk: trust its score, which already includes any
                # goal it saw.
                tally["goals"] = [analysis.score_a, analysis.score_b]
                if goals:
                    tally["scorer"] = scorer_of(goals[0])
                    tally["last_goal_end"] = c1
                return

            last = tally["last_goal_end"]
            same_goal = last is not None and c0 - last <= CHUNK_SECONDS + 0.5
            if goals and not same_goal:
                scorer = scorer_of(goals[0])
                if scorer is not None:
                    tally["goals"][teams.index(scorer)] += 1
                    tally["scorer"] = scorer
                goals = goals[1:]
                tally["last_goal_end"] = c1
            elif same_goal and (goals or any(e.kind in ("celebration", "replay")
                                             for e in analysis.events)):
                tally["last_goal_end"] = c1
                if goals:
                    print(f"  [{c0:.0f}-{c1:.0f}s] repeat goal report "
                          f"dropped (same goal as before)")
                    analysis.summary = (f"Aftermath of the goal by "
                                        f"{tally['scorer']}.")
            for e in goals:
                e.kind, e.team = "celebration", tally["scorer"] or e.team
                e.description = "players react to the goal just scored"
            analysis.score_a, analysis.score_b = tally["goals"]

        async def on_chunk(i, c0, c1, analysis, took_ms):
            stats["chunks"] += 1
            settle_goals(c0, c1, analysis)
            voice.prepare(c0, c1, analysis)   # write + synthesise now
            ahead = c1 - max(0.0, video_time())
            print(f"[{ms():6d} ms] chunk {i} [{c0:.0f}-{c1:.0f}s] "
                  f"{took_ms:.0f} ms, {ahead:+.1f}s ahead of playback: "
                  f"{analysis.summary[:60]}")
            await send({"type": "log", "i": i, "t0": c0, "t1": c1,
                        "summary": analysis.summary,
                        "events": [e.model_dump() for e in analysis.events],
                        "score": [analysis.score_a, analysis.score_b],
                        "teams": [analysis.team_a, analysis.team_b],
                        "took_ms": round(took_ms), "ms": ms()})

        eyes = Eyes(client, concurrency=CONCURRENCY, on_chunk=on_chunk, src=src)
        eyes_task = asyncio.create_task(eyes.run(duration))

        # ---- tell the browser when to start rolling ------------------
        async def start_playback():
            await asyncio.sleep(pre_roll)
            print(f"[{ms():6d} ms] playback starts now")
            await send({"type": "play", "ms": ms()})

        play_task = asyncio.create_task(start_playback())

        # ---- the director: decide what the voice says, and when ------
        async def direct():
            # Lines arrive in match order, already written and synthesising.
            # Each plays when the video reaches the event it calls (not the
            # start of its chunk: a goal late in a chunk was being called up
            # to 4 s early) and the previous line has finished. A line that is hopelessly late is dropped
            # rather than describing play the viewer saw seconds ago,
            # unless it is a goal, shot or save.
            while True:
                line = await voice.lines.get()
                while (video_time() < line.at - LEAD_S
                       or voice.backlog() > MAX_BACKLOG_S):
                    await asyncio.sleep(0.05)
                await line.ready.wait()
                late = video_time() - line.c1
                if late > STALE_S and not line.important:
                    print(f"[{ms():6d} ms] dropped stale line for "
                          f"[{line.c0:.0f}-{line.c1:.0f}s] ({late:.1f}s late)")
                    continue
                if line.text is None:
                    continue
                stats["lines"] += 1
                print(f"[{ms():6d} ms] -> voice: [{line.c0:.0f}-{line.c1:.0f}s] "
                      f"event at {line.at:.1f}s, playing at video "
                      f"t={video_time():.1f}s: {line.text}")
                await voice.play(line)

        async def listen():
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    m = json.loads(msg.data)
                    if m.get("type") == "playhead":
                        playhead["t"] = float(m["t"])
                        playhead["at"] = time.perf_counter()

        tasks = [asyncio.create_task(direct()), asyncio.create_task(listen())]

        # run until the video finishes plus a drain for the last line
        await asyncio.sleep(pre_roll + duration + 6)
        for t in tasks + [eyes_task, play_task]:
            t.cancel()
        voice.stop()

    except asyncio.CancelledError:
        raise
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        print(f"[{ms():6d} ms] ERROR {detail}")
        traceback.print_exc()
        await send({"type": "error", "detail": detail, "ms": ms()})

    spoken = voice.audio_bytes / 2 / 24000
    summary = (f"{stats['chunks']} chunks analysed, {stats['lines']} commentary "
               f"lines, {spoken:.1f}s of speech, {voice.truncations} truncations")
    print(f"=== session end: {summary}\n")
    await send({"type": "closed", "summary": summary, "ms": ms()})
    if not ws.closed:
        await ws.close()
    return ws


async def voices(request):
    """The designed commentators saved in this project folder."""
    designed = [{"id": v["id"], "name": v.get("name") or v["description"][:44],
                 "description": v["description"]}
                for v in _cache().values()]
    return web.json_response({"default": VOICE, "designed": designed})


async def index(request):
    return web.FileResponse(STATIC / "index.html")


async def video(request):
    path = video_path(request.match_info.get("key", DEFAULT_VIDEO))
    if not path.exists():
        raise web.HTTPNotFound(text=f"{path.name} not found in the project root")
    return web.FileResponse(path)


async def videos(request):
    return web.json_response({
        "default": DEFAULT_VIDEO,
        "videos": [{"key": k, "name": label, "available": path.exists()}
                   for k, (path, label) in CLIPS.items()],
    })


def main():
    missing = [v.name for v in VIDEOS.values() if not v.exists()]
    if len(missing) == len(VIDEOS):
        raise SystemExit(f"no videos found in {ROOT}: {missing}")
    if missing:
        print(f"  (missing, will not be selectable: {', '.join(missing)})")
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/video/{key}", video)
    app.router.add_get("/videos", videos)
    app.router.add_get("/voices", voices)
    app.router.add_get("/ws/commentary", ws_handler)
    app.router.add_static("/static/", STATIC)
    print("\n  Hybrid football commentator -> http://localhost:8001\n")
    web.run_app(app, host="127.0.0.1", port=8001, print=None)


if __name__ == "__main__":
    main()
