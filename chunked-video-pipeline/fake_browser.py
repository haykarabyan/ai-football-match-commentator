"""Stand in for the browser: connect, log everything, save the commentary audio."""
import asyncio, json, time
import aiohttp
from common import OUT, VOICE, write_wav

t0 = time.perf_counter()
def log(*a): print(f"[{(time.perf_counter()-t0):6.1f}s]", *a, flush=True)

async def main():
    pcm = bytearray(); said = []; played_at = None
    async with aiohttp.ClientSession() as http:
        async with http.ws_connect("http://localhost:8001/ws/commentary") as ws:
            import sys
            clip = sys.argv[1] if len(sys.argv) > 1 else "final"
            voice = sys.argv[2] if len(sys.argv) > 2 else VOICE
            await ws.send_str(json.dumps({"type": "start", "clip": clip, "voice": voice}))
            while True:
                try:
                    m = await asyncio.wait_for(ws.receive(), timeout=60)
                except asyncio.TimeoutError:
                    log("timeout"); break
                if m.type == aiohttp.WSMsgType.BINARY:
                    pcm.extend(m.data)
                elif m.type == aiohttp.WSMsgType.TEXT:
                    e = json.loads(m.data)
                    if e["type"] == "says":
                        said.append((e["video_t"], e["text"]))
                    elif e["type"] == "log":
                        goals = [v for v in e["events"] if v["kind"] == "goal"]
                        log(f"EYES  [{e['t0']:.0f}-{e['t1']:.0f}s] {e['took_ms']}ms  "
                            f"{e['summary'][:60]}" + ("   <-- GOAL" if goals else ""))
                    elif e["type"] == "play":
                        played_at = time.perf_counter() - t0
                        log("PLAYBACK STARTS")
                    elif e["type"] == "error":
                        log("ERROR", e["detail"])
                    elif e["type"] == "closed":
                        log("closed:", e["summary"]); break
                elif m.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                    break

    path = write_wav(OUT / "hybrid_run.wav", bytes(pcm))
    print(f"\n=== {len(pcm)/2/24000:.1f}s of commentary -> {path}")
    print("=== transcript (video time it was spoken at):")
    cur = None
    for t, txt in said:
        if cur is None or abs(t - cur) > 0.05:
            print(f"\n  {t:5.1f}s  {txt}", end="")
            cur = t
        else:
            print(txt, end="")
    print()

asyncio.run(main())
