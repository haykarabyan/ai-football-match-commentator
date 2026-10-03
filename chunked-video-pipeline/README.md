# Chunked video pipeline

The first working approach. For the final one, see [`../live-video-stream`](../live-video-stream).

Two models with separate jobs:

- **eyes** — `gemini-3.8-flash` watches real 4-second **video** chunks and returns
  structured events (kickoff, cross, shot, goal, celebration…), team by shirt
  colour, and a running score.
- **voice** — `gemini-3.5-flash-lite` writes one line per chunk from that match
  log and `gemini-3.8-flash-tts` speaks it in a Voice Design commentator voice.
  It never sees the video; it only has to talk (see "Voices").

An earlier prototype (not kept) streamed 1 fps stills straight into the
Live session. See "Why not frames into Live" below.

## Setup

```bash
cd /Users/haykarabyan/Projects/personal/tech_europe
.venv/bin/pip install -U google-genai python-dotenv aiohttp pydantic
brew install ffmpeg          # ffmpeg AND ffprobe are both required
```

`.env` in the project root holds `GEMINI_API_KEY`; only the server reads it.
The clips listed under "Clips" must be in the project root.

## Run

```bash
cd chunked-video-pipeline
../.venv/bin/python server.py
```

Open **http://localhost:8001**, pick a clip and a voice, and press **Start**.

The video waits ~10.5 s before it rolls — that is the pre-roll, while the eyes build a head start. The server
then tells the browser to play.

### Pieces on their own

```bash
../.venv/bin/python eyes.py        # sequential analysis of the whole clip
../.venv/bin/python pipeline.py 3  # pipelined, prints how far ahead it stays
../.venv/bin/python fake_browser.py  # drives the real server without a browser
afplay out/hybrid_run.wav
```

## Clips

Picked from the **clip** dropdown. All but `demo` are 30 s cuts of the full
103-minute `final_game.mp4` (2020 Champions League final, PSG v Bayern, BT Sport
feed with a scoreboard overlay).

| key | file | cut from | what happens |
|---|---|---|---|
| `demo` (default) | `demo_video.mp4` | — | 28 s stock clip, red v white, a goal and celebration |
| `neymar` | `final_neymar_chance.mp4` | 21:00 (17') | Neymar through on goal, Neuer saves (~24 s in), replays |
| `lewandowski` | `final_lewandowski_chance.mp4` | 24:44 (21') | Lewandowski chance, Navas save (~24 s in) |
| `midfield` | `final_game_20s.mp4` | 51:44 (45') | 20 s, second-half restart, open midfield play, 0-0 |
| `goal` | `final_coman_goal.mp4` | 1:04:24 (59') | build-up, Coman header (~22 s in), celebration, 0-1 |
| `whistle` | `final_final_whistle.mp4` | 1:40:46 (90+') | last seconds, final whistle, Bayern celebrate |

Cut with (swap the start time and name):

```bash
ffmpeg -ss 3864 -t 30 -i final_game.mp4 -c:v libx264 -preset medium -crf 23 \
       -an -movflags +faststart final_coman_goal.mp4 -y
```

Match clock = video time − 3:23 in the first half, − 5:31 in the second.

## Voices

Every commentator is a **Voice Design** voice: a voice written in prose, stored
on the project and referred to by a `voice_…` id. The dropdown lists the ones
in `out/designed_voices.json`:

| name | id |
|---|---|
| Roaring stadium announcer (default) | `voice_83ilus4bci19` |
| British TV commentator | `voice_yshgis75mcg9` |
| Excitable English grandmother | `voice_bbajjhxju52c` |
| High-octane female commentator | `voice_s61irdupxk2c` |
| Scottish veteran | `voice_zdcugfqamp9z` |
| Latin American goal screamer | `voice_vrjgo7m9s5dl` |

Each line is written by `gemini-3.5-flash-lite` (~2.5 s) and streamed by
`gemini-3.8-flash-tts` (~2.8 s to first audio). That is too slow to do on
demand, so each line is written and synthesised as soon as its chunk is
analysed, then held until the video reaches it (`TtsVoice` in `voice.py`).
Delivery style follows the events: roaring for a goal, rising for a shot.

`gemini-3.8-live` is not used for the voice: it silently ignores Voice Design
ids — any `voice_…` value, even a made-up one, connects without error and falls
back to a stock voice.

**The voice's description is also the commentator's character.** The prose
that designed a voice is added to the writer's instructions, so what is said
matches how it sounds: the Scottish veteran says "HE'S DIRLED IT IN, COMAN'S
SCORED FOR BAYERN!", the grandmother "Oh, what a brilliant header from Kingsley
Coman!". The one-line and facts-only rules still come first
(`commentator_prompt` in `voice.py`). `gemini-3.1-flash-lite` wrote nearly
identical lines for every persona; `gemini-3.5-flash-lite` commits to the
character at the same speed, so it is the writer.

**Adding a commentator.** There is no design box in the page any more. Create a
voice from Python and give it a name; it then appears in the dropdown:

```python
from common import design_voice, make_client, VOICE_CACHE
import json
r = design_voice(make_client(), "A booming Irish commentator ...", "male")
cache = json.loads(VOICE_CACHE.read_text())
for v in cache.values():
    if v["id"] == r["id"]: v["name"] = "Irish legend"
VOICE_CACHE.write_text(json.dumps(cache, indent=2))
```

Creating a voice takes ~15–25 s; identical descriptions are reused from
`out/designed_voices.json` instead of using up another of the 200 stored voices
per project. Voice Design occasionally returns `500 Voice synthesis service
failed` for a description and keeps failing on retries; rewording it fixed that
("South American … stretches every goal into an endless, breathless roar"
failed three times, the reworded "Latin American" one worked first time).

## Architecture

```
server.py reads demo_video.mp4 directly  (the browser sends nothing)
          │
          ├─ ffmpeg cuts a 4s chunk ──► gemini-3.8-flash ──► ChunkAnalysis (JSON)
          │    (3 chunks in flight)          "eyes"            events, teams, score
          │                                                         │
          │                                                   MatchLog
          │                                                         │
          │                                                         │
          │        writer (flash-lite) ──► one line ──► flash-tts (streaming)
          │                                                         │
          └─ director: plays each line when the playhead reaches its event
                        └─► 24 kHz PCM + transcript ──► browser
```

**Why the server reads the file instead of the browser sending frames.** It can
then analyse *ahead* of playback. The browser only plays video and audio.

**Pre-roll.** Pipelined, the eyes run at ~0.58× real time overall, but the first
two chunks are not ready until ~10 s of wall clock, and each line then takes
~2.5 s to write and ~2.8 s to its first audio. `PRE_ROLL_S = 10.5` delays
playback so the first line is ready when the video starts; after that the eyes
stay 4–18 s ahead and later lines are prepared while earlier ones play.

**Score.** The server owns it, not the eyes. Chunks are analysed in parallel,
so a chunk cannot see that its neighbour already reported a goal, and the
celebration after a goal used to come back as a second goal — once for the
wrong team. A goal reported in a chunk next to the previous goal is treated as
the same goal. The starting score is read from the first chunk (the broadcast
clips start mid-match with a scoreboard), and replays are reported as
`replay`, not as new events.

**Backlog.** The voice gate mirrors the browser's audio queue (when the last
received audio will finish playing). Measuring it against time since Start
instead credited the silent pre-roll as spare time, which let ~7 s of speech
pile up and put the whole second half of the clip behind the picture.

**Concurrency.** Chunk 0 runs alone because it fixes the team colours every later
chunk inherits. The rest run 3 at a time but are **emitted strictly in order**,
so commentary never jumps around.

**Timing.** The eyes timestamp every event (`at`, seconds into the chunk). Each
line is written about one event (`focus_event` in `voice.py`: goal first, then
save, shot, …) and starts playing when playback reaches that event, minus
`LEAD_S` (0.1 s), once the previous line has finished. Timing lines to the
start of the chunk instead called a goal late in a chunk up to 4 s before the
ball went in. "Playback" is the browser's real playhead, which it reports a
few times a second, so a slow video start does not push the voice ahead of the
picture. A routine line that would land more than `STALE_S` after its
chunk's end is dropped; goals, shots and saves never are.

Knobs: `PRE_ROLL_S`, `CONCURRENCY`, `LEAD_S`, `MAX_BACKLOG_S`, `STALE_S` in
`server.py`; `CHUNK_SECONDS`, `WRITER_MODEL`, `VOICE` in `common.py`;
`COMMENTATOR` and `STYLES` in `voice.py`.

## Why not frames into Live

The early prototype streamed 1 fps JPEG frames into the Live session. That is genuinely the
only video input the Live API takes — `send_realtime_input(video=...)` routes
through `t_image_blob()`, which requires `image/*`; `video/mp4` and `video/webm`
are both rejected outright, and mp4 bytes mislabelled as JPEG close the socket
with a 1007. So the prototype was not doing it the wrong way; it was doing the only way.

> **Later finding:** the 1 fps limit in the docs is not enforced, and 2 fps and up
> catch goals that 1 fps misses. That is what [`../live-video-stream`](../live-video-stream)
> is built on, and it replaces this pipeline.

The cost is that at 1 fps the ball crossing the line is often in no frame at
all. Across runs the prototype would sometimes call the goal, sometimes miss the shot
and infer it only from the celebration, and sometimes attribute it to the wrong
team and correct itself on air.

Here the eyes see actual motion. The goal was detected and correctly attributed
to red on the first attempt in every run so far, with the score advanced to 1-0.

| | prototype (1 fps frames into Live) | this pipeline (video chunks, then a voice) |
|---|---|---|
| what the model sees | 1 fps stills | real 4 s video |
| goal detection | variable; sometimes only the celebration | reliable so far, with correct team |
| felt lag | ~3 s, hidden with a scout-video delay | commentary lands in its own window |
| models per session | 1 | 2 |
| cost | one Live session | 7 flash calls + a writer and TTS call per line |

## Measured

```
7 chunks analysed, 14 commentary turns, ~40s of speech, 0 truncations
eyes: mean 5.1 s per 4 s chunk sequentially (1.27x real time — cannot keep up)
      0.58x real time pipelined at concurrency 3 — keeps up comfortably
```

Commentary landing against the action, from a real run (video time it was spoken):

```
11.8s  Suddenly, a long ball targets the right flank.     [chunk 12-16s]
15.0s  The winger chases it down the flank.               [chunk 12-16s]
17.1s  But the keeper safely gathers the cross.           [chunk 16-20s]
21.2s  They break the deadlock with a clinical finish!    [chunk 20-24s = the goal]
22.8s  The stadium erupts as the players celebrate.
25.9s  They share a jubilant embrace near the corner flag.
```

## Known limitations

- **The video waits ~10.5 s before it starts** while the eyes, writer and TTS
  build their lead.
- **One line per 4 s chunk**, with short gaps between lines; there is no
  filler.
- **The eyes cost 7 extra model calls** per 28 s clip and ~5 s each. For a full
  90-minute match that is ~1,350 calls; chunk length and concurrency would need
  rethinking.
- **Commentary granularity is one chunk**, so anything inside a 4 s window that
  is not in the analyst's event list is simply never mentioned. Shorter chunks
  mean more calls and less motion context per call.
- **Goal detection is reliable on this clip but unproven generally.** One clip,
  one goal, a handful of runs.
- **No user interaction.** No mic, no questions about the match.
- **Browser path unverified by me** — no browser automation in this session. All
  numbers above come from `fake_browser.py`, which drives the same WebSocket the
  page uses.
