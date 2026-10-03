# AI Football Match Commentator

A live AI commentator that **watches a football match and calls it as it
happens**, in a voice you design, and **answers your questions out loud**
mid-match ("Who is Bayern's number 9?"), then picks the commentary back up.

Built on **Gemini 3.8 Live**: one real-time session sees the match as a video
stream, hears you through the mic, and speaks.

```
21.7s  He whips a dangerous cross into the center...
25.0s  ...and Coman heads it into the net! GOAL!
28.9s  Coman is mobbed by his teammates in celebration.
       You: Who is Bayern's number nine?
       Commentator: That's Robert Lewandowski, Bayern's centre forward.
```

## Features

- **Watches real video.** The match goes into Gemini Live as a live frame
  stream, the same way AI Studio's *Share Screen* works. Play one of the
  included clips, or share any tab (a match on YouTube works).
- **Lands on the action.** For clips, the model watches a copy that runs a few
  seconds ahead of yours, tuned automatically, so a goal is called as you see
  it go in, not after.
- **Flows like a broadcast.** Short bursts that follow the ball by name and
  build to the big moments, without repeating itself.
- **Designed voices.** Pick a commentator (a roaring stadium announcer, a
  Scottish veteran, a Latin American goal screamer…) or describe a new one and
  Gemini Voice Design creates it in ~20 s. The description also shapes how the
  commentator talks.
- **Talk to it.** The mic is always listening. Ask a question and the
  commentator stops, answers, and carries on.

## Quick start

You need **Python 3.11+**, **ffmpeg**, a **Gemini API key**, and Chrome (for
the mic and screen sharing).

```bash
git clone https://github.com/haykarabyan/ai-football-match-commentator.git
cd ai-football-match-commentator

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
echo "GEMINI_API_KEY=your-key-here" > .env      # never committed (.gitignore)

cd live-video-stream
../.venv/bin/python server.py
```

Open **http://localhost:8002**, pick a match and a commentator, press
**Start**, and allow the microphone. Headphones work best: from speakers,
the commentary can reach the mic.

**No clips, or another match?** Choose *Share a screen or tab*, type who's playing, and
share a tab with the match in it (a YouTube highlight works).

### The match clips

The video files are **not in the repo**: they're cut from a broadcast
recording of the 2020 Champions League final (PSG v Bayern). Put your
recording in the repo root as `final_game.mp4` and cut the clips from it:

```bash
cut() { ffmpeg -ss "$1" -t "$2" -i final_game.mp4 -c:v libx264 -crf 23 -an -movflags +faststart "$3"; }
cut 3864  30 final_coman_goal.mp4            # Coman's header (59'), goal at ~0:23
cut 1260  30 final_neymar_chance.mp4         # Neymar through on Neuer (17')
cut 1484  30 final_lewandowski_chance.mp4    # Lewandowski's chance (21')
cut 3104  20 final_game_20s.mp4              # second-half kick-off
cut 6046  30 final_final_whistle.mp4         # full time
cut 3760 180 final_long_coman_goal.mp4       # 3 min, past Live's 2 min video limit
cut 1230 150 final_long_neymar.mp4
cut 5900 180 final_long_final_whistle.mp4
```

These offsets fit our recording, where the match clock is video time − 3:23
in the first half and − 5:31 in the second; shift them for yours. The app
only lists the clips it finds. Any other football clip can be added in
`live-video-stream/common.py`.

## How it works

```
 browser                                   server                         Gemini
 ───────                                   ──────                         ──────
 clip (copy running ahead) ─ JPEG 2 fps ─►┐
 microphone ──────────────── 16 kHz PCM ─►┴─► one Live session ──────► gemini-3.8-live
                                                                       watches · writes · listens
 your screen + speakers ◄── commentary ◄──┬── stock voice: Live speaks itself
                                          └── designed voice: Live's words ─► gemini-3.8-flash-tts
```

1. The page grabs frames from a hidden copy of the clip that runs a few
   seconds ahead of the one you watch, and streams them, with your mic, to a
   small Python server.
2. The server keeps one `gemini-3.8-live` session open. Whenever the
   commentary is about to run dry, it asks for the next burst, passing the
   commentary so far so it flows and never repeats.
3. A **stock voice** is Live's own audio (fastest). A **designed voice** takes
   Live's words and speaks them with `gemini-3.8-flash-tts` in a Voice Design
   voice.
4. When you speak, Live's voice activity detection notices within ~0.4 s;
   the commentary stops, Live answers, and the commentary resumes.

Details and design decisions: [`live-video-stream/README.md`](live-video-stream/README.md).

## Repository

| folder | what it is |
|---|---|
| [`live-video-stream/`](live-video-stream) | **The app.** One Gemini Live session on a live frame stream. |
| [`chunked-video-pipeline/`](chunked-video-pipeline) | The earlier approach: `gemini-3.8-flash` analyses 4 s video chunks into match events, a writer model turns them into lines, TTS speaks them. Works, but needs a 10 s head start and 2 models per line. |
| [`experiments/`](experiments) | One script per Gemini model tried along the way (TTS and Voice Design, image, thinking levels, batch transcription). |
| [`server/`](server) | A small mic playground for Live chat, live translation and live captions. |
| [`NOTES.md`](NOTES.md) | Measured latencies and API gotchas from that exploration. |

## What we learned about the Live API

Tested against the real API (`google-genai` 2.28, October 2026):

- **Video goes in only as JPEG/PNG frames.** mp4, webm, MPEG-TS and H.264 on
  `realtime_input.video` all close the session (1007), even when streamed.
- **More than 1 fps works, and matters.** The docs say max 1 fps; at 1 fps the
  model called a goal a save, at 2 fps and up it saw the ball go in.
- **Live can't speak Voice Design voices.** It accepts a `voice_...` id and
  silently ignores it, so designed voices need a TTS step.
- **Live is audio-out only.** `TEXT` responses are rejected. For text, use the
  output transcription, which is complete well before the turn ends
  (~1.7 s vs ~4 s), so don't wait for `turn_complete`.
- **`voice_activity` events are the fast barge-in signal** (~0.4 s in); the
  input transcription only arrives after the speaker stops.
- **With automatic VAD off, text sent between `activity_start` and
  `activity_end` kills the session**, which rules out push-to-talk alongside
  text-driven commentary.

## Limitations

- The commentary comes in bursts every ~4 s. A goal that lands just after a
  burst starts is called a moment late, and occasionally a scramble is called
  wrong (a header that went in called as a penalty appeal).
- Sharing a live screen has no look-ahead: a stock voice lands ~1.3 s after
  the action, a designed voice ~4.5 s.
- Sessions over ~10 minutes rely on Live's session resumption, which is
  enabled but not yet tested at that length.
