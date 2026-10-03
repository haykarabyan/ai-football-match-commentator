# Live video stream

The final approach: **one `gemini-3.8-live` session watches the match as a live
video stream, commentates it, and answers viewers who talk to it.**

```bash
cd live-video-stream
../.venv/bin/python server.py        # → http://localhost:8002
```

Setup, clips and the big picture are in the [top-level README](../README.md).
This page covers how it works and why.

## Architecture

```
 browser                                server.py                       Gemini
 ───────                                ─────────                       ──────
 hidden copy of the clip,  JPEG 2 fps ─►┐
 a few seconds ahead                    ├─► realtime_input ─────────► gemini-3.8-live
 mic, always on        16 kHz PCM ─────►┘   (video + audio)            watches, writes,
                                                                       hears the viewer
 visible copy of the clip                    ◄── audio + transcript ──┘
 speakers ◄──── 24 kHz PCM ◄──┬── stock voice: Live's own audio
                              └── designed voice: transcript ─► gemini-3.8-flash-tts
```

| file | role |
|---|---|
| `server.py` | WebSocket session, the director that cues lines, question handling, voice endpoints |
| `live.py` | the Live session: prompt, frames and mic in, lines and answers out, resumption |
| `speaker.py` | stock voice pass-through, or designed-voice TTS (parallel synthesis, in-order playback) |
| `common.py` | clips and match context, Voice Design helpers |
| `static/` | the page: frame capture, look-ahead sync, mic, playback, transcript |
| `out/designed_voices.json` | the designed commentators (ids from Voice Design) |

## How the video gets in

The Live API takes video **only as a stream of JPEG/PNG frames** on
`realtime_input.video`, which is also what AI Studio's Share Screen sends.
Tested against the real API, all of these close the socket with 1007: mp4
(whole or fragmented), webm (whole or streamed like MediaRecorder), MPEG-TS,
raw H.264, codec-tagged mime types, and video parts in `client_content`.

**Frame rate matters.** The docs say "max 1 fps", but faster is accepted. On
the same goal, 1 fps called it a save; 2, 4, 8 and 12 fps saw the ball go in.
The page sends 2 fps at up to 768 px.

## Keeping the commentary on the picture

- **Cueing.** Live only speaks when asked, so the server sends `continue`
  (plus the commentary so far, as one running paragraph) whenever the audio
  still to be heard runs low. Stock voice: when < 1 s is queued. Designed
  voice: early enough to cover Live writing the line (~1.7 s) plus TTS (~2.9 s).
- **Look-ahead.** The model watches a hidden copy of the clip that runs ahead
  of the one you see by about one cue→heard delay (starts at 1.5 s stock,
  5 s designed). The page measures every line and keeps it tuned. A shared
  screen is live, so it has no look-ahead.
- **Designed voice doesn't wait for Live's turn to end.** Live is audio-out
  only (TEXT is rejected), and its turn ends only once that audio is
  generated (~4 s). The line's transcript is complete after ~1.7 s, so it
  goes to TTS at the first sentence end.
- **Flow.** Each request gets one short burst (3–12 words), but the prompt
  makes the bursts one continuous stream: fragments that pick up where the
  last stopped, follow the ball by name, build and pay off. One rule sits
  above the flow: what's in the latest frames beats continuing the story.
- **No repeats.** The prompt forbids re-calling a moment, and in designed mode
  a line that mostly restates one of the last three is dropped.

## Voices

- **Stock voices** (Fenrir, Puck, …) are Live's own: lowest latency.
- **Designed voices** come from Voice Design on `gemini-3.8-flash-tts`. Live
  accepts `voice_config.voice = "voice_..."` but ignores it (a made-up id
  works the same, and every id comes out as one stock voice), so designed
  voices go through TTS. The prose that designed the voice also becomes the
  commentator's character in the prompt.
- **Create a commentator** in the page: ~20 s, uses one of the project's 200
  stored voices; identical descriptions are reused.

## Talking to the commentator

The mic streams all the time and Live's automatic voice activity detection
stays on. `voice_activity: ACTIVITY_START` arrives ~0.4 s after the viewer
starts talking; the server then stops everything queued, and Live's next
turn is treated as the answer. The question's transcript only arrives when
the viewer stops, so if no words come within 2 s it was a cough and the
commentary carries on. The match context includes both starting line-ups
with shirt numbers (not the result), so "number 9" questions are grounded.

Why no push-to-talk button: with automatic VAD off, any text sent between
`activity_start` and `activity_end` closes the socket, and the commentary
runs on text cues.

## Sessions

Video sessions end after ~2 min without compression, so sliding-window
context compression and session resumption (reconnect on `go_away`) are on.
The 2.5–3 min clips exist to exercise this.
