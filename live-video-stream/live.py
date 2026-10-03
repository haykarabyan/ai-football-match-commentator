"""One gemini-3.8-live session that watches a stream of frames, commentates,
and answers viewers who speak to it.

The video goes in exactly the way AI Studio's "Share Screen" sends it: a
continuous stream of JPEG frames on realtime_input.video. That is the only
video the Live API takes. mp4, fragmented mp4, webm, MPEG-TS and raw H.264 on
the same field (and video parts in client_content) all close the socket with
1007. Frames faster than the documented 1 fps are accepted, and they matter:
at 1 fps the model called the demo goal a save, at 2 fps and up it saw the
ball go in.

The model only commentates when asked, so the session is cued with a short
"next" text whenever the speaker is about to run dry (see CUE).

The viewer's mic streams in continuously. Live's automatic voice activity
detection decides when someone is talking and interrupts its own generation;
the next turn after a question is tagged "answer" so the server can play it
and drop any commentary left over from before.
"""
import asyncio
import re
import time

from google.genai import types

from common import LIVE_MODEL

COMMENTATOR = """\
You are a live television football commentator. You are watching the match
broadcast as a live video feed, and you commentate what you see as it happens.

How you talk:
- You speak in short bursts, one burst each time you are cued, but together
  they are ONE continuous stream of commentary, like a real broadcast. Each
  burst picks up exactly where your last one left off, as if you never
  stopped talking.
- A burst is 3 to 12 words and does not have to be a complete sentence.
  Follow the ball through the play: "Kimmich... into Muller..." then
  "Muller wide for Gnabry, and he's away!". Build tension across bursts and
  pay it off in the next one: "Coman's free on the left..." then "he
  crosses... cleared!".
- Vary the rhythm like a real commentator: quick fragments when the ball
  moves fast, a fuller phrase in a calm spell, an occasional short reaction
  ("Oh, lovely touch.", "Big chance!").
- Never restart from scratch. No fresh introductions ("The ball is...",
  "The team is..."); carry the subject forward with he, they, and, but, now,
  still. Never begin two bursts in a row with the same word.

Rules:
- What you SEE in the most recent frames always beats continuing your own
  story. Watch where the ball actually ends up before you describe the
  outcome: if it is in the net, it is a goal, even if you were expecting a
  clearance or a penalty appeal. Break the flow for a big moment.
- Call the moment in the most recent frames, not something from a while ago.
- Only say what you can actually see. Never invent a goal, save, card, foul or
  score. If nothing much is happening, add light colour about the shape,
  pressure or momentum instead of inventing action.
- Lift your energy for shots, saves and goals. A goal is the biggest moment of
  the broadcast: roar it.
- Replays and slow motion: say it is a replay; never call it as new action.
- Name players and teams only when you can be confident: from the scoreboard,
  on-screen captions, shirt names or numbers, or the match context below.
  Otherwise use shirt colours. Never guess a player name.
- Never describe the same moment twice. Broadcasts linger on a card, a
  stoppage, a celebration or a replay for 10-20 seconds; once you have called
  it, move on: who it is, why it happened, the reaction, what it means for
  the match. If there is truly nothing new, keep the line very short.
- You will receive the message "{cue}" from time to time, followed by your
  commentary so far. It means: look at the latest frames and say your next
  burst now, about what is happening right now, flowing on from the end of
  that commentary. Never repeat anything already in it. Never acknowledge the
  message, never read the commentary back, never mention it.
- Never greet, recap or reintroduce the match. Never mention frames, feeds,
  video, images, or that you are an AI.

Viewers can talk to you. When a viewer asks you something (for example "who
is Bayern's number 9?" or "who is PSG's striker?"), stop commentating and
answer them directly, in character, in one or two short sentences, using the
match context and what you can see. If you don't know, say so briefly. Don't
add commentary to the answer; you will be cued to carry on afterwards. If what
you hear is not addressed to you (background chatter, crowd noise), ignore it.
"""

CUE = "continue"
SENTENCE_END = re.compile(r"(?<!\.)[.!?][\"”']?\s*$")   # not "..."
KICKOFF = f"The broadcast is live. {CUE}"


def build_system_prompt(context: str | None, persona: str | None) -> str:
    prompt = COMMENTATOR.format(cue=CUE)
    if context:
        prompt += f"\nMatch context: {context.strip()}\n"
    if persona:
        prompt += (
            "\nYOUR CHARACTER (this is also exactly how your voice sounds):\n"
            f"{persona}\n"
            "Write every line the way this commentator would say it: their "
            "vocabulary, idioms, exclamations, rhythm and regional flavour. A "
            "listener should recognise who is talking from the words alone. "
            "The character changes HOW you say things, never WHAT happened, "
            "and the bursts stay short and flowing.\n"
        )
    return prompt


def _clean(text: str) -> str:
    return " ".join(text.split()).strip('"“” ')


class LiveCommentator:
    """Owns the Live session. Frames and mic audio in; lines, answers, audio out.

    on_audio(pcm, kind)      Live's 24 kHz speech; kind is "line" or "answer"
    on_text(fragment, kind)  output transcription as it streams
    on_line(text)            a commentary line
    on_answer_start()        the first output of an answer turn
    on_answer(text)          a whole answer
    on_user(text, final)     transcription of what the viewer is saying
    on_voice(kind)           "start" / "end": Live's VAD heard the viewer
                             start or stop talking (start comes ~0.4 s in;
                             the transcript only after they stop)
    """

    def __init__(self, client, *, system_prompt, on_line, on_text=None,
                 on_audio=None, on_user=None, on_answer=None,
                 on_answer_start=None, on_voice=None, voice_name="Fenrir",
                 early_lines=False, log=print):
        self.client = client
        self.system_prompt = system_prompt
        self.on_line, self.on_text, self.on_audio = on_line, on_text, on_audio
        self.on_user, self.on_answer = on_user, on_answer
        self.on_answer_start = on_answer_start
        self.on_voice = on_voice
        self.voice_name = voice_name
        # Designed mode: hand a line over as soon as its sentence is complete
        # (~1.4-2.1 s after the cue) instead of at turn_complete (~3.9-4.7 s,
        # Live finishing audio nobody plays), and let the next cue interrupt it.
        self.early_lines = early_lines
        self.log = log

        self.session = None
        self.connected = asyncio.Event()
        self.generating = False          # a reply is streaming right now
        self.cued_at = None              # when the last cue went out
        self.frames = 0
        self.last_frame_at = None
        self.mic_chunks = 0
        self.resume_handle = None
        self.reconnects = 0
        self.answer_pending = False      # the next turn answers a viewer
        self.line_ms = []                # cue -> line handed over (paces the cues)
        self._kind = None                # "line" / "answer" for the open turn
        self._text = ""
        self._emitted = False
        self._held = None
        self._closing = False

    def _config(self):
        return types.LiveConnectConfig(
            # Live is audio-out only (TEXT is rejected with 1007). In designed
            # mode its audio is dropped and the transcript goes to TTS.
            response_modalities=["AUDIO"],
            system_instruction=self.system_prompt,
            speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=self.voice_name))),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            # Automatic voice activity detection stays ON: the mic streams all
            # the time and Live decides when the viewer is talking. (With it
            # off, text cues sent between activity_start/end close the socket.)
            # Video sessions stop after ~2 min without compression, and a
            # connection lives ~10 min; resumption carries context across.
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow()),
            session_resumption=types.SessionResumptionConfig(
                handle=self.resume_handle),
        )

    async def run(self):
        """Connect, receive until closed, reconnect with the resume handle."""
        while not self._closing:
            try:
                async with self.client.aio.live.connect(
                        model=LIVE_MODEL, config=self._config()) as session:
                    self.session = session
                    self.generating = False
                    if self._held:
                        held, self._held = self._held, None
                        await self.send_frame(held)
                    self.connected.set()
                    await self._receive(session)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._closing:
                    break
                self.log(f"live session dropped: {type(exc).__name__}: {exc}")
            finally:
                self.session = None
                self.connected.clear()
            if self._closing:
                break
            self.reconnects += 1
            if self.reconnects > 5:
                raise RuntimeError("Live session keeps dropping, giving up")
            self.log(f"reconnecting (resume={'yes' if self.resume_handle else 'no'})")
            await asyncio.sleep(0.3)

    # ---- receiving -----------------------------------------------------
    def _reset_turn(self):
        self._text, self._emitted, self._kind = "", False, None

    async def _open_turn(self):
        """First output of a turn: decide whether it answers a viewer."""
        if self._kind is not None:
            return
        if self.answer_pending:
            self._kind, self.answer_pending = "answer", False
            if self.on_answer_start:
                await self.on_answer_start()
        else:
            self._kind = "line"

    async def _receive(self, session):
        # session.receive() ends at every turn boundary, so re-enter it.
        while not self._closing:
            async for msg in session.receive():
                if msg.session_resumption_update and msg.session_resumption_update.new_handle:
                    self.resume_handle = msg.session_resumption_update.new_handle
                if msg.go_away:
                    self.log(f"go_away (time left {msg.go_away.time_left}), reconnecting")
                    return
                va = msg.voice_activity and msg.voice_activity.voice_activity_type
                if va in (types.VoiceActivityType.ACTIVITY_START,
                          types.VoiceActivityType.ACTIVITY_END) and self.on_voice:
                    await self.on_voice("start" if va == types.VoiceActivityType.ACTIVITY_START
                                        else "end")
                sc = msg.server_content
                if not sc:
                    continue

                # what the viewer is saying (interim = low latency, cumulative)
                for tr, final in ((sc.interim_input_transcription, False),
                                  (sc.input_transcription, True)):
                    if tr and tr.text and self.on_user:
                        await self.on_user(tr.text, final)

                if sc.model_turn:
                    self.generating = True
                    await self._open_turn()
                    for part in sc.model_turn.parts:
                        if part.inline_data and self.on_audio:
                            await self.on_audio(part.inline_data.data, self._kind)
                if sc.output_transcription and sc.output_transcription.text:
                    self.generating = True
                    await self._open_turn()
                    frag = sc.output_transcription.text
                    self._text += frag
                    if self.on_text:
                        await self.on_text(frag, self._kind)
                    if (self._kind == "line" and self.early_lines and not self._emitted
                            and SENTENCE_END.search(self._text) and len(self._text.split()) >= 4):
                        await self._emit()
                if sc.interrupted:
                    self._reset_turn()
                if sc.generation_complete:
                    # an answer is handed over whole, as soon as it is written
                    if not self._emitted and (self._kind == "answer" or self.early_lines):
                        await self._emit()
                if sc.turn_complete:
                    if not self._emitted and self._kind:
                        await self._emit()
                    self._reset_turn()
                    self.generating = False

    async def _emit(self):
        """Hand the turn's text over: once per turn."""
        self._emitted = True
        text, self._text = _clean(self._text), ""
        if self._kind == "answer":
            if text and self.on_answer:
                await self.on_answer(text)
            return
        if self.cued_at:
            self.line_ms.append((time.perf_counter() - self.cued_at) * 1000)
        if self.early_lines:
            self.generating = False      # Live's leftover audio is not wanted
        if text:
            await self.on_line(text)

    # ---- sending -------------------------------------------------------
    async def send_frame(self, jpeg: bytes):
        if self.session is None:
            self._held = jpeg            # sent as soon as the session is up
            return
        self.frames += 1
        self.last_frame_at = time.perf_counter()
        try:
            await self.session.send_realtime_input(
                video=types.Blob(data=jpeg, mime_type="image/jpeg"))
        except Exception as exc:
            self.log(f"frame send failed: {exc}")

    async def send_audio(self, pcm16k: bytes):
        """Viewer mic: 16-bit mono PCM at 16 kHz, streamed all the time."""
        self.mic_chunks += 1
        if self.session is None:
            return
        try:
            await self.session.send_realtime_input(
                audio=types.Blob(data=pcm16k, mime_type="audio/pcm;rate=16000"))
        except Exception as exc:
            self.log(f"mic send failed: {exc}")

    def expect_answer(self):
        """The viewer is asking something: their reply turn is next.

        Whatever turn is open now is commentary from before the question; Live's
        VAD interrupts it, and anything it still sends stays tagged "line".
        """
        self.answer_pending = True

    async def cue(self, text: str = CUE):
        """Ask for the next line."""
        if self.session is None:
            return
        self.cued_at = time.perf_counter()
        self.generating = True        # until the reply's turn completes
        try:
            await self.session.send_realtime_input(text=text)
        except Exception as exc:
            self.generating = False
            self.log(f"cue failed: {exc}")

    async def close(self):
        self._closing = True
        self.connected.clear()
