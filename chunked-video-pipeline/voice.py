"""The voice: speaks commentary from the match log in a designed voice.

It is told what happened; it never looks at the video. That split is the whole
point of this design — the eyes see real motion, the voice only has to talk.

A writer model turns each analysed chunk into one line, and streaming TTS
speaks it in a Voice Design voice. That takes ~5.5 s per line, so each line is
prepared as soon as its chunk is analysed and held until playback reaches it.
"""
import asyncio
import base64
import time

from common import TTS_MODEL, WRITER_MODEL

COMMENTATOR = (
    "You are a live television football commentator calling a match as it happens.\n"
    "You are fed a running feed of events from the match analyst. Commentate them.\n"
    "Style: energetic, natural, short punchy lines. Lift your voice for shots and "
    "goals.\n"
    "CRITICAL: say exactly ONE short sentence, 5 to 10 words. Never two. You get "
    "the next chunk of play within a few seconds, so never try to cover "
    "everything at once.\n"
    "Only ever state what is in the events you are given. Never invent a goal, "
    "card, save or score. If you are given no new events, add light colour about "
    "the build-up, the shape or the momentum instead of inventing action.\n"
    "Refer to each team exactly as the events name it — a club name if the "
    "events use one, otherwise the shirt colour. Never invent a club name.\n"
    "FLOW: you can see everything you have already said. Each line is the next "
    "beat of one continuous broadcast, not a standalone caption. Link to your "
    "previous line with pronouns (he, they, it) and connectives (now, still, "
    "again, suddenly).\n"
    "VARIETY: never begin two lines in a row with the same word, and do not start "
    "line after line with a team name. 'Red does X. Red does Y.' is the worst "
    "failure mode here.\n"
    "You are mid-broadcast. Never greet, never recap, never reintroduce the match. "
    "Never mention events, feeds, analysts, or that you are an AI."
)


def commentator_prompt(persona: str | None) -> str:
    """COMMENTATOR, written in the character of the voice that will speak it.

    persona is the same prose that designed the voice, so what is said matches
    how it sounds. The rules above it still win: one short line, facts only.
    """
    if not persona:
        return COMMENTATOR
    return (
        COMMENTATOR + "\n\n"
        "YOUR CHARACTER — this is also exactly how your voice sounds:\n"
        f"{persona}\n"
        "Write every line the way this commentator would say it. Make the "
        "character unmistakable: a listener should recognise who is talking "
        "from the words alone, before hearing the voice. Use their vocabulary, "
        "idioms, exclamations, rhythm and regional flavour, and lean into it "
        "rather than playing it safe — a Scot sounds Scottish, a Latin American "
        "screamer stretches his goal calls (\"GOOOOL!\"), a gentle grandmother "
        "is warm and a little fussy. Let the character colour HOW you say "
        "things, never WHAT happened: the one-sentence limit and the "
        "facts-only rules above still apply."
    )


# Which event a line is about, most important first. The line is played at
# that event's moment, so the writer is told to call exactly that event.
FOCUS_ORDER = ["goal", "save", "shot", "card", "foul", "corner", "cross",
               "celebration", "dribble", "tackle", "interception", "kickoff"]


def focus_event(analysis):
    for kind in FOCUS_ORDER:
        for e in analysis.events:
            if e.kind == kind:
                return e
    live = [e for e in analysis.events if e.kind not in ("replay", "close_up")]
    return (live or analysis.events or [None])[0]


def events_to_prompt(t0, t1, analysis) -> str:
    """Turn one chunk of analyst output into something speakable."""
    lines = [f"[{t0:.0f}-{t1:.0f}s] {analysis.summary}"]
    for e in analysis.events:
        lines.append(f"- {e.kind} ({e.team}): {e.description}")
    lines.append(f"Score {analysis.team_a} {analysis.score_a} - "
                 f"{analysis.score_b} {analysis.team_b}.")
    focus = focus_event(analysis)
    if focus is not None:
        lines.append(f"Your line is heard at the moment of this event, so call "
                     f"it: {focus.kind} ({focus.team}): {focus.description}. "
                     "One short line.")
    else:
        lines.append("Commentate the most important new thing here. One short line.")
    return "\n".join(lines)


# How each line is delivered, picked from the chunk's events.
STYLES = [
    ({"goal"}, "ecstatic, roaring at the top of the voice, a goal has just gone in"),
    ({"celebration"}, "elated and joyful, still buzzing from the goal"),
    ({"shot", "save"}, "excited and rising fast, a real chance at goal"),
    ({"cross", "corner", "dribble"}, "building anticipation, quickening"),
]
CALM_STYLE = "lively, conversational live football commentary"


def style_for(analysis) -> str:
    kinds = {e.kind for e in analysis.events}
    for wanted, style in STYLES:
        if kinds & wanted:
            return style
    return CALM_STYLE


class Line:
    """One prepared line: text from the writer, audio streaming in from TTS."""

    def __init__(self, c0, c1, at, important):
        self.c0, self.c1 = c0, c1
        self.at = at                      # video time of the event it calls
        self.important = important        # never dropped for being late
        self.text = None
        self.audio = asyncio.Queue()      # PCM chunks, then None when done
        self.ready = asyncio.Event()      # first audio has arrived (or failed)


class TtsVoice:
    """Designed-voice commentator.

    prepare() is called for every analysed chunk, in match order. Lines are
    written one at a time (each sees the lines before it, for flow), and each
    line's TTS starts the moment its text exists. The caller pulls lines with
    next_line() and plays them when the video reaches them.
    """

    def __init__(self, client, on_audio, on_text, voice_name, persona=None):
        self.client = client
        self.on_audio = on_audio
        self.on_text = on_text
        self.voice_name = voice_name
        self.system = commentator_prompt(persona)
        self.t0 = time.perf_counter()
        self.audio_bytes = 0
        self.truncations = 0
        self.play_end = 0.0
        self.history = []                  # lines written so far, in order
        self.lines = asyncio.Queue()       # Line objects, in match order
        self._jobs = asyncio.Queue()
        self._writer = None

    def start(self):
        self._writer = asyncio.create_task(self._write_loop())

    def stop(self):
        if self._writer:
            self._writer.cancel()

    def prepare(self, c0, c1, analysis):
        important = any(e.kind in ("goal", "shot", "save") for e in analysis.events)
        focus = focus_event(analysis)
        at = c0 + min(max(focus.at, 0.0), c1 - c0) if focus is not None else c0
        line = Line(c0, c1, at, important)
        self.lines.put_nowait(line)
        self._jobs.put_nowait((line, analysis))

    async def _write_loop(self):
        while True:
            line, analysis = await self._jobs.get()
            try:
                line.text = await self._write(line, analysis)
            except Exception as exc:
                print(f"  writer failed for [{line.c0:.0f}-{line.c1:.0f}s]: {exc}")
                line.audio.put_nowait(None)
                line.ready.set()
                continue
            self.history.append(line.text)
            asyncio.create_task(self._speak(line, style_for(analysis)))

    async def _write(self, line, analysis) -> str:
        said = "\n".join(f"- {t}" for t in self.history[-4:]) or "(nothing yet)"
        interaction = await self.client.aio.interactions.create(
            model=WRITER_MODEL,
            system_instruction=self.system,
            input=(f"Your previous lines:\n{said}\n\nNew from the analyst:\n"
                   f"{events_to_prompt(line.c0, line.c1, analysis)}\n\n"
                   "Write the line you say next. Output only that sentence."),
            generation_config={"thinking_level": "minimal"},
        )
        return interaction.output_text.strip().strip('"')

    async def _speak(self, line, style):
        try:
            stream = await self.client.aio.interactions.create(
                model=TTS_MODEL,
                input=[{"type": "user_input", "content": [{
                    "type": "text", "text": line.text,
                    "annotations": [{"type": "speech_metadata", "style": style}],
                }]}],
                response_format={"type": "audio", "mime_type": "audio/l16",
                                 "sample_rate": 24000},
                generation_config={"speech_config": [{"voice": self.voice_name}]},
                stream=True,
            )
            async for ev in stream:
                if ev.event_type == "step.delta" and ev.delta.type == "audio":
                    line.audio.put_nowait(base64.b64decode(ev.delta.data))
                    line.ready.set()
        except Exception as exc:
            print(f"  TTS failed for [{line.c0:.0f}-{line.c1:.0f}s]: {exc}")
        finally:
            line.audio.put_nowait(None)
            line.ready.set()

    def backlog(self) -> float:
        return self.play_end - time.perf_counter()

    async def play(self, line):
        """Forward one prepared line to the listener, as fast as it streams."""
        await self.on_text(line.text)
        while (chunk := await line.audio.get()) is not None:
            self.audio_bytes += len(chunk)
            self.play_end = (max(self.play_end, time.perf_counter())
                             + len(chunk) / 2 / 24000)
            await self.on_audio(chunk)
