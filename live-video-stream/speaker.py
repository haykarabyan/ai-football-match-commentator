"""Who actually speaks the lines Live writes.

NativeSpeaker   Live's own audio, passed straight through (stock voice, ~1 s)
DesignedSpeaker each line -> gemini-3.8-flash-tts in a Voice Design voice
                (~1-3 s more to first audio, hidden by the look-ahead)

Both track play_end: when the browser's audio queue will run dry, assuming it
plays everything as soon as it arrives. The director cues the next line off it.
"""
import asyncio
import base64
import re
import time

from common import OUTPUT_RATE, TTS_MODEL

BYTES_PER_S = OUTPUT_RATE * 2


class _Clock:
    def __init__(self):
        self.play_end = 0.0

    def add(self, nbytes: int):
        self.play_end = max(self.play_end, time.perf_counter()) + nbytes / BYTES_PER_S

    def backlog(self) -> float:
        """Seconds of audio still queued in the browser."""
        return max(0.0, self.play_end - time.perf_counter())


class NativeSpeaker(_Clock):
    def __init__(self, on_audio):
        super().__init__()
        self.on_audio = on_audio

    async def audio(self, pcm: bytes):
        self.add(len(pcm))
        await self.on_audio(pcm)

    async def line(self, text: str, cue_t=None, kind="line"):
        pass                       # already spoken by Live

    def busy(self) -> bool:
        return False

    def flush(self):
        """A viewer cut in: the page drops its queued audio."""
        self.play_end = 0.0

    def stop(self):
        pass


# Delivery style for TTS, picked from the words of the line.
STYLES = [
    (r"\bgo+a+l+\b|\bscores?\b|\bin the net\b|\bback of the net\b",
     "ecstatic, roaring at the top of the voice, a goal has just gone in"),
    (r"\bsave[sd]?\b|\bshoots?\b|\bshot\b|\bchance\b|\bwide\b|\bpost\b|\bbar\b|\bheader\b",
     "excited and rising fast, a real chance at goal"),
    (r"\breplay\b|\bslow motion\b|\banother look\b",
     "measured and analytical, looking back at the replay"),
    (r"\bwhistle\b|\bfull time\b|\bit'?s over\b|\bchampions\b",
     "emotional and elated, a historic moment"),
]
CALM = "lively, conversational live football commentary"
ANSWER_STYLE = "friendly and direct, turning to answer a viewer's question mid-broadcast"
IMPORTANT = re.compile(STYLES[0][0] + "|" + STYLES[1][0], re.I)


def speech_seconds(text: str) -> float:
    """Rough speaking time of a commentary line (~2.8 words a second)."""
    return len(text.split()) / 2.8


# Each line is synthesised on its own, so tell TTS it is mid-flow: otherwise
# every line starts like a fresh announcement.
FLOW = ("; carrying straight on from the previous phrase, mid-broadcast, "
        "no fresh start, one continuous stream of commentary")


def style_for(text: str) -> str:
    for pattern, style in STYLES:
        if re.search(pattern, text, re.I):
            return style + FLOW
    return CALM + FLOW


class _Line:
    def __init__(self, text, cue_t, kind, epoch):
        self.text, self.cue_t, self.kind, self.epoch = text, cue_t, kind, epoch
        self.at = time.perf_counter()        # when the text arrived
        self.audio: asyncio.Queue = asyncio.Queue()   # PCM chunks, then None
        self.started = False                 # TTS audio has begun
        self.task = None


class DesignedSpeaker(_Clock):
    """Speaks each line in a designed voice, in order.

    TTS starts the moment a line arrives, even while earlier lines are still
    playing, so a line never waits behind another line's synthesis. A routine
    line that would start more than stale_s late is dropped: by then it
    describes play the viewer has already seen. Goals, shots and saves are
    always spoken.
    """

    def __init__(self, client, voice_id, on_audio, on_spoken, log=print,
                 stale_s=2.5):
        super().__init__()
        self.client = client
        self.voice_id = voice_id
        self.on_audio = on_audio
        self.on_spoken = on_spoken          # (text, cue_t, kind) as a line starts
        self.log = log
        self.stale_s = stale_s
        self.order: asyncio.Queue = asyncio.Queue()
        self.pending: list[_Line] = []      # arrived, not yet started playing
        self.tts_s: list[float] = []        # measured time to first audio
        self.expected_tts_s = 2.9           # updated from measured first-audio times
        self.epoch = 0                      # bumped by flush(); older lines are dead
        self._task = asyncio.create_task(self._forward())

    async def audio(self, pcm: bytes):
        pass                       # Live's own audio is not played

    async def line(self, text: str, cue_t=None, kind="line"):
        ln = _Line(text, cue_t, kind, self.epoch)
        ln.task = asyncio.create_task(self._synth(ln))
        self.pending.append(ln)
        self.order.put_nowait(ln)

    def busy(self) -> bool:
        return len(self.pending) >= 2

    def backlog(self) -> float:
        """Audio still to be heard, counting lines that are still in TTS."""
        now = time.perf_counter()
        end = max(self.play_end, now)
        for ln in self.pending:
            if not ln.started:
                end = max(end, ln.at + self.expected_tts_s)
            end += speech_seconds(ln.text)
        return end - now

    def flush(self):
        """A viewer cut in: cancel every line not yet heard, stop the current one."""
        self.epoch += 1
        for ln in self.pending:
            ln.task.cancel()
        self.pending.clear()
        self.play_end = 0.0

    def stop(self):
        self._task.cancel()
        for ln in self.pending:
            ln.task.cancel()

    async def _forward(self):
        while True:
            ln = await self.order.get()
            if ln.epoch != self.epoch:
                continue                          # flushed
            first = await ln.audio.get()
            if ln in self.pending:
                self.pending.remove(ln)
            if first is None or ln.epoch != self.epoch:
                continue                          # TTS failed, or flushed
            late = time.perf_counter() - max(self.play_end, ln.at + self.expected_tts_s)
            if ln.kind == "line" and late > self.stale_s and not IMPORTANT.search(ln.text):
                self.log(f"  dropped stale line ({late:.1f}s late): {ln.text}")
                ln.task.cancel()
                continue
            await self.on_spoken(ln.text, ln.cue_t, ln.kind)
            chunk = first
            while chunk is not None and ln.epoch == self.epoch:
                self.add(len(chunk))
                await self.on_audio(chunk)
                chunk = await ln.audio.get()

    async def _synth(self, ln: _Line):
        try:
            stream = await self.client.aio.interactions.create(
                model=TTS_MODEL,
                input=[{"type": "user_input", "content": [{
                    "type": "text", "text": ln.text,
                    "annotations": [{"type": "speech_metadata",
                                     "style": ANSWER_STYLE if ln.kind == "answer" else style_for(ln.text)}],
                }]}],
                response_format={"type": "audio", "mime_type": "audio/l16",
                                 "sample_rate": OUTPUT_RATE},
                generation_config={"speech_config": [{"voice": self.voice_id}]},
                stream=True,
            )
            async for ev in stream:
                if ev.event_type == "step.delta" and ev.delta.type == "audio":
                    if not ln.started:
                        ln.started = True
                        self.tts_s.append(time.perf_counter() - ln.at)
                        recent = sorted(self.tts_s[-5:])
                        self.expected_tts_s = recent[len(recent) // 2]
                    ln.audio.put_nowait(base64.b64decode(ev.delta.data))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log(f"  TTS failed: {type(exc).__name__}: {exc}")
        finally:
            ln.audio.put_nowait(None)
