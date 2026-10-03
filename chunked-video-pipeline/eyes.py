"""The eyes: send real video chunks to gemini-3.8-flash, get structured events.

Unlike 1 fps stills streamed into a Live session, this sees actual motion, so it
can tell a shot from a cross and watch the ball cross the line.

Run over the whole clip:
    ../.venv/bin/python eyes.py
"""
import base64
import json
import sys
import time
from typing import List

from pydantic import BaseModel, Field

from common import (CHUNK_SECONDS, EYES_MODEL, OUT, VIDEO, cut_chunk,
                    make_client, video_duration)


class Event(BaseModel):
    kind: str = Field(description=(
        "one of: kickoff, pass, cross, dribble, tackle, interception, shot, "
        "save, goal, celebration, foul, card, corner, goal_kick, offside, "
        "throw_in, buildup, pressure, replay, close_up"))
    team: str = Field(description=(
        "the team involved, named EXACTLY as team_a or team_b below, or 'unknown'"))
    description: str = Field(description="max 12 words, what actually happened")
    at: float = Field(description=(
        "seconds from the START of this clip when it happens (for a goal, the "
        "moment the ball crosses the line), between 0 and the clip length"))


class ChunkAnalysis(BaseModel):
    team_a: str = Field(description=(
        "how to call one team for the WHOLE match: its club name if the broadcast "
        "makes it identifiable (scoreboard, badges, known kit), otherwise its "
        "shirt colour. Must match the running context exactly if given there."))
    team_b: str = Field(description="the same, for the other team")
    summary: str = Field(description="one sentence describing this chunk")
    events: List[Event] = Field(description="events in order, most important first")
    score_a: int = Field(description="goals scored so far by team_a")
    score_b: int = Field(description="goals scored so far by team_b")
    shots_a: int = Field(description="shots so far by team_a")
    shots_b: int = Field(description="shots so far by team_b")


PROMPT = """You are the video analyst for a live football broadcast.

You are watching seconds {t0:.1f}-{t1:.1f} of the match. Report ONLY what is
visibly in THIS clip. Do not invent events. If you cannot tell which team did
something, set team to "unknown".

A goal is only a goal if you actually see the ball cross the line or see players
celebrating. The team CELEBRATING is the team that scored.

This is a TV broadcast. Slow-motion or alternate-angle REPLAYS of an earlier
moment are not new events: report them as a single "replay" event describing
what is being replayed, never as a new goal, shot or save. Close-up shots of
players, coaches or fans are "close_up". If a scoreboard is visible, the score
fields must match it.

Running context from earlier in the match:
{context}

Name the two teams IDENTICALLY to the running context above for every chunk of
this match. If the broadcast identifies the clubs (scoreboard, badges, famous
kits), use the club names; otherwise use shirt colours. Never switch between
the two once established."""


class MatchLog:
    """Running state the commentator and the analyst both read."""

    def __init__(self):
        self.chunks = []          # list of (t0, t1, ChunkAnalysis)
        self.team_a = None
        self.team_b = None

    def context(self, n: int = 3) -> str:
        if not self.chunks:
            return "(nothing yet - this is the opening of the match)"
        lines = []
        if self.team_a:
            lines.append(f"team_a is {self.team_a}; team_b is {self.team_b}. "
                         f"Use these exact names.")
        last = self.chunks[-1][2]
        lines.append(f"Score so far {self.team_a} {last.score_a} - "
                     f"{last.score_b} {self.team_b}; "
                     f"shots {last.shots_a}-{last.shots_b}.")
        lines.append("Recent play:")
        for t0, t1, a in self.chunks[-n:]:
            lines.append(f"  {t0:.0f}-{t1:.0f}s: {a.summary}")
        return "\n".join(lines)

    def add(self, t0, t1, analysis: ChunkAnalysis):
        if self.team_a is None:
            self.team_a, self.team_b = analysis.team_a, analysis.team_b
        self.chunks.append((t0, t1, analysis))


def analyse(client, log: MatchLog, index: int, t0: float, t1: float,
            src=VIDEO) -> ChunkAnalysis:
    path = cut_chunk(t0, t1 - t0, OUT / f"chunk_{index:03d}.mp4", src=src)
    data = base64.b64encode(path.read_bytes()).decode()
    interaction = client.interactions.create(
        model=EYES_MODEL,
        input=[
            {"type": "text",
             "text": PROMPT.format(t0=t0, t1=t1, context=log.context())},
            {"type": "video", "data": data, "mime_type": "video/mp4"},
        ],
        generation_config={"thinking_level": "low"},
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": ChunkAnalysis.model_json_schema(),
        },
    )
    return ChunkAnalysis.model_validate_json(interaction.output_text)


def main():
    client = make_client()
    log = MatchLog()
    duration = video_duration()
    n = int(duration // CHUNK_SECONDS) + (1 if duration % CHUNK_SECONDS > 0.5 else 0)
    print(f"{VIDEO.name}: {duration:.1f}s -> {n} chunks of {CHUNK_SECONDS}s "
          f"with {EYES_MODEL}\n")

    times = []
    for i in range(n):
        t0 = i * CHUNK_SECONDS
        t1 = min(t0 + CHUNK_SECONDS, duration)
        s = time.perf_counter()
        try:
            a = analyse(client, log, i, t0, t1)
        except Exception as exc:
            print(f"  chunk {i} ({t0:.0f}-{t1:.0f}s) FAILED: "
                  f"{type(exc).__name__}: {exc}")
            continue
        ms = (time.perf_counter() - s) * 1000
        times.append(ms)
        log.add(t0, t1, a)
        print(f"  [{t0:5.1f}-{t1:5.1f}s] {ms:6.0f} ms  {a.summary}")
        for e in a.events:
            print(f"              - {e.kind:12s} {e.team:8s} {e.description}")
        print(f"              score {a.score_a}-{a.score_b}  shots {a.shots_a}-{a.shots_b}")

    if times:
        print(f"\n  {len(times)} chunks, mean {sum(times)/len(times):,.0f} ms, "
              f"max {max(times):,.0f} ms  (chunk is {CHUNK_SECONDS}s of video)")
        if sum(times)/len(times) > CHUNK_SECONDS * 1000:
            print("  WARNING: slower than real time - the eyes cannot keep up live")
    (OUT / "match_log.json").write_text(json.dumps(
        [{"t0": t0, "t1": t1, **a.model_dump()} for t0, t1, a in log.chunks], indent=2))
    print(f"\n  match log -> {OUT / 'match_log.json'}")


if __name__ == "__main__":
    sys.exit(main())
