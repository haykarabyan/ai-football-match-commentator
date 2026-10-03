"""Async pipelined eyes: keep N chunk analyses in flight, emit them in order.

Sequentially the eyes run ~1.27x slower than real time, so they fall behind.
Chunk 0 runs alone to establish the team colours; after that a bounded number
run concurrently, which is what lets the eyes stay ahead of playback.

Measure the whole clip:  ../.venv/bin/python pipeline.py [concurrency]
"""
import asyncio
import sys
import time

from common import CHUNK_SECONDS, VIDEO, make_client, video_duration
from eyes import MatchLog, analyse


class Eyes:
    def __init__(self, client, concurrency: int = 2, on_chunk=None, src=VIDEO):
        self.client = client
        self.src = src
        self.log = MatchLog()
        self.sem = asyncio.Semaphore(concurrency)
        self.on_chunk = on_chunk          # async callback(index, t0, t1, analysis)

    async def _one(self, i, t0, t1):
        async with self.sem:
            s = time.perf_counter()
            # analyse() is blocking (ffmpeg + a REST call), so keep it off the loop
            a = await asyncio.to_thread(analyse, self.client, self.log, i, t0, t1,
                                        self.src)
            return i, t0, t1, a, (time.perf_counter() - s) * 1000

    async def run(self, duration: float):
        n = int(duration // CHUNK_SECONDS) + (1 if duration % CHUNK_SECONDS > 0.5 else 0)

        # Chunk 0 alone: it fixes the team colours every later chunk inherits.
        i0, t0, t1, a0, ms0 = await self._one(0, 0.0, min(CHUNK_SECONDS, duration))
        self.log.add(t0, t1, a0)
        if self.on_chunk:
            await self.on_chunk(0, t0, t1, a0, ms0)

        tasks = [
            asyncio.create_task(self._one(i, i * CHUNK_SECONDS,
                                          min((i + 1) * CHUNK_SECONDS, duration)))
            for i in range(1, n)
        ]
        # Emit strictly in order so the commentary never jumps around.
        for t in tasks:
            i, ct0, ct1, a, ms = await t
            self.log.add(ct0, ct1, a)
            if self.on_chunk:
                await self.on_chunk(i, ct0, ct1, a, ms)


async def main():
    conc = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    client = make_client()
    duration = video_duration()
    t0 = time.perf_counter()

    async def show(i, a0, a1, a, ms):
        el = (time.perf_counter() - t0)
        ahead = a1 - el
        print(f"  chunk {i} [{a0:5.1f}-{a1:5.1f}s]  took {ms:6.0f} ms  "
              f"ready at {el:5.1f}s wall  ({ahead:+5.1f}s vs playback)  {a.summary[:60]}")

    print(f"concurrency={conc}, {duration:.1f}s of video\n")
    eyes = Eyes(client, concurrency=conc, on_chunk=show)
    await eyes.run(duration)
    total = time.perf_counter() - t0
    print(f"\n  whole clip analysed in {total:.1f}s of wall time "
          f"for {duration:.1f}s of video = {total/duration:.2f}x real time")
    print("  -> eyes keep up" if total < duration else
          "  -> eyes still too slow; raise concurrency or chunk length")

if __name__ == "__main__":
    asyncio.run(main())
