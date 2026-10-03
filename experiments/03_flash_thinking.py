"""Flash: same prompt at thinking_level low vs high, with latency and token counts.

Run:  .venv/bin/python experiments/03_flash_thinking.py
"""
import sys

from _common import make_client, timed

MODEL = "gemini-3.8-flash"

PROMPT = (
    "I'm building a voice-first app. A user says: 'book me a table for four"
    " tomorrow night, somewhere my vegetarian friend won't hate.' List the"
    " tool calls you'd need, in order, and the one piece of information you"
    " must ask the user for before any of them. Be brief."
)


def run(client, level):
    print(f"\n--- thinking_level={level} ---")
    with timed(f"{MODEL} {level}") as t:
        interaction = client.interactions.create(
            model=MODEL,
            input=PROMPT,
            generation_config={
                "thinking_level": level,
                "thinking_summaries": "auto",
            },
        )

    usage = interaction.usage
    print(f"     thought tokens: {usage.total_thought_tokens}"
          f"   output tokens: {usage.total_output_tokens}"
          f"   total: {usage.total_tokens}")

    for step in interaction.steps:
        if step.type == "thought" and step.summary:
            text = " ".join(c.text for c in step.summary if c.type == "text")
            print(f"     thought summary: {text[:200]}...")

    print(f"\n{interaction.output_text}\n")
    return t.ms


def main():
    client = make_client()
    low = run(client, "low")
    high = run(client, "high")
    print(f"=== low {low:,.0f} ms   high {high:,.0f} ms   "
          f"high is {high/low:.1f}x slower ===")


if __name__ == "__main__":
    sys.exit(main())
