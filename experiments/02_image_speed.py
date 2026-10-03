"""Image: generate 3 images with Nano Banana 2 Lite and time each one.

Run:  .venv/bin/python experiments/02_image_speed.py
"""
import base64
import sys

from _common import OUT, make_client, timed

MODEL = "gemini-3.1-flash-lite-image"   # Nano Banana 2 Lite (1K only)
# NB: this model rejects image/png — image/jpeg is the only supported mime_type,
# even though the docs example shows png.

PROMPTS = [
    ("mic", "A glowing retro microphone on a dark stage, volumetric light, cinematic"),
    ("waveform", "An abstract audio waveform made of liquid gold on deep navy, minimal poster art"),
    ("kitchen", "A cozy Armenian grandmother's kitchen at golden hour, warm film photography"),
]


def main():
    client = make_client()
    print(f"{MODEL} — 3 generations, 16:9, 1K\n")

    times = []
    for key, prompt in PROMPTS:
        path = OUT / f"img_{key}.jpg"
        with timed(key) as t:
            interaction = client.interactions.create(
                model=MODEL,
                input=prompt,
                response_format={
                    "type": "image",
                    "mime_type": "image/jpeg",
                    "aspect_ratio": "16:9",
                    "image_size": "1K",
                },
            )
            image = interaction.output_image
            if image is None:
                raise RuntimeError(f"no image returned for {key!r}")
            path.write_bytes(base64.b64decode(image.data))
        times.append(t.ms)
        print(f"     {path.name}  {image.mime_type}  {path.stat().st_size:,} bytes")

    print(f"\n  mean {sum(times)/len(times):,.0f} ms   min {min(times):,.0f}   max {max(times):,.0f}")
    print(f"\nOpen them:  open {OUT}/img_*.jpg")


if __name__ == "__main__":
    sys.exit(main())
