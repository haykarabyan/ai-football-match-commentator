"""TTS: design 3 voices from text descriptions, say the same line in each,
plus one line in Armenian. Writes .wav files into experiments/out/.

Run:  .venv/bin/python experiments/01_tts_voices.py
"""
import base64
import sys

from _common import OUT, make_client, timed

MODEL = "gemini-3.8-flash-tts"
FAST_MODEL = "gemini-3.8-flash-lite-tts"

LINE = "Welcome to the hackathon. Let me show you what this voice can do."
ARMENIAN_LINE = "Բարի գալուստ հաքաթոն։ Եկեք տեսնենք, թե ինչ կարող է անել այս ձայնը։"

VOICES = [
    {
        "key": "grandmother",
        "display_name": "Warm Armenian Grandmother",
        "gender": "female",
        "language_code": "hy-AM",
        "description": (
            "A warm, loving Armenian grandmother in her late seventies. She speaks"
            " slowly and gently with a soft Armenian accent, lots of affection in"
            " her voice, as if talking to a grandchild she has not seen in months."
        ),
        "style": "warm, affectionate, unhurried",
    },
    {
        "key": "commentator",
        "display_name": "Energetic Sports Commentator",
        "gender": "male",
        "language_code": "en-US",
        "description": (
            "A high-energy stadium sports commentator in his forties. Fast, punchy"
            " delivery, rising excitement, big breath support, the kind of voice"
            " that makes a routine pass sound like a championship moment."
        ),
        "style": "excited and fast, building to a peak",
    },
    {
        "key": "nurse",
        "display_name": "Calm Swedish Nurse",
        "gender": "female",
        "language_code": "sv-SE",
        "description": (
            "A calm, reassuring Swedish nurse in her thirties speaking English with"
            " a light Swedish accent. Measured pace, low and steady pitch, the"
            " unflappable tone of someone who has handled every emergency twice."
        ),
        "style": "calm, steady, reassuring",
    },
]


def synthesize(client, model, text, voice_id, style, path):
    content = {"type": "text", "text": text}
    if style:
        content["annotations"] = [{"type": "speech_metadata", "style": style}]

    interaction = client.interactions.create(
        model=model,
        input=[{"type": "user_input", "content": [content]}],
        response_format={"type": "audio"},          # defaults to audio/wav
        generation_config={"speech_config": [{"voice": voice_id}]},
    )
    audio = interaction.output_audio
    if audio is None:
        raise RuntimeError("no audio in response; steps=%r" % (interaction.steps,))
    path.write_bytes(base64.b64decode(audio.data))
    return audio.mime_type, path.stat().st_size


def main():
    client = make_client()
    designed = []

    print(f"Designing {len(VOICES)} voices with {MODEL} (Voice Design)")
    for spec in VOICES:
        with timed(f"design {spec['key']}"):
            created = client.voices.create(
                store=True,
                voice={
                    "model": MODEL,
                    "type": "prompted",
                    "display_name": spec["display_name"],
                    "gender": spec["gender"],
                    "language_code": spec["language_code"],
                    "prompted": {"input": spec["description"]},
                },
            )
        print(f"     voice id: {created.id}")
        designed.append((spec, created))

        # Voice Design hands back a preview sample for free — keep it.
        if created.sample_audio and created.sample_audio.data:
            p = OUT / f"voice_preview_{spec['key']}.wav"
            p.write_bytes(base64.b64decode(created.sample_audio.data))
            print(f"     preview -> {p.name}")

    print(f"\nSynthesizing the same English line in each voice ({MODEL})")
    for spec, created in designed:
        path = OUT / f"tts_{spec['key']}.wav"
        with timed(f"say  {spec['key']}"):
            mime, size = synthesize(client, MODEL, LINE, created.id, spec["style"], path)
        print(f"     {path.name}  {mime}  {size:,} bytes")

    print("\nSame grandmother voice, Armenian text")
    gm_spec, gm_voice = designed[0]
    path = OUT / "tts_armenian_grandmother.wav"
    with timed("say  armenian"):
        mime, size = synthesize(client, MODEL, ARMENIAN_LINE, gm_voice.id, gm_spec["style"], path)
    print(f"     {path.name}  {mime}  {size:,} bytes")

    print(f"\nSame Armenian line on the fast model ({FAST_MODEL}) for comparison")
    path = OUT / "tts_armenian_grandmother_lite.wav"
    with timed("say  armenian (lite)"):
        mime, size = synthesize(client, FAST_MODEL, ARMENIAN_LINE, gm_voice.id, gm_spec["style"], path)
    print(f"     {path.name}  {mime}  {size:,} bytes")

    print(f"\nWav files are in {OUT}")
    print("Play one:  afplay experiments/out/tts_grandmother.wav")


if __name__ == "__main__":
    sys.exit(main())
