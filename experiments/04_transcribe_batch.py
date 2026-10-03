"""Batch transcribe: speaker diarization + word timestamps on a TTS-generated file.

Generates a two-speaker dialogue with gemini-3.8-flash-tts (so diarization has
something to separate), then transcribes it with gemini-3.5-transcribe.

Run:  .venv/bin/python experiments/04_transcribe_batch.py
"""
import base64
import sys

from _common import OUT, make_client, timed

TTS_MODEL = "gemini-3.8-flash-tts"
MODEL = "gemini-3.5-transcribe"

DIALOGUE = [
    ("Joe", "cheerful and curious",
     "So this whole thing runs on Gemini three point eight Live?"),
    ("Jane", "calm and confident",
     "Yes, and the transcription is a separate model called Gemini three point five Transcribe."),
    ("Joe", "surprised",
     "Wait, it does speaker labels and word timestamps in the same call?"),
    ("Jane", "calm and confident",
     "It does, as long as you stay in verbatim mode."),
]

DIALOGUE_WAV = OUT / "dialogue_two_speakers.wav"


def make_dialogue(client):
    if DIALOGUE_WAV.exists():
        print(f"Reusing {DIALOGUE_WAV.name}")
        return
    print(f"Generating a 2-speaker dialogue with {TTS_MODEL}")
    with timed("multi-speaker tts"):
        interaction = client.interactions.create(
            model=TTS_MODEL,
            input=[{
                "type": "user_input",
                "content": [
                    {
                        "type": "text",
                        "text": text,
                        "annotations": [{
                            "type": "speech_metadata",
                            "speaker": speaker,
                            "style": style,
                        }],
                    }
                    for speaker, style, text in DIALOGUE
                ],
            }],
            response_format={"type": "audio"},
            generation_config={
                "speech_config": {
                    "speakers": [
                        {"speaker": "Joe", "voice": "Puck"},
                        {"speaker": "Jane", "voice": "Kore"},
                    ],
                }
            },
        )
    DIALOGUE_WAV.write_bytes(base64.b64decode(interaction.output_audio.data))
    print(f"     {DIALOGUE_WAV.name}  {DIALOGUE_WAV.stat().st_size:,} bytes")


def word_annotations(interaction):
    words = []
    for step in getattr(interaction, "steps", []) or []:
        for content in getattr(step, "content", []) or []:
            for ann in getattr(content, "annotations", []) or []:
                if getattr(ann, "type", None) == "word_info":
                    words.append(ann)
    return words


def main():
    client = make_client()
    make_dialogue(client)

    print(f"\nUploading to the Files API")
    with timed("upload"):
        audio_file = client.files.upload(file=str(DIALOGUE_WAV))
    print(f"     {audio_file.uri}  ({audio_file.mime_type})")

    # Pass 1: verbatim + diarization + word timestamps.
    print(f"\n{MODEL} — verbatim, diarization, word timestamps")
    with timed("transcribe"):
        interaction = client.interactions.create(
            model=MODEL,
            input=[{
                "type": "audio",
                "uri": audio_file.uri,
                "mime_type": audio_file.mime_type,
            }],
            generation_config={
                "transcription_config": {
                    "language_codes": [],          # auto-detect
                    "mode": {
                        "type": "verbatim",
                        "diarization_mode": "speaker",
                        "timestamp_granularities": ["word"],
                    },
                }
            },
        )

    print(f"\n  transcript:\n    {interaction.output_text}\n")

    words = word_annotations(interaction)
    print(f"  {len(words)} word_info annotations")
    if words:
        print("\n  grouped by speaker turn:")
        current, buf, t0 = None, [], None
        for w in words:
            spk = getattr(w, "speaker", None)
            if spk != current:
                if buf:
                    print(f"    [{current}] {t0} -> {last_end}  {' '.join(buf)}")
                current, buf, t0 = spk, [], getattr(w, "start_offset", "?")
            buf.append(w.text)
            last_end = getattr(w, "end_offset", "?")
        if buf:
            print(f"    [{current}] {t0} -> {last_end}  {' '.join(buf)}")

        print("\n  first 8 words raw:")
        for w in words[:8]:
            print(f"    {getattr(w, 'speaker', '-'):>6}  "
                  f"{getattr(w, 'start_offset', '?'):>8} -> {getattr(w, 'end_offset', '?'):<8}  {w.text}")

    # Pass 2: custom vocabulary (cannot be combined with the options above).
    print(f"\n{MODEL} — custom_vocabulary pass (incompatible with diarization/timestamps)")
    with timed("transcribe + vocab"):
        vocab_interaction = client.interactions.create(
            model=MODEL,
            input=[{
                "type": "audio",
                "uri": audio_file.uri,
                "mime_type": audio_file.mime_type,
            }],
            generation_config={
                "transcription_config": {
                    "custom_vocabulary": ["Gemini", "Live API", "diarization", "verbatim"],
                }
            },
        )
    print(f"\n  transcript:\n    {vocab_interaction.output_text}")


if __name__ == "__main__":
    sys.exit(main())
