"""Shared bits for the hybrid commentator (eyes = flash on video, voice = TTS)."""
import base64
import hashlib
import json
import os
import pathlib
import subprocess
import wave

from dotenv import load_dotenv
from google import genai
from google.genai import types

ROOT = pathlib.Path(__file__).resolve().parent.parent
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")
if not os.environ.get("GEMINI_API_KEY"):
    raise SystemExit(f"GEMINI_API_KEY not found in {ROOT / '.env'}")

# Clips the UI can pick between. Keys are what the browser sends. The "final_*"
# clips are 30s cuts of final_game.mp4 (2020 UCL final, PSG v Bayern, BT Sport);
# see the README for the timestamps they were cut from.
CLIPS = {
    "demo":        (ROOT / "demo_video.mp4", "Stock clip: red v white, goal"),
    "neymar":      (ROOT / "final_neymar_chance.mp4", "UCL final 17': Neymar through on Neuer"),
    "lewandowski": (ROOT / "final_lewandowski_chance.mp4", "UCL final 21': Lewandowski chance"),
    "midfield":    (ROOT / "final_game_20s.mp4", "UCL final 45': second-half restart, 20s"),
    "goal":        (ROOT / "final_coman_goal.mp4", "UCL final 59': Coman header, 1-0"),
    "whistle":     (ROOT / "final_final_whistle.mp4", "UCL final 90+': final whistle"),
}
VIDEOS = {k: path for k, (path, _) in CLIPS.items()}
DEFAULT_VIDEO = "demo"     # the clip with a goal in it: the best demo moment
VIDEO = VIDEOS[DEFAULT_VIDEO]


def video_path(key: str) -> pathlib.Path:
    return VIDEOS.get(key, VIDEOS[DEFAULT_VIDEO])

EYES_MODEL = "gemini-3.8-flash"        # sees real video chunks
OUTPUT_RATE = 24000                    # TTS audio comes back at 24 kHz

CHUNK_SECONDS = 4.0

# Default commentator: a Voice Design id from out/designed_voices.json
# ("Roaring stadium announcer").
VOICE = "voice_83ilus4bci19"

TTS_MODEL = "gemini-3.8-flash-tts"      # Voice Design runs on the TTS model
# Writes each line, which TTS then speaks. (gemini-3.8-live could do both in one
# hop, but it ignores Voice Design ids: any "voice_..." value, even a made-up
# one, connects without error and falls back to a stock voice.)
WRITER_MODEL = "gemini-3.5-flash-lite"

# What the preview says. Voice Design's own sample_audio is a minute of generic
# talk unrelated to football, so the preview speaks a commentary line instead.
PREVIEW_LINE = ("Here he comes, cutting inside, past one, past two... he shoots! "
                "What a save from the keeper!")

# Designing a voice takes 13-21 s, so cache by description. Also protects the
# 200-stored-voices-per-project limit from repeated identical prompts.
VOICE_CACHE = OUT / "designed_voices.json"


def _cache() -> dict:
    if VOICE_CACHE.exists():
        try:
            return json.loads(VOICE_CACHE.read_text())
        except Exception:
            return {}
    return {}


def design_voice(client, description: str, gender: str = "male",
                 language_code: str = "en-GB") -> dict:
    """Create (or reuse) a voice from a prose description.

    Returns {"id", "preview_b64", "cached"}. preview_b64 is a ready-to-play WAV
    of the new voice calling a bit of football (see commentary_preview).
    """
    description = description.strip()
    if not description:
        raise ValueError("voice description is empty")
    key = hashlib.sha256(
        f"{description}|{gender}|{language_code}".encode()).hexdigest()[:16]

    cache = _cache()
    if key in cache:
        vid = cache[key]["id"]
        return {"id": vid, "preview_b64": commentary_preview(client, vid),
                "cached": True}

    created = client.voices.create(
        store=True,
        voice={
            "model": TTS_MODEL,
            "type": "prompted",
            "display_name": description[:48],
            "gender": gender,
            "language_code": language_code,
            "prompted": {"input": description},
        },
    )
    cache[key] = {"id": created.id, "description": description}
    VOICE_CACHE.write_text(json.dumps(cache, indent=2))
    return {"id": created.id,
            "preview_b64": commentary_preview(client, created.id),
            "cached": False}


def commentary_preview(client, voice: str) -> str:
    """Base64 WAV of PREVIEW_LINE spoken in this voice."""
    interaction = client.interactions.create(
        model=TTS_MODEL,
        input=[{"type": "user_input", "content": [{
            "type": "text", "text": PREVIEW_LINE,
            "annotations": [{"type": "speech_metadata",
                             "style": "live football commentary, rising excitement"}],
        }]}],
        generation_config={"speech_config": [{"voice": voice}]},
    )
    data = interaction.output_audio.data
    return data if isinstance(data, str) else base64.b64encode(data).decode()

FFMPEG = "ffmpeg"


def make_client() -> genai.Client:
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def video_duration(path: pathlib.Path = VIDEO) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def cut_chunk(start: float, seconds: float, dest: pathlib.Path,
              src: pathlib.Path = VIDEO) -> pathlib.Path:
    """Re-encode a short standalone chunk (copy would break on keyframes)."""
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error",
         "-ss", f"{start}", "-t", f"{seconds}", "-i", str(src),
         "-c:v", "libx264", "-preset", "ultrafast", "-an", str(dest), "-y"],
        check=True,
    )
    return dest


def write_wav(path: pathlib.Path, pcm: bytes, rate: int = OUTPUT_RATE):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return path
