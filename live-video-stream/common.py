"""Shared bits for the Live video commentator."""
import base64
import hashlib
import json
import os
import pathlib

from dotenv import load_dotenv
from google import genai

ROOT = pathlib.Path(__file__).resolve().parent.parent
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")
if not os.environ.get("GEMINI_API_KEY"):
    raise SystemExit(f"GEMINI_API_KEY not found in {ROOT / '.env'}")

LIVE_MODEL = "gemini-3.8-live"        # watches the frame stream, writes the lines
TTS_MODEL = "gemini-3.8-flash-tts"    # speaks them in a Voice Design voice
OUTPUT_RATE = 24000                   # both Live and TTS return 24 kHz PCM

# Clips the UI can pick: key -> (file, label, match context). The final_* clips
# are cut from the 2020 UCL final (see the top-level README). The context goes
# into the system prompt so the model can name teams and players.
UCL = ("2020 Champions League final in Lisbon, Paris Saint-Germain (dark blue "
       "shirts) v Bayern Munich (red shirts), BT Sport broadcast with a "
       "scoreboard overlay. Starting line-ups, shirt numbers: "
       "PSG (coach Thomas Tuchel): 1 Keylor Navas; 4 Thilo Kehrer, 2 Thiago "
       "Silva (captain), 3 Presnel Kimpembe, 14 Juan Bernat; 21 Ander Herrera, "
       "5 Marquinhos, 8 Leandro Paredes; 11 Angel Di Maria, 7 Kylian Mbappe "
       "(centre forward), 10 Neymar. "
       "Bayern (coach Hansi Flick): 1 Manuel Neuer (captain); 32 Joshua Kimmich, "
       "17 Jerome Boateng (replaced early by 4 Niklas Sule), 27 David Alaba, "
       "19 Alphonso Davies; 6 Thiago Alcantara, 18 Leon Goretzka; 22 Serge "
       "Gnabry, 25 Thomas Muller, 29 Kingsley Coman; 9 Robert Lewandowski "
       "(centre forward).")
CLIPS = {
    "goal":        (ROOT / "final_coman_goal.mp4", "Coman's winning header · 0:30", UCL),
    "neymar":      (ROOT / "final_neymar_chance.mp4", "Neymar through on Neuer · 0:30", UCL),
    "lewandowski": (ROOT / "final_lewandowski_chance.mp4", "Lewandowski's chance · 0:30", UCL),
    "midfield":    (ROOT / "final_game_20s.mp4", "Second-half kick-off · 0:20", UCL),
    "whistle":     (ROOT / "final_final_whistle.mp4", "The final whistle · 0:30", UCL),
    # Longer cuts, for sessions past Live's 2 min video limit.
    "long_goal":   (ROOT / "final_long_coman_goal.mp4", "Build-up to Coman's goal · 3:00", UCL),
    "long_neymar": (ROOT / "final_long_neymar.mp4", "Neymar's chance, extended · 2:30", UCL),
    "long_whistle": (ROOT / "final_long_final_whistle.mp4", "Stoppage time to full time · 3:00", UCL),
    "demo":        (ROOT / "demo_video.mp4", "Red v white, stock footage · 0:28",
                    "A football match, red shirts v white shirts. Club names are unknown."),
}
DEFAULT_CLIP = "goal"

# Native mode: Live speaks for itself in one of these stock voices.
PREBUILT_VOICES = ["Puck", "Charon", "Fenrir", "Orus", "Kore", "Aoede", "Leda", "Zephyr"]
DEFAULT_PREBUILT = "Fenrir"   # also Live's (unheard) voice in designed mode

# Designed mode. Live accepts voice_config.voice="voice_..." but ignores it:
# a made-up id connects just the same and every id comes out as one stock voice.
# So designed voices go through TTS.
VOICE_CACHE = OUT / "designed_voices.json"
DEFAULT_DESIGNED = "voice_83ilus4bci19"   # "Roaring stadium announcer"

PREVIEW_LINE = ("Here he comes, cutting inside, past one, past two... he shoots! "
                "What a save from the keeper!")


def make_client() -> genai.Client:
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def clip(key: str) -> tuple[pathlib.Path, str, str]:
    """(file, label, match context) for a clip key, or the default clip."""
    return CLIPS.get(key, CLIPS[DEFAULT_CLIP])


def voice_cache() -> dict:
    try:
        return json.loads(VOICE_CACHE.read_text())
    except Exception:
        return {}


def designed_voices() -> list[dict]:
    return [{"id": v["id"], "name": v.get("name") or v["description"][:44],
             "description": v["description"]} for v in voice_cache().values()]


def persona_of(voice_id: str) -> str | None:
    """The prose that designed a voice; it also shapes what the voice says."""
    return next((v["description"] for v in voice_cache().values()
                 if v["id"] == voice_id), None)


def design_voice(client, description: str, name: str = "", gender: str = "male",
                 language_code: str = "en-GB") -> dict:
    """Create (or reuse) a voice from a prose description. Takes ~15-25 s.

    Identical descriptions reuse the cached id: a project can store only 200.
    Returns {"id", "name", "cached"}.
    """
    description = description.strip()
    if not description:
        raise ValueError("voice description is empty")
    key = hashlib.sha256(f"{description}|{gender}|{language_code}".encode()).hexdigest()[:16]
    cache = voice_cache()
    if key in cache:
        if name:
            cache[key]["name"] = name
            VOICE_CACHE.write_text(json.dumps(cache, indent=2))
        return {"id": cache[key]["id"], "name": cache[key].get("name", name), "cached": True}

    created = client.voices.create(
        store=True,
        voice={
            "model": TTS_MODEL,
            "type": "prompted",
            "display_name": (name or description)[:48],
            "gender": gender,
            "language_code": language_code,
            "prompted": {"input": description},
        },
    )
    cache[key] = {"id": created.id, "description": description,
                  "name": name or description[:44]}
    VOICE_CACHE.write_text(json.dumps(cache, indent=2))
    return {"id": created.id, "name": cache[key]["name"], "cached": False}


async def commentary_preview(client, voice: str) -> str:
    """Base64 WAV of PREVIEW_LINE in this designed voice."""
    interaction = await client.aio.interactions.create(
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

