"""Gemini Live playground — local server proxy.

The API key never leaves this process. The browser talks to us over a plain
WebSocket on localhost; we hold the Gemini Live session.

Protocol, browser -> server:
    first message : JSON  {"type": "start", ...mode options}
    then          : binary frames of raw 16-bit mono LE PCM @ 16 kHz
    JSON          : {"type": "audio_stream_end"}   (mic paused / turn done)
    JSON          : {"type": "text", "text": "..."} (live chat only)

Protocol, server -> browser:
    binary frames : raw 16-bit mono LE PCM @ 24 kHz (playback)
    JSON          : {"type": "status"|"you"|"gemini"|"caption"|"tool"|
                             "interrupted"|"thinking"|"error"|"closed", ...}

Run:  .venv/bin/python server/app.py
"""
import asyncio
import datetime
import json
import os
import pathlib
import time
import traceback

from aiohttp import web, WSMsgType
from dotenv import load_dotenv
from google import genai
from google.genai import types

ROOT = pathlib.Path(__file__).resolve().parent.parent
STATIC = ROOT / "server" / "static"
load_dotenv(ROOT / ".env")

if not os.environ.get("GEMINI_API_KEY"):
    raise SystemExit(f"GEMINI_API_KEY not found in {ROOT / '.env'}")

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

LIVE_MODEL = "gemini-3.8-live"
THINKING_MODEL = "gemini-3.8-live-extended-thinking"
TRANSLATE_MODEL = "gemini-3.5-live-translate-preview"
TRANSCRIBE_MODEL = "gemini-3.5-transcribe-live"
LANG_ID_MODEL = "gemini-3.5-flash-lite"

INPUT_MIME = "audio/pcm;rate=16000"   # what the browser sends us


# --------------------------------------------------------------------------
# Tools for the Live chat tab
# --------------------------------------------------------------------------
GET_TIME = types.FunctionDeclaration(
    name="get_time",
    description="Get the current local date and time. Returns instantly.",
    behavior=types.Behavior.NON_BLOCKING,
)

SLOW_SEARCH = types.FunctionDeclaration(
    name="lookup_hackathon_schedule",
    description=(
        "Search the hackathon schedule for what is happening. This is a slow"
        " lookup and takes about six seconds, so tell the user what you are"
        " doing while you wait."
    ),
    behavior=types.Behavior.NON_BLOCKING,
    parameters=types.Schema(
        type="OBJECT",
        properties={"topic": types.Schema(type="STRING", description="what to look up")},
    ),
)

FAKE_SCHEDULE = (
    "10:00 Opening keynote, Hall A. "
    "13:00 Gemini Live API workshop, Hall B. "
    "15:00 Voice agents office hours, Hall B. "
    "19:00 Demos and judging, main stage."
)


async def run_tool(name: str, args: dict, notify) -> dict:
    """Execute a demo tool. The slow one deliberately takes ~6 s."""
    if name == "get_time":
        return {"now": datetime.datetime.now().strftime("%A %d %B %Y, %H:%M:%S")}

    if name == "lookup_hackathon_schedule":
        await notify({"type": "tool", "state": "running", "name": name,
                      "detail": "sleeping 6 s on purpose — listen, it keeps talking"})
        await asyncio.sleep(6)
        return {"topic": args.get("topic", "schedule"), "schedule": FAKE_SCHEDULE}

    return {"error": f"unknown tool {name}"}


# --------------------------------------------------------------------------
# Per-mode session configs
# --------------------------------------------------------------------------
def live_chat_config(opts):
    extended = bool(opts.get("extended_thinking"))
    model = THINKING_MODEL if extended else LIVE_MODEL

    cfg = dict(
        response_modalities=[types.Modality.AUDIO],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        tools=[types.Tool(function_declarations=[GET_TIME, SLOW_SEARCH])],
        system_instruction=types.Content(parts=[types.Part(text=(
            "You are a friendly, concise hackathon buddy exploring the Gemini"
            " Live API. Keep answers short and conversational. When you call a"
            " slow tool, keep talking to the user while you wait — say what"
            " you're doing. Never go silent for more than a second or two."
        ))]),
        # 15 min audio-only limit; compression buys more than that.
        context_window_compression=types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow(),
        ),
        # Push-to-talk: the button is the VAD. Automatic VAD only ends a turn
        # when it hears trailing silence *in the stream*, and releasing the
        # button stops the stream dead — so with auto VAD the turn never ends.
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(disabled=True),
        ),
    )
    if extended:
        cfg["thinking_config"] = types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.LOW
        )
    return model, types.LiveConnectConfig(**cfg)


def translate_config(opts):
    return TRANSLATE_MODEL, types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        translation_config=types.TranslationConfig(
            target_language_code=opts.get("target", "hy"),
            echo_target_language=True,
        ),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(disabled=True),
        ),
    )


def captions_config(opts):
    vocab = [v.strip() for v in (opts.get("vocabulary") or "").split(",") if v.strip()]
    transcription = types.AudioTranscriptionConfig(
        language_codes=[],                       # [] = automatic detection
        mode=types.AudioTranscriptionConfigMode.SMART
        if opts.get("smart") else types.AudioTranscriptionConfigMode.VERBATIM,
    )
    if vocab:
        transcription.custom_vocabulary = vocab[:1000]
    return TRANSCRIBE_MODEL, types.LiveConnectConfig(
        response_modalities=[types.Modality.TEXT],
        input_audio_transcription=transcription,
    )


MODES = {
    "live": live_chat_config,
    "translate": translate_config,
    "captions": captions_config,
}


# --------------------------------------------------------------------------
# Language identification (the Live API does not report the detected
# language — Transcription.language_code comes back None — so we label
# finalized captions with a cheap side call. ~2 s, runs out of band.)
# --------------------------------------------------------------------------
async def identify_language(text: str) -> str | None:
    try:
        interaction = await asyncio.to_thread(
            client.interactions.create,
            model=LANG_ID_MODEL,
            input=("Language of this text? Reply with the BCP-47 code and the"
                   " English name, like 'sv - Swedish'. Nothing else.\n\n" + text),
            generation_config={"thinking_level": "minimal"},
        )
        return (interaction.output_text or "").strip() or None
    except Exception as exc:                       # non-fatal, it's a badge
        print(f"  lang-id failed: {exc}")
        return None


# --------------------------------------------------------------------------
# WebSocket handler
# --------------------------------------------------------------------------
async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    mode = request.match_info["mode"]
    if mode not in MODES:
        raise web.HTTPNotFound(text=f"unknown mode {mode}")

    ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
    await ws.prepare(request)
    lock = asyncio.Lock()

    async def send(obj: dict):
        if not ws.closed:
            async with lock:
                await ws.send_str(json.dumps(obj))

    # ---- wait for the start message -------------------------------------
    opts = {}
    async for msg in ws:
        if msg.type == WSMsgType.TEXT:
            payload = json.loads(msg.data)
            if payload.get("type") == "start":
                opts = payload
                break
    else:
        return ws

    model, config = MODES[mode](opts)
    t0 = time.perf_counter()

    def ms():
        return round((time.perf_counter() - t0) * 1000)

    print(f"[{mode}] connecting to {model} opts={ {k: v for k, v in opts.items() if k != 'type'} }")

    try:
        async with client.aio.live.connect(model=model, config=config) as session:
            await send({"type": "status", "state": "connected", "model": model,
                        "ms": ms()})
            print(f"[{mode}] connected in {ms()} ms")

            pending_tools: set[asyncio.Task] = set()

            # ---- tool execution, off the main receive loop --------------
            async def handle_tool_call(fc):
                await send({"type": "tool", "state": "called", "name": fc.name,
                            "args": dict(fc.args or {}), "ms": ms()})
                result = await run_tool(fc.name, dict(fc.args or {}), send)
                response = types.FunctionResponse(
                    id=fc.id, name=fc.name, response=result
                )
                # gemini-3.8-live-extended-thinking rejects `scheduling`
                # outright (closes the socket with 1007), so only the plain
                # Live model gets it.
                if model == LIVE_MODEL:
                    response.scheduling = types.FunctionResponseScheduling.INTERRUPT
                await session.send_tool_response(function_responses=[response])
                await send({"type": "tool", "state": "returned", "name": fc.name,
                            "result": result, "ms": ms()})

            # ---- browser -> gemini --------------------------------------
            async def pump_in():
                async for msg in ws:
                    if msg.type == WSMsgType.BINARY:
                        await session.send_realtime_input(
                            audio=types.Blob(data=msg.data, mime_type=INPUT_MIME)
                        )
                    elif msg.type == WSMsgType.TEXT:
                        payload = json.loads(msg.data)
                        kind = payload.get("type")
                        if kind == "activity_start":
                            await session.send_realtime_input(
                                activity_start=types.ActivityStart())
                        elif kind == "activity_end":
                            await session.send_realtime_input(
                                activity_end=types.ActivityEnd())
                        elif kind == "audio_stream_end":
                            # Hybrid VAD: works on gemini-3.5-transcribe-live.
                            await session.send_realtime_input(audio_stream_end=True)
                        elif kind == "text":
                            await session.send_realtime_input(text=payload["text"])
                    elif msg.type == WSMsgType.ERROR:
                        break
                print(f"[{mode}] browser closed the socket")

            # ---- gemini -> browser --------------------------------------
            async def pump_out():
                # receive() ends at each turn boundary; re-enter it.
                while not ws.closed:
                    async for message in session.receive():
                        if message.tool_call:
                            for fc in message.tool_call.function_calls:
                                task = asyncio.create_task(handle_tool_call(fc))
                                pending_tools.add(task)
                                task.add_done_callback(pending_tools.discard)

                        if message.go_away:
                            await send({"type": "status", "state": "go_away",
                                        "detail": str(message.go_away), "ms": ms()})

                        sc = message.server_content
                        if not sc:
                            continue

                        # Audio out — process every part in the event.
                        if sc.model_turn and sc.model_turn.parts:
                            for part in sc.model_turn.parts:
                                if part.inline_data and part.inline_data.data:
                                    if not ws.closed:
                                        async with lock:
                                            await ws.send_bytes(part.inline_data.data)

                        if sc.interrupted:
                            await send({"type": "interrupted", "ms": ms()})

                        if sc.interim_input_transcription and sc.interim_input_transcription.text:
                            await send({"type": "caption", "final": False,
                                        "text": sc.interim_input_transcription.text,
                                        "ms": ms()})

                        if sc.input_transcription and sc.input_transcription.text:
                            text = sc.input_transcription.text
                            if mode == "captions":
                                cid = f"c{ms()}"
                                await send({"type": "caption", "final": True,
                                            "text": text, "id": cid,
                                            "lang": sc.input_transcription.language_code,
                                            "ms": ms()})
                                # language badge arrives separately
                                async def label(text=text, cid=cid):
                                    guess = await identify_language(text)
                                    if guess:
                                        await send({"type": "lang", "id": cid,
                                                    "lang": guess})
                                task = asyncio.create_task(label())
                                pending_tools.add(task)
                                task.add_done_callback(pending_tools.discard)
                            else:
                                await send({"type": "you", "text": text, "ms": ms()})

                        if sc.output_transcription and sc.output_transcription.text:
                            await send({"type": "gemini",
                                        "text": sc.output_transcription.text,
                                        "ms": ms()})

                        if sc.interaction_status:
                            await send({"type": "thinking",
                                        "status": str(sc.interaction_status.value
                                                      if hasattr(sc.interaction_status, "value")
                                                      else sc.interaction_status),
                                        "ms": ms()})

                        if sc.turn_complete:
                            await send({"type": "status", "state": "turn_complete",
                                        "ms": ms()})

            tasks = [asyncio.create_task(pump_in()), asyncio.create_task(pump_out())]
            done, rest = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in rest:
                task.cancel()
            for task in pending_tools:
                task.cancel()
            for task in done:
                if task.exception():
                    raise task.exception()

    except asyncio.CancelledError:
        raise
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        print(f"[{mode}] ERROR after {ms()} ms -> {detail}")
        traceback.print_exc()
        await send({"type": "error", "detail": detail, "ms": ms()})

    await send({"type": "closed", "ms": ms()})
    if not ws.closed:
        await ws.close()
    print(f"[{mode}] session ended after {ms()/1000:.1f} s")
    return ws


async def index(request):
    return web.FileResponse(STATIC / "index.html")


def main():
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ws/{mode}", ws_handler)
    app.router.add_static("/static/", STATIC)

    print("\n  Gemini Live playground -> http://localhost:8000")
    print("  (localhost counts as a secure origin, so the mic works)\n")
    web.run_app(app, host="127.0.0.1", port=8000, print=None)


if __name__ == "__main__":
    main()
