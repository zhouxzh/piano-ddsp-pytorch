#!/usr/bin/env python3
"""Serve a browser instrument backed by the stateful ONNX control model."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import secrets
import ssl
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from aiohttp import WSMsgType, web

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ddsp_piano.realtime import (
    MidiTimeline,
    RealtimeOnnxSynthesizer,
    apply_scheduled_midi_event,
    encode_wav_chunk,
    load_midi_timeline,
    restore_midi_timeline_state,
)
from ddsp_piano.model_registry import load_model_registry


WEB_ROOT = ROOT / "web" / "realtime_midi"
DEFAULT_ARTIFACTS_DIR = ROOT / "artifacts" / "model-suite-v1.0.1"
DEFAULT_MIDI_DIR = ROOT / "midi"


@dataclass(frozen=True)
class MidiCatalogEntry:
    id: str
    name: str
    path: Path
    timeline: MidiTimeline


@dataclass(frozen=True)
class ModelAsset:
    model_id: str
    model: Path
    metadata: Path


@dataclass(frozen=True)
class ServerConfig:
    models: dict[str, ModelAsset]
    default_model_id: str
    midi_dir: Path
    chunk_frames: int
    seed: int
    keyoff_fade_ms: float
    all_notes_off_fade_ms: float
    warmup_seconds: float
    access_token: str | None
    max_clients: int
    onnx_intra_op_threads: int
    onnx_inter_op_threads: int
    torch_threads: int
    torch_interop_threads: int


class ClientSession:
    def __init__(
        self,
        websocket: web.WebSocketResponse,
        config: ServerConfig,
        midi_catalog: dict[str, MidiCatalogEntry],
    ) -> None:
        self.websocket = websocket
        self.config = config
        self.midi_catalog = midi_catalog
        self.synth: RealtimeOnnxSynthesizer | None = None
        self.audio_task: asyncio.Task[None] | None = None
        self.midi_task: asyncio.Task[None] | None = None
        self.server_gain = 1.0
        self._sequence = 0
        self._clipped_samples = 0
        self._late_blocks = 0
        self._midi_id: str | None = None
        self._midi_started_at: float | None = None
        self._midi_duration = 0.0
        self._midi_position = 0.0
        self._midi_tempo_scale = 1.0
        self._midi_loop = False
        self._midi_state = "stopped"
        self.model_id = getattr(config, "default_model_id", "gru_ir_96_64")

    async def start(self, model_id: str, piano_model: int, server_gain: float) -> None:
        await self.stop(notify=False)
        if model_id not in self.config.models:
            raise ValueError(f"Unknown model ID: {model_id!r}")
        self.model_id = model_id
        asset = self.config.models[model_id]
        self.server_gain = _bounded_float(server_gain, "server_gain", 0.05, 16.0)
        await self._send_json({"type": "status", "state": "loading"})

        def create_and_warm_up() -> tuple[RealtimeOnnxSynthesizer, int]:
            synth = RealtimeOnnxSynthesizer(
                asset.model,
                asset.metadata,
                piano_model=piano_model,
                chunk_frames=self.config.chunk_frames,
                seed=self.config.seed,
                keyoff_fade_ms=self.config.keyoff_fade_ms,
                all_notes_off_fade_ms=self.config.all_notes_off_fade_ms,
                onnx_intra_op_threads=self.config.onnx_intra_op_threads,
                onnx_inter_op_threads=self.config.onnx_inter_op_threads,
            )
            chunks = synth.warm_up(self.config.warmup_seconds)
            return synth, chunks

        self.synth, warmup_chunks = await asyncio.to_thread(create_and_warm_up)
        self._sequence = 0
        self._clipped_samples = 0
        self._late_blocks = 0
        await self._send_json(
            {
                "type": "status",
                "state": "streaming",
                "warmup_chunks": warmup_chunks,
                "contract": self.synth.describe(),
            }
        )
        self.audio_task = asyncio.create_task(self._stream_audio())

    async def stop(self, notify: bool = True) -> None:
        await self._stop_midi_playback(reset=False, notify=False)
        if self.synth is not None:
            self.synth.midi.panic()
        task, self.audio_task = self.audio_task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self.synth = None
        if notify and not self.websocket.closed:
            await self._send_json({"type": "status", "state": "stopped"})

    async def handle_event(self, payload: dict[str, Any]) -> None:
        event_type = payload.get("type")
        if event_type == "start":
            await self.start(
                str(payload.get("model_id", self.config.default_model_id)),
                _bounded_int(payload.get("piano_model", 9), "piano_model", 0, 127),
                payload.get("server_gain", 1.0),
            )
            return
        if event_type == "stop":
            await self.stop()
            return
        if event_type == "ping":
            await self._send_json(
                {
                    "type": "pong",
                    "client_time": payload.get("client_time"),
                    "server_time": time.time() * 1000.0,
                }
            )
            return
        if self.synth is None:
            raise ValueError("Start the audio stream before sending MIDI events")
        if event_type == "note_on":
            pitch = _bounded_int(payload.get("pitch"), "pitch", 0, 127)
            velocity = _bounded_int(payload.get("velocity"), "velocity", 1, 127)
            if not self.synth.midi.note_on(pitch, velocity):
                raise ValueError("The model accepts piano pitches 21 through 108")
        elif event_type == "note_off":
            pitch = _bounded_int(payload.get("pitch"), "pitch", 0, 127)
            self.synth.midi.note_off(pitch)
        elif event_type == "control_change":
            controller = _bounded_int(payload.get("controller"), "controller", 0, 127)
            value = _bounded_int(payload.get("value"), "value", 0, 127)
            if not self.synth.midi.control_change(controller, value):
                raise ValueError("Only pedal controllers 64 through 67 are supported")
        elif event_type == "panic":
            await self._stop_midi_playback(reset=True, notify=True)
            await self._send_json({"type": "panic_ack"})
        elif event_type == "play_midi":
            midi_id = payload.get("midi_id")
            if not isinstance(midi_id, str) or midi_id not in self.midi_catalog:
                raise ValueError("Unknown MIDI score")
            entry = self.midi_catalog[midi_id]
            await self._start_midi_playback(
                entry,
                _bounded_float(
                    payload.get("position_seconds", 0.0),
                    "position_seconds",
                    0.0,
                    entry.timeline.duration_seconds,
                ),
                _bounded_float(
                    payload.get("tempo_scale", 1.0), "tempo_scale", 0.5, 2.0
                ),
                _boolean(payload.get("loop", False), "loop"),
            )
        elif event_type == "pause_midi":
            await self._pause_midi_playback()
        elif event_type == "resume_midi":
            await self._resume_midi_playback()
        elif event_type == "seek_midi":
            if self._midi_id is None:
                raise ValueError("No MIDI score is loaded")
            await self._seek_midi_playback(
                _bounded_float(
                    payload.get("position_seconds"),
                    "position_seconds",
                    0.0,
                    self._midi_duration,
                )
            )
        elif event_type == "set_midi_transport":
            await self._configure_midi_playback(payload)
        elif event_type == "stop_midi":
            await self._stop_midi_playback(reset=True, notify=True)
        elif event_type == "set_server_gain":
            self.server_gain = _bounded_float(
                payload.get("server_gain"), "server_gain", 0.05, 16.0
            )
        else:
            raise ValueError(f"Unsupported event type: {event_type!r}")

    async def _stream_audio(self) -> None:
        if self.synth is None:
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time()
        next_metrics = deadline
        try:
            while not self.websocket.closed and self.synth is not None:
                chunk = await asyncio.to_thread(self.synth.render_chunk)
                encoded, clipped = encode_wav_chunk(
                    chunk.audio, self.synth.sample_rate, self.server_gain
                )
                self._clipped_samples += clipped
                await self.websocket.send_bytes(encoded)
                self._sequence += 1
                now = loop.time()
                if now >= next_metrics:
                    await self._send_json(
                        {
                            "type": "metrics",
                            "sequence": self._sequence,
                            "render_ms": chunk.render_seconds * 1000.0,
                            "realtime_factor": chunk.render_seconds
                            / self.synth.chunk_seconds,
                            "active_notes": chunk.snapshot.active_notes,
                            "sustain": chunk.snapshot.sustain,
                            "voice_steals": chunk.snapshot.voice_steals,
                            "late_blocks": self._late_blocks,
                            "clipped_samples": self._clipped_samples,
                            "midi_playback": self._midi_playback_snapshot(now),
                        }
                    )
                    next_metrics = now + 0.5

                deadline += self.synth.chunk_seconds
                delay = deadline - loop.time()
                if delay < -self.synth.chunk_seconds:
                    self._late_blocks += 1
                    deadline = loop.time()
                    delay = 0.0
                if delay > 0:
                    await asyncio.sleep(delay)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if self.synth is not None:
                self.synth.midi.panic()
            self.synth = None
            self.audio_task = None
            if not self.websocket.closed:
                await self._send_json(
                    {"type": "error", "message": f"Audio stream failed: {error}"}
                )

    async def _start_midi_playback(
        self,
        entry: MidiCatalogEntry,
        position_seconds: float,
        tempo_scale: float,
        loop_enabled: bool,
    ) -> None:
        await self._stop_midi_playback(reset=True, notify=False)
        self._midi_id = entry.id
        self._midi_duration = entry.timeline.duration_seconds
        self._midi_position = position_seconds
        self._midi_tempo_scale = tempo_scale
        self._midi_loop = loop_enabled
        await self._resume_midi_playback()

    async def _pause_midi_playback(self) -> None:
        if self._midi_id is None or self._midi_state != "playing":
            raise ValueError("No MIDI score is currently playing")
        loop = asyncio.get_running_loop()
        self._midi_position = self._current_midi_position(loop.time())
        await self._cancel_midi_task()
        self._midi_started_at = None
        self._midi_state = "paused"
        if self.synth is not None:
            await asyncio.to_thread(self.synth.hard_reset)
        await self._send_midi_playback("paused")

    async def _resume_midi_playback(self) -> None:
        if self._midi_id is None:
            raise ValueError("No MIDI score is loaded")
        if self._midi_state == "playing":
            return
        entry = self.midi_catalog[self._midi_id]
        await self._prepare_midi_position(entry, self._midi_position)
        self._midi_state = "playing"
        self._midi_started_at = asyncio.get_running_loop().time()
        self.midi_task = asyncio.create_task(self._play_midi_timeline(entry))
        await self._send_midi_playback("playing")

    async def _seek_midi_playback(self, position_seconds: float) -> None:
        if self._midi_id is None:
            raise ValueError("No MIDI score is loaded")
        was_playing = self._midi_state == "playing"
        entry = self.midi_catalog[self._midi_id]
        await self._cancel_midi_task()
        self._midi_position = position_seconds
        self._midi_started_at = None
        if self.synth is not None:
            await asyncio.to_thread(self.synth.hard_reset)
        if was_playing:
            await self._prepare_midi_position(entry, position_seconds, reset=False)
            self._midi_state = "playing"
            self._midi_started_at = asyncio.get_running_loop().time()
            self.midi_task = asyncio.create_task(self._play_midi_timeline(entry))
        else:
            self._midi_state = "paused"
        await self._send_midi_playback(self._midi_state, reason="seek")

    async def _configure_midi_playback(self, payload: dict[str, Any]) -> None:
        if self._midi_id is None:
            raise ValueError("No MIDI score is loaded")
        previous_tempo = self._midi_tempo_scale
        tempo_scale = _bounded_float(
            payload.get("tempo_scale", previous_tempo),
            "tempo_scale",
            0.5,
            2.0,
        )
        loop_enabled = _boolean(payload.get("loop", self._midi_loop), "loop")
        if self._midi_state == "playing" and tempo_scale != previous_tempo:
            now = asyncio.get_running_loop().time()
            position = self._current_midi_position(now)
            await self._cancel_midi_task()
            self._midi_position = position
            self._midi_started_at = None
            self._midi_tempo_scale = tempo_scale
            self._midi_loop = loop_enabled
            entry = self.midi_catalog[self._midi_id]
            await self._prepare_midi_position(entry, position)
            self._midi_started_at = asyncio.get_running_loop().time()
            self.midi_task = asyncio.create_task(self._play_midi_timeline(entry))
        else:
            self._midi_tempo_scale = tempo_scale
            self._midi_loop = loop_enabled
        await self._send_midi_playback(self._midi_state, reason="configured")

    async def _prepare_midi_position(
        self,
        entry: MidiCatalogEntry,
        position_seconds: float,
        reset: bool = True,
    ) -> None:
        if self.synth is None:
            raise ValueError("Start the audio stream before playing a MIDI score")
        if reset:
            await asyncio.to_thread(self.synth.hard_reset)
        restore_midi_timeline_state(
            self.synth.midi, entry.timeline, position_seconds
        )

    async def _cancel_midi_task(self) -> None:
        task, self.midi_task = self.midi_task, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _stop_midi_playback(self, reset: bool, notify: bool) -> None:
        await self._cancel_midi_task()
        stopped_id = self._midi_id
        stopped_duration = self._midi_duration
        self._midi_id = None
        self._midi_started_at = None
        self._midi_duration = 0.0
        self._midi_position = 0.0
        self._midi_state = "stopped"
        if reset and self.synth is not None:
            await asyncio.to_thread(self.synth.hard_reset)
        if notify and stopped_id and not self.websocket.closed:
            await self._send_json(
                {
                    "type": "midi_playback",
                    "state": "stopped",
                    "midi_id": stopped_id,
                    "position_seconds": 0.0,
                    "duration_seconds": stopped_duration,
                    "tempo_scale": self._midi_tempo_scale,
                    "loop": self._midi_loop,
                }
            )

    async def _play_midi_timeline(self, entry: MidiCatalogEntry) -> None:
        if self.synth is None:
            return
        loop = asyncio.get_running_loop()
        try:
            while self.synth is not None:
                start_position = self._midi_position
                started_at = self._midi_started_at
                if started_at is None:
                    return
                tempo_scale = self._midi_tempo_scale
                for event in entry.timeline.events:
                    if event.time_seconds <= start_position:
                        continue
                    target = started_at + (
                        event.time_seconds - start_position
                    ) / tempo_scale
                    delay = target - loop.time()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    if self.synth is None:
                        return
                    apply_scheduled_midi_event(self.synth.midi, event)
                target_end = started_at + (
                    entry.timeline.duration_seconds - start_position
                ) / tempo_scale
                remaining = target_end - loop.time()
                if remaining > 0:
                    await asyncio.sleep(remaining)
                if not self._midi_loop:
                    break
                await asyncio.to_thread(self.synth.hard_reset)
                self._midi_position = 0.0
                restore_midi_timeline_state(self.synth.midi, entry.timeline, 0.0)
                self._midi_started_at = loop.time()
                await self._send_midi_playback("playing", reason="loop")

            if self.synth is not None:
                self.synth.midi.panic()
            self._midi_position = entry.timeline.duration_seconds
            self._midi_started_at = None
            self._midi_state = "ended"
            await self._send_midi_playback("ended")
        except asyncio.CancelledError:
            raise
        finally:
            if self.midi_task is asyncio.current_task():
                self.midi_task = None

    def _current_midi_position(self, now: float) -> float:
        position = self._midi_position
        if self._midi_state == "playing" and self._midi_started_at is not None:
            position += (now - self._midi_started_at) * self._midi_tempo_scale
        return min(max(position, 0.0), self._midi_duration)

    def _midi_playback_snapshot(self, now: float) -> dict[str, Any] | None:
        if self._midi_id is None:
            return None
        return {
            "midi_id": self._midi_id,
            "state": self._midi_state,
            "position_seconds": self._current_midi_position(now),
            "duration_seconds": self._midi_duration,
            "tempo_scale": self._midi_tempo_scale,
            "loop": self._midi_loop,
        }

    async def _send_midi_playback(
        self,
        state: str,
        reason: str | None = None,
    ) -> None:
        if self._midi_id is None or self.websocket.closed:
            return
        now = asyncio.get_running_loop().time()
        payload = {
            "type": "midi_playback",
            "state": state,
            "midi_id": self._midi_id,
            "position_seconds": self._current_midi_position(now),
            "duration_seconds": self._midi_duration,
            "tempo_scale": self._midi_tempo_scale,
            "loop": self._midi_loop,
        }
        if reason is not None:
            payload["reason"] = reason
        await self._send_json(payload)

    async def _send_json(self, payload: dict[str, Any]) -> None:
        await self.websocket.send_str(json.dumps(payload, separators=(",", ":")))


def _bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _bounded_float(value: Any, name: str, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number") from error
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def create_app(config: ServerConfig) -> web.Application:
    app = web.Application(client_max_size=8 * 1024)
    app["config"] = config
    app["clients"] = set()
    app["midi_catalog"] = _load_midi_catalog(config.midi_dir)

    async def page(_: web.Request) -> web.FileResponse:
        return web.FileResponse(WEB_ROOT / "index.html")

    async def asset(request: web.Request) -> web.FileResponse:
        return web.FileResponse(WEB_ROOT / request.path.removeprefix("/"))

    async def health(_: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "ok",
                "models": list(config.models),
                "default_model_id": config.default_model_id,
                "active_clients": len(app["clients"]),
                "max_clients": config.max_clients,
                "midi_files": len(app["midi_catalog"]),
                "runtime_threads": {
                    "onnx_intra_op": config.onnx_intra_op_threads,
                    "onnx_inter_op": config.onnx_inter_op_threads,
                    "torch": config.torch_threads,
                    "torch_interop": config.torch_interop_threads,
                },
            }
        )

    async def websocket_handler(request: web.Request) -> web.StreamResponse:
        if config.access_token and not secrets.compare_digest(
            request.query.get("token", ""), config.access_token
        ):
            raise web.HTTPUnauthorized(text="Invalid WebSocket access token")
        if len(app["clients"]) >= config.max_clients:
            raise web.HTTPServiceUnavailable(text="The realtime session is already in use")
        websocket = web.WebSocketResponse(heartbeat=20.0, max_msg_size=8 * 1024)
        await websocket.prepare(request)
        app["clients"].add(websocket)
        client = ClientSession(websocket, config, app["midi_catalog"])
        try:
            default_asset = config.models[config.default_model_id]
            metadata = json.loads(default_asset.metadata.read_text(encoding="utf-8"))
            await websocket.send_json(
                {
                    "type": "hello",
                    "protocol_version": 2,
                    "model": default_asset.model.name,
                    "model_id": config.default_model_id,
                    "models": [
                        {
                            "model_id": asset.model_id,
                            "display_name": json.loads(
                                asset.metadata.read_text(encoding="utf-8")
                            ).get("display_name", asset.model_id),
                        }
                        for asset in config.models.values()
                    ],
                    "host_dsp_profile": metadata.get("host_dsp_profile", "legacy"),
                    "piano_model_years": metadata.get(
                        "piano_model_index_to_maestro_year", []
                    ),
                    "chunk_frames": config.chunk_frames,
                    "sample_rate": metadata.get("sample_rate"),
                    "binary_format": "mono-pcm16-wav",
                    "runtime_threads": {
                        "onnx_intra_op": config.onnx_intra_op_threads,
                        "onnx_inter_op": config.onnx_inter_op_threads,
                        "torch": config.torch_threads,
                        "torch_interop": config.torch_interop_threads,
                    },
                    "midi_files": [
                        {
                            "id": entry.id,
                            "name": entry.name,
                            "duration_seconds": entry.timeline.duration_seconds,
                            "event_count": len(entry.timeline.events),
                        }
                        for entry in app["midi_catalog"].values()
                    ],
                }
            )
            async for message in websocket:
                if message.type == WSMsgType.TEXT:
                    try:
                        payload = json.loads(message.data)
                        if not isinstance(payload, dict):
                            raise ValueError("WebSocket payload must be a JSON object")
                        await client.handle_event(payload)
                    except (json.JSONDecodeError, ValueError) as error:
                        await websocket.send_json({"type": "error", "message": str(error)})
                    except Exception as error:
                        await websocket.send_json(
                            {"type": "error", "message": f"Server operation failed: {error}"}
                        )
                elif message.type == WSMsgType.ERROR:
                    break
        finally:
            await client.stop(notify=False)
            app["clients"].discard(websocket)
        return websocket

    app.router.add_get("/", page)
    app.router.add_get("/healthz", health)
    app.router.add_get("/ws", websocket_handler)
    for name in ("app.js", "styles.css", "pcm-worklet.js"):
        app.router.add_get(f"/{name}", asset, name=f"asset-{name}")
    return app


def _load_midi_catalog(midi_dir: Path) -> dict[str, MidiCatalogEntry]:
    midi_dir = midi_dir.resolve()
    catalog: dict[str, MidiCatalogEntry] = {}
    paths = sorted(
        path for path in midi_dir.rglob("*") if path.suffix.lower() in {".mid", ".midi"}
    )
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file() or not resolved.is_relative_to(midi_dir):
            continue
        midi_id = resolved.relative_to(midi_dir).as_posix()
        catalog[midi_id] = MidiCatalogEntry(
            id=midi_id,
            name=resolved.stem.replace("-", " "),
            path=resolved,
            timeline=load_midi_timeline(resolved),
        )
    return catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument(
        "--model-id",
        help="Initial model ID; defaults to the registry's default_model_id",
    )
    parser.add_argument("--midi-dir", type=Path, default=DEFAULT_MIDI_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--chunk-frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--keyoff-fade-ms",
        type=float,
        default=60.0,
        help="Per-voice dry-signal fade after Note Off (default: 60 ms)",
    )
    parser.add_argument(
        "--all-notes-off-fade-ms",
        type=float,
        default=120.0,
        help="Master fade after every Note Off gate closes (default: 120 ms)",
    )
    parser.add_argument("--warmup-seconds", type=float, default=0.5)
    parser.add_argument("--max-clients", type=int, default=1)
    parser.add_argument("--onnx-intra-op-threads", type=int, default=1)
    parser.add_argument("--onnx-inter-op-threads", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--access-token")
    parser.add_argument("--ssl-cert", type=Path)
    parser.add_argument("--ssl-key", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not WEB_ROOT.is_dir():
        raise FileNotFoundError(f"Browser application not found: {WEB_ROOT}")
    registry = load_model_registry()
    default_spec = registry.require(args.model_id or registry.default_model_id)
    artifacts_dir = args.artifacts_dir.resolve()
    models: dict[str, ModelAsset] = {}
    for model_id, spec in registry.models.items():
        model = spec.asset_path(artifacts_dir, ".onnx")
        metadata = spec.asset_path(artifacts_dir, ".json")
        if model.is_file() and metadata.is_file():
            models[model_id] = ModelAsset(model_id, model, metadata)
    if default_spec.model_id not in models:
        raise FileNotFoundError(
            f"Model {default_spec.model_id!r} is missing from {artifacts_dir}; "
            "run scripts/prepare_release.py first"
        )
    config = ServerConfig(
        models=models,
        default_model_id=default_spec.model_id,
        midi_dir=args.midi_dir.resolve(),
        chunk_frames=args.chunk_frames,
        seed=args.seed,
        keyoff_fade_ms=args.keyoff_fade_ms,
        all_notes_off_fade_ms=args.all_notes_off_fade_ms,
        warmup_seconds=args.warmup_seconds,
        access_token=args.access_token,
        max_clients=args.max_clients,
        onnx_intra_op_threads=args.onnx_intra_op_threads,
        onnx_inter_op_threads=args.onnx_inter_op_threads,
        torch_threads=args.torch_threads,
        torch_interop_threads=args.torch_interop_threads,
    )
    if not config.midi_dir.is_dir():
        raise FileNotFoundError(f"MIDI directory not found: {config.midi_dir}")
    if (
        config.chunk_frames <= 0
        or config.max_clients <= 0
        or config.keyoff_fade_ms <= 0
        or config.all_notes_off_fade_ms <= 0
        or config.warmup_seconds < 0
        or config.onnx_intra_op_threads <= 0
        or config.onnx_inter_op_threads <= 0
        or config.torch_threads <= 0
        or config.torch_interop_threads <= 0
    ):
        raise ValueError(
            "chunk frames/max clients/fades/runtime threads must be positive "
            "and warmup non-negative"
        )

    torch.set_num_threads(config.torch_threads)
    try:
        torch.set_num_interop_threads(config.torch_interop_threads)
    except RuntimeError:
        if torch.get_num_interop_threads() != config.torch_interop_threads:
            raise

    ssl_context = None
    if bool(args.ssl_cert) != bool(args.ssl_key):
        raise ValueError("--ssl-cert and --ssl-key must be supplied together")
    if args.ssl_cert:
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(args.ssl_cert, args.ssl_key)
    web.run_app(create_app(config), host=args.host, port=args.port, ssl_context=ssl_context)


if __name__ == "__main__":
    main()
