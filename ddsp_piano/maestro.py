"""TensorFlow-free MAESTRO loading, MIDI conditioning, and cache management."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import mido
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm


MAESTRO_CSV = "maestro-v3.0.0.csv"
REQUIRED_COLUMNS = {"split", "year", "midi_filename", "audio_filename"}


@dataclass(frozen=True)
class PreprocessConfig:
    sample_rate: int = 16_000
    frame_rate: int = 250
    segment_seconds: float = 3.0
    overlap: float = 0.5
    max_polyphony: int = 16

    @property
    def segment_samples(self) -> int:
        return int(round(self.sample_rate * self.segment_seconds))

    @property
    def segment_frames(self) -> int:
        return int(round(self.frame_rate * self.segment_seconds))

    @property
    def hop_samples(self) -> int:
        return int(round(self.segment_samples * (1.0 - self.overlap)))

    @property
    def hop_frames(self) -> int:
        return int(round(self.segment_frames * (1.0 - self.overlap)))

    def validate(self) -> None:
        if self.sample_rate <= 0 or self.frame_rate <= 0:
            raise ValueError("sample_rate and frame_rate must be positive")
        if self.segment_samples <= 0 or self.segment_frames <= 0:
            raise ValueError("segment_seconds must produce non-empty segments")
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError("overlap must be in [0, 1)")
        if self.max_polyphony <= 0:
            raise ValueError("max_polyphony must be positive")
        if self.sample_rate % self.frame_rate != 0:
            raise ValueError("sample_rate must be divisible by frame_rate")

    def signature(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class MaestroTrack:
    audio_rel: str
    midi_rel: str
    year: int
    piano_id: int
    split: str


def _read_rows(maestro_root: Path) -> list[dict[str, str]]:
    csv_path = maestro_root / MAESTRO_CSV
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing MAESTRO metadata: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(f"{csv_path} does not have the MAESTRO v3 columns")
        return list(reader)


def load_tracks(maestro_root: Path, split: str) -> tuple[list[MaestroTrack], list[int]]:
    rows = _read_rows(maestro_root)
    years = sorted({int(row["year"]) for row in rows})
    piano_ids = {year: index for index, year in enumerate(years)}
    tracks = [
        MaestroTrack(
            audio_rel=row["audio_filename"],
            midi_rel=row["midi_filename"],
            year=int(row["year"]),
            piano_id=piano_ids[int(row["year"])],
            split=row["split"],
        )
        for row in rows
        if row["split"] == split
    ]
    if not tracks:
        raise ValueError(f"No MAESTRO rows found for split={split!r}")
    return tracks, years


def validate_maestro(maestro_root: Path, splits: Iterable[str]) -> dict[str, int]:
    report: dict[str, int] = {}
    for split in splits:
        tracks, _ = load_tracks(maestro_root, split)
        missing_audio = sum(not (maestro_root / track.audio_rel).is_file() for track in tracks)
        missing_midi = sum(not (maestro_root / track.midi_rel).is_file() for track in tracks)
        report[f"{split}_tracks"] = len(tracks)
        report[f"{split}_missing_audio"] = missing_audio
        report[f"{split}_missing_midi"] = missing_midi
    return report


def _cache_key(track: MaestroTrack, config: PreprocessConfig) -> str:
    value = "\0".join((track.audio_rel, track.midi_rel, str(track.year), config.signature()))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _cache_dir(cache_root: Path, track: MaestroTrack, config: PreprocessConfig) -> Path:
    return cache_root / "tracks" / _cache_key(track, config)


def _cache_complete(path: Path, config: PreprocessConfig) -> bool:
    required = ("audio.npy", "conditioning.npy", "pedal.npy", "polyphony.npy", "metadata.json")
    if not path.is_dir() or not all((path / name).is_file() for name in required):
        return False
    try:
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        return metadata.get("config") == asdict(config)
    except (OSError, ValueError, TypeError):
        return False


def _load_audio(audio_path: Path, sample_rate: int) -> np.ndarray:
    waveform, source_rate = torchaudio.load(str(audio_path))
    if waveform.numel() == 0:
        raise ValueError(f"Empty audio file: {audio_path}")
    waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    return waveform.squeeze(0).contiguous().numpy().astype(np.float32, copy=False)


def _midi_roll(midi_path: Path, n_frames: int, frame_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Return active notes/onset velocities and four pedal controls per frame."""
    midi = mido.MidiFile(str(midi_path))
    scheduled: dict[int, list[object]] = defaultdict(list)
    elapsed = 0.0
    tempo = 500_000

    for message in mido.merge_tracks(midi.tracks):
        elapsed += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == "set_tempo":
            tempo = message.tempo
            continue
        if message.type in {"note_on", "note_off", "control_change"}:
            frame = int(elapsed * frame_rate)
            if frame < n_frames:
                scheduled[frame].append(message)

    activity = np.zeros((n_frames, 88), dtype=np.float32)
    onset = np.zeros((n_frames, 88), dtype=np.float32)
    pedals = np.zeros((n_frames, 4), dtype=np.float32)
    active: dict[int, float] = {}
    deferred_note_off: set[int] = set()
    controls = np.zeros(4, dtype=np.float32)
    sustain_on = False

    for frame in range(n_frames):
        for message in scheduled.get(frame, []):
            if message.type == "control_change" and 64 <= message.control <= 67:
                control_index = message.control - 64
                previous_sustain = sustain_on
                controls[control_index] = message.value / 127.0
                if message.control == 64:
                    sustain_on = message.value >= 64
                    if previous_sustain and not sustain_on:
                        for note in deferred_note_off:
                            active.pop(note, None)
                        deferred_note_off.clear()
                continue

            if not 21 <= message.note <= 108:
                continue
            if message.type == "note_on" and message.velocity > 0:
                active[message.note] = message.velocity / 127.0
                deferred_note_off.discard(message.note)
                onset[frame, message.note - 21] = message.velocity / 127.0
            else:
                if sustain_on:
                    deferred_note_off.add(message.note)
                else:
                    active.pop(message.note, None)

        pedals[frame] = controls
        for note in active:
            activity[frame, note - 21] = 1.0

    return np.stack((activity, onset), axis=-1), pedals


def _pack_polyphony(midi_roll: np.ndarray, max_polyphony: int) -> tuple[np.ndarray, np.ndarray]:
    """Pack active notes into stable synthesizer slots across time."""
    frames = midi_roll.shape[0]
    conditioning = np.zeros((frames, max_polyphony, 2), dtype=np.float32)
    polyphony = midi_roll[..., 0].sum(axis=1).astype(np.int16)
    slots = np.zeros(max_polyphony, dtype=np.int16)

    for frame in range(frames):
        active_indices = np.flatnonzero(midi_roll[frame, :, 0])
        active_pitches = {int(index + 21) for index in active_indices}
        for slot, pitch in enumerate(slots):
            if pitch and pitch not in active_pitches:
                slots[slot] = 0

        assigned = {int(pitch) for pitch in slots if pitch}
        candidates = sorted(active_pitches - assigned, reverse=True)
        for pitch in candidates:
            free_slots = np.flatnonzero(slots == 0)
            if free_slots.size == 0:
                break
            slots[int(free_slots[0])] = pitch

        for slot, pitch in enumerate(slots):
            if pitch:
                midi_index = pitch - 21
                conditioning[frame, slot, 0] = pitch
                conditioning[frame, slot, 1] = midi_roll[frame, midi_index, 1]

    return conditioning, polyphony


def build_track_cache(
    maestro_root: Path,
    cache_root: Path,
    track: MaestroTrack,
    config: PreprocessConfig,
) -> Path:
    config.validate()
    destination = _cache_dir(cache_root, track, config)
    if _cache_complete(destination, config):
        return destination

    audio_path = maestro_root / track.audio_rel
    midi_path = maestro_root / track.midi_rel
    if not audio_path.is_file() or not midi_path.is_file():
        missing = [str(path) for path in (audio_path, midi_path) if not path.is_file()]
        raise FileNotFoundError("Missing MAESTRO track file(s): " + ", ".join(missing))

    audio = _load_audio(audio_path, config.sample_rate)
    n_frames = int(np.ceil(audio.size / config.sample_rate * config.frame_rate))
    target_samples = n_frames * (config.sample_rate // config.frame_rate)
    if audio.size < target_samples:
        audio = np.pad(audio, (0, target_samples - audio.size))
    else:
        audio = audio[:target_samples]

    midi_roll, pedal = _midi_roll(midi_path, n_frames, config.frame_rate)
    conditioning, polyphony = _pack_polyphony(midi_roll, config.max_polyphony)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise RuntimeError(f"Temporary cache path already exists: {temporary}")
    temporary.mkdir()
    try:
        np.save(temporary / "audio.npy", audio)
        np.save(temporary / "conditioning.npy", conditioning)
        np.save(temporary / "pedal.npy", pedal)
        np.save(temporary / "polyphony.npy", polyphony)
        metadata = {
            "audio_rel": track.audio_rel,
            "midi_rel": track.midi_rel,
            "year": track.year,
            "piano_id": track.piano_id,
            "split": track.split,
            "config": asdict(config),
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        try:
            os.replace(temporary, destination)
        except FileExistsError:
            if not _cache_complete(destination, config):
                raise
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
    return destination


def prepare_split(
    maestro_root: Path,
    cache_root: Path,
    split: str,
    config: PreprocessConfig,
    limit: int | None = None,
) -> dict[str, int]:
    tracks, years = load_tracks(maestro_root, split)
    if limit is not None:
        tracks = tracks[:limit]
    created = 0
    for track in tqdm(tracks, desc=f"Caching {split}"):
        path = _cache_dir(cache_root, track, config)
        existed = _cache_complete(path, config)
        build_track_cache(maestro_root, cache_root, track, config)
        created += int(not existed)
    return {"tracks": len(tracks), "created": created, "piano_models": len(years)}


@lru_cache(maxsize=4)
def _open_cache(cache_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = Path(cache_path)
    return (
        np.load(path / "audio.npy", mmap_mode="r"),
        np.load(path / "conditioning.npy", mmap_mode="r"),
        np.load(path / "pedal.npy", mmap_mode="r"),
        np.load(path / "polyphony.npy", mmap_mode="r"),
    )


class MaestroSegmentDataset(Dataset):
    """A disk-backed dataset of cached, aligned MAESTRO audio/MIDI segments."""

    def __init__(
        self,
        maestro_root: Path,
        cache_root: Path,
        split: str,
        config: PreprocessConfig,
        require_cache: bool = True,
        limit_tracks: int | None = None,
    ) -> None:
        self.config = config
        self.config.validate()
        tracks, years = load_tracks(maestro_root, split)
        self.piano_models = years
        if limit_tracks is not None:
            tracks = tracks[:limit_tracks]

        self.index: list[tuple[str, int, int, int]] = []
        for track in tqdm(tracks, desc=f"Indexing {split}"):
            track_cache = _cache_dir(cache_root, track, config)
            if not _cache_complete(track_cache, config):
                if require_cache:
                    raise FileNotFoundError(
                        f"Missing cache for {track.audio_rel}. Run scripts/prepare_maestro.py first."
                    )
                build_track_cache(maestro_root, cache_root, track, config)

            _, conditioning, _, polyphony = _open_cache(str(track_cache))
            for frame_start in range(0, conditioning.shape[0] - config.segment_frames + 1, config.hop_frames):
                frame_end = frame_start + config.segment_frames
                if int(polyphony[frame_start:frame_end].max()) <= config.max_polyphony:
                    sample_start = frame_start * (config.sample_rate // config.frame_rate)
                    self.index.append((str(track_cache), sample_start, frame_start, track.piano_id))

        if not self.index:
            raise ValueError(f"No usable {split} segments found in the selected cache")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_path, sample_start, frame_start, piano_id = self.index[item]
        audio, conditioning, pedal, _ = _open_cache(cache_path)
        sample_end = sample_start + self.config.segment_samples
        frame_end = frame_start + self.config.segment_frames
        return (
            torch.from_numpy(np.array(audio[sample_start:sample_end], dtype=np.float32, copy=True)),
            torch.from_numpy(np.array(conditioning[frame_start:frame_end], dtype=np.float32, copy=True)),
            torch.from_numpy(np.array(pedal[frame_start:frame_end], dtype=np.float32, copy=True)),
            torch.tensor(piano_id, dtype=torch.long),
        )
