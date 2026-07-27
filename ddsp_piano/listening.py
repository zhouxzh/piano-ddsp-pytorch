"""Generate and finalize deterministic blind-listening packages."""

from __future__ import annotations

import json
import hashlib
import math
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import torchaudio

from ddsp_piano.evaluation import LISTENING_SCHEMA, read_json, utc_now, write_json


DIMENSIONS = ("timbre", "attack", "dynamics", "sustain", "reverb", "artifacts")


def write_pcm16(path: Path, audio: np.ndarray, sample_rate: int) -> dict:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not audio.size:
        raise ValueError("Listening audio must be a non-empty mono signal")
    if not np.isfinite(audio).all():
        raise ValueError("Listening audio contains NaN or Inf")
    clipped = int(np.count_nonzero(np.abs(audio) > 1.0))
    pcm = np.round(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return {
        "path": str(path.resolve()),
        "peak": float(np.max(np.abs(audio))),
        "rms": rms,
        "clipped_samples": clipped,
    }


def integrated_loudness(audio: np.ndarray, sample_rate: int) -> float:
    tensor = torch.from_numpy(np.array(audio, dtype=np.float32, copy=True)).unsqueeze(0)
    try:
        value = float(torchaudio.functional.loudness(tensor, sample_rate).item())
    except (RuntimeError, ValueError):
        value = math.nan
    if not math.isfinite(value):
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        value = 20.0 * math.log10(max(rms, 1e-8))
    return value


def loudness_match(
    audio: np.ndarray,
    sample_rate: int,
    target_lufs: float = -23.0,
    peak_ceiling_dbfs: float = -1.0,
) -> tuple[np.ndarray, dict]:
    measured = integrated_loudness(audio, sample_rate)
    gain_db = target_lufs - measured
    scaled = np.asarray(audio, dtype=np.float32) * math.pow(10.0, gain_db / 20.0)
    ceiling = math.pow(10.0, peak_ceiling_dbfs / 20.0)
    peak = float(np.max(np.abs(scaled), initial=0.0))
    ceiling_reduction_db = 0.0
    if peak > ceiling:
        ceiling_reduction_db = 20.0 * math.log10(ceiling / peak)
        scaled = scaled * (ceiling / peak)
    return scaled, {
        "source_lufs": measured,
        "target_lufs": target_lufs,
        "requested_gain_db": gain_db,
        "ceiling_reduction_db": ceiling_reduction_db,
        "applied_gain_db": gain_db + ceiling_reduction_db,
    }


def select_excerpt_frames(
    conditioning: np.ndarray,
    pedal: np.ndarray,
    frame_rate: int,
    duration_seconds: float,
) -> tuple[int, int]:
    excerpt_frames = min(conditioning.shape[0], max(1, int(round(duration_seconds * frame_rate))))
    if excerpt_frames == conditioning.shape[0]:
        return 0, excerpt_frames
    hop = frame_rate
    onset = (conditioning[..., 1] > 0).sum(axis=-1).astype(np.float32)
    active = (conditioning[..., 0] > 0).sum(axis=-1).astype(np.float32)
    sustain = pedal[:, 0].astype(np.float32)
    best_start = 0
    best_score = -math.inf
    margin = min(conditioning.shape[0] // 4, 5 * frame_rate)
    last_start = conditioning.shape[0] - excerpt_frames
    for start in range(0, last_start + 1, hop):
        if last_start > 2 * margin and (start < margin or start > last_start - margin):
            continue
        end = start + excerpt_frames
        score = (
            0.5 * float(onset[start:end].mean())
            + 0.3 * float(active[start:end].std())
            + 0.2 * float(sustain[start:end].mean())
        )
        if score > best_score:
            best_score = score
            best_start = start
    return best_start, best_start + excerpt_frames


def _html(trials: list[dict], evaluation_id: str, deadline: str | None) -> str:
    public_trials = [
        {
            "id": trial["id"],
            "title": trial["title"],
            "mode": trial["mode"],
            "a": trial["a"],
            "b": trial["b"],
        }
        for trial in trials
    ]
    payload = json.dumps(public_trials, ensure_ascii=False)
    dimensions = json.dumps(DIMENSIONS)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DDSP-Piano Blind Listening</title>
<style>
:root {{ color-scheme: light; --ink:#191c1d; --muted:#687076; --line:#d9dddf; --paper:#f6f7f5; --accent:#176b4d; --warn:#a13d2d; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink); font:15px/1.45 system-ui,sans-serif; letter-spacing:0; }}
header {{ border-bottom:1px solid var(--line); background:#fff; }}
header > div, main {{ width:min(920px, calc(100% - 32px)); margin:auto; }}
header > div {{ min-height:64px; display:flex; align-items:center; justify-content:space-between; gap:16px; }}
h1 {{ margin:0; font-size:20px; }}
.status {{ display:flex; align-items:flex-end; flex-direction:column; color:var(--muted); font-size:12px; }}
#progress {{ font-size:14px; font-variant-numeric:tabular-nums; }}
main {{ padding:28px 0 48px; }}
.trial {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:20px; }}
.meta {{ display:flex; justify-content:space-between; gap:16px; margin-bottom:18px; }}
.mode {{ color:var(--accent); font-weight:650; }}
.players {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
.player {{ border-top:3px solid var(--ink); padding-top:12px; }}
.player h2 {{ font-size:16px; margin:0 0 8px; }}
audio {{ width:100%; }}
.preference {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin:22px 0; }}
button {{ border:1px solid var(--line); background:#fff; color:var(--ink); min-height:42px; border-radius:6px; cursor:pointer; font-weight:650; }}
button[aria-pressed="true"] {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
button:disabled {{ cursor:default; opacity:.4; }}
.ratings {{ display:grid; grid-template-columns:110px 1fr 1fr; gap:10px 16px; align-items:center; }}
.ratings strong {{ font-size:13px; }}
input[type="range"] {{ width:100%; accent-color:var(--accent); }}
.rating-control {{ display:grid; grid-template-columns:1fr 24px; gap:8px; align-items:center; }}
.rating-control output {{ color:var(--muted); text-align:right; font-variant-numeric:tabular-nums; }}
.flags {{ display:flex; gap:24px; margin:20px 0; color:var(--warn); }}
textarea {{ width:100%; min-height:72px; resize:vertical; border:1px solid var(--line); border-radius:6px; padding:10px; font:inherit; }}
.actions {{ display:flex; justify-content:space-between; gap:10px; margin-top:18px; }}
.primary {{ background:var(--ink); color:#fff; border-color:var(--ink); padding:0 18px; }}
@media (max-width:700px) {{ .players {{ grid-template-columns:1fr; }} .ratings {{ grid-template-columns:80px 1fr 1fr; gap:10px 8px; }} .rating-control {{ grid-template-columns:1fr 18px; gap:4px; }} .trial {{ padding:16px; }} }}
</style>
</head>
<body>
<header><div><h1>DDSP-Piano 盲听</h1><div class="status"><span id="progress"></span><span id="deadline"></span></div></div></header>
<main><section class="trial" id="trial"></section></main>
<script>
const schema={json.dumps(LISTENING_SCHEMA)}, evaluationId={json.dumps(evaluation_id)}, deadline={json.dumps(deadline)};
const trials={payload}, dimensions={dimensions};
const labels={{timbre:'音色',attack:'起音',dynamics:'力度',sustain:'延音',reverb:'混响',artifacts:'纯净度'}};
const key='ddsp-piano-listening:'+evaluationId;
let state=JSON.parse(localStorage.getItem(key)||'{{"index":0,"answers":{{}}}}');
const root=document.getElementById('trial'), progress=document.getElementById('progress');
document.getElementById('deadline').textContent=deadline?'截止 '+new Date(deadline).toLocaleString():'等待全部训练完成后开启';
function esc(value) {{ const el=document.createElement('span'); el.textContent=value; return el.innerHTML; }}
function emptyAnswer() {{ const ratings={{}},rating_touched={{}}; dimensions.forEach(d=>{{ratings[d]={{A:3,B:3}};rating_touched[d]={{A:false,B:false}};}}); return {{preference:null,ratings,rating_touched,severe_artifact:{{A:false,B:false}},notes:''}}; }}
function save() {{ localStorage.setItem(key,JSON.stringify(state)); }}
function render() {{
  const t=trials[state.index], a=state.answers[t.id]||emptyAnswer(); state.answers[t.id]=a;
  progress.textContent=`${{state.index+1}} / ${{trials.length}}`;
  const control=(d,side)=>`<label class="rating-control"><input aria-label="${{labels[d]}} ${{side}}" data-d="${{d}}" data-side="${{side}}" type="range" min="1" max="5" value="${{a.ratings[d][side]}}"><output>${{a.ratings[d][side]}}</output></label>`;
  const rows=dimensions.map(d=>`<strong>${{labels[d]}}</strong>${{control(d,'A')}}${{control(d,'B')}}`).join('');
  root.innerHTML=`<div class="meta"><strong>${{esc(t.title)}}</strong><span class="mode">${{t.mode==='fixed_gain'?'固定增益':'响度匹配'}}</span></div><div class="players"><div class="player"><h2>A</h2><audio controls preload="metadata" src="${{t.a}}"></audio></div><div class="player"><h2>B</h2><audio controls preload="metadata" src="${{t.b}}"></audio></div></div><div class="preference"><button data-pref="A" aria-pressed="${{a.preference==='A'}}">A</button><button data-pref="tie" aria-pressed="${{a.preference==='tie'}}">平局</button><button data-pref="B" aria-pressed="${{a.preference==='B'}}">B</button></div><div class="ratings"><span></span><strong>A</strong><strong>B</strong>${{rows}}</div><div class="flags"><label><input data-flag="A" type="checkbox" ${{a.severe_artifact.A?'checked':''}}> A 严重缺陷</label><label><input data-flag="B" type="checkbox" ${{a.severe_artifact.B?'checked':''}}> B 严重缺陷</label></div><textarea placeholder="备注">${{esc(a.notes)}}</textarea><div class="actions"><button id="previous" ${{state.index===0?'disabled':''}}>上一项</button><button class="primary" id="next">${{state.index===trials.length-1?'导出评分':'下一项'}}</button></div>`;
  root.querySelectorAll('[data-pref]').forEach(el=>el.onclick=()=>{{a.preference=el.dataset.pref;save();render();}});
  root.querySelectorAll('input[type=range]').forEach(el=>el.oninput=()=>{{a.ratings[el.dataset.d][el.dataset.side]=Number(el.value);a.rating_touched=a.rating_touched||{{}};a.rating_touched[el.dataset.d]=a.rating_touched[el.dataset.d]||{{A:false,B:false}};a.rating_touched[el.dataset.d][el.dataset.side]=true;el.nextElementSibling.value=el.value;save();}});
  root.querySelectorAll('[data-flag]').forEach(el=>el.onchange=()=>{{a.severe_artifact[el.dataset.flag]=el.checked;save();}});
  root.querySelector('textarea').oninput=e=>{{a.notes=e.target.value;save();}};
  root.querySelector('#previous').onclick=()=>{{state.index=Math.max(0,state.index-1);save();render();}};
  root.querySelector('#next').onclick=()=>{{if(!a.preference){{alert('请选择 A、B 或平局');return;}} if(state.index<trials.length-1){{state.index++;save();render();}}else{{download();}}}};
}}
function download() {{
  const answers=trials.map(t=>Object.assign({{trial_id:t.id}},state.answers[t.id]||emptyAnswer()));
  if(answers.some(a=>!a.preference)){{alert('仍有未完成项目');return;}}
  const data={{schema,evaluation_id:evaluationId,completed_at:new Date().toISOString(),trials:answers}};
  const blob=new Blob([JSON.stringify(data,null,2)],{{type:'application/json'}}), link=document.createElement('a');
  link.href=URL.createObjectURL(blob);link.download='listening_scores.json';link.click();URL.revokeObjectURL(link.href);
}}
render();
</script>
</body>
</html>
"""


def create_listening_package(
    output_dir: Path,
    evaluation_id: str,
    items: list[dict],
    timeout_minutes: int,
    target_lufs: float,
    fixed_target_peak_dbfs: float,
    start_review: bool = True,
) -> dict:
    listening_dir = output_dir / "listening"
    private_dir = output_dir / "private"
    listening_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc)
    deadline = (
        (created + timedelta(minutes=timeout_minutes)).isoformat()
        if start_review
        else None
    )
    target_peak = math.pow(10.0, fixed_target_peak_dbfs / 20.0)
    baseline_peak = max(float(item["baseline_full_peak"]) for item in items)
    candidate_peak = max(float(item["candidate_full_peak"]) for item in items)
    reference_peak = max(baseline_peak, candidate_peak)
    fixed_gain = target_peak / max(reference_peak, 1e-8)
    trials = []
    mapping = {}
    clipping = 0
    for item in items:
        for mode in ("fixed_gain", "loudness_matched"):
            trial_id = f"{item['id']}-{mode}"
            swap = hashlib.sha256(trial_id.encode("utf-8")).digest()[0] % 2 == 1
            rendered = {}
            adjustments = {}
            for role in ("baseline", "candidate"):
                audio = item[role]["wet"]
                if mode == "fixed_gain":
                    value = audio * fixed_gain
                    adjustments[role] = {"applied_gain_db": 20.0 * math.log10(fixed_gain)}
                else:
                    value, adjustments[role] = loudness_match(
                        audio, item["sample_rate"], target_lufs
                    )
                side = "A" if (role == "baseline") == swap else "B"
                relative = Path("audio") / f"{trial_id}_{side.lower()}.wav"
                rendered[side] = str(relative)
                wav_report = write_pcm16(listening_dir / relative, value, item["sample_rate"])
                clipping += int(wav_report["clipped_samples"])
            trials.append(
                {
                    "id": trial_id,
                    "title": item["title"],
                    "mode": mode,
                    "a": rendered["A"],
                    "b": rendered["B"],
                }
            )
            mapping[trial_id] = {
                "A": "baseline" if swap else "candidate",
                "B": "candidate" if swap else "baseline",
                "adjustments": adjustments,
            }

        for role in ("baseline", "candidate"):
            for stem in ("harmonic", "noise", "dry", "wet"):
                stem_path = output_dir / "stems" / role / f"{item['id']}_{stem}.wav"
                write_pcm16(stem_path, item[role][stem] * fixed_gain, item["sample_rate"])

    (listening_dir / "index.html").write_text(
        _html(trials, evaluation_id, deadline), encoding="utf-8"
    )
    write_json(
        listening_dir / "trials.json",
        {
            "schema": LISTENING_SCHEMA,
            "evaluation_id": evaluation_id,
            "trials": trials,
        },
    )
    write_json(private_dir / "blind_mapping.json", {"schema": LISTENING_SCHEMA, "mapping": mapping})
    return {
        "schema": LISTENING_SCHEMA,
        "status": "pending" if start_review else "prepared",
        "created_at": created.isoformat(),
        "activated_at": created.isoformat() if start_review else None,
        "deadline": deadline,
        "timeout_minutes": timeout_minutes,
        "trials": len(trials),
        "fixed_gain": fixed_gain,
        "fixed_gain_reference_peak": reference_peak,
        "baseline_full_peak": baseline_peak,
        "candidate_full_peak": candidate_peak,
        "fixed_target_peak_dbfs": fixed_target_peak_dbfs,
        "target_lufs": target_lufs,
        "clipped_samples": clipping,
        "page": str((listening_dir / "index.html").resolve()),
        "scores_file": str((listening_dir / "listening_scores.json").resolve()),
    }


def activate_review(report_dir: Path, timeout_minutes: int) -> dict:
    """Start the human-review clock after every finalist has finished training."""
    report_path = report_dir / "report.json"
    report = read_json(report_path)
    review = report.get("human_review")
    if not review:
        raise ValueError("The report has no listening task")
    if review.get("status") != "prepared":
        return report
    if timeout_minutes < 0:
        raise ValueError("Review timeout must be non-negative")
    manifest = read_json(report_dir / "listening" / "trials.json")
    if manifest.get("evaluation_id") != report.get("evaluation_id"):
        raise ValueError("Listening trial manifest belongs to another evaluation")
    activated = datetime.now(timezone.utc)
    deadline = (activated + timedelta(minutes=timeout_minutes)).isoformat()
    review.update(
        {
            "status": "pending",
            "activated_at": activated.isoformat(),
            "deadline": deadline,
            "timeout_minutes": timeout_minutes,
        }
    )
    report["verdict"]["human_status"] = "pending"
    report["verdict"]["promotion_eligible"] = False
    (report_dir / "listening" / "index.html").write_text(
        _html(manifest["trials"], report["evaluation_id"], deadline),
        encoding="utf-8",
    )
    write_json(report_path, report)
    return report


def defer_review(report_dir: Path) -> dict:
    report_path = report_dir / "report.json"
    report = read_json(report_path)
    review = report.get("human_review")
    if not review:
        raise ValueError("The report has no listening task")
    if review["status"] == "pending":
        review["status"] = "deferred"
        review["deferred_at"] = utc_now()
        report["verdict"]["human_status"] = "deferred"
        write_json(report_path, report)
    return report


def finalize_review(report_dir: Path, scores_path: Path, config: dict) -> dict:
    report_path = report_dir / "report.json"
    report = read_json(report_path)
    mapping = read_json(report_dir / "private" / "blind_mapping.json")["mapping"]
    scores = read_json(scores_path)
    if scores.get("schema") != LISTENING_SCHEMA:
        raise ValueError("Listening score schema mismatch")
    if scores.get("evaluation_id") != report["evaluation_id"]:
        raise ValueError("Listening scores belong to another evaluation")
    expected = set(mapping)
    trials = scores.get("trials", [])
    received = [entry.get("trial_id") for entry in trials]
    if len(received) != len(expected) or set(received) != expected:
        raise ValueError("Listening scores are incomplete or contain unknown trials")

    points = 0.0
    dimension_deltas = {name: [] for name in DIMENSIONS}
    repeated_artifacts = 0
    for entry in trials:
        trial_map = mapping[entry["trial_id"]]
        preference = entry.get("preference")
        if preference not in {"A", "B", "tie"}:
            raise ValueError(f"Invalid preference for {entry['trial_id']}")
        if preference == "tie":
            points += 0.5
        elif trial_map[preference] == "candidate":
            points += 1.0
        for dimension in DIMENSIONS:
            ratings = entry["ratings"][dimension]
            candidate_side = "A" if trial_map["A"] == "candidate" else "B"
            baseline_side = "B" if candidate_side == "A" else "A"
            touched = entry.get("rating_touched")
            if touched is None or (
                bool(touched.get(dimension, {}).get(candidate_side))
                and bool(touched.get(dimension, {}).get(baseline_side))
            ):
                dimension_deltas[dimension].append(
                    float(ratings[candidate_side]) - float(ratings[baseline_side])
                )
        candidate_side = "A" if trial_map["A"] == "candidate" else "B"
        repeated_artifacts += int(bool(entry["severe_artifact"][candidate_side]))

    preference_rate = points / len(expected)
    mean_deltas = {
        name: float(np.mean(values)) if values else None
        for name, values in dimension_deltas.items()
    }
    rated_deltas = [value for value in mean_deltas.values() if value is not None]
    dimensions_passed = not rated_deltas or min(rated_deltas) >= -float(
        config["gates"]["human_dimension_regression"]
    )
    human_passed = (
        preference_rate >= float(config["gates"]["human_preference_rate"])
        and dimensions_passed
        and repeated_artifacts < int(config["gates"]["human_repeated_artifacts"])
    )
    destination = report_dir / "listening" / "listening_scores.json"
    write_json(destination, scores)
    deadline = report["human_review"].get("deadline")
    report["human_review"].update(
        {
            "status": "passed" if human_passed else "failed",
            "completed_at": scores.get("completed_at", utc_now()),
            "submitted_after_deadline": bool(
                deadline
                and datetime.now(timezone.utc) > datetime.fromisoformat(deadline)
            ),
            "preference_rate": preference_rate,
            "dimension_mean_delta": mean_deltas,
            "dimension_rating_counts": {
                name: len(values) for name, values in dimension_deltas.items()
            },
            "candidate_severe_artifacts": repeated_artifacts,
        }
    )
    report["verdict"]["human_status"] = report["human_review"]["status"]
    report["verdict"]["promotion_eligible"] = bool(
        report["verdict"]["objective_eligible"] and human_passed
    )
    write_json(report_path, report)
    return report
