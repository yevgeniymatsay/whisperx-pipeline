#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from pipeline.call_extractor_wavlm.io import utc_now_compact

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Segment:
    start_s: float
    end_s: float
    speaker: str

    @property
    def dur_s(self) -> float:
        return max(0.0, float(self.end_s - self.start_s))


def _read_metadata_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _load_diarized_segments(path: Path) -> tuple[float, list[Segment]]:
    data = json.loads(path.read_text())
    duration = float(data.get("duration", 0.0) or 0.0)
    segs: list[Segment] = []
    for s in data.get("segments") or []:
        spk = str(s.get("speaker") or "").strip()
        if not spk:
            continue
        start = float(s.get("start", 0.0) or 0.0)
        end = float(s.get("end", start) or start)
        if end <= start:
            continue
        segs.append(Segment(start_s=start, end_s=end, speaker=spk))
    segs.sort(key=lambda x: (x.start_s, x.end_s, x.speaker))
    if duration <= 0.0 and segs:
        duration = max(s.end_s for s in segs)
    return duration, segs


def _read_wav_mono_16k(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except Exception as e:  # pragma: no cover
        raise RuntimeError("soundfile is required: pip install soundfile") from e

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono wav, got shape={audio.shape} ({path})")
    return audio, int(sr)


def _concat_speaker_audio(
    audio: np.ndarray,
    sr: int,
    segments: Iterable[Segment],
    *,
    max_audio_s: float,
    inter_segment_silence_s: float,
) -> tuple[np.ndarray, float, float]:
    """Return concatenated speaker audio sample + (used_s, total_s)."""
    segments = sorted(list(segments), key=lambda s: (s.start_s, s.end_s))
    total_s = float(sum(s.dur_s for s in segments))
    if not segments:
        return np.zeros(0, dtype=np.float32), 0.0, 0.0

    parts: list[np.ndarray] = []
    used_s = 0.0
    silence = np.zeros(int(round(float(inter_segment_silence_s) * float(sr))), dtype=np.float32)
    for seg in segments:
        start_f = int(round(seg.start_s * sr))
        end_f = int(round(seg.end_s * sr))
        start_f = max(0, min(start_f, audio.shape[0]))
        end_f = max(0, min(end_f, audio.shape[0]))
        if end_f <= start_f:
            continue
        chunk = audio[start_f:end_f]
        if chunk.size == 0:
            continue
        chunk_s = float(chunk.size) / float(sr)
        remaining_s = float(max_audio_s - used_s)
        if remaining_s <= 0.0:
            break
        if chunk_s > remaining_s:
            n = int(round(remaining_s * sr))
            if n <= 0:
                break
            chunk = chunk[:n]
            chunk_s = float(chunk.size) / float(sr)
        parts.append(chunk)
        parts.append(silence)
        used_s += chunk_s

    if not parts:
        return np.zeros(0, dtype=np.float32), 0.0, total_s
    out = np.concatenate(parts, axis=0)
    return out.astype(np.float32, copy=False), float(used_s), float(total_s)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = float(np.linalg.norm(x) + 1e-12)
    return x / n


class Embedder:
    name: str

    def min_speech_s(self) -> float:
        return 1.0

    def embed(self, audio_f32: np.ndarray, sr: int) -> np.ndarray:
        raise NotImplementedError


class SpeechBrainECAPA(Embedder):
    name = "speechbrain/spkrec-ecapa-voxceleb"

    def __init__(self, device: str):
        # SpeechBrain imports call torchaudio.list_audio_backends(), but torchaudio>=2.9
        # removed that function (moved to torchcodec). Patch it in for compatibility.
        try:
            import torchaudio  # type: ignore

            if not hasattr(torchaudio, "list_audio_backends"):
                torchaudio.list_audio_backends = lambda: ["torchcodec"]  # type: ignore[attr-defined]
        except Exception:
            pass

        from speechbrain.inference.speaker import EncoderClassifier

        self._model = EncoderClassifier.from_hparams(source=self.name, run_opts={"device": device})
        self._device = device

    def embed(self, audio_f32: np.ndarray, sr: int) -> np.ndarray:
        import torch

        wav = torch.from_numpy(np.asarray(audio_f32, dtype=np.float32)).to(self._device).unsqueeze(0)
        with torch.no_grad():
            emb = self._model.encode_batch(wav).squeeze(0).squeeze(0).detach().cpu().numpy()
        return _l2_normalize(np.asarray(emb, dtype=np.float32))


class NeMoTitaNet(Embedder):
    name = "nvidia/speakerverification_en_titanet_large"

    def __init__(self, device: str):
        import torch
        from nemo.collections.asr.models import EncDecSpeakerLabelModel

        self._torch = torch
        self._device = device
        self._model = EncDecSpeakerLabelModel.from_pretrained(model_name=self.name).to(device)
        self._model.eval()

    def embed(self, audio_f32: np.ndarray, sr: int) -> np.ndarray:
        # NeMo's `EncDecSpeakerLabelModel.get_embedding()` expects a *wav file path*.
        # Writing a short temp wav is OK for this pilot (small scale).
        import tempfile

        import soundfile as sf

        if int(sr) != 16000:
            raise ValueError(f"NeMo TitaNet expects 16kHz audio (got sr={sr})")

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            sf.write(tmp.name, np.asarray(audio_f32, dtype=np.float32), int(sr), subtype="PCM_16")
            with self._torch.no_grad():
                emb = self._model.get_embedding(tmp.name)

        if hasattr(emb, "detach"):
            emb = emb.detach().cpu().numpy()
        emb = np.asarray(emb, dtype=np.float32).squeeze()
        return _l2_normalize(np.asarray(emb, dtype=np.float32))


class HFCaseEmbedding(Embedder):
    name = "bigstorm/case-speaker-embedding-v2-512"

    def __init__(self, device: str):
        # This model repo is a minimal PyTorch module (no Transformers feature extractor).
        # Use its provided `CASESpeakerEncoder` wrapper from `model.py`.
        import importlib.util

        from huggingface_hub import hf_hub_download

        model_py = hf_hub_download(self.name, "model.py")
        hf_hub_download(self.name, "config.json")
        hf_hub_download(self.name, "pytorch_model.bin")
        self._model_dir = str(Path(model_py).parent)

        spec = importlib.util.spec_from_file_location("case_model", model_py)
        if spec is None or spec.loader is None:  # pragma: no cover
            raise RuntimeError(f"Failed to load CASE model.py from {model_py}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[misc]

        if not hasattr(mod, "CASESpeakerEncoder"):  # pragma: no cover
            raise RuntimeError("CASE model.py missing CASESpeakerEncoder")
        self._encoder = mod.CASESpeakerEncoder.from_pretrained(self._model_dir, device=device)

    def min_speech_s(self) -> float:
        # Model card says ~0.5s recommended; use that as the minimum for the pilot.
        return 0.5

    def embed(self, audio_f32: np.ndarray, sr: int) -> np.ndarray:
        if int(sr) != 16000:
            raise ValueError(f"CASE expects 16kHz audio (got sr={sr})")
        emb = self._encoder.encode(np.asarray(audio_f32, dtype=np.float32))
        return _l2_normalize(np.asarray(emb, dtype=np.float32))


def _cluster_raw_speakers(
    raw_speakers: list[str],
    embeddings: dict[str, np.ndarray],
    weights: dict[str, float],
    *,
    allow_k1: bool,
) -> dict[str, str]:
    """Return raw_speaker -> canonical speaker label mapping."""
    valid = [s for s in raw_speakers if s in embeddings]
    if len(valid) < 2:
        return {s: "S0" for s in raw_speakers}
    if allow_k1 is False and len(valid) < 2:
        raise RuntimeError("Need >=2 speakers with embeddings for K=2 clustering")

    X = np.stack([embeddings[s] for s in valid], axis=0)
    w = np.asarray([float(weights.get(s, 1.0)) for s in valid], dtype=np.float64)
    w = np.clip(w, 1e-3, None)

    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=2, random_state=0, n_init="auto")
    km.fit(X, sample_weight=w)
    labels = km.labels_.tolist()

    # Assign S0 to the cluster with larger total weight (more speech).
    w0 = float(sum(wi for wi, li in zip(w.tolist(), labels) if li == 0))
    w1 = float(sum(wi for wi, li in zip(w.tolist(), labels) if li == 1))
    s0_cluster = 0 if w0 >= w1 else 1
    mapping: dict[str, str] = {}
    for raw, li in zip(valid, labels):
        mapping[raw] = "S0" if li == s0_cluster else "S1"
    # Any raw speakers without embeddings get folded into S0 (conservative).
    for raw in raw_speakers:
        mapping.setdefault(raw, "S0")
    return mapping


def _merge_consecutive(segments: list[Segment]) -> list[Segment]:
    if not segments:
        return []
    out = [segments[0]]
    for s in segments[1:]:
        prev = out[-1]
        if s.speaker == prev.speaker and s.start_s <= prev.end_s + 1e-6:
            out[-1] = Segment(start_s=prev.start_s, end_s=max(prev.end_s, s.end_s), speaker=prev.speaker)
        else:
            out.append(s)
    return out


def _switch_rate_per_min(segments: list[Segment], duration_s: float) -> float:
    if duration_s <= 0:
        return 0.0
    merged = _merge_consecutive(segments)
    switches = 0
    for a, b in zip(merged, merged[1:]):
        if a.speaker != b.speaker:
            switches += 1
    return float(switches) / max(1e-6, float(duration_s) / 60.0)


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp_{utc_now_compact()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(xs[int(k)])
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return float(d0 + d1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Canonicalize GPT diarized clips to 1-2 speakers and write QA report.")
    parser.add_argument("--pilot-dir", type=Path, required=True, help="Pilot dir created by build + diarize steps")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Report output dir (default: reports/diarization_pilot/<pilot_name>/)",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Embedding device: cpu or cuda (if available)")
    parser.add_argument("--max-speaker-audio-s", type=float, default=30.0, help="Max audio seconds sampled per raw speaker label")
    parser.add_argument("--silence-s", type=float, default=0.05, help="Silence inserted between concatenated segments")
    parser.add_argument("--tiny-speaker-s", type=float, default=3.0, help="Flag if second canonical speaker has < this many seconds")
    parser.add_argument("--allow-k1", action=argparse.BooleanOptionalAction, default=True, help="Allow collapsing to 1 speaker when needed")
    parser.add_argument("--limit", type=int, default=None, help="Optional max clips to include")
    parser.add_argument(
        "--embedder",
        action="append",
        default=None,
        help="Which embedders to run: case|ecapa|titanet (repeatable)",
    )
    args = parser.parse_args()

    pilot_dir = Path(args.pilot_dir)
    meta_path = pilot_dir / "metadata.jsonl"
    diarized_dir = pilot_dir / "diarized"
    clips_dir = pilot_dir / "clips"
    if not meta_path.exists():
        raise SystemExit(f"Missing metadata: {meta_path}")
    if not diarized_dir.exists():
        raise SystemExit(f"Missing diarized/: {diarized_dir}")

    pilot_name = pilot_dir.name
    report_dir = Path(args.report_dir) if args.report_dir else Path("reports/diarization_pilot") / pilot_name
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_metadata_jsonl(meta_path)
    if args.limit is not None:
        rows = rows[: int(args.limit)]

    device = str(args.device)
    embedder_names_raw = args.embedder if args.embedder is not None else ["case", "ecapa", "titanet"]
    embedder_names = [e.strip().lower() for e in embedder_names_raw]
    embedders: list[tuple[str, Embedder]] = []
    if "case" in embedder_names:
        embedders.append(("case", HFCaseEmbedding(device=device)))
    if "ecapa" in embedder_names:
        embedders.append(("ecapa", SpeechBrainECAPA(device=device)))
    if "titanet" in embedder_names:
        embedders.append(("titanet", NeMoTitaNet(device=device)))
    if not embedders:
        raise SystemExit("No embedders enabled")

    clip_results: list[dict[str, Any]] = []
    # Collect switch rates for summary
    switch_rates: dict[str, list[float]] = {k: [] for k, _ in embedders}
    tiny_second_counts: dict[str, int] = {k: 0 for k, _ in embedders}
    k1_counts: dict[str, int] = {k: 0 for k, _ in embedders}

    for idx, row in enumerate(rows):
        clip_id = str(row["clip_id"])
        clip_path = Path(row.get("clip_path") or (clips_dir / f"{clip_id}.wav"))
        diarized_path = diarized_dir / f"{clip_id}.diarized_json.json"
        if not diarized_path.exists():
            raise SystemExit(f"Missing diarized file: {diarized_path}")
        if not clip_path.exists():
            raise SystemExit(f"Missing clip audio: {clip_path}")

        clip_audio, sr = _read_wav_mono_16k(clip_path)
        if sr != 16000:
            logger.warning(f"Non-16k clip detected (sr={sr}) for {clip_id} ({clip_path})")
        duration_s, segments = _load_diarized_segments(diarized_path)
        raw_speakers = sorted({s.speaker for s in segments})
        by_speaker: dict[str, list[Segment]] = {s: [] for s in raw_speakers}
        for seg in segments:
            by_speaker[seg.speaker].append(seg)

        # Build one audio sample per raw speaker label.
        speaker_samples: dict[str, dict[str, Any]] = {}
        for spk in raw_speakers:
            sample, used_s, total_s = _concat_speaker_audio(
                clip_audio,
                sr,
                by_speaker[spk],
                max_audio_s=float(args.max_speaker_audio_s),
                inter_segment_silence_s=float(args.silence_s),
            )
            speaker_samples[spk] = {
                "used_s": float(used_s),
                "total_s": float(total_s),
                "sample_len_s": float(sample.size) / float(sr) if sample.size else 0.0,
                "sample": sample,
            }

        per_model: dict[str, Any] = {}
        for key, embedder in embedders:
            embeddings: dict[str, np.ndarray] = {}
            weights: dict[str, float] = {}
            too_short: list[str] = []
            for spk in raw_speakers:
                sample = speaker_samples[spk]["sample"]
                used_s = float(speaker_samples[spk]["used_s"])
                total_s = float(speaker_samples[spk]["total_s"])
                weights[spk] = max(1e-3, total_s)
                if used_s < float(embedder.min_speech_s()):
                    too_short.append(spk)
                    continue
                try:
                    emb = embedder.embed(np.asarray(sample, dtype=np.float32), sr)
                    embeddings[spk] = np.asarray(emb, dtype=np.float32)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"[{key}] embed failed clip={clip_id} spk={spk}: {type(e).__name__}: {e}")

            mapping = _cluster_raw_speakers(raw_speakers, embeddings, weights, allow_k1=bool(args.allow_k1))
            canon_segments = [Segment(start_s=s.start_s, end_s=s.end_s, speaker=mapping.get(s.speaker, "S0")) for s in segments]
            canon_segments.sort(key=lambda s: (s.start_s, s.end_s, s.speaker))

            # Canonical talk time.
            canon_dur: dict[str, float] = {}
            for s in canon_segments:
                canon_dur[s.speaker] = canon_dur.get(s.speaker, 0.0) + s.dur_s
            canon_speakers = sorted(canon_dur.keys())
            canon_count = len(canon_speakers)
            if canon_count <= 1:
                k1_counts[key] += 1
            tiny_second = False
            if canon_count >= 2:
                dur_sorted = sorted(canon_dur.values())
                tiny_second = float(dur_sorted[0]) < float(args.tiny_speaker_s)
                if tiny_second:
                    tiny_second_counts[key] += 1

            srpm = _switch_rate_per_min(canon_segments, duration_s)
            switch_rates[key].append(float(srpm))

            per_model[key] = {
                "raw_speaker_count": int(len(raw_speakers)),
                "raw_speakers": raw_speakers,
                "raw_speaker_total_s": {s: float(speaker_samples[s]["total_s"]) for s in raw_speakers},
                "raw_speaker_used_s": {s: float(speaker_samples[s]["used_s"]) for s in raw_speakers},
                "raw_speaker_too_short": too_short,
                "canonical_mapping": mapping,
                "canonical_speaker_count": int(canon_count),
                "canonical_speaker_total_s": {k: float(v) for k, v in canon_dur.items()},
                "tiny_second_speaker": bool(tiny_second),
                "switch_rate_per_min": float(srpm),
            }

        clip_results.append({
            "clip_id": clip_id,
            "video_id": row.get("video_id"),
            "call_index": row.get("call_index"),
            "offset_s": row.get("offset_s"),
            "clip_duration_s": row.get("clip_duration_s"),
            "diarized_duration_s": float(duration_s),
            "per_model": per_model,
        })
        if (idx + 1) % 10 == 0 or (idx + 1) == len(rows):
            logger.info(f"Processed {idx+1}/{len(rows)} clips")

    summary: dict[str, Any] = {
        "pilot_dir": str(pilot_dir),
        "clip_count": int(len(clip_results)),
        "embedders": [k for k, _ in embedders],
        "tiny_second_s_threshold_s": float(args.tiny_speaker_s),
        "switch_rate_per_min_p50": {k: _percentile(v, 50) for k, v in switch_rates.items()},
        "switch_rate_per_min_p90": {k: _percentile(v, 90) for k, v in switch_rates.items()},
        "k1_count": k1_counts,
        "tiny_second_count": tiny_second_counts,
    }

    out_json = report_dir / "qa_report.json"
    out_md = report_dir / "qa_report.md"
    _dump_json(out_json, {"summary": summary, "clips": clip_results})

    lines: list[str] = []
    lines.append(f"# Diarization Pilot QA: {pilot_name}\n")
    lines.append("## Summary\n")
    lines.append(f"- Clips: {len(clip_results)}")
    lines.append(f"- Embedders: {', '.join(summary['embedders'])}")
    lines.append(f"- Tiny second speaker threshold: {float(args.tiny_speaker_s):.1f}s\n")
    lines.append("### Aggregate\n")
    for k in summary["embedders"]:
        lines.append(f"- `{k}`: k=1 clips={k1_counts[k]} tiny_second={tiny_second_counts[k]} switch_rate_p50={summary['switch_rate_per_min_p50'][k]:.2f}/min switch_rate_p90={summary['switch_rate_per_min_p90'][k]:.2f}/min")
    lines.append("\n## Per Clip (first 20)\n")
    lines.append("| clip_id | raw_spk | case(k) tiny | ecapa(k) tiny | titanet(k) tiny |")
    lines.append("|---|---:|---|---|---|")
    for clip in clip_results[:20]:
        per = clip["per_model"]
        raw = per["case"]["raw_speaker_count"] if "case" in per else next(iter(per.values()))["raw_speaker_count"]
        def _fmt(model: str) -> str:
            if model not in per:
                return "n/a"
            k = per[model]["canonical_speaker_count"]
            tiny = per[model]["tiny_second_speaker"]
            return f"{k}{' tiny' if tiny else ''}"
        lines.append(f"| {clip['clip_id']} | {raw} | {_fmt('case')} | {_fmt('ecapa')} | {_fmt('titanet')} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info(f"Wrote {out_json}")
    logger.info(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
