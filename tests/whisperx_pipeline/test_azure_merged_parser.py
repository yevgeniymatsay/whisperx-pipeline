import json
from pathlib import Path

from pipeline.call_segmenter.upstream import parse_azure_merged_diarized


def test_parse_azure_merged_diarized_segments_sorted_and_valid():
    p = Path("artifacts/azure_diarize_eval/20260205_20calls/videos/-POaWp9_UaM/merged.diarized.json")
    assert p.exists(), f"Missing test fixture: {p}"

    doc = json.loads(p.read_text())
    segs = parse_azure_merged_diarized(doc)

    assert len(segs) > 0
    assert all(s.t1_abs > s.t0_abs for s in segs)
    assert all(s.t0_abs >= 0.0 for s in segs)

    # Monotonic start times.
    t0s = [s.t0_abs for s in segs]
    assert t0s == sorted(t0s)

    # Speaker ids preserved (stringified).
    first_raw = doc["segments"][0]["speaker"]
    assert segs[0].spk == str(first_raw)

