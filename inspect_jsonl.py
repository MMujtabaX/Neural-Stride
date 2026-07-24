"""
================================================================================
NeuralStride AI - JSONL Health Report (Training-Data Verification Tool)
================================================================================

Standalone script that reads a schema v2 .jsonl file (from LandmarkExporter
or video_landmark_extractor.py) and prints a health report. This is how you
verify the Phase 1 export layer worked BEFORE trusting the file enough to
add it to a training dataset — it trains nothing and builds no tensors.

REPORTS:
    - Schema version and the full session block
    - Total frame lines, total rep lines (completed vs aborted)
    - Feature vector length observed vs feature_count in metadata (must match
      on every frame)
    - Count of frames with NaN/inf in "features" (must be zero)
    - feature_quality distribution (min / median / max)
    - Per-feature min / mean / max, printed with feature_names
    - Rep duration distribution (frames and seconds)
    - Issue-code frequency across reps
    - Continuity check: every rep's [start_frame, end_frame] must correspond
      to frames that exist in the file, and rep windows must not overlap

USAGE:
    python inspect_jsonl.py Landmarks\\landmarks_Squat_s01_24-07-2026_02-30-00PM.jsonl

DEPENDENCIES:
    json, math, statistics, sys — all Python standard library
================================================================================
"""

import json
import math
import statistics
import sys


def load_lines(path):
    metadata = None
    frames = []
    reps = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARNING: line {line_number} is not valid JSON ({e}) - skipped")
                continue

            line_type = obj.get("type")
            if line_type == "metadata":
                metadata = obj
            elif line_type == "frame":
                frames.append(obj)
            elif line_type == "rep":
                reps.append(obj)
            else:
                print(f"WARNING: line {line_number} has unrecognised type {line_type!r} - skipped")

    return metadata, frames, reps


def report_session(metadata):
    print("=" * 70)
    print("SCHEMA / SESSION")
    print("=" * 70)
    if metadata is None:
        print("No metadata line found (expected the first line, type: metadata).")
        return
    print(f"format:          {metadata.get('format')}")
    print(f"schema_version:  {metadata.get('schema_version')}")
    session = metadata.get("session", {})
    if session:
        for key, value in session.items():
            print(f"session.{key:<14} {value}")
    else:
        print("No session block present.")


def report_feature_health(metadata, frames):
    print("\n" + "=" * 70)
    print("FEATURE HEALTH")
    print("=" * 70)

    expected_count = metadata.get("feature_count") if metadata else None
    feature_names = metadata.get("feature_names") if metadata else None
    print(f"feature_count (metadata): {expected_count}")

    mismatched = 0
    nan_inf_frames = 0
    qualities = []
    per_feature_values = None

    for frame in frames:
        features = frame.get("features", [])
        if expected_count is not None and len(features) != expected_count:
            mismatched += 1

        has_bad = any((not math.isfinite(v)) for v in features)
        if has_bad:
            nan_inf_frames += 1

        quality = frame.get("feature_quality")
        if quality is not None:
            qualities.append(quality)

        if features and not has_bad:
            if per_feature_values is None:
                per_feature_values = [[] for _ in features]
            if len(features) == len(per_feature_values):
                for i, v in enumerate(features):
                    per_feature_values[i].append(v)

    print(f"Frames with mismatched feature length: {mismatched} / {len(frames)}")
    print(f"Frames with NaN/inf in features:       {nan_inf_frames} / {len(frames)}")

    if qualities:
        print(
            f"feature_quality: min={min(qualities):.3f} "
            f"median={statistics.median(qualities):.3f} max={max(qualities):.3f}"
        )
    else:
        print("feature_quality: no data")

    print("\nPer-feature min / mean / max:")
    if per_feature_values:
        names = feature_names or [f"feature_{i}" for i in range(len(per_feature_values))]
        for i, values in enumerate(per_feature_values):
            if not values:
                continue
            name = names[i] if i < len(names) else f"feature_{i}"
            print(f"  {name:<26} min={min(values):8.3f}  mean={statistics.mean(values):8.3f}  max={max(values):8.3f}")
    else:
        print("  No feature data available.")


def report_reps(reps):
    print("\n" + "=" * 70)
    print("REP EVENTS")
    print("=" * 70)

    completed = [r for r in reps if r.get("completed")]
    aborted = [r for r in reps if r.get("aborted")]

    print(f"Total rep lines: {len(reps)} ({len(completed)} completed, {len(aborted)} aborted)")

    durations_frames = [r["end_frame"] - r["start_frame"] for r in reps if "start_frame" in r and "end_frame" in r]
    durations_seconds = [r["duration"] for r in reps if "duration" in r]

    if durations_frames:
        print(
            f"Rep duration (frames):  min={min(durations_frames)} "
            f"median={statistics.median(durations_frames):.1f} max={max(durations_frames)}"
        )
    if durations_seconds:
        print(
            f"Rep duration (seconds): min={min(durations_seconds):.2f} "
            f"median={statistics.median(durations_seconds):.2f} max={max(durations_seconds):.2f}"
        )

    issue_counts = {}
    for rep in reps:
        for issue in rep.get("issues") or []:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

    print("\nIssue-code frequency across reps:")
    if issue_counts:
        for issue, count in sorted(issue_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {issue:<16} {count}")
    else:
        print("  None recorded.")


def report_continuity(frames, reps):
    print("\n" + "=" * 70)
    print("CONTINUITY CHECK")
    print("=" * 70)

    frame_numbers = {f["frame"] for f in frames if "frame" in f}
    problems = 0

    for rep in reps:
        start = rep.get("start_frame")
        end = rep.get("end_frame")
        if start is None or end is None:
            print(f"  ERROR: rep {rep.get('rep_id')} missing start_frame/end_frame")
            problems += 1
            continue
        if start not in frame_numbers or end not in frame_numbers:
            print(
                f"  ERROR: rep {rep.get('rep_id')} window [{start}, {end}] "
                f"references frames not present in the file"
            )
            problems += 1

    sorted_reps = sorted(
        (r for r in reps if "start_frame" in r and "end_frame" in r),
        key=lambda r: r["start_frame"]
    )
    for a, b in zip(sorted_reps, sorted_reps[1:]):
        if a["end_frame"] >= b["start_frame"]:
            print(
                f"  ERROR: rep {a.get('rep_id')} (ends at {a['end_frame']}) overlaps "
                f"rep {b.get('rep_id')} (starts at {b['start_frame']})"
            )
            problems += 1

    if problems == 0:
        print("  OK - all rep windows reference existing frames and do not overlap.")
    else:
        print(f"  {problems} problem(s) found.")

    return problems


def main():
    if len(sys.argv) != 2:
        print("Usage: python inspect_jsonl.py <path-to-file.jsonl>")
        sys.exit(1)

    path = sys.argv[1]

    try:
        metadata, frames, reps = load_lines(path)
    except FileNotFoundError:
        print(f"Error: file not found: {path}")
        sys.exit(1)

    report_session(metadata)
    print(f"\nTotal frame lines: {len(frames)}")

    report_feature_health(metadata, frames)
    report_reps(reps)
    problems = report_continuity(frames, reps)

    print("\n" + "=" * 70)
    if problems == 0:
        print("HEALTH CHECK: PASSED")
    else:
        print("HEALTH CHECK: FAILED")
    print("=" * 70)


if __name__ == "__main__":
    main()
