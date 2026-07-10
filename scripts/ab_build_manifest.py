#!/usr/bin/env python3.9
"""Build an A/B manifest by mining validate-spatial prompt blocks from the logs.

Each `[Claude:scene-game-validate-spatial] >>> PROMPT` block in the trace logs
carries the round's obj_a / obj_b / relation / toy_list and a HH:MM:SS stamp.
Captured frames are named scene_answer_YYYYMMDD_HHMMSS.jpg, so we pair a block
to its frame by timestamp (date from the log filename when available).

Output: manifest.json = [{image, obj_a, obj_b, relation, toy_list, source, ts}]

  .venv39/bin/python scripts/ab_build_manifest.py \
        --logs src/logs --user-data src/user_data --out scripts/ab_out/manifest.json
"""

import argparse
import glob
import json
import os
import re

RE_TOYS = re.compile(r"The valid game objects are:\s*(.+?)\.\s*Only identify")
RE_AB = re.compile(r"actual spatial relation of the (.+?) TO the (.+?)\?")
RE_REL = re.compile(r"requested relation '([^']+)'")
RE_PROMPT_START = re.compile(r"scene-game-validate-spatial\]\s*>>>\s*PROMPT")
RE_TIME = re.compile(r"(\d{2}):(\d{2}):(\d{2})")
RE_DATE_IN_NAME = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
RE_FRAME_TS = re.compile(r"scene_answer_(\d{8})_(\d{6})\.jpg$")


def index_frames(user_data):
    """Return (by_full_ts, by_hhmmss) dicts mapping ts -> [paths]."""
    by_full, by_hhmmss = {}, {}
    for p in glob.glob(os.path.join(user_data, "*", "captured_scenes", "scene_answer_*.jpg")):
        m = RE_FRAME_TS.search(os.path.basename(p))
        if not m:
            continue
        ymd, hms = m.group(1), m.group(2)
        by_full.setdefault(f"{ymd}_{hms}", []).append(p)
        by_hhmmss.setdefault(hms, []).append(p)
    return by_full, by_hhmmss


def find_frame(by_full, by_hhmmss, ymd, hms):
    """Exact ts, then ±2s (same date), then unique HHMMSS across all dates."""
    if ymd:
        key = f"{ymd}_{hms}"
        if by_full.get(key):
            return by_full[key][0]
        base = int(hms)
        h, m, s = int(hms[:2]), int(hms[2:4]), int(hms[4:6])
        for delta in (1, -1, 2, -2):
            ss = s + delta
            hh, mm = h, m
            if ss < 0:
                ss += 60; mm -= 1
            elif ss > 59:
                ss -= 60; mm += 1
            if 0 <= mm <= 59:
                cand = f"{ymd}_{hh:02d}{mm:02d}{ss:02d}"
                if by_full.get(cand):
                    return by_full[cand][0]
    hits = by_hhmmss.get(hms, [])
    if len(hits) == 1:
        return hits[0]
    return None


def parse_log(path):
    """Yield dicts {time_hms, obj_a, obj_b, relation, toy_list} per prompt block."""
    with open(path, errors="replace") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if RE_PROMPT_START.search(lines[i]):
            tm = RE_TIME.search(lines[i])
            hms = "".join(tm.groups()) if tm else None
            obj_a = obj_b = relation = None
            toy_list = None
            j = i + 1
            while j < len(lines) and j < i + 60:
                ln = lines[j]
                if RE_PROMPT_START.search(ln) or "<<< RESPONSE" in ln:
                    break
                if toy_list is None:
                    mt = RE_TOYS.search(ln)
                    if mt:
                        toy_list = [t.strip() for t in mt.group(1).split(",") if t.strip()]
                if obj_a is None:
                    ma = RE_AB.search(ln)
                    if ma:
                        obj_a, obj_b = ma.group(1).strip(), ma.group(2).strip()
                if relation is None:
                    mr = RE_REL.search(ln)
                    if mr:
                        relation = mr.group(1).strip()
                j += 1
            if obj_a and obj_b and relation and hms:
                yield {"time_hms": hms, "obj_a": obj_a, "obj_b": obj_b,
                       "relation": relation, "toy_list": toy_list}
            i = j
        else:
            i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="src/logs")
    ap.add_argument("--user-data", default="src/user_data")
    ap.add_argument("--out", default="scripts/ab_out/manifest.json")
    args = ap.parse_args()

    by_full, by_hhmmss = index_frames(args.user_data)
    print(f"indexed {sum(len(v) for v in by_full.values())} frames")

    manifest, seen = [], set()
    unpaired = 0
    for logf in sorted(glob.glob(os.path.join(args.logs, "*.log"))):
        dm = RE_DATE_IN_NAME.search(os.path.basename(logf))
        ymd = "".join(dm.groups()) if dm else None
        for blk in parse_log(logf):
            img = find_frame(by_full, by_hhmmss, ymd, blk["time_hms"])
            if not img:
                unpaired += 1
                continue
            if img in seen:
                continue
            seen.add(img)
            manifest.append({
                "image": img,
                "obj_a": blk["obj_a"], "obj_b": blk["obj_b"],
                "relation": blk["relation"], "toy_list": blk["toy_list"],
                "source": os.path.basename(logf), "ts": blk["time_hms"],
            })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"paired {len(manifest)} rounds to frames  (unpaired: {unpaired})")
    by_rel = {}
    for m in manifest:
        by_rel[m["relation"]] = by_rel.get(m["relation"], 0) + 1
    print("by relation:", dict(sorted(by_rel.items())))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
