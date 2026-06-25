#!/usr/bin/env python3
"""Create year-based train/val/test COCO splits for MAGFILO.

Years are parsed from each COCO image id (e.g. "030101-20110109104734Ch" -> 2011).
Default split:
  train = 2011-2019, val = 2020, test = 2021-2022
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def year_of(image_id: str) -> int:
    if "-" in image_id:
        return int(image_id.split("-")[1][:4])
    return int(image_id[:4])


def build_subset(coco: dict, ids: set) -> dict:
    out = {k: v for k, v in coco.items() if k not in {"images", "annotations"}}
    out["images"] = [i for i in coco.get("images", []) if i["id"] in ids]
    out["annotations"] = [
        a for a in coco.get("annotations", []) if a.get("image_id") in ids
    ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-json", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--train-years", type=int, nargs="+", default=list(range(2011, 2020)))
    ap.add_argument("--val-years", type=int, nargs="+", default=[2020])
    ap.add_argument("--test-years", type=int, nargs="+", default=[2021, 2022])
    args = ap.parse_args()

    coco = json.loads(args.coco_json.read_text())
    buckets: dict[str, set] = {"train": set(), "val": set(), "test": set()}
    skipped = 0
    for img in coco.get("images", []):
        try:
            y = year_of(img["id"])
        except Exception:
            skipped += 1
            continue
        if y in args.train_years:
            buckets["train"].add(img["id"])
        elif y in args.val_years:
            buckets["val"].add(img["id"])
        elif y in args.test_years:
            buckets["test"].add(img["id"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "source": str(args.coco_json),
        "train_years": args.train_years,
        "val_years": args.val_years,
        "test_years": args.test_years,
        "skipped_images": skipped,
    }
    for name, ids in buckets.items():
        sub = build_subset(coco, ids)
        (args.out_dir / f"{name}.json").write_text(json.dumps(sub))
        summary[f"{name}_images"] = len(sub["images"])
        summary[f"{name}_annotations"] = len(sub["annotations"])
    (args.out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
