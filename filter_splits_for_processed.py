#!/usr/bin/env python3
"""Filter year-based COCO splits to images present in processed-H-alpha."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def resolve_magfilo_image_path(image_meta: dict, image_dir: Path) -> Path | None:
    image_dir = Path(image_dir)
    image_id = image_meta.get("id", "")

    if image_id and "-" in image_id:
        ts = image_id.split("-", 1)[1]
        if len(ts) >= 8 and ts[:8].isdigit():
            y, m, d = ts[0:4], ts[4:6], ts[6:8]
            for ext in (".jpg", ".jpeg"):
                path = image_dir / y / m / d / f"{image_id}{ext}"
                if path.is_file() and path.stat().st_size > 0:
                    return path

    file_name = image_meta.get("file_name")
    if file_name:
        path = image_dir / Path(file_name).name
        if path.is_file() and path.stat().st_size > 0:
            return path

    return None


def filter_split(coco: dict, image_dir: Path) -> tuple[dict, list[str]]:
    kept_images = []
    missing = []
    for image in coco.get("images", []):
        if resolve_magfilo_image_path(image, image_dir) is not None:
            kept_images.append(image)
        else:
            missing.append(image["id"])

    kept_ids = {image["id"] for image in kept_images}
    out = {k: v for k, v in coco.items() if k not in {"images", "annotations"}}
    out["images"] = kept_images
    out["annotations"] = [
        ann for ann in coco.get("annotations", []) if ann.get("image_id") in kept_ids
    ]
    return out, missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split-dir",
        type=Path,
        default=Path("splits/year_2011-2019_val2020_test2021-2022"),
    )
    ap.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/processed-H-alpha"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("splits/year_2011-2019_val2020_test2021-2022_prep"),
    )
    ap.add_argument(
        "--copy-to-filament",
        type=Path,
        default=Path("../filament/splits/year_2011-2019_val2020_test2021-2022_prep"),
        help="Also copy split JSONs to filament repo (metadata only, not images).",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_split": str(args.split_dir),
        "image_dir": str(args.image_dir.resolve()),
        "splits": {},
        "total_missing": 0,
    }

    for split_name in ("train", "val", "test"):
        src = args.split_dir / f"{split_name}.json"
        coco = json.loads(src.read_text())
        filtered, missing = filter_split(coco, args.image_dir)
        (args.out_dir / f"{split_name}.json").write_text(json.dumps(filtered))
        summary["splits"][split_name] = {
            "original_images": len(coco.get("images", [])),
            "kept_images": len(filtered.get("images", [])),
            "missing_images": len(missing),
            "annotations": len(filtered.get("annotations", [])),
        }
        summary["total_missing"] += len(missing)
        print(
            f"{split_name}: kept {len(filtered['images'])}/{len(coco['images'])} "
            f"(missing {len(missing)})"
        )

    (args.out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote filtered splits to {args.out_dir}")

    if args.copy_to_filament:
        args.copy_to_filament.mkdir(parents=True, exist_ok=True)
        for name in ("train.json", "val.json", "test.json", "split_summary.json"):
            shutil.copy2(args.out_dir / name, args.copy_to_filament / name)
        print(f"Copied split JSONs to {args.copy_to_filament}")


if __name__ == "__main__":
    main()
