#!/usr/bin/env python3
"""Generate side-by-side Phase 1 test comparisons (same image, U-Net vs EdgeAttNet)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader import CocoFilamentDataset, ImageMaskTransform
from edgeattnet_model import UNetEdgeTransformer

FILAMENT_ROOT = Path(__file__).resolve().parents[2] / "filament"
sys.path.insert(0, str(FILAMENT_ROOT))
from train_unet import UNet  # noqa: E402


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    split = root / "splits/year_2011-2019_val2020_test2021-2022"
    phase1 = root / "runs/phase1_fair_raw_year512_ep50"
    filament_phase1 = FILAMENT_ROOT / "runs/phase1_fair_raw_year512_ep50"

    parser = argparse.ArgumentParser(description="Phase 1 U-Net vs EdgeAttNet image comparisons")
    parser.add_argument("--test-json", type=Path, default=split / "test.json")
    parser.add_argument("--image-dir", type=Path, default=Path("/media/data/magfilo_dataset/images"))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--unet-ckpt", type=Path, default=filament_phase1 / "best.pt")
    parser.add_argument("--edgeattnet-ckpt", type=Path, default=phase1 / "best_model.pth")
    parser.add_argument(
        "--out-dirs",
        type=Path,
        nargs="+",
        default=[phase1 / "comparisons", filament_phase1 / "comparisons"],
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--grid-count", type=int, default=8, help="Images in summary_grid.png")
    return parser.parse_args()


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = float(np.logical_and(pred, gt).sum())
    return (2.0 * inter) / (pred.sum() + gt.sum() + 1e-8)


def tensor_to_gray(image: torch.Tensor) -> np.ndarray:
    x = image.squeeze().cpu().numpy()
    x = np.clip(x * 0.5 + 0.5, 0.0, 1.0)
    return x


def overlay_pred(gray: np.ndarray, pred: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    rgb = np.stack([gray, gray, gray], axis=-1)
    overlay = np.zeros_like(rgb)
    mask = pred.astype(bool)
    overlay[mask] = color
    return np.clip(rgb * 0.55 + overlay * 0.45, 0.0, 1.0)


def overlay_gt(gray: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return overlay_pred(gray, gt, (0.0, 1.0, 0.0))


@torch.no_grad()
def predict_mask(model: torch.nn.Module, image: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    logits = model(image.unsqueeze(0).to(device))
    if isinstance(logits, tuple):
        logits = logits[0]
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    return (prob > 0.5).astype(np.uint8)


def save_quad_panel(
    out_path: Path,
    image_id: str,
    gray: np.ndarray,
    gt: np.ndarray,
    unet_pred: np.ndarray,
    edge_pred: np.ndarray,
    unet_dice: float,
    edge_dice: float,
) -> None:
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    panels = [
        (gray, "Input"),
        (overlay_gt(gray, gt), "Ground truth (green)"),
        (overlay_pred(gray, unet_pred, (1.0, 0.2, 0.1)), f"U-Net  Dice {unet_dice:.3f}"),
        (overlay_pred(gray, edge_pred, (0.1, 0.55, 1.0)), f"EdgeAttNet  Dice {edge_dice:.3f}"),
    ]
    for ax, (img, title) in zip(axs, panels):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(image_id, fontsize=13, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_single_overlay(out_path: Path, gray: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> None:
    rgb = np.stack([gray, gray, gray], axis=-1)
    rgb[gt.astype(bool), 1] = 1.0
    rgb[pred.astype(bool), 0] = 1.0
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.clip(rgb, 0.0, 1.0))
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_summary_grid(rows: list[dict], out_path: Path, count: int) -> None:
    if not rows:
        return
    ranked = sorted(rows, key=lambda r: r["edge_dice"] - r["unet_dice"], reverse=True)
    picks = []
    picks.extend(ranked[: max(1, count // 2)])
    picks.extend(sorted(rows, key=lambda r: r["unet_dice"] - r["edge_dice"], reverse=True)[: max(1, count // 2)])
    seen = set()
    selected = []
    for row in picks + ranked:
        if row["image_id"] in seen:
            continue
        seen.add(row["image_id"])
        selected.append(row)
        if len(selected) >= count:
            break

    n = len(selected)
    fig, axs = plt.subplots(n, 4, figsize=(20, 4.2 * n))
    if n == 1:
        axs = np.expand_dims(axs, axis=0)
    for row_ax, row in zip(axs, selected):
        gray = row["gray"]
        gt = row["gt"]
        panels = [
            (gray, row["image_id"]),
            (overlay_gt(gray, gt), "GT"),
            (overlay_pred(gray, row["unet_pred"], (1.0, 0.2, 0.1)), f"U-Net {row['unet_dice']:.3f}"),
            (overlay_pred(gray, row["edge_pred"], (0.1, 0.55, 1.0)), f"EdgeAttNet {row['edge_dice']:.3f}"),
        ]
        for ax, (img, title) in zip(row_ax, panels):
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
    fig.suptitle("Phase 1 test comparisons (512, year split)", fontsize=14, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_unet(ckpt_path: Path, device: torch.device) -> UNet:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model = UNet(
        base_channels=int(config.get("base_channels", 32)),
        depth=int(config.get("depth", 5)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def load_edgeattnet(ckpt_path: Path, device: torch.device) -> UNetEdgeTransformer:
    model = UNetEdgeTransformer().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = ImageMaskTransform(image_size=(args.image_size, args.image_size))
    dataset = CocoFilamentDataset(args.test_json, args.image_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    unet = load_unet(args.unet_ckpt, device)
    edgeattnet = load_edgeattnet(args.edgeattnet_ckpt, device)
    print(f"Loaded U-Net from {args.unet_ckpt}")
    print(f"Loaded EdgeAttNet from {args.edgeattnet_ckpt}")

    manifest: list[dict] = []
    cache_rows: list[dict] = []

    for images, masks, image_ids in loader:
        for i, image_id in enumerate(image_ids):
            image = images[i]
            mask = masks[i].squeeze().cpu().numpy() > 0.5
            gray = tensor_to_gray(image)

            unet_pred = predict_mask(unet, image, device)
            edge_pred = predict_mask(edgeattnet, image, device)

            unet_dice = dice_score(unet_pred, mask)
            edge_dice = dice_score(edge_pred, mask)

            row = {
                "image_id": str(image_id),
                "unet_dice": unet_dice,
                "edge_dice": edge_dice,
                "unet_iou": float(np.logical_and(unet_pred, mask).sum())
                / float(np.logical_or(unet_pred, mask).sum() + 1e-8),
                "edge_iou": float(np.logical_and(edge_pred, mask).sum())
                / float(np.logical_or(edge_pred, mask).sum() + 1e-8),
            }
            manifest.append(row)
            cache_rows.append(
                {
                    **row,
                    "gray": gray,
                    "gt": mask,
                    "unet_pred": unet_pred,
                    "edge_pred": edge_pred,
                }
            )

            for out_root in args.out_dirs:
                per_image = out_root / "per_image" / f"{image_id}_compare.png"
                save_quad_panel(
                    per_image,
                    str(image_id),
                    gray,
                    mask,
                    unet_pred,
                    edge_pred,
                    unet_dice,
                    edge_dice,
                )
                save_single_overlay(out_root / "unet" / f"{image_id}_overlay.png", gray, unet_pred, mask)
                save_single_overlay(
                    out_root / "edgeattnet" / f"{image_id}_overlay.png", gray, edge_pred, mask
                )

    summary = {
        "phase": "phase1_fair_raw_year512_ep50",
        "test_json": str(args.test_json),
        "num_images": len(manifest),
        "mean_unet_dice": float(np.mean([m["unet_dice"] for m in manifest])),
        "mean_edge_dice": float(np.mean([m["edge_dice"] for m in manifest])),
        "images": manifest,
    }

    for out_root in args.out_dirs:
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "manifest.json").write_text(json.dumps(summary, indent=2))
        save_summary_grid(cache_rows, out_root / "summary_grid.png", args.grid_count)
        print(f"Wrote comparisons to {out_root}")


if __name__ == "__main__":
    main()
