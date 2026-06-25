#!/usr/bin/env python3
"""Train EdgeAttNet on MAGFILO COCO annotations."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from edgeattnet_model import UNetEdgeTransformer
from data_loader import DEFAULT_COCO_JSON, DEFAULT_IMAGE_DIR, create_data_loaders
from total_params import count_parameters
from tqdm import tqdm


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = (2, 3)
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    return 1.0 - ((2.0 * inter + eps) / (denom + eps)).mean()


def segmentation_stats(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict:
    pred = (torch.sigmoid(logits) >= threshold).float()
    target = (target >= 0.5).float()
    tp = (pred * target).sum().item()
    fp = (pred * (1.0 - target)).sum().item()
    fn = ((1.0 - pred) * target).sum().item()
    return {"tp": tp, "fp": fp, "fn": fn}


def finalize_metrics(stats: dict, loss: float, n_batches: int) -> dict:
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    return {
        "loss": loss / max(n_batches, 1),
        "dice": (2.0 * tp) / max(2.0 * tp + fp + fn, 1.0),
        "iou": tp / max(tp + fp + fn, 1.0),
    }


def run_epoch(model, loader, optimizer, device, training: bool) -> dict:
    model.train(training)
    loss_sum = 0.0
    stats = {"tp": 0.0, "fp": 0.0, "fn": 0.0}
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, masks, _ in tqdm(loader, desc="train" if training else "eval", leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits, _ = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, masks) + dice_loss(logits, masks)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            loss_sum += float(loss.detach().item())
            batch_stats = segmentation_stats(logits.detach(), masks)
            for key, value in batch_stats.items():
                stats[key] += value

    return finalize_metrics(stats, loss_sum, len(loader))


def parse_args():
    parser = argparse.ArgumentParser(description="Train EdgeAttNet")
    parser.add_argument("--coco-json", type=Path, default=Path(DEFAULT_COCO_JSON))
    parser.add_argument("--image-dir", type=Path, default=Path(DEFAULT_IMAGE_DIR))
    parser.add_argument("--train-json", type=Path, default=None)
    parser.add_argument("--val-json", type=Path, default=None)
    parser.add_argument("--test-json", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("../runs/phase1_fair_raw_year2048_ep50"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--small-subset", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = create_data_loaders(
        coco_json=str(args.coco_json),
        image_dir=str(args.image_dir),
        train_json=str(args.train_json) if args.train_json else None,
        val_json=str(args.val_json) if args.val_json else None,
        test_json=str(args.test_json) if args.test_json else None,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        use_small_subset=args.small_subset,
        num_workers=args.num_workers,
    )
    if train_loader is None or val_loader is None:
        raise RuntimeError("Training requires non-empty train and validation splits.")

    model = UNetEdgeTransformer().to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    config = vars(args)
    config["device"] = str(device)
    config["parameters"] = count_parameters(model)
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

    best_val_dice = -1.0
    metrics_path = args.out_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "train_dice", "train_iou", "val_loss", "val_dice", "val_iou"],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, optimizer, device, training=True)
            val_metrics = run_epoch(model, val_loader, None, device, training=False)
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_dice": train_metrics["dice"],
                "train_iou": train_metrics["iou"],
                "val_loss": val_metrics["loss"],
                "val_dice": val_metrics["dice"],
                "val_iou": val_metrics["iou"],
            }
            writer.writerow(row)
            handle.flush()

            print(
                f"Epoch {epoch:03d} | "
                f"train dice {train_metrics['dice']:.4f} iou {train_metrics['iou']:.4f} | "
                f"val dice {val_metrics['dice']:.4f} iou {val_metrics['iou']:.4f}"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                },
                args.out_dir / "last_model.pth",
            )

            if val_metrics["dice"] > best_val_dice:
                best_val_dice = val_metrics["dice"]
                torch.save(model.state_dict(), args.out_dir / "best_model.pth")
                print(f"  Saved new best model (val dice={best_val_dice:.4f})")

    if test_loader is not None:
        best_path = args.out_dir / "best_model.pth"
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device))
            print(f"Loaded best checkpoint for test eval: {best_path}")
        test_metrics = run_epoch(model, test_loader, None, device, training=False)
        print(
            f"Test dice {test_metrics['dice']:.4f} | "
            f"test iou {test_metrics['iou']:.4f}"
        )
        (args.out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
