#!/usr/bin/env python3
"""Evaluate a trained EdgeAttNet checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from data_loader import DEFAULT_COCO_JSON, DEFAULT_IMAGE_DIR, create_data_loaders
from edgeattnet_model import UNetEdgeTransformer
from evaluation import evaluate_model
from visualize import visualize_prediction_by_filename


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EdgeAttNet")
    parser.add_argument("--coco-json", type=Path, default=Path(DEFAULT_COCO_JSON))
    parser.add_argument("--image-dir", type=Path, default=Path(DEFAULT_IMAGE_DIR))
    parser.add_argument("--model-path", type=Path, default=Path("../models/best_model.pth"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--visualize-id", type=str, default=None)
    parser.add_argument(
        "--save-viz",
        type=Path,
        default=None,
        help="Save visualization PNG (default: ../models/visualizations/<visualize-id>.png)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip full-dataset evaluation when only visualizing",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {args.model_path}. Train first with: python train.py"
        )

    _, _, test_loader = create_data_loaders(
        coco_json=str(args.coco_json),
        image_dir=str(args.image_dir),
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        train_years=[],
        val_years=[],
        test_years=list(range(2011, 2023)),
    )
    if test_loader is None:
        raise RuntimeError("No test samples found for the requested year split.")

    model = UNetEdgeTransformer().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {args.model_path}")

    if not args.skip_eval:
        print("Evaluating the model...")
        results = evaluate_model(model, test_loader, device)
        print("Evaluation results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    if args.visualize_id:
        save_path = args.save_viz
        if save_path is None:
            save_path = Path("../models/visualizations") / f"{args.visualize_id}.png"
        print(f"Visualizing prediction for {args.visualize_id}...")
        visualize_prediction_by_filename(
            model,
            test_loader.dataset,
            device,
            args.visualize_id,
            save_path=save_path,
        )


if __name__ == "__main__":
    main()
