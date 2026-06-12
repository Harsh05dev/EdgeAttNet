import time
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from utils import (
    collect_predictions_and_labels,
    compute_average_recall_iou_thresholds,
    compute_dataset_multiscale_iou,
    compute_dataset_pairwise_miou,
    compute_pairwise_iou,
)


def compute_ap(y_true, y_score):
    y_true_flat = y_true.flatten()
    y_score_flat = y_score.flatten()
    precision, recall, thresholds = precision_recall_curve(y_true_flat, y_score_flat)
    ap = average_precision_score(y_true_flat, y_score_flat)
    return ap, precision, recall, thresholds


def evaluate_model(model, test_loader, device):
    start_time = time.time()
    print("Collecting predictions")
    y_true, y_score, preds_05, y_targets = collect_predictions_and_labels(
        model, test_loader, device
    )

    print("Computing Average Precision")
    ap, precision, recall, thresholds_raw = compute_ap(y_true, y_score)

    thresholds = np.arange(0.5, 1.0, 0.05)
    print("Computing Average Recall (object-level)")
    ar = compute_average_recall_iou_thresholds(y_targets, preds_05, thresholds)

    print("Computing Pairwise Mean IoU (old method)")
    pairwise_miou = compute_dataset_pairwise_miou(y_targets, preds_05, iou_threshold=0.5)

    print("Computing Multiscale IoU (NEW paper-aligned method)")
    ms_iou = compute_dataset_multiscale_iou(y_targets, preds_05, iou_threshold=0.5)

    duration = time.time() - start_time
    print(f"Evaluation done in {duration:.2f}s")

    return {
        "AP": ap,
        "Average Recall": ar,
        "Pairwise mIoU (old)": pairwise_miou,
        "Multiscale IoU (new)": ms_iou,
    }
