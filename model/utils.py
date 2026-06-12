import numpy as np
import torch
from skimage.feature import canny
from skimage.measure import label
from tqdm import tqdm


def connected_components(mask_np):
    labeled = label(mask_np.astype(np.uint8))
    return labeled, int(labeled.max())


def extract_contour(mask_np):
    return canny(mask_np.astype(np.float32), sigma=1.0, low_threshold=0.1, high_threshold=0.3)


def box_count_downsample(mask_np, cell_size):
    height, width = mask_np.shape
    new_h = height // cell_size
    new_w = width // cell_size
    if new_h == 0 or new_w == 0:
        return mask_np
    mask_cropped = mask_np[: new_h * cell_size, : new_w * cell_size]
    blocks = mask_cropped.reshape(new_h, cell_size, new_w, cell_size)
    return blocks.max(axis=(1, 3))


def compute_intersection_ratio(gt_cells, pred_cells):
    intersection = np.logical_and(gt_cells, pred_cells).sum()
    gt_area = gt_cells.sum()
    if gt_area == 0:
        return 1.0
    return intersection / gt_area


def match_objects(gt_mask, pred_mask, iou_threshold=0.5):
    gt_labeled, n_gt = connected_components(gt_mask)
    pred_labeled, n_pred = connected_components(pred_mask)
    matches = []
    matched_pred = set()

    for gt_obj in range(1, n_gt + 1):
        gt_component = gt_labeled == gt_obj
        best_iou = 0.0
        best_pred_obj = None
        for pred_obj in range(1, n_pred + 1):
            if pred_obj in matched_pred:
                continue
            pred_component = pred_labeled == pred_obj
            intersection = np.logical_and(gt_component, pred_component).sum()
            union = np.logical_or(gt_component, pred_component).sum()
            iou = intersection / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_pred_obj = pred_obj
        if best_iou >= iou_threshold and best_pred_obj is not None:
            matches.append((gt_obj, best_pred_obj))
            matched_pred.add(best_pred_obj)

    return gt_labeled, pred_labeled, matches


def compute_pairwise_iou(gt_mask, pred_mask, iou_threshold=0.5):
    gt_labeled, n_gt = connected_components(gt_mask)
    pred_labeled, n_pred = connected_components(pred_mask)
    ious = []

    for gt_obj in range(1, n_gt + 1):
        gt_component = gt_labeled == gt_obj
        for pred_obj in range(1, n_pred + 1):
            pred_component = pred_labeled == pred_obj
            intersection = np.logical_and(gt_component, pred_component).sum()
            if intersection <= 0:
                continue
            union = np.logical_or(gt_component, pred_component).sum()
            iou = intersection / union if union > 0 else 0.0
            if iou >= iou_threshold:
                ious.append(float(iou))
            elif iou > 0:
                ious.append(float(iou))

    if not ious and np.logical_and(gt_mask, pred_mask).any():
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        ious.append(float(intersection / union) if union > 0 else 0.0)

    return ious


def compute_miou_per_object(gt_contour, pred_contour, scales):
    ratios = []
    for cell_size in scales:
        gt_down = box_count_downsample(gt_contour, cell_size)
        pred_down = box_count_downsample(pred_contour, cell_size)
        ratios.append(compute_intersection_ratio(gt_down, pred_down))
    return float(np.mean(ratios)) if ratios else 0.0


def compute_multiscale_iou(gt_mask, pred_mask, scales=(32, 16, 8, 4, 2, 1)):
    gt_labeled, pred_labeled, matches = match_objects(gt_mask, pred_mask, iou_threshold=0.5)
    if not matches:
        gt_contour = extract_contour(gt_mask)
        pred_contour = extract_contour(pred_mask)
        return compute_miou_per_object(gt_contour, pred_contour, scales)

    scores = []
    for gt_obj, pred_obj in matches:
        gt_obj_mask = gt_labeled == gt_obj
        pred_obj_mask = pred_labeled == pred_obj
        scores.append(
            compute_miou_per_object(
                extract_contour(gt_obj_mask),
                extract_contour(pred_obj_mask),
                scales,
            )
        )
    return float(np.mean(scores)) if scores else 0.0


def compute_dataset_pairwise_miou(y_true, y_pred, iou_threshold=0.5):
    scores = []
    for gt, pred in zip(y_true, y_pred):
        gt_mask = gt[0] > 0.5
        pred_mask = pred[0] > 0.5
        ious = compute_pairwise_iou(gt_mask, pred_mask, iou_threshold=iou_threshold)
        if ious:
            scores.append(float(np.mean(ious)))
    return float(np.mean(scores)) if scores else 0.0


def compute_dataset_multiscale_iou(y_true, y_pred, iou_threshold=0.5):
    scores = []
    for gt, pred in zip(y_true, y_pred):
        gt_mask = gt[0] > 0.5
        pred_mask = pred[0] > 0.5
        scores.append(compute_multiscale_iou(gt_mask, pred_mask))
    return float(np.mean(scores)) if scores else 0.0


def compute_average_recall_iou_thresholds(y_true, y_pred, thresholds=np.arange(0.5, 1.0, 0.05)):
    recalls = []
    for thresh in thresholds:
        recall_sum = 0.0
        for gt, pred in zip(y_true, y_pred):
            gt_mask = gt[0] > 0.5
            pred_mask = pred[0] > 0.5
            ious = compute_pairwise_iou(gt_mask, pred_mask, iou_threshold=thresh)
            recall_sum += len(ious)
        recalls.append(recall_sum / max(len(y_true), 1))
    return float(np.mean(recalls))


def collect_predictions_and_labels(model, test_loader, device):
    model.eval()
    y_true, y_score, preds_05, y_targets = [], [], [], []

    with torch.no_grad():
        for images, masks, _ in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            logits, _ = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            masks_np = masks.cpu().numpy()

            y_true.append(masks_np)
            y_score.append(probs)
            preds_05.append((probs > 0.5).astype(np.float32))
            y_targets.append(masks_np)

    return (
        np.concatenate(y_true, axis=0),
        np.concatenate(y_score, axis=0),
        np.concatenate(preds_05, axis=0),
        np.concatenate(y_targets, axis=0),
    )
