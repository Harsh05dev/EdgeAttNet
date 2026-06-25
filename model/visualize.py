# Required imports for visualization
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.transform import resize


def visualize_prediction_by_filename(model, dataset, device, filename, save_path=None):
    """
    Visualizes the prediction for a given image filename from the dataset.

    Args:
        model: Trained model.
        dataset: Dataset containing images and masks.
        device: CPU or GPU device.
        filename: COCO image id to visualize (without extension).
        save_path: Optional path to save the figure as PNG.
    """
    model.eval()

    # Search for the image in the dataset using the filename (without extension)
    image, gt_mask, image_filename = None, None, None
    for img, mask, fname in dataset:
        if fname.split('.')[0] == filename:  # Match filename without extension
            image, gt_mask, image_filename = img, mask, fname
            break

    if image is None:
        print(f"Filename '{filename}' not found in the dataset.")
        return None

    image_tensor = image.unsqueeze(0).to(device)

    with torch.no_grad():
        pred, _ = model(image_tensor)  # Unpack the tuple
        pred_mask = torch.sigmoid(pred).squeeze().cpu().numpy()

    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    gt_mask = gt_mask.squeeze().cpu().numpy().astype(np.uint8)

    img_np = image.squeeze().cpu().numpy()
    if img_np.ndim == 3:
        img_np_show = np.moveaxis(img_np, 0, -1)
    else:
        img_np_show = np.clip(img_np * 0.5 + 0.5, 0.0, 1.0)

    H_img, W_img = img_np_show.shape[:2]

    # Resize masks if needed
    if gt_mask.shape != (H_img, W_img):
        gt_mask = resize(gt_mask, (H_img, W_img), order=0, preserve_range=True).astype(np.uint8)
    if pred_mask.shape != (H_img, W_img):
        pred_mask = resize(pred_mask, (H_img, W_img), order=0, preserve_range=True).astype(np.uint8)

    # Create the figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot the input image
    for ax in axs:
        if img_np_show.ndim == 2:
            ax.imshow(img_np_show, cmap='gray')
        else:
            ax.imshow(img_np_show)
        ax.axis('off')

    axs[0].set_title(f"Input Image: {filename}")

    # Ground truth overlay (green)
    gt_overlay = np.zeros((H_img, W_img, 4), dtype=np.float32)
    gt_overlay[gt_mask == 1] = [0, 0.5, 0, 0.6]
    axs[1].imshow(gt_overlay)
    axs[1].set_title("Image + Ground Truth Mask")

    # Predicted overlay (red)
    pred_overlay = np.zeros((H_img, W_img, 4), dtype=np.float32)
    pred_overlay[pred_mask == 1] = [1, 0, 0, 1]

    axs[2].imshow(pred_overlay)
    axs[2].set_title("Image + Predicted Mask")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved visualization to {save_path}")
        return save_path

    plt.show()
    plt.close(fig)
    return None
