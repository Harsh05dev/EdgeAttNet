import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


DEFAULT_COCO_JSON = "/media/data/magfilo_dataset/magfilo_2024_v1.0.json"
DEFAULT_IMAGE_DIR = "/media/data/magfilo_dataset/images"


class ImageMaskTransform:
    def __init__(self, image_size=(512, 512)):
        self.image_size = image_size
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __call__(self, image, mask):
        image = self.image_transform(image)
        mask = transforms.functional.resize(
            mask,
            self.image_size,
            interpolation=transforms.InterpolationMode.NEAREST,
        )
        mask = transforms.ToTensor()(mask)
        return image, mask


class CocoFilamentDataset(Dataset):
    """Load MAGFILO COCO annotations with images from a flat image directory."""

    def __init__(self, coco_json, image_dir, transform=None, image_ids=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        coco = json.loads(Path(coco_json).read_text())

        self.images = list(coco.get("images", []))
        if image_ids is not None:
            allowed = set(image_ids)
            self.images = [img for img in self.images if img["id"] in allowed]

        self.anns_by_image = defaultdict(list)
        for ann in coco.get("annotations", []):
            if ann.get("iscrowd", 0):
                continue
            self.anns_by_image[ann["image_id"]].append(ann)

    def __len__(self):
        return len(self.images)

    def _resolve_image_path(self, image_meta):
        file_name = Path(image_meta["file_name"]).name
        path = self.image_dir / file_name
        if not path.exists():
            raise FileNotFoundError(
                f"Image for id '{image_meta['id']}' not found at {path}"
            )
        return path

    def _rasterize_mask(self, image_meta):
        width = int(image_meta["width"])
        height = int(image_meta["height"])
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        for ann in self.anns_by_image.get(image_meta["id"], []):
            segmentation = ann.get("segmentation", [])
            if not isinstance(segmentation, list):
                continue
            for poly in segmentation:
                if isinstance(poly, list) and len(poly) >= 6:
                    points = [
                        (float(poly[i]), float(poly[i + 1]))
                        for i in range(0, len(poly), 2)
                    ]
                    draw.polygon(points, outline=1, fill=1)
        return mask

    def __getitem__(self, idx):
        image_meta = self.images[idx]
        image = Image.open(self._resolve_image_path(image_meta)).convert("L")
        mask = self._rasterize_mask(image_meta)

        if self.transform:
            image, mask = self.transform(image, mask)

        mask = (mask > 0).float()
        return image, mask, image_meta["id"]


def extract_year_from_image_id(image_id):
    try:
        if "-" in image_id:
            return int(image_id.split("-")[1][:4])
        return int(image_id[:4])
    except Exception as exc:
        raise ValueError(f"Failed to extract year from image ID '{image_id}': {exc}") from exc


def split_indices_by_year(dataset, train_years, val_years, test_years):
    train_indices, val_indices, test_indices = [], [], []
    for idx, image_meta in enumerate(dataset.images):
        year = extract_year_from_image_id(image_meta["id"])
        if year in train_years:
            train_indices.append(idx)
        elif year in val_years:
            val_indices.append(idx)
        elif year in test_years:
            test_indices.append(idx)
    return train_indices, val_indices, test_indices


def create_data_loaders(
    coco_json=DEFAULT_COCO_JSON,
    image_dir=DEFAULT_IMAGE_DIR,
    train_years=None,
    val_years=None,
    test_years=None,
    batch_size=4,
    image_size=(512, 512),
    use_small_subset=False,
    num_workers=4,
):
    if train_years is None:
        train_years = list(range(2011, 2020))
    if val_years is None:
        val_years = [2020]
    if test_years is None:
        test_years = list(range(2021, 2023))

    full_dataset = CocoFilamentDataset(coco_json=coco_json, image_dir=image_dir, transform=None)
    train_idx, val_idx, test_idx = split_indices_by_year(
        full_dataset, train_years, val_years, test_years
    )

    if use_small_subset:
        random.seed(42)
        train_idx = random.sample(train_idx, min(50, len(train_idx)))
        val_idx = random.sample(val_idx, min(10, len(val_idx)))
        test_idx = random.sample(test_idx, min(20, len(test_idx)))

    shared_transform = ImageMaskTransform(image_size=image_size)

    train_dataset = Subset(
        CocoFilamentDataset(coco_json, image_dir, transform=shared_transform),
        train_idx,
    )
    val_dataset = Subset(
        CocoFilamentDataset(coco_json, image_dir, transform=shared_transform),
        val_idx,
    )
    test_dataset = Subset(
        CocoFilamentDataset(coco_json, image_dir, transform=shared_transform),
        test_idx,
    )

    train_loader = None
    if len(train_dataset) > 0:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(1, num_workers // 2),
        )

    test_loader = None
    if len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(1, num_workers // 2),
        )

    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(val_dataset)}")
    print(f"Total test samples: {len(test_dataset)}")

    return train_loader, val_loader, test_loader
