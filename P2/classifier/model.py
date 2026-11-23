from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from .config import CLASS_NAMES, CROP_SIZE, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES


def build(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(params.numel() for params in model.parameters() if params.requires_grad or not trainable_only)

def freeze_backbone(model: nn.Module, frozen: bool) -> None:
    for params in model.features.parameters():
        params.requires_grad_(not frozen)

    model.features.train(not frozen)

def to_model_input(batch_u8: torch.Tensor) -> torch.Tensor:
    """uint8 NHWC -> float NCHW normalized"""
    x = batch_u8.permute(0, 3, 1, 2).float().div_(255)
    
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std

def center_crop(batch: torch.Tensor, size: int = CROP_SIZE) -> torch.Tensor:
    h, w = batch.shape[1], batch.shape[2]
    top, left = (h - size) // 2, (w - size) // 2

    return batch[:, top : top + size, left : left + size]

def save(path: Path, model: nn.Module, config: dict, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(), 
            "class_names": CLASS_NAMES, 
            "config": config, 
            "metrics": metrics
        },
        path,
    )

def load(path: Path, device: str | torch.device) -> nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if list(checkpoint["class_names"]) != CLASS_NAMES:
        raise ValueError(f"ISSUE WIH CLASS NAMES: {checkpoint['class_names']}")
    
    model = build(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()

@torch.no_grad()
def predict_logits(model: nn.Module, images_u8: np.ndarray | torch.Tensor, batch_size: int, device: str | torch.device) -> np.ndarray:
    device = device or next(model.parameters()).device
    model.eval()
    out = []

    for index in range(0, len(images_u8), batch_size):
        batch = torch.as_tensor(np.asarray(images_u8[index : index + batch_size])).to(device, non_blocking=True)
        input = to_model_input(center_crop(batch)).contiguous(memory_format=torch.channels_last)
        out.append(model(input).float().cpu())
    
    return torch.cat(out).numpy()