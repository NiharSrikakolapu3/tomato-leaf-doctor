"""The classifier networks the stand-in runs were trained with (tools/train_stand_in.py).

pipeline.py rebuilds the network named in the weights folder's classifier.json, so a
backbone change needs no change here beyond its entry in BACKBONES.
"""
from torch import nn
from torchvision import models, transforms

from labels import CLASSES

IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# Fine-tuning rates are the usual ones for each family, not tuned on development photos.
BACKBONES = {
    "effnet_b0": {"kind": "torchvision", "lr": 2e-4, "wd": 1e-4},
    "convnext_tiny": {"kind": "timm", "name": "convnext_tiny.fb_in22k_ft_in1k", "lr": 1e-4, "wd": 0.05},
    "dinov2_vitb": {"kind": "timm", "name": "vit_base_patch14_dinov2.lvd142m", "lr": 3e-5, "wd": 0.05,
                    "args": {"img_size": 224}},
    "dinov3_vitb": {"kind": "timm", "name": "vit_base_patch16_dinov3.lvd1689m", "lr": 3e-5, "wd": 0.05},
    "siglip2_vitb": {"kind": "timm", "name": "vit_base_patch16_siglip_224.v2_webli", "lr": 3e-5, "wd": 0.05},
    "bioclip_vitb": {"kind": "open_clip", "name": "hf-hub:imageomics/bioclip", "lr": 3e-5, "wd": 0.05},
}


class ClipClassifier(nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual
        self.head = nn.Linear(visual.output_dim, len(CLASSES))

    def forward(self, images):
        return self.head(self.visual(images))


def build_classifier(backbone, pretrained):
    """Return (model, head parameters, mean, std) for an 8-class classifier."""
    spec = BACKBONES[backbone]
    if spec["kind"] == "torchvision":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
        return model, list(model.classifier.parameters()), *IMAGENET
    if spec["kind"] == "timm":
        import timm
        model = timm.create_model(spec["name"], pretrained=pretrained, num_classes=len(CLASSES),
                                  **spec.get("args", {}))
        config = timm.data.resolve_model_data_config(model)
        return model, list(model.get_classifier().parameters()), list(config["mean"]), list(config["std"])
    import open_clip
    # An hf-hub model always loads its released weights; the pipeline then loads ours over them.
    clip, _, preprocess = open_clip.create_model_and_transforms(spec["name"])
    normalize = next(t for t in preprocess.transforms if isinstance(t, transforms.Normalize))
    model = ClipClassifier(clip.visual)
    return model, list(model.head.parameters()), list(normalize.mean), list(normalize.std)
