import os
from pathlib import Path
import torch
from torchvision import transforms


def make_dinov3_transform(resize_size: int = 224):
    to_tensor = transforms.Lambda(lambda x: x / 255.)
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose([to_tensor, resize, normalize])


MODEL_NAMES = {
    'dinov3_vits16',
    "dinov3_vits16plus",
    "dinov3_vitb16",
    "dinov3_vitl16",
    "dinov3_vith16plus",
    "dinov3_vit7b16",
}
SHA_CHECKSUM = {
    "dinov3_vits16": "08c60483",
    "dinov3_vits16plus": "4057cbaa",
    "dinov3_vitb16": "73cec8be",
    "dinov3_vitl16": "8aa4cbdd",
    "dinov3_vith16plus": "7c1da9a5",
    "dinov3_vit7b16": "a955f4ea",
}
METHOD_ROOT = Path(__file__).resolve().parents[1]
print(f"using METHOD_ROOT: {METHOD_ROOT}")
REPO_DIR = os.environ.get("DINOV3_REPO_DIR", os.path.join(METHOD_ROOT.parent, "dinov3"))
CHECKPOINT_DIR = os.environ.get("DINOV3_CKPT_DIR", os.path.join(METHOD_ROOT, "..", "pretrained_models"))

def load_dinov3(model_name):
    assert model_name in MODEL_NAMES
    model = torch.hub.load(
        REPO_DIR,
        model_name,
        source='local',
        weights=os.path.join(CHECKPOINT_DIR, f"{model_name}_pretrain_lvd1689m-{SHA_CHECKSUM[model_name]}.pth")
    )
    return model
