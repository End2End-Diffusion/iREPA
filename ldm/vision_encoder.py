import torch
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from pathlib import Path
from torchvision.transforms import Normalize
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

# Compute pretrained models directory relative to this file
_FILE_DIR = Path(__file__).resolve().parent
PRETRAINED_DIR = _FILE_DIR.parent / "pretrained_models"


def fix_mocov3_state_dict(state_dict):
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder'):
            # fix naming bug in checkpoint
            new_k = k[len("module.base_encoder."):]
            if "blocks.13.norm13" in new_k:
                new_k = new_k.replace("norm13", "norm1")
            if "blocks.13.mlp.fc13" in k:
                new_k = new_k.replace("fc13", "fc1")
            if "blocks.14.norm14" in k:
                new_k = new_k.replace("norm14", "norm2")
            if "blocks.14.mlp.fc14" in k:
                new_k = new_k.replace("fc14", "fc2")
            # remove prefix
            if 'head' not in new_k and new_k.split('.')[0] != 'fc':
                state_dict[new_k] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    if 'pos_embed' in state_dict.keys():
        state_dict['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
            state_dict['pos_embed'], [16, 16],
        )
    return state_dict


class VisionEncoder(ABC):
    """Base class for all vision encoders"""
    
    def __init__(self, encoder_type: str, architecture: str, model_config: str, 
                 device: torch.device, resolution: int = 256, accelerator=None):
        self.encoder_type = encoder_type
        self.architecture = architecture
        self.model_config = model_config
        self.device = device
        self.resolution = resolution
        self.accelerator = accelerator
        self._embed_dim = None
        self.model = None
        
    @abstractmethod
    def load_model(self):
        """Load and initialize the encoder model"""
        pass
        
    @abstractmethod
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess raw images
        Args:
            x: Raw images tensor (B, C, H, W) in range [0, 255]
        Returns:
            Preprocessed tensor ready for encoder
        """
        pass
        
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """
        Forward pass through encoder
        Args:
            x: Preprocessed images
        Returns:
            Dictionary with:
                - 'x_norm_clstoken': (B, D) CLS token or None if not available
                - 'x_norm_patchtokens': (B, T, D) patch tokens
        """
        # Default implementation - subclasses should override if needed
        out = self.model.forward_features(x)
        if isinstance(out, dict):
            return out
        else:
            # Assume it's just patch tokens
            return {
                'x_norm_clstoken': None,
                'x_norm_patchtokens': out
            }
    
    @property
    def embed_dim(self) -> int:
        return self._embed_dim
    
    def eval(self):
        """Set model to eval mode"""
        if self.model is not None:
            self.model.eval()
        return self
    
    def to(self, device):
        """Move model to device"""
        if self.model is not None:
            self.model = self.model.to(device)
        self.device = device
        return self


class DINOEncoder(VisionEncoder):
    """DINO encoder implementation"""
    
    def load_model(self):
        import timm
        
        # Load model from torch hub
        model_name = f'dino_vit{self.model_config}16'
        
        if self.accelerator is not None:
            with self.accelerator.main_process_first():
                self.model = torch.hub.load('facebookresearch/dino:main', model_name)
        else:
            self.model = torch.hub.load('facebookresearch/dino:main', model_name)
        
        # Remove head
        del self.model.head
        self.model.head = torch.nn.Identity()
        
        # Resample position embeddings if needed
        patch_resolution = 16 * (self.resolution // 256)
        self.model.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            self.model.pos_embed.data, [patch_resolution, patch_resolution],
        )
        
        # Set embed dim
        self._embed_dim = self.model.embed_dim
        
        # Move to device and set to eval
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize to [0, 1]
        x = x / 255.
        # Apply ImageNet normalization
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        # Interpolate if needed
        x = torch.nn.functional.interpolate(x, self.resolution, mode='bicubic')
        return x
    
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        # DINO returns a dictionary with cls and patch tokens
        out = self.model.get_intermediate_layers(x)[0]
        return {
            'x_norm_clstoken': out[:, 0],
            'x_norm_patchtokens': out[:, 1:]
        }


class DINOv2Encoder(VisionEncoder):
    """DINOv2 encoder implementation"""
    
    def load_model(self):
        import timm
        
        # Determine if using register tokens
        use_reg = 'reg' in self.encoder_type
        
        # Load model from torch hub
        model_name = f'dinov2_vit{self.model_config}14{"_reg" if use_reg else ""}'
        
        if self.accelerator is not None:
            with self.accelerator.main_process_first():
                self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        else:
            self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        
        # Remove head
        del self.model.head
        self.model.head = torch.nn.Identity()
        
        # Resample position embeddings if needed
        patch_resolution = 16 * (self.resolution // 256)
        self.model.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            self.model.pos_embed.data, [patch_resolution, patch_resolution],
        )
        
        # Set embed dim
        self._embed_dim = self.model.embed_dim
        
        # Move to device and set to eval
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize to [0, 1]
        x = x / 255.
        # Apply ImageNet normalization
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        # Interpolate if needed
        x = torch.nn.functional.interpolate(x, 224 * (self.resolution // 256), mode='bicubic')
        return x
    
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        # DINOv2 returns a dictionary with cls and patch tokens
        out = self.model.forward_features(x)
        return {
            'x_norm_clstoken': out.get('x_norm_clstoken'),
            'x_norm_patchtokens': out.get('x_norm_patchtokens')
        }


class DINOv2MixedEncoder(DINOv2Encoder):
    """DINOv2 encoder with mixed CLS and patch tokens"""
    
    def __init__(self, *args, alpha: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
    
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        out = self.model.forward_features(x)
        cls_token = out['x_norm_clstoken']
        patch_tokens = out['x_norm_patchtokens']
        
        # Mix CLS token into patch tokens
        mixed_patch_tokens = cls_token[:, None, :] * self.alpha + patch_tokens * (1 - self.alpha)
        
        return {
            'x_norm_clstoken': cls_token,
            'x_norm_patchtokens': mixed_patch_tokens
        }


class DINOv3Encoder(VisionEncoder):
    """DINOv3 encoder implementation"""
    
    def load_model(self):
        from models.dinov3_loader import load_dinov3
        
        self.model = load_dinov3(f"dinov3_vit{self.model_config}")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Set embed dim
        self._embed_dim = self.model.embed_dim
        
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        from models.dinov3_loader import make_dinov3_transform
        transform_func = make_dinov3_transform(resize_size=self.resolution)
        return transform_func(x)
    
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        out = self.model.forward_features(x)
        return {
            'x_norm_clstoken': out.get('x_norm_clstoken'),
            'x_norm_patchtokens': out.get('x_norm_patchtokens')
        }


class SigLIPEncoder(VisionEncoder):
    """SigLIP encoder implementation"""
    
    def load_model(self):
        from transformers import SiglipVisionModel

        assert self.resolution == 256, "SigLIP2 only supports 256 resolution"

        # Map model config to full model name
        model_map = {
            'b': 'google/siglip-base-patch16-256',
            'l': 'google/siglip-large-patch16-256',
            'so400m': 'google/siglip-so400m-patch14-224',
        }
        
        if self.model_config not in model_map:
            raise ValueError(f"Unknown SigLIP model config: {self.model_config}")

        self.patch_size = 14 if self.model_config == "so400m" else 16

        self.model = SiglipVisionModel.from_pretrained(model_map[self.model_config])
        self.model.to(self.device)
        self.model.eval()
        
        self._embed_dim = self.model.config.hidden_size
        
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize to [0, 1]
        x = x / 255.
        # Apply ImageNet normalization
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, self.patch_size * (self.resolution // 16), mode='bicubic')
        return x
    
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        out = self.model(x).last_hidden_state
        return {
            'x_norm_clstoken': None,  # SigLIP has no CLS token
            'x_norm_patchtokens': out
        }


class SigLIP2Encoder(VisionEncoder):
    """SigLIP2 encoder implementation"""
    
    def load_model(self):
        from transformers import SiglipVisionModel
        
        assert self.resolution == 256, "SigLIP2 only supports 256 resolution"
        
        # Map model config to full model name
        model_map = {
            'b': 'google/siglip2-base-patch16-256',
            'l': 'google/siglip2-large-patch16-256',
            'so400m': 'google/siglip2-so400m-patch16-256',
            'g': 'google/siglip2-giant-opt-patch16-256'
        }
        
        if self.model_config not in model_map:
            raise ValueError(f"Unknown SigLIP2 model config: {self.model_config}")

        self.model = SiglipVisionModel.from_pretrained(model_map[self.model_config])
        self.model.to(self.device)
        self.model.eval()
        
        self._embed_dim = self.model.config.hidden_size
        
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize to [0, 1]
        x = x / 255.
        # Apply ImageNet normalization
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, self.resolution, mode='bicubic')
        return x
    
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        out = self.model(x).last_hidden_state
        return {
            'x_norm_clstoken': None,  # SigLIP has no CLS token
            'x_norm_patchtokens': out
        }


class CLIPEncoder(VisionEncoder):
    """CLIP encoder implementation"""
    
    def load_model(self):
        import clip
        from models.clip_vit import UpdatedVisionTransformer
        import timm
        
        encoder_ = clip.load(f"ViT-{self.model_config}/14", device='cpu')[0].visual
        self.model = UpdatedVisionTransformer(encoder_).to(self.device)

        patch_resolution = 16 * (self.resolution // 256)
        self.model.model.positional_embedding.data = timm.layers.pos_embed.resample_abs_pos_embed(
            self.model.model.positional_embedding.data.unsqueeze(0), [patch_resolution, patch_resolution],
        ).squeeze(0)
        self._embed_dim = self.model.model.transformer.width
        self.model.eval()
        
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize to [0, 1]
        x = x / 255.
        # Interpolate for CLIP
        resolution = x.shape[-1]
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        # Apply CLIP normalization
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
        return x
    
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        # CLIP returns patch tokens directly from custom forward
        out = self.model.forward(x)
        cls_token = out[:, 0]
        patch_tokens = out[:, 1:]
        return {
            'x_norm_clstoken': cls_token,
            'x_norm_patchtokens': patch_tokens
        }


class MoCoV3Encoder(VisionEncoder):
    """MoCoV3 encoder implementation"""
    
    def load_model(self):
        from models import mocov3_vit
        import timm
        
        # Create model based on config
        if self.model_config == 's':
            self.model = mocov3_vit.vit_small()
        elif self.model_config == 'b':
            self.model = mocov3_vit.vit_base()
        elif self.model_config == 'l':
            self.model = mocov3_vit.vit_large()
        else:
            raise ValueError(f"Unknown MoCoV3 model config: {self.model_config}")
        
        # Load checkpoint
        ckpt = torch.load(PRETRAINED_DIR / f'mocov3_vit{self.model_config}.pth', weights_only=False)
        state_dict = fix_mocov3_state_dict(ckpt['state_dict'])
        del self.model.head
        self.model.load_state_dict(state_dict, strict=True)
        self.model.head = torch.nn.Identity()
        
        self.model = self.model.to(self.device)
        self.model.eval()

        patch_resolution = 16 * (self.resolution // 256)
        self.model.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            self.model.pos_embed.data, [patch_resolution, patch_resolution],
        )

        # Set embed dim
        self._embed_dim = self.model.embed_dim
        
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        return x
    
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        # MoCoV3 returns CLS + patch tokens, we skip CLS (index 0)
        out = self.model.forward_features(x)  # By default MoCov3 uses CLS token
        cls_token = out[:, 0]
        patch_tokens = out[:, 1:]
        return {
            'x_norm_clstoken': cls_token,
            'x_norm_patchtokens': patch_tokens,
        }


class SAMEncoder(VisionEncoder):
    def load_model(self):
        from transformers import SamModel

        if self.model_config == "b":
            model_name = "facebook/sam-vit-base"
        elif self.model_config == "l":
            model_name = "facebook/sam-vit-large"
        elif self.model_config == "h":
            model_name = "facebook/sam-vit-huge"
        else:
            raise NotImplementedError(f"model size {self.model_config} not supported")

        sam_model = SamModel.from_pretrained(model_name).to(self.device).eval()
        self.model = sam_model.vision_encoder
        self._embed_dim = self.model.config.output_channels

    def preprocess(self, x):
        # SAM only takes 1024 input
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 1024, mode='bicubic')
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        return x

    def forward_features(self, x):
        out = self.model(x)
        hidden_states = out.last_hidden_state
        if hidden_states.shape[-1] != self.resolution // 16:
            hidden_states = torch.nn.functional.interpolate(
                hidden_states, 
                size=(self.resolution // 16, self.resolution // 16), 
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
        return {
        'x_norm_clstoken': None,
        'x_norm_patchtokens': hidden_states.view(hidden_states.shape[0], -1, hidden_states.shape[-1]),
    }


class SAM2Encoder(VisionEncoder):
    def load_model(self):
        from transformers import Sam2Model

        if self.model_config == "s":
            model_name = "facebook/sam2-hiera-small"
        elif self.model_config == "b":
            model_name = "facebook/sam2-hiera-base-plus"
        elif self.model_config == "l":
            model_name = "facebook/sam2-hiera-large"
        else:
            raise NotImplementedError(f"model size {self.model_config} not supported")

        sam_model = Sam2Model.from_pretrained(model_name).to(self.device).eval()
        self.model = sam_model.vision_encoder
        self._embed_dim = self.model.config.backbone_config.embed_dim_per_stage[-1]

    def preprocess(self, x):
        # SAM2 has 32x downsample rate
        x = x / 255.
        x = torch.nn.functional.interpolate(x, self.resolution * 2, mode='bicubic')
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        return x

    def forward_features(self, x):
        out = self.model(x)
        hidden_states = out.last_hidden_state
        return {
        'x_norm_clstoken': None,
        'x_norm_patchtokens': hidden_states.view(hidden_states.shape[0], -1, hidden_states.shape[-1]),
    }


class SAM2LogitEncoder(VisionEncoder):
    @staticmethod
    def make_grids(grid_size, H=1024, W=1024):
        y_coords = torch.linspace(0, H-1, grid_size)
        x_coords = torch.linspace(0, W-1, grid_size)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # [grid_size*grid_size, 2]
        return grid_points

    def load_model(self):
        from transformers import Sam2Model, Sam2Processor

        model_config = self.model_config[0]
        grid_size = int(self.model_config[1:])

        if model_config == "s":
            model_name = "facebook/sam2-hiera-small"
        elif model_config == "b":
            model_name = "facebook/sam2-hiera-base-plus"
        elif model_config == "l":
            model_name = "facebook/sam2-hiera-large"
        else:
            raise NotImplementedError(f"model size {model_config} not supported")

        self.model = Sam2Model.from_pretrained(model_name).to(self.device).eval()
        self.processor = Sam2Processor.from_pretrained(model_name)
        self._embed_dim = grid_size * grid_size
        self.grid_points = self.make_grids(grid_size)
        self.target_resolution = self.resolution // 16

    def preprocess(self, x):
        # Preprocess is in the forward
        return x

    def forward_features(self, x):
        B = x.shape[0]
        num_points = self.grid_points.shape[0]
        max_batch = 64  # NOTE: Points per batch, should be smaller on 80G VRAM

        all_results = {i: [] for i in range(B)}  # Store results per image

        # Process points in batches
        for point_idx in range(0, num_points, max_batch):
            end_idx = min(point_idx + max_batch, num_points)
            batch_points = self.grid_points[point_idx:end_idx]
            current_batch = end_idx - point_idx
            
            # Format for batch processing: each image gets same points as separate objects
            input_points = []
            input_labels = []
            
            for img_idx in range(B):
                # Each point is a separate object with 1 point
                # Format: [[[x1, y1]], [[x2, y2]], ...] for image
                points_for_image = []
                labels_for_image = []
                
                for i in range(current_batch):
                    points_for_image.append([[batch_points[i, 0].item(), batch_points[i, 1].item()]])
                    labels_for_image.append([1])
                
                input_points.append(points_for_image)
                input_labels.append(labels_for_image)
            
            with torch.no_grad():
                inputs = self.processor(
                    images=[x[i] for i in range(B)],
                    input_points=input_points,
                    input_labels=input_labels,
                    return_tensors="pt"
                ).to("cuda")
                
                outputs = self.model(**inputs, multimask_output=False)
                # Shape: [B, current_batch, 1, H, W] - one mask per point per image
                mask_logits = outputs.pred_masks.squeeze(2)  # Remove the channel dimension -> [B, current_batch, H, W]
            
            # Distribute masks to correct image
            for img_idx in range(B):
                # Get masks for this image from this batch
                image_masks = mask_logits[img_idx]  # [current_batch, H, W]
                all_results[img_idx].append(image_masks)

        # Combine results per image: [B, num_points, H, W]
        dense_features = torch.stack([
            torch.cat(all_results[i], dim=0) for i in range(B)
        ], dim=0)

        # Downsample if needed
        dense_features = torch.nn.functional.interpolate(
            dense_features, 
            size=(self.target_resolution, self.target_resolution), 
            mode='bilinear',
            align_corners=False
        )
        return {
            'x_norm_clstoken': None,
            'x_norm_patchtokens': dense_features.view(B, dense_features.shape[1], -1).permute(0, 2, 1),
        }

class MAEEncoder(VisionEncoder):
    """MAE encoder implementation"""
    
    def load_model(self):
        from models.mae_vit import vit_large_patch16
        import timm

        assert self.resolution == 256, "MAE only supports 256 resolution"

        kwargs = dict(img_size=256)
        self.model = vit_large_patch16(**kwargs).to(self.device)
        
        with open(PRETRAINED_DIR / f"mae_vit{self.model_config}.pth", "rb") as f:
            state_dict = torch.load(f, weights_only=False)
        
        if 'pos_embed' in state_dict["model"].keys():
            state_dict["model"]['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
                state_dict["model"]['pos_embed'], [16, 16],
            )
        
        self.model.load_state_dict(state_dict["model"])
        self.model.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            self.model.pos_embed.data, [16, 16],
        )
        
        self._embed_dim = self.model.embed_dim
        self.model.eval()
        
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        return x
    
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        out = self.model.forward_features(x)
        cls_token = out[:, 0]
        patch_tokens = out[:, 1:]
        return {
            'x_norm_clstoken': cls_token,
            'x_norm_patchtokens': patch_tokens
        }


class JEPAEncoder(VisionEncoder):
    """I-JEPA encoder implementation"""
    
    def load_model(self):
        from models.jepa import vit_huge
        import timm

        kwargs = dict(img_size=[224, 224], patch_size=14)
        self.model = vit_huge(**kwargs).to(self.device)
        
        with open(PRETRAINED_DIR / f"ijepa_vit{self.model_config}.pth", "rb") as f:
            state_dict = torch.load(f, map_location=self.device, weights_only=False)
        
        new_state_dict = dict()
        for key, value in state_dict['encoder'].items():
            new_state_dict[key[7:]] = value
        
        self.model.load_state_dict(new_state_dict)
        self.model.eval()

        patch_resolution = 16 * (self.resolution // 256)
        self.model.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            self.model.pos_embed.data, [patch_resolution, patch_resolution], num_prefix_tokens=0
        )

        self._embed_dim = self.model.embed_dim
        
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (self.resolution // 256), mode='bicubic')
        return x
    
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        out = self.model.forward(x)  # JEPA doesn't have CLS token
        return {
            'x_norm_clstoken': None,
            'x_norm_patchtokens': out
        }


class WebSSLEncoder(VisionEncoder):
    """WebSSL encoder implementation"""
    
    def load_model(self):
        from transformers import Dinov2Model, AutoImageProcessor
        
        model_name = f"facebook/webssl-{self.model_config.replace('_', '-')}"
        self.model = Dinov2Model.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self._embed_dim = self.model.config.hidden_size
        
        # Also load processor for preprocessing
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (self.resolution // 256), mode='bicubic')
        return x
    
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        # Skip CLS token (index 0)
        out = self.model.forward(x).last_hidden_state
        cls_token = out[:, 0]
        patch_tokens = out[:, 1:]
        return {
            'x_norm_clstoken': cls_token,
            'x_norm_patchtokens': patch_tokens
        }


class PEEncoder(VisionEncoder):
    """PE (Perceptual Encoder) implementation"""
    
    def load_model(self):
        import models.pe as pe
        
        # Check if using normalization
        self.use_norm = self.model_config.endswith("norm")
        if self.use_norm:
            config_name = self.model_config[:-4]
        else:
            config_name = self.model_config
        
        # Map config to model name
        if self.encoder_type == "pe":
            config_map = {
                "t": "PE-Core-T16-384",
                "s": "PE-Core-S16-384",
                "b": "PE-Core-B16-224",
                "l": "PE-Core-L14-336",
                "g": "PE-Core-G14-448"
            }
        elif self.encoder_type == "spatialpe":
            config_map = {
                "b": "PE-Spatial-B16-512",
                "l": "PE-Spatial-L14-448",
                "g": "PE-Spatial-G14-448"
            }
        elif self.encoder_type == "langpe":
            config_map = {
                "l": "PE-Lang-L14-448",
                "g": "PE-Lang-G14-448"
            }
        else:
            raise ValueError(f"Unknown PE encoder type: {self.encoder_type}")
        
        if config_name not in config_map:
            raise ValueError(f"Unknown PE model config: {config_name}")
        
        self.model = pe.VisionTransformer.from_config(config_map[config_name], pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self._embed_dim = self.model.width
        
        # Get patch size for preprocessing
        if config_name in {"t", "s", "b", "tnorm", "snorm", "bnorm"}:
            self.patch_size = 16
        elif config_name in {"l", "g", "lnorm", "gnorm"}:
            self.patch_size = 14
        else:
            raise NotImplementedError()
        
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.
        x = torch.nn.functional.interpolate(
            x, self.patch_size * (self.resolution // 16), mode='bilinear'
        )
        x = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(x)
        return x
    
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        # PE returns patch tokens without CLS
        out = self.model.forward_features(x, norm=self.use_norm, strip_cls_token=False)
        if self.model.use_cls_token:
            cls_token = out[:, 0]
            patch_tokens = out[:, 1:]
        else:
            cls_token = None
            patch_tokens = out
        return {
            'x_norm_clstoken': cls_token,
            'x_norm_patchtokens': patch_tokens
        }


class CRadioEncoder(VisionEncoder):

    def load_model(self):
        # C-RADIOv3-g model (ViT-G/16)
        # C-RADIOv3-H model (ViT-H/16)
        # C-RADIOv3-L model (ViT-L/16)
        # C-RADIOv3-B model (ViT-B/16)
        model_version = f"c-radio_v3-{self.model_config}"
        self.patch_size = 16
        self.model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=model_version, progress=True, skip_validation=True)
        self.model.to(self.device).eval()

        self._embed_dim = self.model.embed_dim

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.
        x = torch.nn.functional.interpolate(
            x, self.patch_size * (self.resolution // 16), mode='bilinear', align_corners=False
        )
        return x

    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        _, spatial_features = self.model(x)
        # NOTE: Similar to SigLip2, summary token dimension doesn't match patch tokens, so return None
        return {
            'x_norm_clstoken': None,
            'x_norm_patchtokens': spatial_features,
        }


class SIFTOpenCVEncoder(VisionEncoder):
    """Dense SIFT on a 16px grid. Output: [B, T, 128] with T=(res/16)^2."""
    def load_model(self):
        import cv2  # lazy import

        # just verify SIFT exists; keep handle for potential params
        self._cv2 = cv2
        self._sift = cv2.SIFT_create()  # defaults work well
        self.grid_stride = 16           # -> 16x16 at 256, 32x32 at 512
        self._embed_dim = 128

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Keep raw RGB in [0,255]; just resize to target resolution.
        # (OpenCV SIFT needs uint8; we convert in forward_features.)
        x = torch.nn.functional.interpolate(
            x, self.resolution, mode='bilinear', align_corners=False
        )
        return x

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        # x is [B,3,H,W] in [0,255] (float). Convert to numpy uint8 BHWC.
        B, _, H, W = x.shape
        device = x.device
        imgs = x.clamp(0, 255).round().to(torch.uint8).permute(0,2,3,1).cpu().numpy()

        step = self.grid_stride
        ys = list(range(step//2, H, step))
        xs = list(range(step//2, W, step))
        Hgrid, Wgrid = len(ys), len(xs)
        assert Hgrid * Wgrid == (self.resolution // 16) ** 2, "Grid mismatch"

        feats_out = []
        for b in range(B):
            # OpenCV expects BGR; SIFT actually works on grayscale internally,
            # but we pass BGR to be consistent with cv2 APIs.
            img_bgr = self._cv2.cvtColor(imgs[b], self._cv2.COLOR_RGB2BGR)
            kps = [self._cv2.KeyPoint(float(x0), float(y0), step)
                   for y0 in ys for x0 in xs]
            _, desc = self._sift.compute(img_bgr, kps)     # [T,128]
            if desc is None:
                # extremely rare; fallback to zeros to keep shapes consistent
                desc = np.zeros((Hgrid*Wgrid, 128), dtype=np.float32)
            feats_out.append(torch.from_numpy(desc.astype('float32')))

        tokens = torch.stack(feats_out, dim=0).to(device)  # [B, T, 128]
        return {'x_norm_clstoken': None, 'x_norm_patchtokens': tokens}


class SIFTEncoder(VisionEncoder):
    """Dense SIFT on a 16px grid using Kornia. Output: [B, T, 128] with T=(res/16)^2."""

    def load_model(self):
        import torch
        import kornia as K
        import kornia.feature as KF

        self.torch = torch
        self.K = K
        self.KF = KF

        self._embed_dim = 128
        self.grid_stride = 16
        self.patch_size = 41  # SIFTDescriptor default patch
        # Descriptor (matches your params)
        self.sift = KF.SIFTDescriptor(
            patch_size=self.patch_size,
            num_ang_bins=8,
            num_spatial_bins=4,
            rootsift=False,
            clipval=0.2,
        ).to(self.device)
        # Compute dominant orientation like OpenCV (angle = -1)
        self.orienter = KF.LAFOrienter(self.patch_size // 2).to(self.device)

        self._setup_grid()

    def _setup_grid(self):
        """Precompute grid keypoints, scales, orientations."""
        step = self.grid_stride
        H = W = self.resolution

        # centers (x, y) exactly like your OpenCV version: start at step//2
        ys = self.torch.arange(step // 2, H, step, device=self.device, dtype=self.torch.float32)
        xs = self.torch.arange(step // 2, W, step, device=self.device, dtype=self.torch.float32)
        yy, xx = self.torch.meshgrid(ys, xs, indexing='ij')
        kpts = self.torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # [T, 2], (x,y)

        self.keypoints = kpts                               # [T, 2] on self.device
        self.num_keypoints = int(kpts.shape[0])

        expected = (self.resolution // 16) ** 2
        assert self.num_keypoints == expected, f"Grid mismatch: {self.num_keypoints} vs {expected}"

        # LAF expects scales [B, T, 1, 1] and orientations [B, T, 1].
        # Using 'step' as a reasonable support size; tune if you need tighter OpenCV matching.
        s = float(step)
        self.base_scales = self.torch.full((1, self.num_keypoints, 1, 1), s, device=self.device)
        self.base_orients = self.torch.zeros((1, self.num_keypoints, 1), device=self.device)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Match OpenCV encoder: resize to target resolution, keep [0,255] float range.
        return self.torch.nn.functional.interpolate(
            x, self.resolution, mode='bilinear', align_corners=False
        )

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        K, KF, torch = self.K, self.KF, self.torch
        B, _, H, W = x.shape
        device = x.device

        # Ensure grid tensors are on the same device as the batch
        if self.keypoints.device != device:
            self.keypoints = self.keypoints.to(device)
            self.base_scales = self.base_scales.to(device)
            self.base_orients = self.base_orients.to(device)
            self.sift = self.sift.to(device)
            self.orienter = self.orienter.to(device)

        # Kornia expects grayscale in [0,1]
        gray = K.color.rgb_to_grayscale(x.float() / 255.0)  # [B, 1, H, W]

        # Expand grid for the batch
        centers = self.keypoints.unsqueeze(0).expand(B, -1, -1)         # [B, T, 2]
        scales = self.base_scales.expand(B, -1, -1, -1)                 # [B, T, 1, 1]
        orients = self.base_orients.expand(B, -1, -1)                   # [B, T, 1]

        # Build LAFs
        lafs = KF.laf_from_center_scale_ori(centers, scales, orients)   # [B, T, 2, 3]

        # Assign dominant orientation (closer to OpenCV SIFT when angle=-1)
        lafs = self.orienter(lafs, gray)

        # Sample 41x41 patches at each LAF
        patches = KF.extract_patches_from_pyramid(gray, lafs, PS=self.patch_size)  # [B, T, 1, 41, 41]

        B, T = patches.shape[0], patches.shape[1]
        patches_bt = patches.reshape(B * T, 1, self.patch_size, self.patch_size).contiguous()  # [B*T, 1, 41, 41]

        # Describe with SIFT
        desc_bt = self.sift(patches_bt)                         # [B*T, 128]
        desc = desc_bt.view(B, T, 128)                          # [B, T, 128]
        desc = torch.nn.functional.normalize(desc, p=2, dim=-1) # optional

        return {'x_norm_clstoken': None, 'x_norm_patchtokens': desc}

class HOGVisionEncoder(VisionEncoder):
    """
    Dense HOG at full image resolution, then downsample to a target grid:
      - Compute HOG on grayscale at (resolution x resolution) with small cells (default 2x2)
      - cells_per_block defaults to 4x4 → per-token dim D = (4*4*orientations)
      - Choose D via model_config; orientations = D / (cb_r*cb_c)
      - Bilinear-resize the (HB,WB) HOG map to (grid,grid) and flatten → (B, T=grid^2, D)

    model_config examples:
      - "g16_d128"         → grid=16, D=128 (orientations=8 with cb=4x4)
      - "g16_d128_ppc2"    → same, explicitly pixels_per_cell=(2,2)
      - "g16_d128_ppc1"    → even denser base (HB≈253), still downsampled to 16x16
      - "g16_d128_cb4"     → explicit 4x4 cells per block (default)

    Output dict:
      - 'x_norm_clstoken': None
      - 'x_norm_patchtokens': Float tensor (B, grid*grid, D), L2-normalized per token
    """
    def load_model(self):
        try:
            from skimage.feature import hog as _hog
            self._hog = _hog
        except Exception as e:
            raise ImportError("scikit-image is required for HOGVisionEncoder (pip install scikit-image).") from e

        # Defaults
        self.grid = 16
        self.hog_dim = 128
        ppc = 2
        cb  = 4

        # Parse model_config
        mc = str(self.model_config)
        for part in mc.split('_'):
            if part.startswith('g'):
                self.grid = int(part[1:])
            elif part.startswith('d'):
                self.hog_dim = int(part[1:])
            elif part.startswith('ppc'):
                ppc = int(part[3:])
            elif part.startswith('cb'):
                cb = int(part[2:])

        self.pixels_per_cell = (ppc, ppc)              # small cells → high-res HOG
        self.cells_per_block = (cb, cb)

        div = self.cells_per_block[0] * self.cells_per_block[1]
        if self.hog_dim % div != 0:
            raise ValueError(f"D={self.hog_dim} must be divisible by cells_per_block={self.cells_per_block} product={div}.")
        self.orientations = self.hog_dim // div
        if self.orientations < 1:
            raise ValueError("orientations must be >=1; increase D or decrease cells_per_block.")

        self._embed_dim = self.hog_dim  # per-token channel dim

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Keep [0,255] range; just ensure (H,W) == (resolution,resolution)
        return torch.nn.functional.interpolate(
            x, size=(self.resolution, self.resolution), mode='bilinear', align_corners=False
        )

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        B, C, H, W = x.shape
        assert C == 3, "HOGVisionEncoder expects RGB input (B,3,H,W)."
        device = x.device

        # To numpy grayscale in [0,1]
        imgs = x.detach().cpu().clamp(0, 255).numpy()
        bhwc = np.transpose(imgs, (0, 2, 3, 1))
        gray = (0.2126 * bhwc[..., 0] + 0.7152 * bhwc[..., 1] + 0.0722 * bhwc[..., 2]) / 255.0
        gray = np.clip(gray, 0.0, 1.0).astype(np.float32)

        feats_b_list = []
        for b in range(B):
            feat = self._hog(
                gray[b],
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                transform_sqrt=False,
                feature_vector=False,
                block_norm='L2-Hys',
            )  # (HB, WB, cb_r, cb_c, orientations)
            HB, WB = feat.shape[:2]
            f_b = feat.reshape(HB, WB, -1).astype(np.float32)           # (HB,WB,D)
            feats_b_list.append(torch.from_numpy(f_b))

        feats_bhwd = torch.stack(feats_b_list, dim=0).to(device=device, dtype=torch.float32)  # (B,HB,WB,D)
        feats_bhwd = torch.nn.functional.normalize(feats_bhwd, p=2, dim=-1)                   # L2 per token

        # Downsample HOG map to grid x grid
        feats_bdHW = feats_bhwd.permute(0, 3, 1, 2).contiguous()                               # (B,D,HB,WB)
        feats_bdgg = torch.nn.functional.interpolate(
            feats_bdHW, size=(self.grid, self.grid), mode='bilinear', align_corners=False
        )                                                                                      # (B,D,g,g)
        feats_btg = feats_bdgg.permute(0, 2, 3, 1).contiguous().view(B, self.grid * self.grid, self.hog_dim)

        return {'x_norm_clstoken': None, 'x_norm_patchtokens': feats_btg}
        
class VGGEncoder(VisionEncoder):
    """VGG16 pool4 spatial tokens (stride ~16). Output: [B, T, 512] with T=(res/16)^2."""
    def load_model(self):
        import torch.nn as nn
        from torchvision import models

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)
        feats = vgg.features

        # pool4 ends at index 23 (0-based) in torchvision VGG16.features
        self.slice = nn.Sequential(*[feats[i] for i in range(24)])
        for p in self.slice.parameters():
            p.requires_grad = False
        self.slice.to(self.device).eval()

        self._embed_dim = 512
        self.target_stride = 16   # pool4 downsample ≈ 16×

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # 1) Force the network input to your desired resolution (e.g., 256 or 512)
        x = x / 255.
        x = torch.nn.functional.interpolate(
            x, self.resolution, mode='bilinear', align_corners=False
        )
        # 2) ImageNet normalization
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        return x

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor):
        # x: [B,3,res,res], normalized
        y = self.slice(x)                           # [B, 512, h, w] (≈ res/16)
        # 3) Interpolate feature map to EXACT grid size (res/16, res/16)
        g = self.resolution // self.target_stride   # 16 for res=256, 32 for res=512
        if (y.shape[-2], y.shape[-1]) != (g, g):
            y = torch.nn.functional.interpolate(y, size=(g, g), mode='bilinear', align_corners=False)

        # 4) Flatten to tokens
        y = y.permute(0, 2, 3, 1).contiguous()      # [B, g, g, 512]
        tokens = y.view(y.shape[0], g * g, y.shape[-1])  # [B, T, 512]
        return {'x_norm_clstoken': None, 'x_norm_patchtokens': tokens}


# Registry mapping encoder types to classes
ENCODER_REGISTRY = {
    'dino': DINOEncoder,
    # dinov2 and dinov3 encoders
    'dinov2': DINOv2Encoder,
    'dinov2reg': DINOv2Encoder,
    'dinov2mixed': DINOv2MixedEncoder,
    'dinov2mixedreg': DINOv2MixedEncoder,
    'dinov3': DINOv3Encoder,
    # older encoders
    'siglip': SigLIPEncoder,
    'siglip2': SigLIP2Encoder,
    'clip': CLIPEncoder,
    'mocov3': MoCoV3Encoder,
    'mae': MAEEncoder,
    'jepa': JEPAEncoder,
    # webssl encoder
    'webssl': WebSSLEncoder,
    # PE encoders
    'pe': PEEncoder,
    'spatialpe': PEEncoder,
    'langpe': PEEncoder,
    # C-Radio encoders
    "cradio": CRadioEncoder,
    # sam encoders
    "sam": SAMEncoder,
    "sam2": SAM2Encoder,
    "sam2logit": SAM2LogitEncoder,
    # simpler encoders
    "siftopencv": SIFTOpenCVEncoder,
    "sift": SIFTEncoder,
    "vgg": VGGEncoder,
    "hog": HOGVisionEncoder,
}


def create_encoder(encoder_string: str, device: torch.device, 
                   resolution: int = 256, accelerator=None) -> VisionEncoder:
    """
    Factory function to create encoder from string specification
    
    Args:
        encoder_string: Format "encoder_type-architecture-model_config"
        device: torch device
        resolution: Input image resolution
        accelerator: Optional accelerator for distributed training
    
    Returns:
        VisionEncoder instance
    """
    parts = encoder_string.split('-')
    if len(parts) != 3:
        raise ValueError(f"Invalid encoder string format: {encoder_string}. "
                        f"Expected format: encoder_type-architecture-model_config")
    
    encoder_type, architecture, model_config = parts
    
    if encoder_type not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder type: {encoder_type}. "
                        f"Available types: {list(ENCODER_REGISTRY.keys())}")
    
    encoder_class = ENCODER_REGISTRY[encoder_type]
    encoder = encoder_class(encoder_type, architecture, model_config, 
                            device, resolution, accelerator)
    encoder.load_model()
    
    return encoder


@torch.no_grad()
def load_encoders(enc_type: str, device: torch.device, resolution: int = 256, 
                  accelerator=None) -> List[VisionEncoder]:
    """
    Load multiple encoders from comma-separated string
    
    Args:
        enc_type: Comma-separated encoder specifications
        device: torch device
        resolution: Input image resolution
        use_cls_token: Whether to return CLS tokens (for compatibility)
        accelerator: Optional accelerator for distributed training
    
    Returns:
        List of VisionEncoder instances
    """
    # if resolution not in [256, 512]:
    #     raise ValueError(f"Resolution must be 256 or 512, got {resolution}")

    enc_names = enc_type.split(',')
    encoders = []
    
    for enc_name in enc_names:
        # Parse encoder specification
        parts = enc_name.split('-')
        if len(parts) != 3:
            raise ValueError(f"Invalid encoder format: {enc_name}")
        
        encoder = create_encoder(enc_name, device, resolution, accelerator)
        encoder.eval()
        encoders.append(encoder)
    
    return encoders
