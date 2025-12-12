import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json

import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.autoencoder import VAE_F8D4
from models.sit import SiT_models
from loss import SILoss
from vision_encoder import load_encoders

from dataset import HFImgLatentDataset, HFLatentDataset, ImageFolderLatentDataset
import wandb
import math
from torchvision.utils import make_grid
from utils import SpatialNormalization, ALL_SPNORM_METHODS
from models.sit import ALL_PROJECTION_LAYER_TYPES

logger = get_logger(__name__)

#################################################################################
#                                  Utils                                       #
#################################################################################
def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z - latents_bias) * latents_scale
    return z 


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):    
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8
    in_channels = 4
    if args.repa_loss:
        encoders = load_encoders(
            args.enc_type, device, args.resolution, accelerator=accelerator
        )
    else:
        encoders = []

    z_dims = [encoder.embed_dim for encoder in encoders]
    block_kwargs = {
        "fused_attn": args.fused_attn, 
        "qk_norm": args.qk_norm,
    }
    model = SiT_models[args.model](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=args.num_classes,
        use_cfg = (args.cfg_prob > 0),
        z_dims = z_dims,
        encoder_depth=args.encoder_depth,
        projection_layer_type=args.projection_layer_type,
        proj_kwargs_kernel_size=args.proj_kwargs_kernel_size,
        **block_kwargs
    )

    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    # Load the VAE weights correctly, load the BN stats
    vae = VAE_F8D4().to(device).eval()
    vae_state_dict = torch.load("../pretrained_models/sdvae-ft-mse-f8d4.pt", map_location=device, weights_only=False)
    vae.load_state_dict(vae_state_dict)

    latents_stats = torch.load("../pretrained_models/sdvae-ft-mse-f8d4-latents-stats.pt", map_location=device, weights_only=False)
    latents_scale = latents_stats['latents_scale'].to(device).view(1, -1, 1, 1)
    latents_bias = latents_stats['latents_bias'].to(device).view(1, -1, 1, 1)

    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type, 
        accelerator=accelerator,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        weighting=args.weighting,
        projection_loss_type=args.projection_loss_type,
        proj_coeff=args.proj_coeff,
    )
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )    
    
    # Setup data
    if args.repa_loss:
        try:
            # We can preprocess ImageNet 256/512 here, and directly load from disk
            train_dataset = HFImgLatentDataset("sdvae-ft-mse-f8d4", args.data_dir, split="train")
        except Exception as e:
            print(f"Error loading HFImgLatentDataset: {e}")
            print("Falling back to ImageFolderLatentDataset")
            train_dataset = ImageFolderLatentDataset("sdvae-ft-mse-f8d4", args.data_dir, resolution=args.resolution, split="train")
    else:
        train_dataset = HFLatentDataset("sdvae-ft-mse-f8d4", args.data_dir, split="train")
    print(train_dataset)

    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")
    
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    # resume:
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu',
            weights_only=False,
        )
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']

    if args.compile:
        # Allow larger cache size for DYNAMO compilation
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 512
        model = torch.compile(model, backend="inductor", mode="default")
        # encoders = [torch.compile(encoder, backend="inductor", mode="default") for encoder in encoders]

        @torch.compile(fullgraph=False)
        def optim_step_fn():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    else:
        def optim_step_fn():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="REPA-Baseline", 
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{args.exp_name}"}
            },
        )
        
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Labels to condition the model with:
    sample_batch_size = args.n_samples // accelerator.num_processes
    batch = next(iter(train_dataloader))
    if args.repa_loss:
        _, gt_xs, gt_labels = batch
    else:
        gt_xs, gt_labels = batch
    gt_xs = gt_xs[:sample_batch_size]
    gt_labels = gt_labels[:sample_batch_size]
    gt_xs = sample_posterior(
        gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias
    )
    ys = gt_labels.to(device)
    # Create sampling noise:
    n = ys.size(0)
    xT = torch.randn((n, in_channels, latent_size, latent_size), device=device)

    # No longer need separate processors - encoders handle their own preprocessing

    # define spatial normalization class object
    spnorm = SpatialNormalization(args.spnorm_method)
    
    for epoch in range(args.epochs):
        model.train()
        for batch in train_dataloader:
            if args.repa_loss:
                raw_image, x, y = batch
                raw_image = raw_image.to(device)
            else:
                x, y = batch
            x = x.squeeze(dim=1).to(device)
            y = y.to(device)

            labels = y
            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
                zs = []
                with accelerator.autocast():
                    for encoder in encoders:
                        # Preprocess the image using encoder's built-in method
                        raw_image_ = encoder.preprocess(raw_image)

                        # Encode the features
                        # outputs dictionary with keys: 'x_norm_patchtokens' and 'x_norm_clstoken'
                        features = encoder.forward_features(raw_image_) 

                        # normalize spatial features
                        spnorm_kwargs = {
                            'feat': features['x_norm_patchtokens'],
                            'cls': features['x_norm_clstoken'],
                            'cls_weight': args.cls_token_weight,
                            'zscore_alpha': args.zscore_alpha,
                            'zscore_proj_skip_std': args.zscore_proj_skip_std,
                        }
                        z = spnorm(**spnorm_kwargs)

                        # append to list
                        zs.append(z)

            with accelerator.accumulate(model):
                model_kwargs = dict(y=labels)
                loss, proj_loss, loss_dict = loss_fn(model, x, model_kwargs, zs=zs)
                loss_mean = loss.mean()
                proj_loss_mean = proj_loss.mean()
                loss = loss_mean + proj_loss_mean
                    
                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optim_step_fn()

                if accelerator.sync_gradients and global_step % args.ema_update_freq == 0:
                    unwrapped_model = accelerator.unwrap_model(model)
                    update_ema(
                        ema,
                        unwrapped_model._orig_mod if args.compile else unwrapped_model,
                        decay=args.ema_decay
                    )
            
            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1                
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    original_model = unwrapped_model._orig_mod if args.compile else unwrapped_model

                    checkpoint = {
                        "model": original_model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                from samplers import euler_sampler
                with torch.no_grad():
                    samples = euler_sampler(
                        model, 
                        xT, 
                        ys,
                        num_steps=50, 
                        cfg_scale=4.0,
                        guidance_low=0.,
                        guidance_high=1.,
                        path_type=args.path_type,
                        heun=False,
                    ).to(torch.float32)
                    samples = vae.decode(samples / latents_scale + latents_bias).sample
                    gt_samples = vae.decode(gt_xs / latents_scale + latents_bias).sample
                    samples = (samples + 1) / 2.
                    gt_samples = (gt_samples + 1) / 2.
                out_samples = accelerator.gather(samples.to(torch.float32))
                gt_samples = accelerator.gather(gt_samples.to(torch.float32))
                accelerator.log({"samples": wandb.Image(array2grid(out_samples)),
                                 "gt_samples": wandb.Image(array2grid(gt_samples))})
                logging.info("Generating EMA samples done.")

            # Include the loss and grad norms in the logging
            logs = {
                "denoising_loss": accelerator.gather(loss_mean).mean().detach().item(),
                "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item(),
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item()
            }
            logs.update(loss_dict)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=50000)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=256)

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)
    # add arg for type of projection layer: allowed mlp | linear | conv
    parser.add_argument("--projection-layer-type", type=str, default="mlp", choices=ALL_PROJECTION_LAYER_TYPES)
    parser.add_argument("--proj-kwargs-kernel-size", type=int, default=1, choices=[1, 3, 5, 7])

    # dataset
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--ema-update-freq", type=int, default=1)

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=12)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=str, default="0.5")
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--repa-loss", action=argparse.BooleanOptionalAction, default=False)
    # add loss type
    parser.add_argument("--projection-loss-type", type=str, default="cosine", help="Should be a comma-separated list of projection loss types")

    # whether to normalize spatial features
    parser.add_argument("--spnorm-method", type=str, default="none", choices=ALL_SPNORM_METHODS)
    parser.add_argument("--cls-token-weight", type=float, default=0.2)
    parser.add_argument("--zscore-alpha", type=float, default=0.6)
    parser.add_argument("--zscore-proj-skip-std", action=argparse.BooleanOptionalAction, default=False)

    # config file (YAML)
    parser.add_argument("--config", type=str, default=None,
        help="Path to YAML config file (e.g., configs/irepa.yaml)")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # Load config file if provided (config values are overridden by CLI args)
    if args.config:
        from omegaconf import OmegaConf
        config = OmegaConf.load(args.config)
        for key, value in config.items():
            key_underscore = key.replace('-', '_')
            if hasattr(args, key_underscore):
                setattr(args, key_underscore, value)

    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
