import argparse
import datetime
import numpy as np
import os
import sys
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import wandb
# import torchvision.datasets as datasets
from dataset import HFImageDataset

from util.crop import center_crop_arr
import util.misc as misc

import copy
from engine_jit import train_one_epoch, evaluate, generate_grid

from denoiser import Denoiser

from projectors import ALL_PROJECTION_LAYER_TYPES
from spnorm import ALL_SPNORM_METHODS
from vision_encoder import load_encoders
from spnorm import SpatialNormalization

def get_args_parser():
    parser = argparse.ArgumentParser('JiT', add_help=False)

    # architecture
    parser.add_argument('--model', default='JiT-B/16', type=str, metavar='MODEL',
                        help='Name of the model to train')
    parser.add_argument('--img_size', default=256, type=int, help='Image size')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')

    # training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * # GPUs)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='The second ema to track')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # sampling
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE samping method')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Sampling steps')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--interval_min', default=0.0, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float,
                        help='CFG interval max')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='Number of images to generate')
    parser.add_argument('--eval_freq', type=int, default=40,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--save_interval', type=int, default=20,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--sample_grid', action='store_true')
    parser.add_argument('--sample_grid_freq', type=int, default=20,
                        help='Frequency (in epochs) to save a 8x8 grid of images to tensorboard')
    parser.add_argument('--sample_grid_n_images', type=int, default=64,
                        help='Number of images to sample for the grid')
    parser.add_argument('--gen_bsz', type=int, default=256,
                        help='Generation batch size')
    parser.add_argument('--tmp_gen_path', default='ssd/tmp', type=str,
                        help='Path to save temporary generated images')
    parser.add_argument('--gen_checkpoint_path', default=None, type=str,
                        help='Path to the checkpoint to generate from')
    parser.add_argument('--gen_output_dir', default='samples', type=str,
                        help='Path to save generated images')

    # dataset
    parser.add_argument('--data_path', default='../data', type=str,
                        help='Path to the dataset')
    parser.add_argument('--class_num', default=1000, type=int)

    # checkpointing
    parser.add_argument('--output_dir', default='./exps',
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--exp_name', default=None, type=str,
                        help='Experiment name (used for wandb run name)')
    parser.add_argument('--resume', default='',
                        help='Folder that contains checkpoint to resume from')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--log_freq', default=100, type=int)

    # Auto-requeue for SLURM (h2gpu partition doesn't support --signal)
    parser.add_argument('--requeue_time', type=int, default=None,
                        help='Enable auto-requeue with watchdog timer (time limit in minutes). Set to match #SBATCH --time')
    parser.add_argument('--requeue_safety_margin', type=int, default=10,
                        help='Exit this many minutes before requeue_time to save checkpoint (default: 10)')

    # wandb logging
    parser.add_argument('--report_to', default='all', type=str,
                        choices=['tensorboard', 'wandb', 'all'],
                        help='Logging backend: tensorboard, wandb, or all. Use WANDB_ENTITY and WANDB_PROJECT env vars to configure wandb.')

    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')

    # add arg for type of projection layer: allowed mlp | linear | conv
    parser.add_argument("--enable_repa", action="store_true")
    parser.add_argument("--encoder_depth", type=int, default=8)
    parser.add_argument("--projector_dim", type=int, default=2048)
    parser.add_argument("--enc_type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj_coeff", type=str, default="0.5")
    parser.add_argument("--projection_layer_type", type=str, default="mlp", choices=ALL_PROJECTION_LAYER_TYPES)
    parser.add_argument("--proj_kwargs_kernel_size", type=int, default=3, choices=[1, 3, 5, 7])
    parser.add_argument("--projection_loss_type", type=str, default="cosine", help="Should be a comma-separated list of projection loss types")

    # whether to normalize spatial features
    parser.add_argument("--spnorm_method", type=str, default="none", choices=ALL_SPNORM_METHODS)
    parser.add_argument("--cls_token_weight", type=float, default=0.2)
    parser.add_argument("--zscore_alpha", type=float, default=0.8)
    parser.add_argument("--zscore_proj_skip_std", action=argparse.BooleanOptionalAction, default=False)

    # config file (YAML)
    parser.add_argument("--config", type=str, default=None,
        help="Path to YAML config file (e.g., configs/irepa.yaml)")

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32 = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # Set up logging (only on main process)
    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

        # Set up TensorBoard logging
        if args.report_to in ['tensorboard', 'all']:
            log_writer = SummaryWriter(log_dir=args.output_dir)
        else:
            log_writer = None

        # Set up WandB logging (uses WANDB_ENTITY and WANDB_PROJECT env vars)
        if args.report_to in ['wandb', 'all']:
            # Set run name from exp_name if provided and WANDB_NAME not set
            if 'WANDB_NAME' not in os.environ and args.exp_name is not None:
                os.environ['WANDB_NAME'] = args.exp_name

            # Create centralized wandb directory structure: wandb/{exp_name}/
            wandb_base_dir = "wandb"
            if args.exp_name is not None:
                wandb_dir = os.path.join(wandb_base_dir, args.exp_name)
            else:
                wandb_dir = os.path.join(wandb_base_dir, os.path.basename(args.output_dir))
            os.makedirs(wandb_dir, exist_ok=True)

            # Check if we're resuming an existing wandb run
            wandb_run_id_file = os.path.join(wandb_dir, 'wandb_run_id.txt')
            wandb_run_id = None
            if os.path.exists(wandb_run_id_file):
                with open(wandb_run_id_file, 'r') as f:
                    wandb_run_id = f.read().strip()
                print(f"Resuming wandb run: {wandb_run_id}")

            # Try online logging with graceful fallback to offline mode
            try:
                wandb.init(
                    config=vars(args),
                    dir=wandb_dir,
                    resume="allow",
                    id=wandb_run_id,  # None for new runs, existing ID for resumed runs
                    settings=wandb.Settings(
                        init_timeout=300,  # 5 min timeout for slow networks
                        start_method="thread"
                    )
                )
                print(f"WandB online logging initialized: {wandb.run.url}")
            except Exception as e:
                print(f"WandB online init failed: {e}")
                print("Falling back to offline mode...")
                os.environ['WANDB_MODE'] = 'offline'
                wandb.init(
                    config=vars(args),
                    dir=wandb_dir,
                    resume="allow",
                    id=wandb_run_id
                )
                print(f"WandB offline mode enabled.")
                print(f"To sync later: wandb sync {wandb_dir}")
                print(f"To view locally: cd {wandb_dir} && wandb offline")

            # Save run ID for future resumption
            if wandb_run_id is None:
                with open(wandb_run_id_file, 'w') as f:
                    f.write(wandb.run.id)
                print(f"Saved wandb run ID: {wandb.run.id}")
    else:
        log_writer = None

    try:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
        ])
        dataset_train = HFImageDataset(args.data_path, split="train", transform=transform_train)
    except Exception as e:
        # Data augmentation transforms
        print(f"Error loading HFImageDataset: {e}")
        print("Falling back to ImageFolder")
        transform_train = transforms.Compose([
            transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor()
        ])
        dataset_train = ImageFolder(os.path.join(args.data_path, "imagenet", "train"), transform=transform_train)

    print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train =", sampler_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    gt_xs, gt_ys = None, None
    if args.sample_grid:
        samples_per_gpu = args.sample_grid_n_images // misc.get_world_size()
        batch = next(iter(data_loader_train))
        gt_xs, gt_ys = batch
        gt_xs = gt_xs.to(device)[:samples_per_gpu] / 255.0
        gt_ys = gt_ys.to(device)[:samples_per_gpu].long()

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    # Prepare REPA kwargs
    if args.enable_repa:
        encoders = load_encoders(
            args.enc_type, device, args.img_size
        )
        spnorm = SpatialNormalization(args.spnorm_method)
        repa_kwargs = {
            "encoders": encoders,
            "spnorm": spnorm,
            "cls_token_weight": args.cls_token_weight,
            "zscore_alpha": args.zscore_alpha,
            "zscore_proj_skip_std": args.zscore_proj_skip_std,
        }
        # If the script is run for saving generation or evaluation, we don't load the projectors
        args.z_dims = [encoder.embed_dim for encoder in encoders] if not args.evaluate_gen else []
        print("Z dims:", args.z_dims)
    else:
        repa_kwargs = {
            "encoders": [],
            "spnorm": None,
            "cls_token_weight": 0.0,
            "zscore_alpha": 1.0,
            "zscore_proj_skip_std": False,
        }
        args.z_dims = []

    # Create denoiser
    model = Denoiser(args)

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    # Set up optimizer with weight decay adjustment for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # Resume from checkpoint if provided
    checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

        ema_state_dict1 = checkpoint['model_ema1']
        ema_state_dict2 = checkpoint['model_ema2']
        model_without_ddp.ema_params1 = [ema_state_dict1[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        model_without_ddp.ema_params2 = [ema_state_dict2[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        print("Resumed checkpoint from", args.resume)

        # Manually set the start epoch for generation or evaluation for correct saving, but we are not loading the optimizer state
        if args.evaluate_gen:
            args.start_epoch = checkpoint['epoch']

        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not args.evaluate_gen:
            optimizer.load_state_dict(checkpoint['optimizer'])
            # Resume from the next epoch after the one stored in the checkpoint.
            # Here, checkpoint['epoch'] is interpreted as "number of epochs finished".
            args.start_epoch = checkpoint['epoch']
            print("Loaded optimizer & scaler state!")
        del checkpoint
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        print("Training from scratch")

    # Evaluate generation
    if args.evaluate_gen:
        print("Evaluating checkpoint at {} epoch".format(args.start_epoch))
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            with torch.no_grad():
                evaluate(model_without_ddp, args, 0, batch_size=args.gen_bsz, log_writer=log_writer)
        return

    # Double check the REPA kwargs
    print("REPA kwargs:", repa_kwargs)

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, repa_kwargs, log_writer=log_writer, args=args)

        # Save final checkpoint periodically for easier resuming
        if (epoch + 1) % args.save_last_freq == 0 or (epoch + 1) == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=(epoch + 1),
                epoch_name="last"
            )

        # Every save_interval epochs, we save a persistent checkpoint
        if (epoch + 1) % args.save_interval == 0:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=(epoch + 1)
            )

        # Save a 8x8 grid of images to tensorboard
        if args.sample_grid and ((epoch + 1) % args.sample_grid_freq == 0 or (epoch + 1) == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                generate_grid(model_without_ddp, gt_xs, gt_ys, (epoch + 1), log_writer=log_writer)
            torch.cuda.empty_cache()

        # Perform online evaluation at specified intervals
        if args.online_eval and ((epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate(model_without_ddp, args, (epoch + 1), batch_size=args.gen_bsz, log_writer=log_writer)
            torch.cuda.empty_cache()

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)

    # Finish wandb run
    if global_rank == 0 and wandb.run is not None:
        wandb.finish()


def load_config(args):
    """Load YAML config file if provided."""
    if args.config:
        from omegaconf import OmegaConf
        config = OmegaConf.load(args.config)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    return args


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    args = load_config(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
