import math
import sys
import os
import shutil

import torch
import torch.distributed as dist
import numpy as np
import cv2
import wandb

import util.misc as misc
import util.lr_sched as lr_sched
import torch_fidelity
import copy
from torchvision.utils import make_grid


def train_one_epoch(
    model, model_without_ddp, data_loader, optimizer, device, epoch, repa_kwargs,
    log_writer=None, args=None
):
    # REPA kwargs
    encoders = repa_kwargs.get('encoders', [])
    spnorm = repa_kwargs.get('spnorm', None)
    cls_token_weight = repa_kwargs.get('cls_token_weight', 0.2)
    zscore_alpha = repa_kwargs.get('zscore_alpha', 0.6)
    zscore_proj_skip_std = repa_kwargs.get('zscore_proj_skip_std', False)

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # On-the-fly mode: (x, labels)
        x, labels = batch
        # Prepare zs for projection loss
        with torch.no_grad():
            zs = []
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                for encoder in encoders:
                    # Preprocess the image using encoder's built-in method
                    x_ = x.to(device, non_blocking=True)
                    raw_image_ = encoder.preprocess(x_)

                    # Encode the features
                    # outputs dictionary with keys: 'x_norm_patchtokens' and 'x_norm_clstoken'
                    features = encoder.forward_features(raw_image_)

                    # normalize spatial features
                    spnorm_kwargs = {
                        'feat': features['x_norm_patchtokens'],
                        'cls': features['x_norm_clstoken'],
                        'cls_weight': cls_token_weight,
                        'zscore_alpha': zscore_alpha,
                        'zscore_proj_skip_std': zscore_proj_skip_std,
                    }
                    z = spnorm(**spnorm_kwargs)

                    # append to list
                    zs.append(z)

        # normalize image to [-1, 1]
        x = x.to(device, non_blocking=True).to(torch.float32).div_(255)
        x = x * 2.0 - 1.0
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss, loss_dict = model(x, labels, zs)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # loss_value_reduce = misc.all_reduce_mean(loss_value)
        reduced_loss_dict = {}
        for key, value in loss_dict.items():
            reduced_loss_dict[key] = misc.all_reduce_mean(value)  # called on all ranks

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                # log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                for key, value in reduced_loss_dict.items():
                    log_writer.add_scalar(f'train_loss/{key}', value, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)

        # Log to wandb if available
        if misc.is_main_process() and wandb.run is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                wandb_logs = {'train/lr': lr}
                for key, value in reduced_loss_dict.items():
                    wandb_logs[f'train/{key}'] = value
                wandb.log(wandb_logs, step=epoch_1000x)


def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None):

    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    num_steps = args.num_images // (batch_size * world_size) + 1

    # Construct the folder name for saving generated images.
    save_folder = os.path.join(
        args.tmp_gen_path,
        args.output_dir,
        "{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], args.num_images, args.img_size
        )
    )
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # switch to ema params, hard-coded to be the first one
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)

    # ensure that the number of images per class is equal.
    class_num = args.class_num
    assert args.num_images % class_num == 0, "Number of images per class must be the same"
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        start_idx = world_size * batch_size * i + local_rank * batch_size
        end_idx = start_idx + batch_size
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen = torch.Tensor(labels_gen).long().cuda()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sampled_images = model_without_ddp.generate(labels_gen)

        torch.distributed.barrier()

        # denormalize images
        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()

        # distributed save images
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

    torch.distributed.barrier()

    # back to no ema
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)

    # compute FID and IS
    if log_writer is not None:
        if args.img_size == 256:
            fid_statistics_file = 'fid_stats/jit_in256_stats.npz'
        elif args.img_size == 512:
            fid_statistics_file = 'fid_stats/jit_in512_stats.npz'
        else:
            raise NotImplementedError
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix = "_cfg{}_res{}".format(model_without_ddp.cfg_scale, args.img_size)
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))

        # Log to wandb if available
        if misc.is_main_process() and wandb.run is not None:
            wandb.log({
                f'eval/fid{postfix}': fid,
                f'eval/is{postfix}': inception_score
            }, step=epoch*1000)

        shutil.rmtree(save_folder)

    torch.distributed.barrier()


def array2grid(x):
    # tensorboard requires [C, H, W] tensor in range [0, 1]
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    return x


def gather_tensor_standard(local_tensor):
    # 1. Create a list to hold the tensors from all GPUs
    #    The list size must equal the world_size (number of GPUs)
    world_size = misc.get_world_size()
    tensor_list = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    
    # 2. Gather tensors from all GPUs into the list
    #    (This is a synchronous operation)
    dist.all_gather(tensor_list, local_tensor)
    
    # 3. Concatenate the list along the 0-th dimension
    gathered_tensor = torch.cat(tensor_list, dim=0)
    
    return gathered_tensor


def generate_grid(model_without_ddp, gt_xs, gt_ys, epoch, log_writer=None):
    model_without_ddp.eval()
    
    # switch to ema params, hard-coded to be the first one
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        sampled_images = model_without_ddp.generate(gt_ys)
        sampled_images = (sampled_images + 1) / 2  # [0, 1]

    torch.distributed.barrier()

    gt_xs_ = gather_tensor_standard(gt_xs).cpu()
    sampled_images_ = gather_tensor_standard(sampled_images).cpu()

    # back to no ema
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)

    if log_writer is not None:
        gt_grid = array2grid(gt_xs_)
        log_writer.add_image('gt_grid', gt_grid, epoch)
        sampled_grid = array2grid(sampled_images_)
        log_writer.add_image('sampled_grid', sampled_grid, epoch)

        # Log to wandb if available
        if misc.is_main_process() and wandb.run is not None:
            # Convert tensors to numpy arrays in [H, W, C] format for wandb
            gt_grid_np = gt_grid.permute(1, 2, 0).numpy()
            sampled_grid_np = sampled_grid.permute(1, 2, 0).numpy()
            wandb.log({
                'images/gt_grid': wandb.Image(gt_grid_np),
                'images/sampled_grid': wandb.Image(sampled_grid_np)
            }, step=epoch*1000)

    torch.distributed.barrier()
