import torch
import torch.nn as nn
from model_jit import JiT_models
import projection_loss as pl


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            encoder_depth=args.encoder_depth,
            z_dims=args.z_dims,
            projector_dim=args.projector_dim,
            projection_layer_type=args.projection_layer_type,
            proj_kwargs_kernel_size=args.proj_kwargs_kernel_size,
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

        # projection loss
        self.projection_loss_type = [elem.strip() for elem in args.projection_loss_type.split(",") if elem.strip()]
        self.proj_coeff = [float(elem.strip()) for elem in args.proj_coeff.split(",") if elem.strip()]
        assert len(self.projection_loss_type) == len(self.proj_coeff), \
            f"len(self.projection_loss_type) - {len(self.projection_loss_type)} != len(self.proj_coeff) - {len(self.proj_coeff)}"

        # create projection loss
        self.projection_loss_kwargs = {}
        self.projection_loss = [
            pl.make_projection_loss(projection_loss_type, **self.projection_loss_kwargs)
            for projection_loss_type in self.projection_loss_type
        ]
        assert len(self.projection_loss) == len(self.proj_coeff), \
            f"len(self.projection_loss) - {len(self.projection_loss)} != len(self.proj_coeff) - {len(self.proj_coeff)}"

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels, zs):
        labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred, zs_tilde, zs_tilde_original = self.net(z, t.flatten(), labels_dropped)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # l2 loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        # Projection loss
        total_proj_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        loss_dict = {}
        if zs and zs_tilde and zs_tilde_original:
            assert len(zs) == len(zs_tilde) == len(zs_tilde_original), \
                f"Shape mismatch: {len(zs)=} vs {len(zs_tilde)=} vs {len(zs_tilde_original)=}"

            # loop across different projection losses [e.g. cosine, nt-xent, p2p-gram-cossim]
            for proj_loss_name, proj_loss_fn, coeff in zip(self.projection_loss_type, self.projection_loss, self.proj_coeff):
                proj_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
                if len(zs) > 0 and len(zs_tilde) > 0:
                    # loop across different encoders
                    for z, z_tilde, z_tilde_original in zip(zs, zs_tilde, zs_tilde_original):
                        # zs_tilde_original will be only used for gram-matrix loss, so its shape doesn't matter
                        assert z.shape == z_tilde.shape, f"Shape mismatch: {z.shape=} vs {z_tilde.shape=}"
                        # NOTE: We pass vision_feats, projected_sit_feats, and unprojected_sit_feats, but the last one might not be used
                        proj_loss = proj_loss + proj_loss_fn(z, z_tilde, z_tilde_original)
                    proj_loss /= len(zs)
                loss_dict[proj_loss_name] = proj_loss.detach().item()
                loss_dict[f"{proj_loss_name}_weighted"] = proj_loss.detach().item() * coeff
                total_proj_loss = total_proj_loss + coeff * proj_loss

            loss_dict['denoise_loss'] = loss.detach().item()
            loss_dict['total_proj_loss'] = total_proj_loss.detach().item()

        loss = loss + total_proj_loss
        loss_dict['total_loss'] = loss.detach().item()

        return loss, loss_dict

    @torch.no_grad()
    def generate(self, labels):
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z

    # @torch.no_grad()
    # def _forward_sample(self, z, t, labels):
    #     # conditional
    #     x_cond, _, _ = self.net(z, t.flatten(), labels)
    #     v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

    #     # unconditional
    #     x_uncond, _, _ = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes))
    #     v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

    #     # cfg interval
    #     low, high = self.cfg_interval
    #     interval_mask = (t < high) & ((low == 0) | (t > low))
    #     cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

    #     return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        # 1. Determine if we actually need CFG
        # Check global scale
        is_guidance_active = self.cfg_scale != 1.0
        
        # Check interval (assuming t is uniform across batch, which is standard)
        low, high = self.cfg_interval
        if is_guidance_active:
            # If t is outside the interval, guidance is effectively turned off
            if (t[0] >= high) or (low != 0 and t[0] <= low):
                is_guidance_active = False

        # --- PATH A: Standard Execution (No CFG) ---
        # 50% faster than the original code when active
        if not is_guidance_active:
            x_cond, _, _ = self.net(z, t.flatten(), labels)
            return (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # --- PATH B: CFG Execution (Batched) ---
        # Double batch size, single forward pass
        z_in = torch.cat([z, z])
        t_in = torch.cat([t, t])
        
        # Construct labels: [actual_labels, null_labels]
        null_labels = torch.full_like(labels, self.num_classes)
        c_in = torch.cat([labels, null_labels])

        # Single forward pass
        x_out, _, _ = self.net(z_in, t_in.flatten(), c_in)
        x_cond, x_uncond = x_out.chunk(2)

        # Apply CFG directly to x space (Optimization: calculate velocity once)
        # Note: We know scale is self.cfg_scale here because we passed the interval check
        x_cfg = x_uncond + self.cfg_scale * (x_cond - x_uncond)

        return (x_cfg - z) / (1.0 - t).clamp_min(self.t_eps)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
