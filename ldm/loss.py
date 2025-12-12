import torch
import numpy as np
import torch.nn.functional as F
import projection_loss as pl

########################################################
# Loss for the denoising step
########################################################
def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))


class SILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            projection_loss_type="cosine",
            projection_loss_kwargs={},
            proj_coeff=[0.5],
        ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias

        # parse projection loss type and coeff
        self.projection_loss_type = [elem.strip() for elem in projection_loss_type.split(",") if elem.strip()]
        self.proj_coeff = [float(elem.strip()) for elem in proj_coeff.split(",") if elem.strip()]
        assert len(self.projection_loss_type) == len(self.proj_coeff), \
            f"len(self.projection_loss_type) - {len(self.projection_loss_type)} != len(self.proj_coeff) - {len(self.proj_coeff)}"
        self.projection_loss_kwargs = projection_loss_kwargs
        # create projection loss
        self.projection_loss = [
            pl.make_projection_loss(projection_loss_type, **projection_loss_kwargs)
            for projection_loss_type in self.projection_loss_type
        ]
        assert len(self.projection_loss) == len(self.proj_coeff), \
            f"len(self.projection_loss) - {len(self.projection_loss)} != len(self.proj_coeff) - {len(self.proj_coeff)}"
        

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images, model_kwargs=None, zs=FileNotFoundError):
        if model_kwargs is None:
            model_kwargs = {}

        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
                
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            # TODO: add x or eps prediction
            raise NotImplementedError()
        model_output, zs_tilde, zs_tilde_original = model(model_input, time_input.flatten(), **model_kwargs)
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        # projection loss
        total_proj_loss = 0.
        proj_loss_dict = {}
        # loop across different projection losses [e.g. cosine, nt-xent, p2p-gram-cossim]
        for proj_loss_name, proj_loss_fn, coeff in zip(self.projection_loss_type, self.projection_loss, self.proj_coeff):
            proj_loss = torch.tensor(0.0, device=images.device, dtype=images.dtype)
            if len(zs) > 0 and len(zs_tilde) > 0:
                # loop across different encoders
                for z, z_tilde, z_tilde_original in zip(zs, zs_tilde, zs_tilde_original):
                    # NOTE: We pass vision_feats, projected_sit_feats, and unprojected_sit_feats, but the last one might not be used
                    proj_loss = proj_loss + proj_loss_fn(z, z_tilde, z_tilde_original)
                proj_loss /= len(zs)
            proj_loss_dict[proj_loss_name] = proj_loss.detach().item()
            proj_loss_dict[f"{proj_loss_name}_weighted"] = proj_loss.detach().item() * coeff
            total_proj_loss = total_proj_loss + coeff * proj_loss
        return denoising_loss, total_proj_loss, proj_loss_dict
