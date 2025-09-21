import tqdm
import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

def match_dimensions(tensor, shape):
    """Return view of the input tensor to allow broadcasting with the shape."""
    tensor_shape = tensor.shape
    while len(tensor_shape) < len(shape):
        tensor_shape = tensor_shape + (1,) 
    return tensor.view(tensor_shape)

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

__DIFFUSION__ = {}

def register_diffusion(name):
    def wrapper(cls):   
        if __DIFFUSION__.get(name, None):
            if __DIFFUSION__[name] != cls:
                warnings.warn(f"Name {name} is already registered!", UserWarning)
        __DIFFUSION__[name] = cls
        cls.name = name
        return cls
    return wrapper

def get_diffusion(name: str, **kwargs):
    if __DIFFUSION__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __DIFFUSION__[name](**kwargs)

@register_diffusion("edm")
class DiffusionEDM(nn.Module):
    def __init__(self, estimator, sigma_min=0.001, sigma_max=100, rho=7):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

        self.estimator = estimator


    def sigma_schedule(self, num_steps, rho=7, sigma_min=0.002, sigma_max=80):
        if num_steps == 1:
            return np.array([sigma_max])
        step_indices = np.arange(num_steps)
        return (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    def sample_xt(self, x0, sigma):
        # x0: [B, C, H, W], sigma: [B]
        eps = torch.randn_like(x0, device=x0.device)
        sigma = match_dimensions(sigma, x0.shape)
        x_sigma = x0 + sigma * eps
        return x_sigma, eps


    def estimate_epsilon(self, x, sigma):
        """Estimate the noise at time t, given x.
        Uses the relationship between the score and the noise prediction.
        
        Args:
            x: (N, C, H, W) tensor
            sigma: (N,) tensor of current noise level
        """
        
        return self.estimator(x.to(torch.float32), sigma.to(torch.float32))
    
    def loss_sigma(self, x0, sigma):
        """Compute the loss at time t, given x0 and sigma.
        
        Args:
            x0: (N, C, H, W) tensor of original samples
            sigma: (N,) tensor of current noise level
        """
        x_sigma, eps = self.sample_xt(x0, sigma)
        estimate = self.estimate_epsilon(x_sigma, sigma)
        loss = F.mse_loss(estimate, eps)
        return loss
    
    def compute_loss(self, x0):
        """Compute the loss for a batch of samples.
        
        Args:
            x0: (N, C, H, W) tensor of original samples
        """
        batch_size = x0.shape[0]
        sigma = torch.rand(batch_size, device=x0.device) * (self.sigma_max - self.sigma_min) + self.sigma_min
        return self.loss_sigma(x0, sigma)
    
    def score_estimate(self, xt, sigma):
        """Estimate the score at time t, given x.
        Uses the relationship between the score and the noise prediction.
        
        Args:
            x: (N, C, H, W) tensor
            t: (N,) tensor of current time step
        """
        if not torch.is_tensor(sigma):
            sigma = torch.full((xt.shape[0],), sigma, device=xt.device)
        # c_noise = self.condition_index(sigma)
        
        c_noise = sigma
        c_in = 1
        # c_in = 1 / torch.sqrt(sigma**2 + 1)
        # c_in = match_dimensions(c_in, x_hat.shape)

        eps = self.estimate_epsilon(c_in*xt, c_noise)
        denom = match_dimensions(sigma, xt.shape)
        return - eps / denom


    def denoised_estimate(self, x_hat, sigma):
        """Returns denoised sample D_{\theta}(x, sigmal) as given in EDM paper.

        Args:
            x_hat: (N, C, H, W) tensor, sample before scaling by s(t)
            sigma: noise level float or tensor of shape (N,)

        """
        if not torch.is_tensor(sigma):
            sigma = torch.full((x_hat.shape[0],), sigma, device=x_hat.device)
        # c_noise = self.condition_index(sigma)
        
        c_out = -sigma
        c_noise = sigma

        c_in = 1
        # c_in = 1 / torch.sqrt(sigma**2 + 1)
        # c_in = match_dimensions(c_in, x_hat.shape)

        c_out = match_dimensions(c_out, x_hat.shape)
        eps = self.estimate_epsilon(c_in*x_hat, c_noise)
        return x_hat + c_out*eps
    
    def dx_estimate(self, x, sigma):
        """Returns the dx estimate D_{\theta}(x, sigmal) as given in EDM paper.
        This is used to compute the reverse diffusion step.

        Args:
            x_hat: (N, C, H, W) tensor, sample before scaling by s(t)
            sigma: noise level

        """
        # simplified expression for d_cur, for s(t) = 1, sigma(t) = t
        denoised_x = self.denoised_estimate(x, sigma)
        out = (x - denoised_x)/sigma
        return out
        
    
    def sigma_inv(self, sigma):
        """Returns the inverse of the function sigma(t)."""
        return sigma

    def get_sigma_t(self, t):
        """Returns the inverse of the function sigma(t)."""
        return t
    
    @torch.no_grad()
    def ddpm_ode_sample(
        self, n_evals=100, start_x=None, start_t=None, **kwargs
        ):

        use_euler = kwargs.get("use_euler", False)
        show_progress = kwargs.get("show_progress", False)

        shape = kwargs.get("shape", (1, 3, 256, 256))
        if start_t is not None:
            start_t = self.T
        offset = self.T - start_t
        num_steps = n_evals if use_euler else n_evals//2
        sigma_values = self.iddpm_sigma_schedule(num_steps, offset)
        sigma_values = np.concatenate((sigma_values, np.zeros_like(sigma_values[:1])))  # adding last sigma value of 0.

        if start_x is not None:
            x = start_x.to(self.device, dtype=self.dtype)
        else:
            x = sigma_values[0]*torch.randn(shape, dtype=self.dtype, device=self.device)

        for i, (sigma_current, sigma_next) in tqdm.tqdm(
                enumerate(zip(sigma_values[:-1], sigma_values[1:])),
                total=len(sigma_values)-1,
                desc="Generating...",
                disable=not show_progress
            ):
            
            # get values of time from sigma values
            t_current = self.sigma_inv(sigma_current)
            t_next = self.sigma_inv(sigma_next)

            d_cur = self.dx_estimate(x, sigma_current)

            # Euler step...
            x_prime = x + d_cur * (t_next - t_current)

            if use_euler or sigma_next == 0:
                x = x_prime
            else:
                d_prime = self.dx_estimate(x_prime, sigma_next)
                x = x + 0.5 * (d_cur + d_prime) * (t_next - t_current)
        x = x.contiguous()
        return x
    
    @torch.no_grad()
    def edm_ode_sample(
        self, n_evals=100, start_x=None, **kwargs):
        rho = kwargs.get("rho", 7)
        use_euler = kwargs.get("use_euler", False)
        show_progress = kwargs.get("show_progress", False)

        sigma_min=kwargs.get("sigma_min", 0.002)
        sigma_max=kwargs.get("sigma_max", 80)
        shape = kwargs.get("shape", (1, 3, 256, 256))

        num_steps = n_evals if use_euler else n_evals//2
        sigma_values = self.sigma_schedule(num_steps, rho=rho, sigma_min=sigma_min, sigma_max=sigma_max)
        sigma_values = np.concatenate((sigma_values, np.zeros_like(sigma_values[:1])))  # adding last sigma value of 0.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.estimator = self.estimator.to(device)
        if start_x is None:
            x = sigma_values[0]*torch.randn(shape, device=device)
        else:
            x = start_x.to(device)

        for i, (sigma_current, sigma_next) in tqdm.tqdm(
                enumerate(zip(sigma_values[:-1], sigma_values[1:])),
                total=len(sigma_values)-1,
                desc="Generating...",
                disable=not show_progress
            ):
            
            # get values of time from sigma values
            t_current = self.sigma_inv(sigma_current)
            t_next = self.sigma_inv(sigma_next)

            d_cur = self.dx_estimate(x, sigma_current)

            # Euler step...
            x_prime = x + d_cur * (t_next - t_current)

            if use_euler or sigma_next == 0:
                x = x_prime
            else:
                d_prime = self.dx_estimate(x_prime, sigma_next)
                x = x + 0.5 * (d_cur + d_prime) * (t_next - t_current)
        x = x.contiguous().to(torch.float32)
        return x
    

    @torch.no_grad()
    def ddpm_sde_sample(
        self, n_evals=100, shape=(1, 3, 256, 256), **kwargs):

        use_euler = kwargs.get("use_euler", False)
        show_progress = kwargs.get("show_progress", False)
        s_churn = kwargs.get("s_churn", 80)
        s_min = kwargs.get("s_min", 0.05)
        s_max = kwargs.get("s_max", 50)
        s_noise = kwargs.get("s_noise", 1.003)

        

        num_steps = n_evals if use_euler else n_evals//2
        sigma_values = self.iddpm_sigma_schedule(num_steps)

        sigma_values = np.concatenate((sigma_values, np.zeros_like(sigma_values[:1])))  # adding last sigma value of 0.

        x = sigma_values[0]*torch.randn(shape, dtype=self.dtype, device=self.device)

        for i, (sigma_current, sigma_next) in tqdm.tqdm(
                enumerate(zip(sigma_values[:-1], sigma_values[1:])),
                total=len(sigma_values)-1,
                desc="Generating...",
                disable=not show_progress
            ):
            
            # get values of time from sigma values
            t_current = self.sigma_inv(sigma_current)
            t_next = self.sigma_inv(sigma_next)

            epsilon = torch.randn_like(x) * s_noise

            if sigma_current > s_min and sigma_current < s_max:
                gamma = min(s_churn/num_steps, np.sqrt(2)-1)
                t_hat = t_current + gamma*t_current
                sigma_hat = self.get_sigma_t(t_hat)
                x_hat = x + epsilon*(sigma_hat**2 - sigma_current**2)**0.5
            else:
                gamma = 0.0
                t_hat = t_current
                sigma_hat = sigma_current
                x_hat = x 
        
            d_cur = self.dx_estimate(x_hat, sigma_hat)

            # Euler step...
            x_prime = x_hat + d_cur * (t_next - t_hat)

            if use_euler or sigma_next == 0:
                x = x_prime
            else:
                d_prime = self.dx_estimate(x_prime, sigma_next)
                x = x_hat + 0.5 * (d_cur + d_prime) * (t_next - t_hat)
        x = x.contiguous()
        return x

    @torch.no_grad()
    def edm_sde_sample(
        self, n_evals=100, start_x=None, **kwargs):

        rho = kwargs.get("rho", 7)
        use_euler = kwargs.get("use_euler", False)
        show_progress = kwargs.get("show_progress", False)
        s_churn = kwargs.get("s_churn", 80)
        s_min = kwargs.get("s_min", 0.05)
        s_max = kwargs.get("s_max", 50)
        s_noise = kwargs.get("s_noise", 1.003)

        sigma_min=kwargs.get("sigma_min", 0.0064)
        sigma_max=kwargs.get("sigma_max", 80)
        shape = kwargs.get("shape", (1, 3, 256, 256))

        num_steps = n_evals if use_euler else n_evals//2
        sigma_values = self.sigma_schedule(num_steps, rho=rho, sigma_min=sigma_min, sigma_max=sigma_max)

        sigma_values = np.concatenate((sigma_values, np.zeros_like(sigma_values[:1])))  # adding last sigma value of 0.

        if start_x is None:
            x = sigma_values[0]*torch.randn(shape, dtype=self.dtype, device=self.device)
        else:
            x = start_x.to(self.device, dtype=self.dtype)

        for i, (sigma_current, sigma_next) in tqdm.tqdm(
                enumerate(zip(sigma_values[:-1], sigma_values[1:])),
                total=len(sigma_values)-1,
                desc="Generating...",
                disable=not show_progress
            ):
            
            # get values of time from sigma values
            t_current = self.sigma_inv(sigma_current)
            t_next = self.sigma_inv(sigma_next)

            epsilon = torch.randn_like(x) * s_noise

            if sigma_current > s_min and sigma_current < s_max:
                gamma = min(s_churn/num_steps, np.sqrt(2)-1)
                t_hat = t_current + gamma*t_current
                sigma_hat = self.get_sigma_t(t_hat)
                x_hat = x + epsilon*(sigma_hat**2 - sigma_current**2)**0.5
            else:
                gamma = 0.0
                t_hat = t_current
                sigma_hat = sigma_current
                x_hat = x 
        
            d_cur = self.dx_estimate(x_hat, sigma_hat)

            # Euler step...
            x_prime = x_hat + d_cur * (t_next - t_hat)

            if use_euler or sigma_next == 0:
                x = x_prime
            else:
                d_prime = self.dx_estimate(x_prime, sigma_next)
                x = x_hat + 0.5 * (d_cur + d_prime) * (t_next - t_hat)
        x = x.contiguous()
        return x
    
    @torch.no_grad()
    def generate(self, method, batch_size=1, **kwargs):
        """Generate samples using the diffusion model as a denoiser.
        Args:
            method: method to use for generation, e.g. 'ddpm', 'sde', etc.
            batch_size: number of samples to generate
            **kwargs: additional arguments for the generation method
        """
        shape = (batch_size, 3, 256, 256)
        if method == 'ddpm_ode':
            return self.ddpm_ode_sample(shape=shape, **kwargs)
        elif method == 'edm_ode':
            return self.edm_ode_sample(shape=shape, **kwargs)
        elif method == 'edm_sde':
            return self.edm_sde_sample(shape=shape, **kwargs)
        elif method == 'ddpm_sde':
            return self.ddpm_sde_sample(shape=shape, **kwargs)
        else:
            raise ValueError(f"Unknown generation method: {method}")


# class DiffusionBaseEDM(ABC):
#     def __init__(self, beta_min=0.0001, beta_max=0.02, T=1000, use_float64=False):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.beta_min = beta_min
#         self.beta_max = beta_max
#         self.T = T
#         if use_float64:
#             self.dtype = torch.float64
#         else:
#             self.dtype = torch.float32
#         self.betas = np.linspace(beta_min, beta_max, T)
#         self.alphas = 1 - self.betas     # alpha = 1 - beta
#         self.alpha_bars = np.cumprod(self.alphas)
#         self.sigma_t = np.sqrt((1 - self.alpha_bars) / self.alpha_bars)

#     def sigma_schedule(self, num_steps, rho=7, sigma_min=0.0064, sigma_max=80):
#         if num_steps == 1:
#             return np.array([sigma_max])
#         step_indices = np.arange(num_steps)
#         return (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho


#     def condition_index(self, sigma):
#         """Returns the condition index for the given sigma value.
#         This is refered to as C_cond(sigma) in the EDM paper.
#         DNN in iDDPM is conditioned on index of markov chain (t: 0-T) at which the sample is generated.
#         For DDIM, they condition directory on the sigma value
#         """
#         pass

#     def denoised_estimate(self, x_hat, sigma):
#         """Returns denoised sample D_{\theta}(x, sigmal) as given in EDM paper.

#         Args:
#             x_hat: (N, C, H, W) tensor, sample before scaling by s(t)
#             sigma: noise level

#         """
#         pass

#     def sigma_inv(self, sigma):
#         """Returns the inverse of the function sigma(t)."""
#         pass

# class EDMiDDPM(nn.Module, DiffusionBaseEDM):
#     """Implements the iDDPM model from 'Improved Denoising Diffusion Probabilistic Models' (Nichol et al. 2021).
#     This is a denoising diffusion model that uses the noise prediction loss.
#     """
#     def __init__(self, estimator=None, beta_min=0.0001, beta_max=0.02, T=1000, use_float64=True):
#         super().__init__()
#         DiffusionBaseEDM.__init__(self, beta_min, beta_max, T)
#         self.estimator = estimator
#         if self.estimator is not None:
#             self.estimator = self.estimator.to(self.device)

        

    
    
#     def estimate_epsilon(self, x, t):
#         """Estimate the noise at time t, given x.
#         Uses the relationship between the score and the noise prediction.
        
#         Args:
#             x: (N, C, H, W) tensor
#             t: (N,) tensor of current time step
#         """
        
#         return self.estimator(x.to(torch.float32), t.to(torch.float32))[:,:3].to(self.dtype)

#     def iddpm_sigma_schedule(self, num_steps=1000, idx_offset=0):
#         """Implements the iDDPM sigma schedule, by using linear steps from max_index-j0 to 0.
#         Sigma schedule is defined by the value of sigma at each of these indices.

#         EDM paper uses an offset j0 to start the schedule from a higher value of sigma.
#         For cosine schedule, they used j0=8, which gives maximum value of sigma=80.
#         For linear schedule, this is j0=69 , which gives maximum value of sigma=80.
        
#         Args:
#             num_steps: number of steps to generate the sample
#             idx_offset: offset to start the schedule from, same as j0 in EDM paper.
#         """
#         step_indices = np.arange(num_steps)
#         t_indices =  self.T - 1 - np.floor(idx_offset + (self.T-1-idx_offset)/(num_steps-1)*step_indices).astype(int) 
#         return self.sigma_t[t_indices]
    
#     def get_sigma_index(self, sigma, beta_min=0.1, beta_d=19.9):
#         # sigma = torch.as_tensor(sigma).to(self.dtype)
#         return ((beta_min ** 2 + 2 * beta_d * (1 + sigma ** 2).log()).sqrt() - beta_min) / beta_d


#     def condition_index(self, sigma):
#         """Returns the condition index for the given sigma value.
#         This is refered to as C_cond(sigma) in the EDM paper.
#         DNN in iDDPM is conditioned on index of markov chain (t: 0-T) at which the sample is generated.
#         For DDIM, they condition directory on the sigma value
#         """
#         # return np.argmin(np.abs(self.sigma_t - sigma)) #  # -1 because we start from 0 index
#         beta_d          = 19.9         # Extent of the noise level schedule.
#         beta_min        = 0.1          # Initial slope of the noise level schedule.
#         M               = 1000         # Original number of timesteps in the DDPM formulation.

#         c_noise = (M - 1) * self.get_sigma_index(sigma, beta_min, beta_d)
#         return c_noise
    
