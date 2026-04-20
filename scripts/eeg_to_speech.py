import time
import yaml
import hydra
import logging
from omegaconf import OmegaConf
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# local
from diffusion_hub import BaseTrainer
from diffusion_hub.datasets import get_dataset
from diffusion_hub.models.eeg_spect import ECoGEncoder


import math
def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    # 1) linear warmup
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)
    
    # 2) after decay — floor at min_lr
    if step > max_steps:
        return min_lr
    
    # 3) cosine decay between warmup and max_steps
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 1 → 0
    return min_lr + coeff * (max_lr - min_lr)

def subsample_dataset(dataset, fraction=0.5, seed=42):
    """
    Randomly subsample a fraction of the dataset reproducibly.
    Same seed → always same subset.
    """
    rng = np.random.default_rng(seed)        # ← isolated RNG, doesn't affect global state
    
    n_total  = len(dataset)
    n_sample = int(n_total * fraction)
    
    indices = rng.choice(n_total, size=n_sample, replace=False)
    
    return torch.utils.data.Subset(dataset, indices)


class SpectTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        train_data,
        min_lr,
        warmup_steps,
        lr_decay_steps,
        weight_decay,
        beta1,
        beta2,
        **kwargs,
        ):
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        super().__init__(model, train_data, **kwargs)
        

    def forward_step(self, batch):
        eeg, spect = batch
        eeg = eeg.to(self.device)
        spect = spect.to(self.device)

        out = self.model(eeg)
        loss = F.mse_loss(out, spect)
        return loss
    

    def configure_optimizer(self, lr):
        self.lr = lr
        # ❌ self.model is DDP wrapper — doesn't have configure_optimizers
        # ✅ self.model.module is the actual underlying model
        model = self.model.module if hasattr(self.model, 'module') else self.model
        return model.configure_optimizers(self.weight_decay, lr, (self.beta1, self.beta2), self.device)

    def configure_scheduler(self, current_step):
        """making sure first update uses appropriate lr because schedular_step is called after update"""
        # epochs = int(current_step/self.steps_per_epoch)
        if current_step == 0:
            lr = 0
        else:
            lr = get_lr(current_step, self.warmup_steps, self.lr_decay_steps, self.lr, self.min_lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def scheduler_step(self, current_step):

        lr = get_lr(current_step, self.warmup_steps, self.lr_decay_steps, self.lr, self.min_lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        if self.is_main_process():
            self.writer.add_scalar(
                "Optim/lr",
                lr,
                current_step
                )


@hydra.main(version_base='1.3', config_path="../configs/train_configs", config_name="eeg_to_speech")
def main(args):

    logging.info(yaml.dump(OmegaConf.to_container(args, resolve=True), indent=4))
    logging.info(f"Creating datasets and model...")
    model = ECoGEncoder(args.spect_model.config)

    train_data_config = args.data.train_data
    train_data = get_dataset(train_data_config.name, **train_data_config.dataset_config)
    
    val_data_config = args.data.val_data
    val_data = get_dataset(val_data_config.name, **val_data_config.dataset_config)

    logging.info(f"Number of training samples: {len(train_data)}")
    logging.info(f"Training data duration (hours): {(len(train_data) * val_data.trial_duration)/3600:.2f}")

    trainer_config = args.trainer
    trainer = SpectTrainer(
        model=model,
        train_data=train_data,
        eval_data=val_data,
        **trainer_config.trainer_config,
    )

    trainer.train(
        **trainer_config.train_method,
    )




if __name__ == "__main__":

    START_TIME = time.time()
    main()
    logging.info(f"Total time taken: {(time.time()-START_TIME)/60} minutes")
    