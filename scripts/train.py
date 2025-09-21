import os
import time
import hydra
import logging
import torch


from diffusion_hub.datasets import get_dataset
from diffusion_hub.diffusion import get_diffusion
from diffusion_hub.models.openaimodel import UNetModel

from ml_utils.trainers import BaseTrainer

class DiffTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        train_data,
        lr_decay=0.99,
        **kwargs,
        ):

        super(DiffTrainer, self).__init__(
            model, 
            train_data,
            **kwargs,
        )
        self.lr_decay = lr_decay

    def forward_step(self, batch):
        images, _ = batch
        images = images.to(self.device)
        if hasattr(self.model, "module"):
            loss = self.model.module.compute_loss(images)
        else:
            loss = self.model.compute_loss(images)
        return loss

    # def configure_scheduler(self, current_step):
    #     self.scheduler =  torch.optim.lr_scheduler.ExponentialLR(
    #         self.optimizer, gamma=self.lr_decay
    #         )

    # def scheduler_step(self, current_step):
    #     # Decay the learning rate after each epoch
    #     if current_step % self.steps_per_epoch == 0:
    #         self.scheduler.step()


@hydra.main(version_base='1.3', config_path="../config", config_name="train_config")
def main(cfg):

    train_data = get_dataset(**cfg.data.train_data)
    val_data = get_dataset(**cfg.data.test_data)
    estimator = UNetModel(**cfg.task.model)
    diffusion = get_diffusion(cfg.task.diffusion, estimator=estimator)

    trainer = DiffTrainer(
        model=diffusion,
        train_data=train_data,
        eval_data=val_data,
        **cfg.task.trainer.constructor_args,
    )

    trainer.train(
        **cfg.task.trainer.train_args,
    )



if __name__ == "__main__":

    START_TIME = time.time()
    main()
    logging.info(f"Total time taken: {(time.time()-START_TIME)/60} minutes")
    