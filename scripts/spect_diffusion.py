import torch
import time
import hydra
import logging

# local
from diffusion_hub import BaseTrainer
from diffusion_hub.datasets import get_dataset
from diffusion_hub.diffusion import SDEDiffusion
from diffusion_hub.diffusion.spect_model import GradLogPEstimator2d


class DiffTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        train_data,
        lr_decay=1.0,
        warmup_factor=1.0,
        warmup_epochs=1,
        gradient_clip=1.0,
        **kwargs,
        ):
        super(DiffTrainer, self).__init__(
            model, 
            train_data,
            **kwargs,
        )
        self.lr_decay = lr_decay
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.gradient_clip = gradient_clip
        

    def forward_step(self, batch):
        spect, *_ = batch
        spect = spect.to(self.device)

        loss, _ = self.model(spect)
        # if hasattr(self.model, "module"):
        #     loss, _ = self.model.module.compute_loss(spect)
        # else:
        #     loss, _ = self.model.compute_loss(spect)
        
        return loss
    
    def training_step(self, batch, step):
        loss = self.forward_step(batch)
        loss = loss/self.gradient_accumulation_steps
        loss.backward()

        if (step+1) % self.gradient_accumulation_steps == 0 or (step+1) == len(self.train_dataloader):
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)

            self.optimizer.step()
            self.optimizer.zero_grad()

        return {'train_loss': loss.item()*self.gradient_accumulation_steps} 

    def configure_scheduler(self, last_epoch):


        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 5.38e-5
        
        # self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        #     self.optimizer, start_factor=self.warmup_factor, total_iters=self.warmup_epochs
        #     )

        # self.scheduler =  torch.optim.lr_scheduler.ExponentialLR(
        #     self.optimizer, gamma=self.lr_decay, last_epoch=last_epoch
        #     )

    def scheduler_step(self, current_step):
        # Decay the learning rate after each epoch
        if current_step % self.steps_per_epoch == 0:
            # current_epoch = current_step // self.steps_per_epoch
            # if current_epoch < self.warmup_epochs:
            #     self.warmup_scheduler.step()
            # elif current_epoch > self.warmup_epochs and current_step < 570000:
            #     self.scheduler.step()
            # else:
            #     ...

            if self.is_main_process():
                self.writer.add_scalar(
                    "Optim/lr",
                    self.optimizer.param_groups[0]['lr'],
                    current_step
                    )


@hydra.main(version_base='1.3', config_path="../configs/train_configs", config_name="spect_diffusion")
def main(args):

    model_config = args.model
    estimator = GradLogPEstimator2d(**model_config.net_config)
    diff_model = SDEDiffusion(estimator, **model_config.diffusion_config)

    data_config = args.data
    train_data = get_dataset(data_config.name, **data_config.dataset_config)
    val_data = get_dataset(data_config.name, **data_config.dataset_config, validation=True)

    
    trainer_config = args.trainer
    trainer = DiffTrainer(
        model=diff_model,
        train_data=train_data,
        eval_data=val_data,
        data_collator=train_data.collate_fn,
        **trainer_config.trainer_config,
    )

    trainer.train(
        **trainer_config.train_method,
    )




if __name__ == "__main__":

    START_TIME = time.time()
    main()
    logging.info(f"Total time taken: {(time.time()-START_TIME)/60} minutes")
    