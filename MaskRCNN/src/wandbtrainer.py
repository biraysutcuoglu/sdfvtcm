import os
import wandb
import torch
import numpy as np

from detectron2.engine import DefaultTrainer, hooks
from detectron2.data import build_detection_train_loader, DatasetMapper
from detectron2.utils.events import EventStorage

from detectron2.data import build_detection_test_loader
from detectron2.evaluation import inference_on_dataset
from detectron2.evaluation import COCOEvaluator

# for data augmentation
from detectron2.data.transforms import (
    ResizeShortestEdge, RandomFlip, RandomBrightness, RandomContrast, RandomRotation
)
import detectron2.data.transforms as T
from detectron2.utils.events import EventStorage

import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR

class RandomGuassianNoise(T.Transform):
    def __init__(self, mean=0, std=10):
        """
        Args:
            mean (float): Mean of the guassian noise.
            std (float): Standard deviation of the guassian noise.
        """
        super().__init__()
        self.mean = mean
        self.std = std
    
    def apply_image(self, img):
        """
        Apply Gaussian noise to the image.
        """
        noise = np.random.normal(self.mean, self.std, img.shape).astype(np.float32)
        img = img.astype(np.float32) + noise
        return np.clip(img, 0, 255).astype(np.uint8)

    def apply_coords(self, coords):
        return coords

# https://detectron2.readthedocs.io/en/latest/tutorials/augmentation.html
def build_augmentation():
    """ Builds data augmentation. """
    return [   
        T.RandomCrop("relative_range", (0.5, 0.7)),  # Crop 30% - 70% of the image,
        T.ResizeShortestEdge(short_edge_length=(800, 1000, 1200), max_size=2000, sample_style="choice")
        # T.RandomBrightness(0.9, 1.1), 
        # T.RandomContrast(0.9, 1.1),  
        # T.RandomRotation(angle=[-10, 10]),
        # T.RandomFlip(horizontal=True, vertical=False),
        # RandomGuassianNoise(mean=0, std=10)
    ]  

class WandbTrainer(DefaultTrainer):
    def __init__(self, cfg, val_dataset_name, test_dataset_name, project_name,run_name=None):
        super().__init__(cfg)

        self.init_lr = cfg.SOLVER.BASE_LR
        self.weight_decay = self.init_lr * 100

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.init_lr,
            weight_decay=self.weight_decay
        )

        # Add ReduceLROnPlateau scheduler according to validation AP
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.75, patience=3, verbose=True
        ) # factor for FBA dataset was 0.75

        # # Cosine Annealing Scheduler
        # self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=10, eta_min=self.init_lr * 0.1)

        # self.lr_scheduler = lr_scheduler.ExponentialLR(
        #     self.optimizer, 
        #     gamma=0.95  # Decay LR by 5% each epoch
        # )

        wandb.init(
            project=project_name,  # Your wandb project name
            name=run_name,         # Optional name for the run
            config=cfg             # Automatically log your cfg as wandb config
        )

        self.validation_metrics = []

        self.val_evaluator = COCOEvaluator(val_dataset_name, cfg, False, None)
        self.val_loader = build_detection_test_loader(cfg, val_dataset_name)

        self.test_evaluator = COCOEvaluator(test_dataset_name, cfg, False, None)
        self.test_loader = build_detection_test_loader(cfg, test_dataset_name)
        
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Builds the train loader.
        """
        return build_detection_train_loader(cfg)
        # If data augmentation is needed, return this
        # return build_detection_train_loader(cfg,
        #                                     mapper=DatasetMapper(cfg, is_train=True, augmentations=build_augmentation()))
        
    # Override the trainer of the detectron 2
    def run_step(self):
        """
        Implements the logging of metrics during training.
        """
        assert self.model.training, "[WandbTrainer] Model is not in training mode!"
        
        # Initialize the data loader iterator if not already done
        if not hasattr(self, "_data_loader_iter"):
            self._data_loader_iter = iter(self.data_loader)

        # Fetch the next batch of data
        data = next(self._data_loader_iter)

        with EventStorage(self.iter) as storage:
            # Forward pass to calculate losses
            loss_dict = self.model(data)
            losses = sum(loss_dict.values())
            
            if torch.isnan(losses):
                raise ValueError(f"NaN encountered in training losses: {loss_dict}")

            # Log loss to wandb
            wandb.log(
                {
                    "loss_total": losses.item(),
                    **{f"loss_{k}": v.item() for k, v in loss_dict.items()},
                    "iteration": self.iter,
                }
            )

            # Backpropagation and optimization step
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

    def after_step(self):
        """
        Additional logging after each step, such as learning rate or custom metrics.
        """
        # Call parent method to include default behavior
        super().after_step()

        assert isinstance(self.optimizer, torch.optim.AdamW), "Optimizer is not AdamW!"

        # Log learning rate
        lr = self.optimizer.param_groups[0]["lr"]
        wandb.log({"learning_rate": lr, "iteration": self.iter})

        # print(f"Iteration {self.iter}: Current LR = {lr}")
        
        if self.iter % 500 == 0:  # Log after every 100 iterations
            ap_score = self.evaluate_on_validation_dataset() # evaluate val data
            
            # Step the scheduler
            if ap_score is not None:
                self.lr_scheduler.step(ap_score)


    def evaluate_on_validation_dataset(self):
        """
        Evaluates on validation set and logs metrics to WandB.
        """
        # Evaluate on validation dataset
        results = inference_on_dataset(self.model, self.val_loader, self.val_evaluator)

        # Log validation results to wandb
        wandb.log({
            **{f"val_{k}": v for k, v in results.items()},
            "iteration": self.iter,
        })

        ap_score = results["bbox"]["AP"]
        return ap_score
        

