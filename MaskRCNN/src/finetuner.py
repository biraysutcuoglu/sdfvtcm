import torch, detectron2
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import yaml

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode

import matplotlib.pyplot as plt
from PIL import Image

from src.wandbtrainer import WandbTrainer

class Finetuner:
    def __init__(self, train_dataset_name, train_dataset_path, train_labels, val_dataset_path, val_labels, test_dataset_path, test_labels, wandb_project_name):
        """
        Initalization of finetuner
        """
        self.train_dataset_name = train_dataset_name
        self.val_dataset_name = "val"
        self.test_dataset_name = "test"
        
        register_coco_instances(self.train_dataset_name, {}, train_labels, train_dataset_path)
        register_coco_instances(self.val_dataset_name, {}, val_labels, val_dataset_path)
        register_coco_instances(self.test_dataset_name, {}, test_labels, test_dataset_path)
        
        self.train_metadata = MetadataCatalog.get(self.train_dataset_name)
        self.train_dataset_dicts = DatasetCatalog.get(self.train_dataset_name)

        self.val_metadata = MetadataCatalog.get(self.val_dataset_name)
        self.val_dataset_dicts = DatasetCatalog.get(self.val_dataset_name)

        self.test_metadata = MetadataCatalog.get(self.test_dataset_name)
        self.test_dataset_dicts = DatasetCatalog.get(self.test_dataset_name)

        self.wandbprojectname = wandb_project_name

    def get_trainer(self, experiment_name, model_output_dir, num_workers, batch_size, lr, max_iter, batch_size_per_image, num_classes=4):
        """
        Prepares trainer with hyperparams, here either the default or wandb trainer can be used
        """
        self.cfg = get_cfg()
        self.cfg.OUTPUT_DIR = model_output_dir
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")) # Pretrained model with random weights
        self.cfg.DATASETS.TRAIN = (self.train_dataset_name)
        self.cfg.DATASETS.TEST = () # testing data can be added here
        self.cfg.DATALOADER.NUM_WORKERS = num_workers
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        self.cfg.SOLVER.IMS_PER_BATCH = batch_size  
        self.cfg.SOLVER.BASE_LR = lr  
        self.cfg.SOLVER.MAX_ITER = max_iter   
        self.cfg.SOLVER.STEPS = []        
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size_per_image   # The "RoIHead batch size" : number of region proposals sampled per image
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # 2 classes for Tool dataset: tool and wear, 4 classes for FBA dataset: tool, flank_wear, bue, groove
        # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here because of the background.
        
        # # for detecting small objects 
        # self.cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[4], [8], [16], [32], [64]]
        # self.cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4.0]]

        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        
        # Use default trainer 
        # trainer = DefaultTrainer(self.cfg) 

        # Use custom WandbTrainer
        trainer = WandbTrainer(self.cfg, self.val_dataset_name, self.test_dataset_name, project_name=self.wandbprojectname, run_name=experiment_name)
        trainer.resume_or_load(resume=False)

        return trainer
    
    def save_model(self, model_config_path):
        """
        Saves model config to yaml file
        """
        # write finetuner config to file
        with open(model_config_path, 'w') as file:
            yaml.dump(self.cfg, file)
        
        # NOTE: Detectron automatically saves the model weights
    
    def load_model(self, model_path, num_classes, predictor_threshold, device="cuda"):
        """
        Loads model, can be used in inference
        """
        # Load config of the og model
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        
        # Modify the config to use custom weights
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = predictor_threshold  # Set threshold for this model
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.DEVICE = device

        self.cfg = cfg

        return DefaultPredictor(cfg)
    
    def save_predictor_outputs(self, predictor, output_dir, datasetdicts, visualization_type="instance predictions"):
        """
        Visualizes prediction results on the images also saves them in COCO json format
        """
        if datasetdicts == "test":
            datasetdicts = self.test_dataset_dicts
        elif datasetdicts == "val": 
            datasetdicts = self.val_dataset_dicts
        
        for idx, d in enumerate(datasetdicts):
            im_filepath = d["file_name"]
            im = cv2.imread(im_filepath)
            outputs = predictor(im)

            # 1. Draw ground truth
            gt_image = self.draw_ground_truth(im, d)

            # 2. Draw prediction
            # Create a visualizer instance for prediction
            v = Visualizer(
                im[:, :, ::-1],               # Convert BGR to RGB
                metadata = self.test_metadata,
                scale=0.8,
                instance_mode=ColorMode.IMAGE_BW  # Use black and white for background
            )
            if (visualization_type == "instance predictions"):
                # Draw the instance predictions
                out = v.draw_instance_predictions(outputs["instances"].to("cpu"), False)
                result_image = out.get_image()[:, :, ::-1]

            # 3. Combine ground truth and prediction
            combined_image = np.hstack((gt_image, result_image))

            predictions_dir = os.path.join(output_dir, "predictions")
            os.makedirs(predictions_dir, exist_ok=True)
            comparisons_dir = os.path.join(output_dir, "comparisons")   
            os.makedirs(comparisons_dir, exist_ok=True)

            # 4. Save the combined image
            # Define the output file path
            filename = os.path.split(im_filepath)[-1]
            prediction_output_file_path = os.path.join(predictions_dir, f"predicted_{filename}")
            cv2.imwrite(prediction_output_file_path, result_image)

            comparison_output_file_path = os.path.join(comparisons_dir, f"comparison_{filename}")
            cv2.imwrite(comparison_output_file_path, combined_image)
    
    def draw_ground_truth(self, im, d):
        v_gt = Visualizer(
            im[:, :, ::-1],  # Convert BGR to RGB
            metadata=self.test_metadata,
            scale=0.8
        )

        gt_visualization = v_gt.draw_dataset_dict(d)  # Draw ground truth annotations
        gt_image = gt_visualization.get_image()[:, :, ::-1]  # Convert back to BGR for OpenCV
        
        return gt_image