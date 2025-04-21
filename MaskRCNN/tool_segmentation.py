# import and check torch and detectron version
import argparse
import torch, detectron2

# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import inference_on_dataset

import matplotlib.pyplot as plt
from PIL import Image
import yaml
import wandb

from src.notebook_utils import NotebookUtils
from src.finetuner import Finetuner

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO


class ToolSegmentation:  
    
    @staticmethod
    def define_finetuner(train_dataset_path, train_labels_path, val_dataset_path, val_labels_path, test_dataset_path, test_labels_path):
        """
        Defines finetuner
        """
        wandb_project_name = config["wandb"]["project_name"]
        train_dataset_name = train_dataset_path.replace("/", "_")

        finetuner = Finetuner(train_dataset_name, train_dataset_path, train_labels_path, 
                              val_dataset_path, val_labels_path, test_dataset_path, test_labels_path, wandb_project_name)
        
        return finetuner
    
    @staticmethod
    def setup_experiment_dir(config, experiment_name):
        """
        Sets up experiment dir using hyperparams
        """
        base_output_dir = config["model"]["output_dir"]
        experiment_output_dir = os.path.join(base_output_dir, experiment_name)
        os.makedirs(experiment_output_dir, exist_ok=True)
        return experiment_output_dir

    @staticmethod
    def train(finetuner, config, num_workers, lr, batch_size, max_iter,
                                batch_size_per_image, num_classes):
        """
        Trains and saves the model
        """
    
        # Train model
        model_output_dir = ToolSegmentation.setup_experiment_dir(config, experiment_name)
        trainer = finetuner.get_trainer(experiment_name, model_output_dir, num_workers, 
                                            batch_size, lr, max_iter, batch_size_per_image, num_classes)

        trainer.train()

        model_config_path = os.path.join(model_output_dir, "config.yaml")
        finetuner.save_model(model_config_path)

        #Evaluate on test dataset after training
        # trainer.evaluate_on_test_dataset()

    @staticmethod
    def infer_on_test(finetuner, config, experiment_name, num_classes, predictor_threshold):
        """
        Evaluation method for test dataset
        """
        # Load model
        model_output_dir = config["model"]["output_dir"]
        model_path = os.path.join(model_output_dir, experiment_name, "model_final.pth")
        predictor = finetuner.load_model(model_path, num_classes, predictor_threshold)
        print(f"Using threshold for predictions: {finetuner.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}")
        
        prediction_visualizations_path = os.path.join("./prediction_visualizations/", experiment_name, "inference_on_test")
        
        # Evaluate test dataset
        test_evaluator = COCOEvaluator(finetuner.test_dataset_name, finetuner.cfg, False, prediction_visualizations_path)
        test_loader = build_detection_test_loader(finetuner.cfg, finetuner.test_dataset_name)
        
        results = inference_on_dataset(predictor.model, test_loader, test_evaluator)
        
        finetuner.save_predictor_outputs(predictor, prediction_visualizations_path, "test")
    
    @staticmethod
    def infer_on_val(finetuner, config, experiment_name, num_classes, predictor_threshold):
        """
        Evaluation method for val dataset
        """
        # Load model
        model_output_dir = config["model"]["output_dir"]
        model_path = os.path.join(model_output_dir, experiment_name, "model_final.pth")
        predictor = finetuner.load_model(model_path, num_classes, predictor_threshold)
        print(f"Using threshold for predictions: {finetuner.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}")
        
        prediction_visualizations_path = os.path.join("./prediction_visualizations/", experiment_name, "inference_on_val")
        
        # Evaluate val dataset
        val_evaluator = COCOEvaluator(finetuner.val_dataset_name, finetuner.cfg, False, prediction_visualizations_path)
        val_loader = build_detection_test_loader(finetuner.cfg, finetuner.val_dataset_name)
        
        results = inference_on_dataset(predictor.model, val_loader, val_evaluator)
        
        finetuner.save_predictor_outputs(predictor, prediction_visualizations_path, "val")
    
    @staticmethod
    def read_yaml(config_file_path):
        # Load YAML configuration
        with open(config_file_path, "r") as file:
            config = yaml.safe_load(file)

        return config

if __name__ == "__main__":
    
    # Argument parsing
    parser = argparse.ArgumentParser(description="Tool Segmentation")
    parser.add_argument("--config", type=str, default="./segmenter_config.yaml", help="Path to configuration file")
    parser.add_argument("--mode", choices=["train", "infer-on-val", "infer-on-test"], required=True, help="Mode: train or infer-on-val or infer-on-test")
    args = parser.parse_args()

    # Prepare config
    config = ToolSegmentation.read_yaml(args.config)

    # Extract file paths from the config
    train_dataset_path = config["train_dataset"]["path"]
    train_labels_path = config["train_dataset"]["labels_path"]

    val_dataset_path = config["val_dataset"]["path"]
    val_labels_path = config["val_dataset"]["labels_path"]

    test_dataset_path = config["test_dataset"]["path"]
    test_labels_path = config["test_dataset"]["labels_path"]

    # Check if all paths exist
    paths = [train_dataset_path, train_labels_path, val_dataset_path, val_labels_path, test_dataset_path, test_labels_path]
    for path in paths:
        if not os.path.exists(path):
            print(f"Error: The path {path} does not exist.")

    finetuner = ToolSegmentation.define_finetuner(
        train_dataset_path, train_labels_path,
        val_dataset_path, val_labels_path,
        test_dataset_path, test_labels_path
    )

    # Hyper-params from config 
    # num_train_images = config["train_dataset"]["num_images"]
    num_classes = config["model"]["num_classes"]
    num_workers = config["model"]["num_workers"]
    lr = config["model"]["lr"]
    batch_size = config["model"]["batch_size"]
    max_iter = config["model"]["max_iter"]
    batch_size_per_image = config["model"]["batch_size_per_image"]

    # Predictor threshold
    predictor_threshold = config["predictor"]["threshold"]

    exp_name = config["wandb"]["experiment_name"]
    experiment_name = f"{exp_name}_lr{lr}_bs{batch_size}_maxiter{max_iter}_bspi_{batch_size_per_image}"

    if args.mode == "train":
        ToolSegmentation.train(finetuner, config, num_workers, lr, batch_size, max_iter,
                                batch_size_per_image, num_classes)
    elif args.mode == "infer-on-val":
        ToolSegmentation.infer_on_val(finetuner, config, experiment_name, num_classes, predictor_threshold)
    elif args.mode == "infer-on-test":
        ToolSegmentation.infer_on_test(finetuner, config, experiment_name, num_classes, predictor_threshold)