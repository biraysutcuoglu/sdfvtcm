import torch, detectron2
# Setup detectron2 logger
import detectron2
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

import matplotlib.pyplot as plt
from PIL import Image

class NotebookUtils:
    
    def display_image(example_image_path):
        # Read the image with OpenCV
        im = cv2.imread(example_image_path)

        # Convert BGR to RGB
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # Display the image using Matplotlib
        plt.imshow(im_rgb)
        plt.title("Tool Image")
        plt.axis("off") 
        plt.show()

        return im
    
    def predict_classes_without_finetuning(example_image):
        cfg = get_cfg()

        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)
        outputs = predictor(example_image)

        return cfg, outputs

    def show_predictions_of_model(cfg, outputs, im):
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # Convert the result from BGR to RGB (since OpenCV uses BGR and matplotlib expects RGB)
        result_image = out.get_image()

        # Display the image using matplotlib
        plt.imshow(result_image)
        plt.title("Predictions of the model without finetuning")
        plt.axis('off')  # Hide axes
        plt.show()
    
    
