# Stable Diffusion Data Augmentation for Visual Tool Condition Monitoring

1. Create a virtual env. (python>=3.9,<3.12  # TensorFlow 2.16.1 supports Python 3.8–3.11)
- python3.10 -m venv {name of the venv}

2. activate venv
- source {name of the venv}/bin/activate

3. upgrade pip and install requirements
- pip install --upgrade pip
- pip install -r requirements.txt
- python -m ipykernel install --user --name={name of the venv} --display-name "Python venv"

4. install detectron2 (this may take a while)
- ``python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' ``
5. move Data folder to root

----------------------
# Segmentation with Mask R-CNN
6. For training Mask R-CNN on FBA dataset
- open MaskRCNN/segmenter_config.yaml and check configurations (configure location of train, validation and test sets, hyperparameters for training MaskRCNN and wandb project and experiment name)
- if everything looks fine, for training:
    - cd MaskRCNN/ and run python tool_segmentation.py --mode train
    - the model will be saved in MaskRCNN/{model_output_dir} directory

    - during training terminal will show MaskRCNN model architecture, expected training time, evaluation results for BBox and Segm. 
    - if you'd like to see the wandb monitoring, just after running the train command it will display a wandblink. Through this link training can be monitored. (this step requires having a wandb account)

# After training
7. After training finishes inference on validation and test images can be done via:
- python tool_segmentation.py --mode infer-on-val or python tool_segmentation.py --mode infer-on-test
- the evaluation metrics for val or test sets will be written in terminal
- prediction visualization will be placed under MaskRCNN/prediction_visualizations/{experiment_name}, here comparisons folder will show the ground truth on the left and the prediction result on the right. predictions folder will only show the model predictions.

# If you want to do inference with an already trained model
- copy fba_models folder into MaskRCNN (in this folder the models mentioned in thesis can be found)
- add fba_model to .gitignore
- pick a model and adjust the segmenter_config.yaml 
    - lets say that the picked model is fba_models/real/real_only_{experimentdetails}. The experiment_details part is automatically created based on the hyperparameters in the config. 
    - model/output_dir should be adjusted to "./fba_models/real/"
    - wandb/experiment_name should be adjusted to "real only" 
    - then run python tool_segmentation.py --mode infer-on-val or python tool_segmentation.py --mode infer-on-test to see the results. Again the prediction results will be placed under MaskRCNN/prediction_visualizations/. 

----------------------
# Segmentation with UNet
For training the UNet Jupyter Notebook model_training.ipynb and for model evaluation model_evaluation.ipynb is implemented.
- 

