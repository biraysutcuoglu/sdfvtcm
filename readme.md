# Stable Diffusion Data Augmentation for Visual Tool Condition Monitoring

## Steps for Running the Detection and Segmentation Pipeline
1. Create a virtual environment (venv) with python>=3.9,<3.12  (TensorFlow 2.16.1 supports Python 3.8–3.11) \
`` python3.10 -m venv {name_of_the_venv} ``

2. Activate venv \
`` source {name_of_the_venv}/bin/activate ``

3. Upgrade pip and install requirements \
`` pip install --upgrade pip `` \
`` pip install -r requirements.txt ``\
`` python -m ipykernel install --user --name={name_of_the_venv} --display-name "Python venv" ``\

4. Install detectron2 (this may take a while)
- ``python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' ``
5. Move Data folder to root \
. \
├── Data \
├── DatasetPreprocessor \
├── Experiments

----------------------
### Segmentation with Mask R-CNN
6. For training Mask R-CNN on FBA dataset:
- Open MaskRCNN/segmenter_config.yaml
    - Check configurations (configure location of train, validation and test sets, hyperparameters for training MaskRCNN and Wandb project and experiment name)
- If everything looks fine, for training:
    - ``cd MaskRCNN/`` and run ``python tool_segmentation.py --mode train``
        - The model will be saved in MaskRCNN/{model_output_dir} directory.

        - During training terminal will show MaskRCNN model architecture, expected training time, evaluation results for BBox and Segm. 
        - If you'd like to see the WandB monitoring, just after running the train command, it will display a Wandb link. Through this link, training plots can be monitored (this step requires having a wandb account).

7.1. After training finishes, inference on validation and test images can be done via:
- ``python tool_segmentation.py --mode infer-on-val`` or `` python tool_segmentation.py --mode infer-on-test``
    - The evaluation metrics for val or test sets will be displayed in terminal.
    - Prediction visualization will be placed under MaskRCNN/prediction_visualizations/{experiment_name}
    - Inside of the prediction_visualizations folder, comparisons folder will show the ground truth on the left and the prediction result on the right. Predictions folder will only show the model predictions.

7.2. **If you want to do inference with an already trained model**
- Copy fba_models folder into MaskRCNN (In this folder, the models mentioned in the experiments section of the thesis can be found.)
    .  
    ├── MaskRCNN  
    │   └── fba_models  
    ├── DatasetPreprocessor  
    ├── Experiments

- Add fba_models to .gitignore
- Pick a model and adjust the segmenter_config.yaml:
    - Let's say that you selected fba_models/real/real only_{experiment_details} model. According to this selection :
        - The experiment_details part is automatically created based on the hyperparameters in the config. 
        - model/output_dir should be adjusted to "./fba_models/real/"
        - wandb/experiment_name should be adjusted to "real only" 
    - Then run `` python tool_segmentation.py --mode infer-on-val`` or `` python tool_segmentation.py --mode infer-on-test`` to see the evaluation results.     
        - Again the prediction results will be placed under MaskRCNN/prediction_visualizations/. 

----------------------
### Segmentation with UNet
For training the UNet, Jupyter Notebook **model_training.ipynb** and for model evaluation model_evaluation.ipynb is implemented.
- By running these notebooks, training or inference can be done on the Tool Dataset.
- If you'd like to use already trained models for inference:
    - Copy the V_Unet folder inside of TransferLearningUnet. (if you change the name of the folder, please add it to the .gitignore file)
    - Run model_evaluation.ipynb notebook.

-----------------------

