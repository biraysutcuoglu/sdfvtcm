# How to run the code
- activate conda env 
- run in training mode (training mode plots the training and validation loss and AP plots in wandb): python MaskRCNN/tool_segmentation.py --mode train
- run in inference mode (displays the evaluation metrics and saves the visualizations of predictions on images):
    - Inference on validation set: python MaskRCNN/tool_segmentation.py --mode infer-on-val
    - Inference on test set: python MaskRCNN/tool_segmentation.py --mode infer-on-val

# When dataset is changed
- Do not forget to adjust the number of classes in the finetuner.py
