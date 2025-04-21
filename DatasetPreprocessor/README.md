This module contains functions and classes for:
- binary mask generation (from COCO JSON annotation to binary masks)
- data labelling (wear area calculation for image prompts) --> TODO: Change this by calculating wear using bounding box only change in y is enough
- data loading (converting data to huggingface dataset using images and prompts)
- coco annotation generator (color coded maps to COCO JSON)

Steps:
- If you have a new dataset with colormap annotations
1. Adjust the preprocessor.config
2. Run the notebook DatasetPreprocessor/prepare_dataset_for_segmentation_model_notebook.ipynb
this will generate coco annotations, apply image enhancing and generate image labels for the stable diffusion
