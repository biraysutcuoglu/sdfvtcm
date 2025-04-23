import numpy as np
import os
from PIL import Image, ImageDraw
from common.common_methods import CommonMethods

class COCOtoBinaryConverter:
    def create_masks(coco_data_path, wear_class_id, out_dir):
        '''
        Convert COCO annotations to binary masks for a specific wear class.
        Args:
            coco_data_path (str): Path to the COCO JSON file.
            wear_class_id (int): The category ID for the wear class.
            out_dir (str): Directory to save the binary masks.
        '''
        if not os.path.exists(coco_data_path):
            raise FileNotFoundError(f"Error: The file '{coco_data_path}' does not exist.")
        
        # out dir for binary masks
        os.makedirs(out_dir, exist_ok=True)

        coco_data = CommonMethods.read_json(coco_data_path)

        # read coco json ann file
        for image_info in coco_data["images"]:
            img_id = image_info["id"]
            img_w, img_h = image_info["width"], image_info["height"]
            file_name = image_info["file_name"]

            # Create a blank mask for the image
            binary_mask = Image.new("L", (img_w, img_h), 0) # Mode L (grayscale), default black
            draw = ImageDraw.Draw(binary_mask)

            # Find annotations of this image
            for annotation in coco_data["annotations"]:
                if annotation["image_id"] == img_id and annotation["category_id"] == wear_class_id:
                    segmentation = annotation["segmentation"]
            
                    # Draw each segmentation polygon on the mask
                    for polygon in segmentation:
                        draw.polygon(polygon, outline=255, fill=255)  # White region for wear (255)

                    # Save the mask
                    mask_path = os.path.join(out_dir, f"{file_name.replace('.png', '_mask.png')}")
                    binary_mask.save(mask_path)
        
        print(f"Binary masks saved to {out_dir}")