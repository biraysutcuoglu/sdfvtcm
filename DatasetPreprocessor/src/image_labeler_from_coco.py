import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import random

'''
This class generates labels considering the wear category and wear score from the COCO annotation files.
It is implemented for labelling the FBA Dataset images to be used for training the Stable Diff. model.
'''
# Change project root to root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)
from common.common_methods import CommonMethods

class ImageLabeler:
    # This class gets a coco annotation file of a dataset and generates image labels for the dataset
    
    def __init__(self, config_file):
        config = CommonMethods.read_yaml(config_file)
        self.coco_annotations_path = config['image_labeler']['coco_annotations_path']
        self.coco_annotations = CommonMethods.read_json(config['image_labeler']['coco_annotations_path'])
        self.tool_class = config['image_labeler']['tool_class']
        
        coco_images_field = self.coco_annotations['images']
        coco_annotations_field = self.coco_annotations['annotations']
        coco_categories_field = self.coco_annotations['categories']
        
        self.category_dict = {category["id"]: category["name"] for category in coco_categories_field}

        # generate image dict and assign required values
        self.image_dict = self.generate_image_dict(coco_images_field)
        # below methods will update the image_dict
        self.get_wear_type(coco_annotations_field)
        self.get_tool_size(coco_annotations_field)
        self.get_wear_state(coco_annotations_field)
        self.calculate_wear_score()

    def generate_image_prompts(self):
        # annotation format for huggingface dataset
        # e.g. {"file_name": "4_1_1.png", "text": "turning tool with moderate flank wear"}
        ann_dir = os.path.dirname(self.coco_annotations_path)
        output_file = os.path.join(ann_dir, "metadata.jsonl")
        num_ann_images = 0
        with open(output_file, "w") as f:
            for k, v in self.image_dict.items():
                file_name = v['file_name']
                
                wear_types = ["flank" if wear == "flank_wear" else wear for wear in v['wear_type']]
                random.shuffle(wear_types)
                type_wear = ", ".join(wear_types)
                # wear_score = f"{v['wear_score']:.2f}"
                # prompt = f"{self.tool_class} with {type_wear} wear with a wear score of {wear_score}"
                # NOTE: For now just focus on wear type without wear score
                prompt = f"{self.tool_class} with {type_wear} wear"

                data = {"file_name": file_name, "text": prompt}
                f.write(json.dumps(data) + "\n")
                num_ann_images += 1
        
        print(f"Labels of {num_ann_images} images, written to {output_file}")

    def get_wear_state(self, coco_annotations_field):
        #Get the bounding boxes of wear categories and get the biggest height (x, y, w, h) and calculate the wear state
        for annotation in coco_annotations_field:
            image_id = annotation['image_id']
            
            if image_id in self.image_dict:
                category = self.category_dict.get(annotation['category_id'])
                if category != "tool":
                    bbox = annotation['bbox']
                    wear = bbox[3]
                    if wear > self.image_dict[image_id]['wear_state'] :
                        self.image_dict[image_id]['wear_state'] = wear

    def get_tool_size(self, coco_annotations_field):
        # Get h from bbox of tool category for each id 
        for annotation in coco_annotations_field:
            image_id = annotation['image_id']
            
            if image_id in self.image_dict:
                category = self.category_dict.get(annotation['category_id'])
                if category == "tool":
                    self.image_dict[image_id]["tool_size"] = annotation["bbox"][3]

    def get_wear_type(self, coco_annotations_field):
        for annotation in coco_annotations_field:
            image_id = annotation['image_id']
            
            if image_id in self.image_dict:
                wear_type = self.category_dict.get(annotation['category_id'])
                if wear_type != "tool":
                    self.image_dict[image_id]["wear_type"].append(wear_type)

    def generate_image_dict(self, coco_images_field):
        image_dict = {}

        for image in coco_images_field:
            image_id = image['id']
            file_name = image['file_name']
            
            image_dict[image_id] = {"file_name": file_name, "wear_type": [], "tool_size": 0, "wear_state": 0, "wear_score": 0}
        
        return image_dict

    def calculate_wear_score(self):
        # normalize (wear size / tool size)
        # with log normalization
        for v in self.image_dict.values():
            v['wear_score'] = np.log1p(v['wear_state'] / v['tool_size'])

    def plot_wear_scores(self):
        file_names = [v['file_name'] for v in self.image_dict.values()]
        wear_scores = [v['wear_score'] for v in self.image_dict.values()]

        # Sort data by wear score (optional)
        sorted_indices = np.argsort(wear_scores)
        wear_scores = [wear_scores[i] for i in sorted_indices]
        file_names = [file_names[i] for i in sorted_indices]

        # Use indices for x-axis
        x_indices = np.arange(len(file_names))

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(x_indices, wear_scores, marker='o', linestyle='-', color='blue')

        # Formatting
        plt.xlabel("Image Index")
        plt.ylabel("Wear Score")
        plt.title("Wear Scores for Different Tools")
        
        # Show every nth label to avoid overcrowding
        step = max(1, len(file_names) // 20)  # Adjust dynamically
        plt.xticks(x_indices[::step], file_names[::step], rotation=45, ha="right")

        # Set y-axis limits (since wear score is normalized)
        plt.ylim(0, 1)

        # Show the plot
        plt.tight_layout()
        plt.show()
    
    def plot_wear_score_distribution(self, bins=10):
        wear_scores = [v['wear_score'] for v in self.image_dict.values()]

        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(wear_scores, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)

        # Formatting
        plt.xlabel("Wear Score Range")
        plt.ylabel("Number of Images")
        plt.title("Distribution of Wear Scores")
        
        # Show grid for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        # Show the plot
        plt.show()




        
