import sys
import os 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
import json

# Change project root to root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)
from common.common_methods import CommonMethods

class Colormap_to_COCO_Annotation_Generator:
    """"
    This class generates COCO annotations from color maps.
    """
    def __init__(self, config_path, labels_out_file_name, mask_indication="_5"):
        self.coco_out_filename = labels_out_file_name
        self.config = CommonMethods.read_yaml(config_path)

        metadata_filepath = self.config["dataset_with_color_maps"]["metadata_dir"]
        self.metadata = CommonMethods.read_json(metadata_filepath)
        
        self.image_list = CommonMethods.read_images_in_dir(self.config["dataset_with_color_maps"]["images_dir"])
        self.mask_list = CommonMethods.read_images_in_dir(self.config["dataset_with_color_maps"]["color_maps_dir"])
        self.mask_file_names_list = self.get_mask_file_names(self.config["dataset_with_color_maps"]["color_maps_dir"], ".png", ".png", mask_indication)

        # Extract category mappings
        self.categories_dict = {
            cat_info["color"]: cat_info["id"]
            for cat_info in self.metadata["categories"].values()
        }

        self.tool_color = self.metadata["categories"]["tool"]["color"]
        self.flank_color = self.metadata["categories"].get("flank_wear", {}).get("color")
    
    # retrieved from https://www.immersivelimit.com/create-coco-annotations-from-scratch
    def create_submasks(self, image_mask):
        """
            Creates submask for each color in the given image mask (color mask),
            Takes color segmented mask image as input,
            Identifies different  regions based on color
            Creates binary masks (submasks) for each region (excluding the background)
            Stores submasks in a dictionary, key is the RGB color and the value is the submask
        """
        width, height = image_mask.size

        # Initialize a dict of submasks indexed by RGB colors
        sub_masks = {}
        for x in range(width):
            for y in range(height):
                pixel = image_mask.getpixel((x,y))[:3]

                # Generate a submask for each color except black
                if pixel!= (0, 0, 0):
                    # Colors are converted to string and used as key
                    pixel_str = str(pixel)
                    sub_mask = sub_masks.get(pixel_str)
                    # Check if a submask exists in this color if not create a new one
                    if sub_mask is None:
                        # Create a submask (binary mask) and add it to dict, also add padding (for protecting edges)
                        sub_masks[pixel_str] = Image.new('1', (width+2, height+2))
                    
                    # Shift the pixel by (1, 1) and set it to 1 (belongs to a submask)
                    sub_masks[pixel_str].putpixel((x+1, y+1), 1)

        return sub_masks
    
    # retrieved from https://www.immersivelimit.com/create-coco-annotations-from-scratch
    def create_submask_annotation(self, sub_mask, image_id, category_id, annotation_id, is_crowd):
        """
            Converts submask to COCO annotation
            Identifies object boundaries and creates polygon segmentation
            Calculates bbox coordinates and the area of the region
            Returns annotation dictionary for the submask (for each category in colormask a submask annotation is generated)
        """
        # Find contours (object boundaries) around each submask
        # if there is a change in color by 0.5 (e.g. 0->1), 
        # Low: contour direction is clockwise
        contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

        segmentations = [] # store polygon coordinates
        polygons = [] #polygon objects

        for contour in contours:
            # Convert coordinates from (row, col) -> (x, y)
            # subtract the padding pixel
            for i in range(len(contour)):
                row, col = contour[i]
                # remove padding
                contour[i] = (col - 1, row - 1)
            
            # Make polygon and simplify it
            poly = Polygon(contour)
            if poly.is_valid:
                poly = poly.simplify(1.0, preserve_topology=False) # false: remove small holes
                polygons.append(poly)
            
                # Extract outer boundary coordinates convert them to a flat list
                if not poly.is_empty and hasattr(poly, 'exterior'):
                    segmentation = np.array(poly.exterior.coords).ravel().tolist()
                else:
                    segmentation = []

                # COCO annotation requires at least 3 points
                if len(segmentation) > 6:
                    segmentations.append(segmentation)

        if polygons:
            valid_polygons = [p for p in polygons if isinstance(p, Polygon) and not p.is_empty]
            if valid_polygons:
                # Combine the polygons to calculate the bbox and area
                multi_poly = MultiPolygon(polygons)
                x, y, max_x, max_y = multi_poly.bounds
                width = max_x - x
                height = max_y - y
                bbox = (x, y, width, height)
                area = multi_poly.area
            else:
                bbox = [0, 0, 0, 0]
                area = 0
        else:
            bbox = [0, 0, 0, 0]
            area = 0

        # COCO Annotation dict
        annotation = {
            'id': annotation_id,
            'iscrowd': is_crowd,
            'image_id': image_id,
            'category_id': category_id,
            'segmentation': segmentations,
            'bbox': bbox,
            'area': area
        }

        return annotation

    def generate_annotations_field(self, mask_images):
        """
            For each mask:
            1. Create submasks
            2. Create submask annotation
            Ensures the nested regions are correctly processed.
            Note: It is observed that categories including other categories inside such as tool or flank wear should be processed separately,
            Submasks inside should be subtracted from the outside submask
            For not to label each category inside as the category outside too.
        """
        is_crowd = 0
        annotation_id = 1
        image_id = 1
        annotations = []

        for mask_image in mask_images:
            sub_masks = self.create_submasks(mask_image)

            # Identify the tool mask
            tool_mask = sub_masks.get(self.tool_color)  
            if tool_mask:
                tool_mask_np = np.array(tool_mask, dtype=np.uint8)  # Convert to NumPy array (binary)
                tool_mask_np = (tool_mask_np > 0).astype(np.uint8)  # Ensure binary values (0 or 1)

                # Subtract wear regions from tool mask
                for color, sub_mask in sub_masks.items():
                    if color != self.tool_color:  # Ignore tool mask itself
                        wear_mask_np = np.array(sub_mask, dtype=np.uint8)
                        wear_mask_np = (wear_mask_np > 0).astype(np.uint8)  # Ensure binary

                        # Subtract wear mask from tool mask and clip to keep valid values
                        tool_mask_np = np.clip(tool_mask_np - wear_mask_np, 0, 1)

                # Convert back to PIL image
                # tool_mask_clean = Image.fromarray((tool_mask_np * 255).astype(np.uint8))

                tool_category_id = self.categories_dict.get(self.tool_color)  # Get tool category ID
                if tool_category_id is not None:
                    annotation = self.create_submask_annotation(tool_mask_np, image_id, tool_category_id, annotation_id, is_crowd)
                    annotations.append(annotation)
                    annotation_id += 1

            # Identify the flank_wear mask
            flank_wear_mask = sub_masks.get(self.flank_color)  
            if flank_wear_mask:
                flank_wear_mask_np = np.array(flank_wear_mask, dtype=np.uint8)  # Convert to NumPy array (binary)
                flank_wear_mask_np = (flank_wear_mask_np > 0).astype(np.uint8)  # Ensure binary values (0 or 1)

                # Subtract wear regions from flank mask
                for color, sub_mask in sub_masks.items():
                    if color != self.flank_color:  # Ignore flank mask itself
                        wear_mask_np = np.array(sub_mask, dtype=np.uint8)
                        wear_mask_np = (wear_mask_np > 0).astype(np.uint8)  # Ensure binary

                        # Subtract wear mask from flank_wear mask and clip to keep valid values
                        wear_mask_np = np.clip(flank_wear_mask_np - wear_mask_np, 0, 1)

                flank_wear_category_id = self.categories_dict.get(self.flank_color)  # Get tool category ID
                if flank_wear_category_id is not None:
                    annotation = self.create_submask_annotation(flank_wear_mask_np, image_id, flank_wear_category_id, annotation_id, is_crowd)
                    annotations.append(annotation)
                    annotation_id += 1

            # Process wear categories separately
            for color, sub_mask in sub_masks.items():
                if color == self.tool_color:
                    continue  # Skip tool since it's already processed

                if color == self.flank_color:
                    continue
                
                category_id = self.categories_dict.get(color)
                if category_id is not None:
                    sub_mask_np = np.array(sub_mask)
                    annotation = self.create_submask_annotation(sub_mask_np, image_id, category_id, annotation_id, is_crowd)
                    annotations.append(annotation)
                    annotation_id += 1

            image_id += 1
        
        print("Annotations are generated.")
        return annotations

    
    def write_annotations(self, info, images, annotations, categories):
        # Write annotations to same folder with the images
        out_dir = self.config["dataset_with_color_maps"]["images_dir"]
        out_file_path = os.path.join(out_dir, self.coco_out_filename + ".json")
        with open(out_file_path, "w") as file:
            json.dump({"info":info, "images":images, "annotations": annotations, "categories":categories}, file, indent=4)
        
        print(f"Annotations saved to {out_file_path}")

    def get_mask_file_names(self, files_dir, input_mask_extension, input_image_extension, mask_indication="_5"):
        if not os.path.isdir(files_dir):
            print(f"Error: Directory '{files_dir}' does not exist or is not accessible.")
            return []
        
        files = os.listdir(files_dir)

        # Filter out only the image files
        image_extensions = [".png", ".jpg", ".bmp"]
        filtered_files = [f for f in files if os.path.isfile(os.path.join(files_dir, f)) and f.lower().endswith(tuple(image_extensions))]

        # To which image file does this annotation belong to?
        if(input_mask_extension == input_image_extension):
            return [f.replace(mask_indication, "") for f in filtered_files]  # Remove mask indication from filenames
            # return filtered_files
        else:
            modified_image_paths = []
            for mask_file in filtered_files:
                # Get the file path and change extension
                clean_name = os.path.splitext(mask_file)[0].replace(mask_indication, "")
                image_path = clean_name + input_image_extension
                modified_image_paths.append(image_path)
            
            return modified_image_paths

    def generate_and_save_coco_annotations(self):
        info = self.generate_info_field()
        images_field = self.generate_images_field()
        annotations = self.generate_annotations_field(self.mask_list)
        categories = self.generate_categories_field()
        self.write_annotations(info, images=images_field, annotations=annotations, categories=categories)

    def generate_images_field(self):
        # Generates "images" field in json
        images = []
        id = 1

        # Assumes all the images in the dataset are in the same size
        im_width, im_height = self.mask_list[0].size[0], self.mask_list[0].size[1]
        
        for m_name in self.mask_file_names_list:
            
            image = {
                'id': id,
                'width': im_width,
                'height': im_height,
                'file_name': m_name,
            }
            images.append(image)

            id += 1

        return images

    def generate_categories_field(self):
        categories = []

        # Get the nested categories dictionary
        categories_metadata = self.metadata["categories"]

        # Loop through each item in the categories dictionary
        for key, value in categories_metadata.items():
            cat_id = value["id"]
            cat_name = key
        
            category = {
                "id": cat_id,
                "name": cat_name
            }
            print(category)
            categories.append(category)

        return categories
    
    def generate_info_field(self):
        info = {
            "description": self.coco_out_filename
        }
        return info
    
    def plot_mask(self, mask):
        # Visualizing the mask using Matplotlib
        plt.imshow(mask, cmap='gray')
        plt.title('Binary Mask Visualization')
        plt.axis('off')  # Hide axes
        plt.show()
    
    @staticmethod
    def combine_two_coco_annotation_files(ann_file1, ann_file2, output_file="merged_annotations.json"):
        """
        Combine two COCO annotation files into one.
        Args:
            ann_file1 (str): Path to the first COCO annotation file.
            ann_file2 (str): Path to the second COCO annotation file.
            output_file (str): Path to save the merged COCO annotation file.
        """
        # Read both annotation files
        f1_annotations = CommonMethods.read_json(ann_file1)
        f2_annotations = CommonMethods.read_json(ann_file2)

        # Get the last image ID in f1 and start numbering from there for f2
        last_im_id = max(img["id"] for img in f1_annotations["images"])
        
        # Create an image ID mapping for images in f2
        image_id_mapping = {}
        for img in f2_annotations["images"]:
            last_im_id += 1
            image_id_mapping[img["id"]] = last_im_id  # Store old->new mapping
            img["id"] = last_im_id  # Update the image ID in-place

        # -----------------------------------------------------------------------
        # Change annotation IDs
        last_ann_id = max(ann["id"] for ann in f1_annotations["annotations"])

        # Create a category ID mapping
        f1_categories = f1_annotations["categories"]
        f2_categories = f2_annotations["categories"]

        mapping_cat = {item["name"]: item["id"] for item in f1_categories}
        correspondence = {item["id"]: mapping_cat[item["name"]] for item in f2_categories}

        # Update annotation IDs and their corresponding image/category IDs
        for annotation in f2_annotations["annotations"]:
            last_ann_id += 1  # Assign a new annotation ID
            annotation["id"] = last_ann_id
            annotation["image_id"] = image_id_mapping[annotation["image_id"]]  # Map to new image ID
            annotation["category_id"] = correspondence[annotation["category_id"]]  # Map category ID

        # -----------------------------------------------------------------------
        # Merge the annotations and images from both files
        combined_annotations = {
            "info": f1_annotations["info"],
            "images": f1_annotations["images"] + f2_annotations["images"],
            "annotations": f1_annotations["annotations"] + f2_annotations["annotations"],
            "categories": f1_categories  # Assuming categories are identical
        }

        # Save the combined annotations to a JSON file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(combined_annotations, f, indent=4, ensure_ascii=False)

        print(f"Merged annotations saved as {output_file}")

    @staticmethod 
    def main():
        coco_annotation_generator = Colormap_to_COCO_Annotation_Generator(config_path="../preprocessor_config.yaml", labels_out_file_name="coco_annotations")
        coco_annotation_generator.generate_and_save_coco_annotations()
    
if __name__ == "__main__":
    Colormap_to_COCO_Annotation_Generator.main()



        



        
            



    




    