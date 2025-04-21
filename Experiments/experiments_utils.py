import os
import sys
import random
import shutil

# Change project root to root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

from DatasetPreprocessor.src.colormap_to_coco_annotation_generator import Colormap_to_COCO_Annotation_Generator

class ExperimentUtils:
    
    @staticmethod
    def select_n_images_and_masks(num_images, images_dir, masks_dir):
        '''
        Randomly selects given number of images from the given directory and returns the selected names of the files
        '''
        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
            raise FileNotFoundError(f"Error: Directory {images_dir} does not exist.")
        
        im_filenames_filtered  = [f for f in os.listdir(images_dir) if f.lower().endswith(".bmp")]
        
        random_files = random.sample(im_filenames_filtered, num_images)
        return random_files
    
    @staticmethod
    def copy_selected_images(random_files, images_dir, output_dir):
        '''
        Copies the given files with the given filenames to target directory
        '''
        os.makedirs(output_dir, exist_ok=True)

        num_images_copied = 0
        for file_name in random_files:
            src_path = os.path.join(images_dir, file_name)
            dest_path = os.path.join(output_dir, file_name)

            shutil.copy2(src_path, dest_path)
            num_images_copied += 1
        
        print(f"Copied {num_images_copied} images to {output_dir}")
    
    @staticmethod
    def copy_selected_masks(random_files, masks_dir, output_dir):
        '''
        Checks the masks directory for finding corresponding masks of the given image file names and copies the found masks to given directory
        '''
        os.makedirs(output_dir, exist_ok=True)
        
        num_masks_copied = 0
        for file_name in random_files:
            name, ext = os.path.splitext(file_name)
            mask_name = f"{name}_5.png"
            
            mask_path = os.path.join(masks_dir, mask_name)
            output_path = os.path.join(output_dir, mask_name)

            if os.path.exists(masks_dir):
                shutil.copy2(mask_path, output_path)
                num_masks_copied += 1
            else:
                print(f"Warning: Mask '{mask_name}' not found in {masks_dir}")
        
        print(f"Copied {num_masks_copied} masks to {output_dir}")
    
    @staticmethod
    def generate_coco_annotations(config_file_path, labels_out_filename):
        '''
        Generates COCO annotations from masks and saves them to given output directory
        '''
        coco_annotation_generator = Colormap_to_COCO_Annotation_Generator(config_file_path, labels_out_filename)
        coco_annotation_generator.generate_and_save_coco_annotations()

    @staticmethod
    def merge_coco_annotations(real_ann_file_path, gen_ann_file_path, output_file_path):
        '''
        Merges COCO annotations of given two COCO annotation files
        '''
        Colormap_to_COCO_Annotation_Generator.combine_two_coco_annotation_files(real_ann_file_path, gen_ann_file_path, output_file_path)

    


            


