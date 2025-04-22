'''This class contains common methods that is used in several parts of the project'''
import json
import yaml
from PIL import Image
import os
import sys
import shutil

# Change project root to root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

class CommonMethods:
    image_extensions = (".png", ".jpg", ".bmp")

    @staticmethod
    def read_json(json_file_path):
        try:
            with open(json_file_path) as f:
                data = json.load(f)
            return data
        
        except FileNotFoundError:
            print(f"Error: The file '{json_file_path}' was not found.")
            return None
    
    # For reading config file
    @staticmethod
    def read_yaml(config_file_path):
        # Load YAML configuration
        with open(config_file_path, "r") as file:
            config = yaml.safe_load(file)

        return config
    
    @staticmethod
    def read_images_in_dir(images_dir):
        # reads the images in the given directory 
        # returns the list of images
        images = []

        if not os.path.isdir(images_dir):
            print(f"Error: Directory '{images_dir}' does not exist or is not accessible.")
        
        else:
            files = os.listdir(images_dir)

            # filter only image files
            image_files = [f for f in files if f.lower().endswith(CommonMethods.image_extensions)]

            # Read images with PIL
            for image_file in image_files:
                image_path = os.path.join(images_dir, image_file)
                try:
                    img = Image.open(image_path)
                    images.append(img)
                except Exception as e:
                    print(f"Error loading {image_file}: {e}")
            
            return images

    @staticmethod
    def copy_image_files_to_directory(source_dir, target_dir):
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Error: Directory {source_dir} does not exist.")
        
        os.makedirs(target_dir, exist_ok=True)
        image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(CommonMethods.image_extensions)]

        if not image_files:
            print(f"No image files found in directory {source_dir}!")

        num_im_copied = 0
        for image_file in image_files:
            source_path = os.path.join(source_dir, image_file)
            target_path = os.path.join(target_dir, image_file)

            try:
                shutil.copy2(source_path, target_path)
                num_im_copied +=1 
            except Exception as e:
                print(f"Error copying {image_file}:{e}")
        
        print(f"{num_im_copied} files are copied to {target_dir}.")
    
    @staticmethod
    def convert_to_png_and_copy_image_files_to_directory(source_dir, target_dir):
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Error: Directory {source_dir} does not exist.")
        
        os.makedirs(target_dir, exist_ok=True)
        image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(CommonMethods.image_extensions)]

        if not image_files:
            print(f"No image files found in directory {source_dir} !")

        num_im_copied = 0
        for image_file in image_files:
            
            source_path = os.path.join(source_dir, image_file)
            
            if image_file.lower().endswith(".bmp"):
                png_filename = os.path.splitext(image_file)[0] + ".png"
                target_path = os.path.join(target_dir, png_filename)
                try:
                    with Image.open(source_path) as img:
                        img.save(target_path, "PNG")
                        num_im_copied += 1
                except Exception as e:
                    print(f"Error converting {image_file}:{e}")

            else:
                target_path = os.path.join(target_dir, image_file)
                try:
                    shutil.copy2(source_path, target_path)
                except Exception as e:
                    print(f"Error copying {image_file}: {e}")

        
        print(f"{num_im_copied} files are copied to {target_dir}.")
    

        



        







    



    
