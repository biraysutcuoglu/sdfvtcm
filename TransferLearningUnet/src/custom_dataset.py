import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class CustomDataset(Dataset):

    def __init__(self, images_dir, masks_dir, binary_masks_dir, mask_type="colormask",transform=None):
        self.image_paths = self.read_images_in_dir(images_dir)
        self.mask_paths = self.read_images_in_dir(masks_dir)

        if mask_type == "colormask":
            CustomDataset.save_binary_masks(self.mask_paths, binary_masks_dir)
            self.binary_mask_paths = self.read_images_in_dir(binary_masks_dir)
        elif mask_type == "binary":
            self.binary_mask_paths = self.mask_paths

        self.images_array = self.get_images_array_np()

        self.transform = transform

    def read_images_in_dir(self, img_dir):
        return sorted([os.path.join(img_dir, img) for img in os.listdir(img_dir) if img.endswith(".png")])

    def image_display(image_pathlist, mask_pathlist, index):
        # input: image and mask filepath and the index 
        # output matplotlib images 
        image = Image.open(image_pathlist[index])
        imagearray = np.array(image)
        print('Image shape: ', imagearray.shape)

        mask = Image.open(mask_pathlist[index])
        maskarray = np.array(mask)
        print('Mask shape: ', maskarray.shape)

        fig, ax = plt.subplots(3,figsize=(5,10))
        ax[0].imshow(imagearray, aspect='auto', cmap='gray')
        ax[1].imshow(maskarray, aspect='auto', cmap='gray')
        ax[2].imshow(imagearray, aspect='auto', cmap = 'gray')
        ax[2].imshow(maskarray, cmap = 'Reds', aspect='auto', alpha = 0.4)
        
    def create_df(image_list, mask_list):
        df_ = pd.DataFrame(data={"filename": image_list, "mask": mask_list})
        df = df_.sample(frac=1).reset_index(drop=True)

        return df
    
    def get_images_array_np(self):
        images_array = []
        for img_path in self.image_paths:
            img = Image.open(img_path)
            img = np.array(img)
            images_array.append(img)

        return np.array(images_array)

    def convert_to_binary_masks(mask_paths):
        binary_masks = []
        for mask_path in mask_paths:
            mask = Image.open(mask_path)
            mask_array = np.array(mask)

            # If the mask is already binary
            if len(mask_array.shape) == 2 and np.array_equal(np.unique(mask_array), [0, 1]):
                binary_masks.append(mask_array.astype(np.uint8))
                continue

            # Extract RGB channels
            red_channel = mask_array[:, :, 0]
            green_channel = mask_array[:, :, 1]
            blue_channel = mask_array[:, :, 2]

            # Create a binary mask
            binary_mask = (red_channel == 255) & (green_channel == 0) & (blue_channel == 0)
            binary_masks.append(binary_mask.astype(np.uint8))
        
        return binary_masks
    
    def save_binary_masks(mask_paths, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        for mask_path in mask_paths:
            mask = Image.open(mask_path)
            mask_array = np.array(mask)

            # If the mask is binary
            if len(mask_array.shape) == 2 and np.array_equal(np.unique(mask_array), [0, 255]):
                binary_mask = mask_array.astype(np.uint8)
            
            # If the mask is rgb
            elif len(mask_array.shape) == 3 and mask_array.shape[2] == 3:
                # Extract RGB channels
                red_channel = mask_array[:, :, 0]
                green_channel = mask_array[:, :, 1]
                blue_channel = mask_array[:, :, 2]

                # Create a binary mask
                binary_mask = (red_channel == 255) & (green_channel == 0) & (blue_channel == 0)
                # Convert boolean array to uint8 and scale (0 → 0, 1 → 255)
                binary_mask = (binary_mask * 255).astype(np.uint8)

            else:
                raise ValueError(f"Unexpected mask format: {mask_path}")

            filename = os.path.basename(mask_path)
            save_path = os.path.join(save_dir, filename)
            binary_mask = Image.fromarray(binary_mask).save(save_path)
    
    # Normalising image pixel values to range 0-1 and convert masks pixels to 1 or 0 only (binarize)
    def adjust_data(img,mask):
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        
        return (img, mask)
    
    def train_generator(data_frame, batch_size, train_path, aug_dict,
        save_img_dir, save_mask_dir,
        image_save_prefix, mask_save_prefix,
        save_format,
        target_size,
        image_color_mode="rgb",
        mask_color_mode="grayscale",
        seed=1):
        # return: generator type object
        '''
        Generate image and mask at the same time using the same seed for
        image_datagen and mask_datagen to ensure the transformation for image
        and mask is the same 
        ''' 
        # Apply same augmentations to both image and mask
        image_datagen = ImageDataGenerator(**aug_dict)
        mask_datagen = ImageDataGenerator(**aug_dict)
        
        image_generator = image_datagen.flow_from_dataframe(
            data_frame,
            directory = train_path,
            x_col = "filename",
            class_mode = None,
            color_mode = image_color_mode,
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = save_img_dir,
            save_prefix  = image_save_prefix,
            save_format = save_format, 
            seed = seed)

        mask_generator = mask_datagen.flow_from_dataframe(
            data_frame,
            directory = train_path,
            x_col = "mask",
            class_mode = None,
            color_mode = mask_color_mode,
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = save_mask_dir,
            save_prefix  = mask_save_prefix,
            save_format = save_format, 
            seed = seed)

        train_gen = zip(image_generator, mask_generator)

        # normalize images and masks
        for (img, mask) in train_gen:
            img, mask = CustomDataset.adjust_data(img, mask)
            yield (img,mask)

    
    



