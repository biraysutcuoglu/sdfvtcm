import os

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.models import Model,load_model
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten

from tensorflow.keras.models import Model, load_model
from tensorflow.keras import Input
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

from src.model_unet import ModelUnet
from src.custom_dataset import CustomDataset

class ModelTrainer:
    """
    Class for training and evaluating a UNet model for image segmentation.
    """
    def train_model(df, train_df, test_df, unet_model, 
                    batch_size, epochs, 
                    train_generator_args,aug_img_dir, aug_mask_dir, aug_img_prefix, aug_mask_prefix, aug_format, 
                    height, width,
                    model_dir):
        """
        Train the UNet model using the provided training data.
        Retrieved from: https://github.com/dorltcheng/Transfer-Learning-U-Net-Deep-Learning-for-Lung-Ultrasound-Segmentation/blob/main/V_Unet/V_Unet_v1_2.ipynb 
        Args:
            df (pd.DataFrame): DataFrame containing the training data.
            train_df (pd.DataFrame): DataFrame containing the training data.
            test_df (pd.DataFrame): DataFrame containing the testing data.
            unet_model (tf.keras.Model): UNet model to be trained.
            batch_size (int): Batch size for training.
            epochs (int): Number of epochs to train the model.
            train_generator_args (dict): Arguments for the training generator.
            aug_img_dir (str): Directory for augmented images.
            aug_mask_dir (str): Directory for augmented masks.
            aug_img_prefix (str): Prefix for augmented images.
            aug_mask_prefix (str): Prefix for augmented masks.
            aug_format (str): Format for saving augmented images and masks.
            height (int): Height of input images.
            width (int): Width of input images.
            model_dir (str): Directory to save the trained model.
        Returns:
            history (tf.keras.callbacks.History): Training history.
            checkpoint_path (str): Path to the saved model checkpoint.
        """
        train_gen = CustomDataset.train_generator(train_df, batch_size, 
                          None, 
                          train_generator_args,
                          aug_img_dir, aug_mask_dir, 
                          aug_img_prefix, aug_mask_prefix,
                          aug_format,
                          (height, width))

        test_gen = CustomDataset.train_generator(test_df, batch_size,
                                None, 
                                dict(),
                                None, None, None, None, None, 
                                (height, width))

        # Train the model with `.fit_generator()`
        unet_model.compile(optimizer = Adam(learning_rate = 1e-5), loss=ModelUnet.dice_coef_loss, 
                            metrics=[ModelUnet.iou, ModelUnet.dice_coef, 'binary_accuracy'])

        callbacks_list, checkpoint_path = ModelTrainer.generate_callbackslist(model_dir)

        print("Model input shape:",unet_model.input_shape)
        x_batch, y_batch = next(train_gen)
        print(f"x_batch shape: {x_batch.shape}")  # Should be (batch_size, 256, 256, 3)
        print(f"y_batch shape: {y_batch.shape}")  # Should be (batch_size, 256, 256, 1) or (batch_size, 256, 256, num_classes)

        # keras function for training the model
        history = unet_model.fit(train_gen,
                                            steps_per_epoch=len(df)//batch_size,
                                            epochs=epochs,
                                            callbacks=callbacks_list,
                                            validation_data = test_gen, 
                                            validation_steps = len(test_df)//batch_size,
                                            verbose=1)
        return history, checkpoint_path
    
    def evaluate_model(checkpoint_path, test_df, batch_size, height, width, history=None):
        """
        Evaluate the UNet model using the given testing data.
        Args:
            checkpoint_path (str): Path to the saved model checkpoint.
            test_df (pd.DataFrame): DataFrame containing the testing data.
            batch_size (int): Batch size for evaluation.
            height (int): Height of input images.
            width (int): Width of input images.
            history (tf.keras.callbacks.History, optional): Training history. Defaults to None.
        Returns:
            TLmodel (tf.keras.Model): Loaded UNet model.
            metrics (dict): Evaluation metrics.
            histories (list): List of training histories.
        """
        # Evaluate the generator 
        histories = []
        losses = []
        accuracies = []
        dicecoefs = []
        ious = []

        TLmodel = load_model(checkpoint_path, 
                            custom_objects={'dice_coef_loss': ModelUnet.dice_coef_loss, 'iou': ModelUnet.iou, 'dice_coef': ModelUnet.dice_coef})
        TLmodel.compile(optimizer=Adam(learning_rate = 1e-5), loss=ModelUnet.dice_coef_loss,
                        metrics=[ModelUnet.iou, ModelUnet.dice_coef, 'binary_accuracy'])

        evaluate_gen = CustomDataset.train_generator(test_df, batch_size,
                                    None, 
                                    dict(),
                                    None, None, None, None, None, 
                                    (height, width))

        results = TLmodel.evaluate(evaluate_gen, 
                                            steps=len(test_df)//batch_size,
                                            verbose=1,
                                            return_dict=True)

        # Store results with their names
        metrics = {
            "binary_accuracy": results['binary_accuracy'],
            "loss": results['loss'],
            "dice_coef": results['dice_coef'],
            "iou": results['iou']
        }

        if history is not None:
            histories.append(history)

        accuracies.append(("binary_accuracy", results['binary_accuracy']))
        losses.append(("loss", results['loss']))
        dicecoefs.append(("dice_coef", results['dice_coef']))
        ious.append(("iou", results['iou']))
        ModelTrainer.print_evaluation_results(histories, accuracies, losses, dicecoefs, ious)

        return TLmodel, metrics, histories

    def print_evaluation_results(histories, accuracies, losses, dicecoefs, ious):
        print('Evaluation scores from pretrained model:')
        print('Accuracy: ', accuracies)
        print('Loss: ', losses)
        print('Dice coefficient: ', dicecoefs)
        print('IOU: ', ious)

    def generate_callbackslist(model_dir):
        """
        Generate a list of callbacks for training the model.
        Args:
            model_dir (str): Directory to save the model checkpoints.
        Returns:
            callbacks_list (list): List of callbacks.
            checkpoint_path (str): Path to the saved model checkpoint.
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        checkpoint_path = os.path.join(model_dir, 'tool_dataset.keras')
        model_checkpoint = ModelCheckpoint(checkpoint_path,  
                                            verbose=1,
                                            monitor='val_loss',
                                            save_best_only=True)

        csvlogger_filename = os.path.join(model_dir, 'training_log.csv')
        pretrain_csvlogger = CSVLogger(filename=csvlogger_filename, separator=",", append=True)
        callbacks_list = [model_checkpoint, pretrain_csvlogger]

        return callbacks_list, checkpoint_path
    
