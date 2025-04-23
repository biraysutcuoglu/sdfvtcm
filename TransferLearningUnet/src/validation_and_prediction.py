import os
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model,load_model

class ValidationAndPrediction:

    def plot_histories(histories, save_dir):
        """
        Plot training and validation loss and accuracy for each fold in cross-validation.
        Args:
            histories (list): List of training histories for each fold.
            save_dir (str): Directory to save the plots.
        """
        for h, history in enumerate(histories):
            keys = history.history.keys()
            fig, axs = plt.subplots(1, len(keys)//2, figsize = (25, 5))
            fig.suptitle('No. ' + str(h+1) + ' Fold Training Results' , fontsize=30)

            for k, key in enumerate(list(keys)[:len(keys)//2]):
                training = history.history[key]
                validation = history.history['val_' + key]

                epoch_count = range(1, len(training) + 1)

                axs[k].plot(epoch_count, training, 'r--')
                axs[k].plot(epoch_count, validation, 'b-')
                axs[k].legend(['Training ' + key, 'Validation ' + key])
        
        hist_filename = os.path.join(save_dir, f'{h+1}_trainHistoryDict.pkl')
        with open(hist_filename, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

    def pred_results_tl(TLmodel, test_df, number_of_img, height, width, save_dir, model_name):
        """
        Predict and visualize results using the trained model.
        Args:
            TLmodel (tf.keras.Model): Trained UNet model.
            test_df (pd.DataFrame): DataFrame containing the testing data.
            number_of_img (int): Number of images to predict.
            height (int): Height of input images.
            width (int): Width of input images.
            save_dir (str): Directory to save the predicted images.
            model_name (str): Name of the model.
        """
        os.makedirs(save_dir, exist_ok=True)
        for i in range(number_of_img):
            index = np.random.randint(0,len(test_df.index))
            print("Testing image", i+1, ":", index)

            img = cv2.imread(test_df['filename'].iloc[index])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img = cv2.resize(img, (height, width))
            img = img[np.newaxis, :, :, :]
            img = img / 255
            tl_pred = TLmodel.predict(img)

            plt.figure(figsize=(10,10))
            plt.subplot(1,3,1)
            plt.imshow(np.squeeze(img))
            plt.title('Original Image')
            plt.axis('off')

            plt.subplot(1,3,2)     
            plt.imshow(np.squeeze(cv2.resize(cv2.imread(test_df['mask'].iloc[index]), (height, width))))
            plt.title('Original Mask')
            plt.axis('off')

            plt.subplot(1,3,3)
            plt.imshow(np.squeeze(tl_pred) > .5)
            plt.title(f'Prediction: {model_name}')
            plt.axis('off')
            plt.tight_layout()

            plt.savefig(os.path.join(save_dir, str(i+1) + '_pred.png'), bbox_inches='tight', pad_inches=0.1)
            plt.show()