"""
    Project: InVision AI - Machine Learning Exercise:

        Objective: The this project, we implement an Deep Learning Pixel Classification Model to detect and segment the
        roof of buildings from areal imagery.

    File: utils.py:

        Description: Contains shared Python imports, global variables and utility functions.

    Author: Mohsen Ghazel

    Date: March 23rd, 2022

    Execution: None

    Output: None

    Copyright: InVision.AI copyright 2023
"""

# ===============================================================================
# 1. Initial Setup:
# ===============================================================================
# - Python Imports
# - Global Variables Definitions
# - Check for GPU devices
# -------------------------------------------------------------------------------
# 1.1: Python Imports:
# -------------------------------------------------------------------------------
# numpy
import numpy as np

# matplotlib
import matplotlib.pyplot as plt

# OpenCV
import cv2

# tensorflow
import tensorflow as tf

# sklearn imports
# - Needed for splitting the dataset into training and testing subsets
from sklearn.model_selection import train_test_split
# Needed for model performance evaluation
from sklearn.metrics import classification_report, confusion_matrix

# Needed for formatting and plotting the confusion matrix
import itertools

# I/O
import os
# Needed for sys.exist(")
import sys

# datetime
import datetime

# random
import random

# import pandas
import pandas as pd

# logging
import logging

# Needed for file search in a folder
import glob

# -------------------------------------------------------------------------------
# 1.2: Global variables:
# -------------------------------------------------------------------------------
# Set the Numpy pseudo-random generator at a fixed value:
# - This ensures repeatable results everytime you run the code.
RANDOM_SEED = 41

# Set the random seed
np.random.seed = RANDOM_SEED

# Set the random state to 41
# - This ensures repeatable results everytime you run the code.
RANDOM_STATE = 41

# Set the data folder containing the image.tiff and its binary mask: labels.tif
DATA_PATH = "../data/"

# Set the output directory
OUTPUT_PATH = "../output/"
# Create the folder if it does not exist
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
    print("Directory '%s' created successfully" % OUTPUT_PATH)

# Create training output sub-folder where training results are saved
train_output_folder = os.path.join(OUTPUT_PATH, 'training', '')
# Create the folder if it does not exist
if not os.path.exists(train_output_folder):
    os.makedirs(train_output_folder)
    print("Directory '%s' created successfully" % train_output_folder)

# Create trained-model sub-folder: where the trained model is saved
trained_model_folder = os.path.join(OUTPUT_PATH, 'trained_model', '')
# Create the folder if it does not exist
if not os.path.exists(trained_model_folder):
    os.makedirs(trained_model_folder)
    print("Directory '%s' created successfully" % trained_model_folder)

# Create inference output sub-folder where inference results are saved
inference_output_folder = os.path.join(OUTPUT_PATH, 'inference', '')
# Create the folder if it does not exist
if not os.path.exists(inference_output_folder):
    os.makedirs(inference_output_folder)
    print("Directory '%s' created successfully" % inference_output_folder)

# Set the input images size: RGB images with size: 128x128 pixels
# input image-width
INPUT_IMAGE_WIDTH = 512
# input image-width
INPUT_IMAGE_HEIGHT = 512
# input image-channels
INPUT_IMAGE_CHANNELS = 3

# Set the number of sub-images in the input image partition
NUM_PARTITIONED_SUB_IMAGES = 100

# The input image partition number of rows
IMAGE_PARTITION_NUM_ROWS = 10

# The input image partition number of columns
IMAGE_PARTITION_NUM_COLUMNS = 10

# The number of rows/columns in each partition cell
IMAGE_PARTITION_CELL_HEIGHT_WIDTH = 500

# The overlaps between neighboring partition sub-images (in pixels)
OVERLAP = 6

# -------------------------------------------------------------------------------
# %%%%%%%%%%%%%%%%%%%%%%%% EDIT MODELS HYPER-PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%
# -------------------------------------------------------------------------------
# Edit the number of training epochs
# -------------------------------------------------------------------------------
TRAIN_MODEL_NUM_EPOCHS = 100

# -------------------------------------------------------------------------------
# Edit the batch-size
# -------------------------------------------------------------------------------
TRAIN_MODEL_BATCH_SIZE = 8

# -------------------------------------------------------------------------------
# 1.3: Check for GPU devices
# -------------------------------------------------------------------------------
# TensorFlow refers to the CPU on your local machine as /device:CPU:0 and
# to the first GPU as /GPU:0—additional GPUs will have sequential numbering:
#  - By default, if a GPU is available, TensorFlow will use it for all operations.
# -------------------------------------------------------------------------------
# Display the number of GPUs available
print("------------------------------------------------------------------------")
print("The number of GPUs Available on your device: ", len(tf.config.list_physical_devices('GPU')))
print("------------------------------------------------------------------------")


# ===============================================================================
# 2. Shared utilities functionalities
# ===============================================================================


def display_image(img):
    """
    Displays the input image.
      Parameters:
        img (numpy.ndarray): the input grayscale or RGB image
      Returns:
        None.
    """
    # Create a figure and set its size
    plt.figure(figsize=(10, 10))
    # Set axes off
    plt.axis('off')
    # Display the image
    # If it is an RGB image
    if len(img.shape) > 2:
        plt.imshow(img)
    else:  # For a Grayscale image
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)


def visualize_sample_results(imgs, labels, preds, num_rows, num_cols, plt_filename):
    """
    Displays sample results consisting of num_rows randomly selected images and
    their associated labels and model predictions

        Parameters:
            imgs (numpy.ndarray): array of sub-images
            labels (numpy.ndarray):: array of labels
            preds (numpy.ndarray):: array of model predictions
            num_rows (int): The number rows in the plot grid
            num_cols (int): The number cols in the plot grid
            plt_filename: the file name of teh saved plot

        Returns:
            None.
    """
    # -------------------------------------------------------------------------------
    # Visualize randomly selected sub-images and their associated
    # ground-truth masks and possibly their model predicted masks.
    # -------------------------------------------------------------------------------
    # specify the overall grid size
    plt.figure(figsize=(5 * num_cols, 5 * num_rows))
    # iterate over the 30 images
    for i in range(0, num_rows * num_cols, num_cols):
        # ---------------------------------------------------------------------------
        # The sub-image counter:
        # ---------------------------------------------------------------------------
        # - Generate a random index between 0 and num_partition_images
        image_counter = random.randint(0, imgs.shape[0] - 1)
        # ---------------------------------------------------------------------------
        # step 1: create the subplot for the next image
        # ---------------------------------------------------------------------------
        plt.subplot(num_rows, num_cols, i + 1)
        # display the image
        plt.imshow(imgs[image_counter])
        # figure title
        plt.title("Sub-image #: " + str(image_counter), fontsize=10)
        # turn-off axes
        plt.axis('off')
        # ---------------------------------------------------------------------------
        # step 2: create the subplot for the ground-truth image
        # ---------------------------------------------------------------------------
        plt.subplot(num_rows, num_cols, i + 2)
        # display the image
        plt.imshow(np.squeeze(labels[image_counter]), cmap='gray')
        # figure title
        plt.title("Labels", fontsize=10)
        # turn-off axes
        plt.axis('off')
        # ---------------------------------------------------------------------------
        # step 3: create the subplot for the model prediction image
        # ---------------------------------------------------------------------------
        if num_cols == 3:
            plt.subplot(num_rows, num_cols, i + 3)
            # display the image
            plt.imshow(np.squeeze(preds[image_counter]), cmap='gray')
            # figure title
            plt.title("Model Predictions", fontsize=10)
            # turn-off axes
            plt.axis('off')

    # Save the figure
    plt.savefig(plt_filename)


def partition_image(img, img_type="image", augment_flag=True, save_images_flag=False):
    """
    Partition the input image into:
      - 10x10 partitions of sub-images
      - Each sub-image has the spatial-size: 512x512
      - Consecutive sub-images are overlapping by 6 pixels
      - Sub-images along the image-border overlap by 12 pixels.

      Parameters:
        img (numpy.ndarray): the input RGB image or binary labels image
        img_type (str): the type of the input image: ("image" or "labels")
        augment_flag (bool): a flag to indicate whether perform data
                            augmentation by applying horizontal and
                            vertical flips of the sub-images
        save_images_flag (bool): a flag to indicate whether to save the
                                 extracted sub-images

      Returns:
        images_partition (numpy.ndarray): 4D numpy array containing the images
                                          partition.
    """
    # ---------------------------------------------------------------------------
    # Step 1: Initialize the output: images_partition
    # ---------------------------------------------------------------------------
    # The input image partition
    if img_type == "image":
        # -------------------------------------------------------------------------
        # Allocate a data structure to store the partitioned sub-images:
        # -------------------------------------------------------------------------
        # 4D numpy array to store the images:
        # - The number of partitioned images: 4*NUM_PARTITIONED_SUB_IMAGES
        # - Each sub-image has size:
        #   - INPUT_IMAGE_HEIGHT x INPUT_IMAGE_WIDTH x INPUT_IMAGE_CHANNELS
        # -------------------------------------------------------------------------
        images_partition = np.zeros((4 * NUM_PARTITIONED_SUB_IMAGES,
                                     INPUT_IMAGE_HEIGHT,
                                     INPUT_IMAGE_WIDTH,
                                     INPUT_IMAGE_CHANNELS), dtype=np.uint8)
    else:  # The input labels image partition
        # -------------------------------------------------------------------------
        # 3D numpy array to store the masks:
        # - The number of training images: (len(train_ids)
        # - Each training grayscale/binary image-mask will be resized to:
        #   - INPUT_IMAGE_HEIGHT x INPUT_IMAGE_WIDTH x 1
        # -------------------------------------------------------------------------
        images_partition = np.zeros((4 * NUM_PARTITIONED_SUB_IMAGES,
                                     INPUT_IMAGE_HEIGHT,
                                     INPUT_IMAGE_WIDTH, 1),
                                    dtype=bool)

    # ---------------------------------------------------------------------------
    # Step 2: Partition the input image
    # ---------------------------------------------------------------------------
    # Initialize the image number
    img_num = 0
    # Iterate over the partition number of rows
    for row in range(IMAGE_PARTITION_NUM_ROWS):
        # row start value
        if row == 0:
            row_start = 0
        elif row < IMAGE_PARTITION_NUM_ROWS - 1:
            row_start = row * IMAGE_PARTITION_CELL_HEIGHT_WIDTH - OVERLAP
        else:
            row_start = row * IMAGE_PARTITION_CELL_HEIGHT_WIDTH - 2 * OVERLAP
        # Iterate over the partition number of columns
        for col in range(IMAGE_PARTITION_NUM_COLUMNS):
            # col start value
            if col == 0:
                col_start = 0
            elif col < IMAGE_PARTITION_NUM_COLUMNS - 1:
                col_start = col * IMAGE_PARTITION_CELL_HEIGHT_WIDTH - OVERLAP
            else:
                col_start = col * IMAGE_PARTITION_CELL_HEIGHT_WIDTH - 2 * OVERLAP

            # -----------------------------------------------------------------------
            # Extract the sub-image
            # -----------------------------------------------------------------------
            # The input image partition
            if img_type == "image":
                sub_img = img[row_start: row_start + INPUT_IMAGE_HEIGHT, col_start: col_start + INPUT_IMAGE_WIDTH, :]
            else:
                sub_img = img[row_start: row_start + INPUT_IMAGE_HEIGHT, col_start: col_start + INPUT_IMAGE_WIDTH]
            # -----------------------------------------------------------------------
            # Store the extracted image in images_partition:
            # -----------------------------------------------------------------------
            # The input image partition
            if img_type == "image":
                images_partition[img_num] = sub_img
            else:
                images_partition[img_num] = sub_img.reshape(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, 1)

            # -----------------------------------------------------------------------
            # Data Augmentation: Apply Horizontal and Vertical flips on the image, if desired
            # -----------------------------------------------------------------------
            if augment_flag:
                # -----------------------------------------------------------------------
                # Flip the image horizontally:
                # -----------------------------------------------------------------------
                sub_img_h_flip = cv2.flip(sub_img, 1)
                # -----------------------------------------------------------------------
                # Store the cropped image in images_partition:
                # -----------------------------------------------------------------------
                # Increment the image number
                img_num += 1
                # The input image partition
                if img_type == "image":
                    images_partition[img_num] = sub_img_h_flip
                else:
                    images_partition[img_num] = sub_img_h_flip.reshape(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, 1)

                # -----------------------------------------------------------------------
                # Flip the image vertically:
                # -----------------------------------------------------------------------
                sub_img_v_flip = cv2.flip(sub_img, 0)
                # -----------------------------------------------------------------------
                # Store the cropped image in images_partition:
                # -----------------------------------------------------------------------
                # Increment the image number
                img_num += 1
                # The input image partition
                if img_type == "image":
                    images_partition[img_num] = sub_img_v_flip
                else:
                    images_partition[img_num] = sub_img_v_flip.reshape(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, 1)

                # -----------------------------------------------------------------------
                # Flip the image both horizontally and vertically
                # -----------------------------------------------------------------------
                sub_img_hv_flip = cv2.flip(sub_img, -1)
                # -----------------------------------------------------------------------
                # Store the cropped image in images_partition:
                # -----------------------------------------------------------------------
                # Increment the image number
                img_num += 1
                # The input image partition
                if img_type == "image":
                    images_partition[img_num] = sub_img_hv_flip
                else:
                    images_partition[img_num] = sub_img_hv_flip.reshape(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, 1)

            # -----------------------------------------------------------------------
            # Increment the image number
            # -----------------------------------------------------------------------
            img_num += 1

            # -----------------------------------------------------------------------
            # Save the sub-images if desired
            # -----------------------------------------------------------------------
            if save_images_flag:
                # ---------------------------------------------------------------------
                # The extracted sub-image
                # ---------------------------------------------------------------------
                # the full-path file
                sub_img_file_name = train_output_folder + img_type + "_sub_img_" + str(row) + 'x' + str(col) + '.jpg'
                # save the sub-image
                cv2.imwrite(sub_img_file_name, sub_img)

                # -----------------------------------------------------------------------
                # Data Augmentation: Horizontal and Vertical flips on the image:
                # -----------------------------------------------------------------------
                if augment_flag:
                    # ---------------------------------------------------------------------
                    # The extracted sub-image: After horizontal-flip
                    # ---------------------------------------------------------------------
                    # the full-path file
                    sub_img_file_name = train_output_folder + img_type + "_sub_img_" + str(row) + 'x' + str(
                        col) + '_h_flip.jpg'
                    # save the sub-image
                    cv2.imwrite(sub_img_file_name, sub_img_h_flip)

                    # ---------------------------------------------------------------------
                    # The extracted sub-image: After vertical-flip
                    # ---------------------------------------------------------------------
                    # the full-path file
                    sub_img_file_name = train_output_folder + img_type + "_sub_img_" + str(row) + 'x' + str(
                        col) + '_v_flip.jpg'
                    # save the sub-image
                    cv2.imwrite(sub_img_file_name, sub_img_v_flip)

                    # ---------------------------------------------------------------------
                    # The extracted sub-image: After horizontal+vertical-flip
                    # ---------------------------------------------------------------------
                    # the full-path file
                    sub_img_file_name = train_output_folder + img_type + "_sub_img_" + str(row) + 'x' + str(
                        col) + '_hv_flip.jpg'
                    # save the sub-image
                    cv2.imwrite(sub_img_file_name, sub_img_hv_flip)

    # return
    return images_partition


def generate_full_image_predictions(partition_images_preds, num_rows, num_cols):
    """
    Merge the predictions obtained from the partition sub-images to obtain the full-image model predictions

          Parameters:
            partition_images_preds (numpy.ndarray): the predictions obtained from the partition sub-images
            num_rows (int): the number of rows
            num_cols (int): the number of cols

          Returns:
            full_img_pred_labels (numpy.ndarray): the full-image model prediction labels.
        """
    # -------------------------------------------------------------------------------
    # Merge the model predictions of the partition images together to get
    # the model prediction for the entire input image: image.tif
    # -------------------------------------------------------------------------------
    # Initialize the full-image prediction labels mask
    full_img_pred_labels = np.zeros((num_rows, num_cols), dtype=bool)

    # The partition image number
    partition_img_num = 0
    # Iterate over the partition number of rows
    for row in range(IMAGE_PARTITION_NUM_ROWS):
        # row start value
        if row == 0:
            row_start = 0
        elif row < IMAGE_PARTITION_NUM_ROWS - 1:
            row_start = row * IMAGE_PARTITION_CELL_HEIGHT_WIDTH - OVERLAP
        else:
            row_start = row * IMAGE_PARTITION_CELL_HEIGHT_WIDTH - 2 * OVERLAP
        # Iterate over the partition number of columns
        for col in range(IMAGE_PARTITION_NUM_COLUMNS):
            # Get the next partition image prediction mask
            partition_image_preds = np.squeeze(partition_images_preds[partition_img_num])
            # col start value
            if col == 0:
                col_start = 0
            elif col < IMAGE_PARTITION_NUM_COLUMNS - 1:
                col_start = col * IMAGE_PARTITION_CELL_HEIGHT_WIDTH - OVERLAP
            else:
                col_start = col * IMAGE_PARTITION_CELL_HEIGHT_WIDTH - 2 * OVERLAP

            # ---------------------------------------------------------------------------
            # Update the sub-image pred_labels
            # ---------------------------------------------------------------------------
            # Get the corresponding sub-image of full_img_pred_labels
            pred_labels_sub_img = full_img_pred_labels[row_start: row_start + INPUT_IMAGE_HEIGHT,
                                  col_start: col_start + INPUT_IMAGE_WIDTH]

            # Update the full-image pred-labels
            full_img_pred_labels[row_start: row_start + INPUT_IMAGE_HEIGHT, col_start: col_start + INPUT_IMAGE_WIDTH] =\
                partition_image_preds + pred_labels_sub_img

            # increment partition_img_num
            partition_img_num += 1

    # Ensure that this a binary mask
    full_img_pred_labels = full_img_pred_labels > 0

    # return the full-image prediction labels
    return full_img_pred_labels


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    Plots the formatted confusion matrix in color.

        Parameters:
                cm (numpy.ndarray): the confusion matrix
                classes (list): the classes
                normalize (bool): a flag indicating whether or not to normalize the matrix
                title (str): the figure title
                cmap (cmap: the color-map

          Returns:
            None.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        # print('Confusion matrix, without normalization')
        pass

    # Display the confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # title
    plt.title(title)
    # add a color-bar
    plt.colorbar()
    # set the x-ticks and y-ticks
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # Set the floating numbers format
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    # Adjust the layout
    plt.tight_layout()
    # Axes labels
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


