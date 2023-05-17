"""
    Project: Computer Vision Exercise:

        Objective: The this project, we implement an Deep Learning Pixel Classification Model to detect and segment the
        roof of buildings from areal imagery.

    File: inference.py:

        Description: Deploy the Deep Learning model for detecting and segmenting roof-top from areal imagery.

    Author: Mohsen Ghazel

    Date: May 17th, 2022

    Execution: >python inference.py

    Output:
        - Various logging statements for each module and step
        - Intermediate and final results, which are saved to inference sub-folder of the output folder

    Copyright: mghazel copyright 2023
"""

# -------------------------------------------------------------------------------
# Import shared utilities
# -------------------------------------------------------------------------------
from utils import *


def main():
    # Display a message
    print('==================================================================================')
    print('## Roof-Top Detection & Segmentation: Deep Learning Model Inference:   ##')
    print('==================================================================================')
    print('Start of execution: {0}'.format(datetime.datetime.now()))
    print('==================================================================================')

    # -------------------------------------------------------------------------------
    # Step 1: Load the trained model
    # -------------------------------------------------------------------------------
    # The trained model director
    trained_model_files_list = [f for f in glob.glob(trained_model_folder + "*.h5")]
    print(trained_model_folder)
    print(trained_model_files_list)
    # Initialize the model file name
    trained_model_file_name = ""
    if len(trained_model_files_list) == 0:
        print('ERROR: No trained model file was found in directory:: {0}'.format(trained_model_folder))
        # exits the program
        sys.exit("Inference Aborted!")
    elif len(trained_model_files_list) > 1:
        print('Multiple trained model files are found in directory:: {0}'.format(trained_model_folder))
        # Set the model file name to the last file in the list
        trained_model_file_name = trained_model_files_list[-1]
        print('The following trained model file is used: {0}'.format(trained_model_file_name))
        # exits the program
        # sys.exit("Inference Aborted!")
    else:
        # Set the model file name
        trained_model_file_name = trained_model_files_list[0]
        print('A single trained model file is found is: {0}'.format(trained_model_file_name))

    # -------------------------------------------------------------------------------
    # Load the trained model
    # -------------------------------------------------------------------------------
    model = tf.keras.models.load_model(trained_model_file_name)
    print(f"Trained model file: {trained_model_file_name} is loaded successfully!")

    # -------------------------------------------------------------------------------
    # Step 2: Read the input image: image.tif
    # -------------------------------------------------------------------------------
    # - Width: 5000 pixels
    # - Height: 5000 pixels
    # - RGB: 3 Channels
    # -------------------------------------------------------------------------------
    # Read the input image
    print(f"Reading the input image: 'image.tif':")
    img = cv2.imread(DATA_PATH + 'image.tif', cv2.IMREAD_UNCHANGED)
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Visualize the image
    # display_image(img)
    # Display the shape of the input image
    print(f"The input image: 'image.tif' has shape: {img.shape}")

    # -------------------------------------------------------------------------------
    # Step 3: Partition the input image and label-image into smaller sub-images:
    # -------------------------------------------------------------------------------
    # - It is assumed that the GPU used for training and prediction cannot process
    #   patches larger than 512x512x3 as the input at a time
    # - The input image and its associated labels image are too large
    # - We shall partition the large input image and its associated labels-image into:
    #   - 10x10 partitions of sub-images
    #   - Each sub-image has the spatial-size: 512x512
    #   - Consecutive sub-images are overlapping by 6 pixels
    #   - Sub-images along the image-border overlap by 12 pixels.
    # -------------------------------------------------------------------------------
    # Apply Image Partitioning on the Input Image: image.tif:
    # -------------------------------------------------------------------------------
    # Call the functionality to partition the input image:
    # -------------------------------------------------------------------------------
    X = partition_image(img, img_type="image", augment_flag=False, save_images_flag=False)
    # -------------------------------------------------------------------------------
    # Display a message
    # -------------------------------------------------------------------------------
    print(f"The input image partition shape = {X.shape}")

    # -------------------------------------------------------------------------------
    # Step 4: Deploy the trained model to predict partition sub-images labels
    # -------------------------------------------------------------------------------
    # Compute the model predictions for the partition sub-images in X:
    # -------------------------------------------------------------------------------
    X_preds = model.predict(X, verbose=2)
    print(f"Model predictions for image partition have been generated successfully!")

    # -------------------------------------------------------------------------------
    # Apply a threshold to convert the mask to binary:
    # -------------------------------------------------------------------------------
    # - values > 0.5 are assumed to belong to the mask
    # -------------------------------------------------------------------------------
    X_preds = (X_preds > 0.5).astype(np.uint8)

    # -------------------------------------------------------------------------------
    # Step 5: Merge the model predictions of the partition images together to get
    #         the model prediction for the entire input image: image.tif
    # -------------------------------------------------------------------------------
    # Call the functionality to generate the full-image prediction labels.
    pred_labels = generate_full_image_predictions(X_preds, img.shape[0], img.shape[1])
    print(f"Model predictions for the full input image have been generated successfully!")

    # -------------------------------------------------------------------------------
    # Visualize the model predictions for the entire input image: image.tif
    # -------------------------------------------------------------------------------
    # Save the figure
    # file name
    image_labels_preds_filename = inference_output_folder + "model_preds_full_input_image.jpg"
    cv2.imwrite(image_labels_preds_filename, 255 * pred_labels)
    print(f"Model predictions for the full input image have saved to: {image_labels_preds_filename}")

    # -------------------------------------------------------------------------------
    # End of successful model inference:
    # -------------------------------------------------------------------------------
    # Display a successful end of execution message
    # Current time
    now = datetime.datetime.now()
    # Display a message
    print("-------------------------------------------------------------------------------")
    print('Model Deployed Successfully on: ' + str(now.strftime("%Y-%m-%d %H:%M:%S") + ", Good-bye!"))
    print("-------------------------------------------------------------------------------")


###############################################################################
# Execute the main() function:
###############################################################################
if __name__ == "__main__":
    main()

