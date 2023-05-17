"""
    Project: InVision AI - Machine Learning Exercise:

        Objective: The this project, we implement an Deep Learning Pixel Classification Model to detect and segment the
        roof of buildings from areal imagery.

    File: train.py:

        Description: Train the Deep Learning model for detecting and segmenting roof-top from areal imagery.

    Author: Mohsen Ghazel

    Date: March 23rd, 2022

    Execution: >python train.py

    Output:
        - Various logging statements for each module and step
        - Intermediate and final results, which are saved to train sub-folder of the output folder
        - The trained model saved to trained_model sub-folder of the output folder

    Copyright: InVision.AI copyright 2023
"""


# -------------------------------------------------------------------------------
# Import shared utilities
# -------------------------------------------------------------------------------
from utils import *


def main():
    # Display a message
    print('==================================================================================')
    print('## Roof-Top Detection & Segmentation: Deep Learning Model Training:   ##')
    print('==================================================================================')
    print('Start of execution: {0}'.format(datetime.datetime.now()))
    print('==================================================================================')

    # -------------------------------------------------------------------------------
    # Step 1: Read & Explore the Input Data:
    # -------------------------------------------------------------------------------
    # 1.1: Read & Explore the Input Image: image.tif
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
    # Display the shape of the input image
    print(f"The input image: 'image.tif' has shape: {img.shape}")

    # -------------------------------------------------------------------------------
    # 1.2: Read & Explore the Input Image Labels: labels.tif
    # -------------------------------------------------------------------------------
    # - Width: 5000 pixels
    # - Height: 5000 pixels
    # - Binary: 1 Channel
    # -------------------------------------------------------------------------------
    # Read the input image
    print(f"Reading the input image labels: 'labels.tif':")
    img_labels = cv2.imread(DATA_PATH + 'labels.tif', cv2.IMREAD_UNCHANGED)
    # Visualize the image
    # display_image(img)
    # Display the shape of the input image
    print(f"The input image labels: 'labels.tif' has shape: {img_labels.shape}")

    # -------------------------------------------------------------------------------
    # Step 2: Partition the input image and label-image into smaller sub-images:
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
    # 2.1: Apply Image Partitioning on the Input Image: image.tif:
    # -------------------------------------------------------------------------------
    # Call the functionality to partition the input image:
    X = partition_image(img, img_type="image", augment_flag=True, save_images_flag=False)
    # Display a message
    print(f"The input image partition shape = {X.shape}")

    # -------------------------------------------------------------------------------
    # 2.2: Apply Image Partitioning on the Input Image Labels: labels.tif:
    # -------------------------------------------------------------------------------
    # Call the functionality to partition the input image Labels:
    y = partition_image(img_labels, img_type="labels", augment_flag=True, save_images_flag=False)
    # Display a message
    print(f"The input image partition shape = {y.shape}")

    # -------------------------------------------------------------------------------
    # 2.3: Visualize sample partitioned images and their labels:
    # -------------------------------------------------------------------------------
    # Set the plot file name
    plt_filename = train_output_folder + "sample_images_plus_labels.jpg"
    # Call the function to visualize and save the sample results
    visualize_sample_results(X, y, [], 10, 2, plt_filename)

    # -------------------------------------------------------------------------------
    # Step 3: Build the the Image Segmentation Model:
    # -------------------------------------------------------------------------------
    # 3.1: Define the Model Layers, as Specified by InVision.AI:
    # -------------------------------------------------------------------------------
    # You are asked to implement the following network:
    # 1. Input RGB Image ->
    # 2. Conv 3x3, 16 filters -> ReLu ->
    # 3. Conv 5x5, 16 filters-> ReLu ->
    # 4. Max Pooling ->
    # 5. Conv 3x3, 32 filters -> ReLu ->
    # 6. Transposed Conv 2x2, 32 filters ->
    # 7. Conv 3x3, 16 filters ->
    # 8. Output
    # -------------------------------------------------------------------------------
    # Define sequential layers of the model:
    # -------------------------------------------------------------------------------
    # 1. Input layer with image size: (INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, INPUT_IMAGE_CHANNELS)
    # -------------------------------------------------------------------------------
    inputs = tf.keras.layers.Input((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, INPUT_IMAGE_CHANNELS))
    # -------------------------------------------------------------------------------
    # 2. Normalize the input image to the interval: [0,1]
    # -------------------------------------------------------------------------------
    s0 = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
    # -------------------------------------------------------------------------------
    # 3. Conv 3x3, 16 filters -> ReLu ->
    # -------------------------------------------------------------------------------
    s1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s0)
    # -------------------------------------------------------------------------------
    # 4. Convolution layer with Conv 5x5, 16 filters-> ReLu ->  and preserve image size
    # -------------------------------------------------------------------------------
    s2 = tf.keras.layers.Conv2D(16, (5, 5), activation='relu', kernel_initializer='he_normal', padding='same')(s1)
    # -------------------------------------------------------------------------------
    # 5. Max-Pooling: Apply 2x2 max-pooling
    # -------------------------------------------------------------------------------
    s3 = tf.keras.layers.MaxPooling2D((2, 2))(s2)
    # -------------------------------------------------------------------------------
    # 6. Convolution layer with Conv 3x3, 32 filters -> ReLu -> and preserve image size
    # -------------------------------------------------------------------------------
    s4 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s3)
    # -------------------------------------------------------------------------------
    # 7. Convolution layer with Transposed Conv 2x2, 32 filters -> and preserve image size
    # -------------------------------------------------------------------------------
    s5 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(s4)
    # -------------------------------------------------------------------------------
    # 8. Convolution layer with Conv 3x3, 16 filters -> and preserve image size
    # -------------------------------------------------------------------------------
    s6 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s5)
    # -------------------------------------------------------------------------------
    # 9. Output layer:
    # -------------------------------------------------------------------------------
    # - Use sigmoid activation function because of binary classification.
    # -------------------------------------------------------------------------------
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(s6)

    # -------------------------------------------------------------------------------
    # 3.2: Construct the Keras model using the above defined layers:
    # -------------------------------------------------------------------------------
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    # -------------------------------------------------------------------------------
    # 3.3: Compile the model:
    # -------------------------------------------------------------------------------
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # -------------------------------------------------------------------------------
    # 3.4: Print the model summary:
    # -------------------------------------------------------------------------------
    print(model.summary())

    # -------------------------------------------------------------------------------
    # Step 4: Train the model:
    # -------------------------------------------------------------------------------
    # 4.1: Define callbacks:
    # -------------------------------------------------------------------------------
    # Define callbacks for early for:
    # - Early stopping
    # - Monitoring training
    # -------------------------------------------------------------------------------
    # 4.1.1: Save the model trained model checkpoint in case of failure or early termination
    # -------------------------------------------------------------------------------
    # Save the best trained model checkpoint
    # the full-path file name
    saved_model_file_name = trained_model_folder + 'roof_top_detection_trained_model_early_stop.h5'
    checkpointer = tf.keras.callbacks.ModelCheckpoint(saved_model_file_name, verbose=1, save_best_only=True)

    # -------------------------------------------------------------------------------
    # 4.1.2: Early stopping and Tensorboard monitoring:
    # -------------------------------------------------------------------------------
    # Stop training if validation accuracy does not improved after 10 consecutive epochs
    # - Save files in logs for tensorboard monitoring
    # -------------------------------------------------------------------------------
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss'),
                 tf.keras.callbacks.TensorBoard(log_dir='logs')]

    # -------------------------------------------------------------------------------
    # 4.2: Split data into training and validation subsets:
    # -------------------------------------------------------------------------------
    # - Training: 80%
    # - Validation: 20%
    # -------------------------------------------------------------------------------
    # Split the data into training and validation subsets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, shuffle=True, random_state=RANDOM_STATE)
    # The number of training images
    num_train_images = X_train.shape[0]
    # The number of validation images
    num_valid_images = X_valid.shape[0]
    # Display a message
    print("-------------------------------------------------------------------------")
    print("Training & Validation Data Sets:")
    print("-------------------------------------------------------------------------")
    print("Number of Training Images = ", num_train_images)
    print("Number of Validation Images = ", num_valid_images)
    print("-------------------------------------------------------------------------")

    # -------------------------------------------------------------------------------
    # 4.3: Train the model:
    # -------------------------------------------------------------------------------
    # Train the model for the specified number of training epochs and batch-size:
    results = model.fit(X_train,  # resized training images
                        y_train,  # resized training masks
                        validation_data=(X_valid, y_valid),  # (validation images, validation masks)
                        batch_size=TRAIN_MODEL_BATCH_SIZE,  # batch size
                        epochs=TRAIN_MODEL_NUM_EPOCHS,  # the number of training epochs
                        verbose=2,  # verbose: level of logging details
                        callbacks=callbacks)  # callbacks functions

    # -------------------------------------------------------------------------------
    # 4.4: Save the trained Model for deployment:
    # -------------------------------------------------------------------------------
    # the full-path file name
    saved_model_file_name = trained_model_folder + 'roof_top_detection_trained_model_num_epochs_' + \
                            str(TRAIN_MODEL_NUM_EPOCHS) + '.h5'
    # Save the trained model
    model.save(saved_model_file_name)

    # -------------------------------------------------------------------------------
    # Step 5: Trained Model Performance Evaluation:
    # -------------------------------------------------------------------------------
    # - Evaluate the performance of the trained model.
    # -------------------------------------------------------------------------------
    # 5.1: Training and Validation Accuracy and Loss Plots:
    # -------------------------------------------------------------------------------
    # 5.1.1: Display the results history keys
    # -------------------------------------------------------------------------------
    print(f"Results.history.keys() = \n{results.history.keys()}")

    # -------------------------------------------------------------------------------
    # 5.1.2: Get the performance metrics:
    # -------------------------------------------------------------------------------
    acc = results.history['accuracy']
    val_acc = results.history['val_accuracy']
    loss = results.history['loss']
    val_loss = results.history['val_loss']

    # -------------------------------------------------------------------------------
    # 5.1.3: Set the Epochs
    # -------------------------------------------------------------------------------
    epochs = range(1, len(acc) + 1)

    # -------------------------------------------------------------------------------
    # 5.1.4: Accuracy plots
    # -------------------------------------------------------------------------------
    # Specify the overall grid size
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, acc, 'r-', linewidth=3, marker='o', markersize=10, label='Training')
    plt.plot(epochs, val_acc, 'g-', linewidth=3, marker='d', markersize=10, label='Validation')
    plt.xlabel("Epochs", fontsize=12)
    # Set label locations.
    plt.xticks(np.arange(0, len(epochs) + 1, step=5))
    plt.ylabel("Accuracy", fontsize=12)
    plt.title('Training and Validation Accuracy', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid()
    # Save the figure
    # file name
    plt_filename = train_output_folder + "train_valid_accuracy_vs_epochs.jpg"
    plt.savefig(plt_filename)

    # -------------------------------------------------------------------------------
    # 5.1.5: Loss plots
    # -------------------------------------------------------------------------------
    # specify the overall grid size
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, loss, 'r-', linewidth=3, marker='o', markersize=10, label='Training')
    plt.plot(epochs, val_loss, 'g-', linewidth=3, marker='d', markersize=10, label='Validation')
    plt.xlabel("Epochs", fontsize=12)
    # Set label locations.
    plt.xticks(np.arange(0, len(epochs) + 1, step=5))
    plt.ylabel("Accuracy", fontsize=12)
    plt.title('Training and Validation Loss', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid()
    # plt.show();
    # Save the figure
    # file name
    plt_filename = train_output_folder + "train_valid_loss_vs_epochs.jpg"
    plt.savefig(plt_filename)

    # -------------------------------------------------------------------------------
    # 5.2: Visualize sample model predictions:
    # -------------------------------------------------------------------------------
    # Visualize some of the predictions of the trained model on:
    #   - Sample training images
    #   - Sample validation images.
    # -------------------------------------------------------------------------------
    # 5.2.1: Sample Training Images:
    # -------------------------------------------------------------------------------
    # Model predictions for 10 randomly selected training images.
    # Compute the model predictions for the training images:
    train_preds = model.predict(X_train, verbose=2)
    # Apply a threshold to convert the mask to binary:
    # - values > 0.5 are assumed to belong to the mask
    train_preds = (train_preds > 0.5).astype(np.uint8)

    # -------------------------------------------------------------------------------
    # - Visualize 10 training images and their associated
    #   ground-truth masks as well as their predicted masks:
    # -------------------------------------------------------------------------------
    # Set the plot file name
    plt_filename = train_output_folder + "model_preds_10_train_images.jpg"
    # Call the function to visualize and save the sample results
    visualize_sample_results(X_train, y_train, train_preds, 10, 3, plt_filename)

    # -------------------------------------------------------------------------------
    # 5.2.2: Sample Validation Images:
    # -------------------------------------------------------------------------------
    # Model predictions for 10 randomly selected validation images.
    # Compute the model predictions for the validation images:--
    valid_preds = model.predict(X_valid, verbose=2)
    # Apply a threshold to convert the mask to binary:
    # - values > 0.5 are assumed to belong to the mask
    valid_preds = (valid_preds > 0.5).astype(np.uint8)

    # -------------------------------------------------------------------------------
    # - Visualize 10 validation images and their associated
    #   ground-truth masks as well as their predicted masks:
    # -------------------------------------------------------------------------------
    # Set the plot file name
    plt_filename = train_output_folder + "model_preds_10_valid_images.jpg"
    # Call the function to visualize and save the sample results
    visualize_sample_results(X_valid, y_valid, valid_preds, 10, 3, plt_filename)

    # -------------------------------------------------------------------------------
    # 5.3: Confusion Matrix and Classification Report:
    # -------------------------------------------------------------------------------
    # Compute the Confusion Matrix and Classification Report for:
    # - The training images
    # - The validation images.
    # -------------------------------------------------------------------------------
    # 5.3.1: Performance Evaluation using the Training Images:
    # -------------------------------------------------------------------------------
    # 5.3.1.1: Compute the Confusion Matrix using the Training Images:
    # -------------------------------------------------------------------------------
    # Get the true and predicted labels for the Training Images:
    #  - Reshape the training images true masks
    y_train_true = np.squeeze(y_train.reshape(-1, ))
    # Reshape the training images predicted masks
    y_train_pred = np.squeeze(train_preds.reshape(-1, ))

    # -------------------------------------------------------------------------------
    # Compute the Confusion Matrix
    # -------------------------------------------------------------------------------
    # Compute confusion matrix
    cnf_matrix_train = confusion_matrix(y_train_true, y_train_pred)

    # -------------------------------------------------------------------------------
    # Plot the Confusion Matrix
    # -------------------------------------------------------------------------------
    # Plot Non-Normalized Confusion Matrix
    # -------------------------------------------------------------------------------
    plt.figure()
    plot_confusion_matrix(cnf_matrix_train, classes=['Roof-Top(1)', 'Background(0)'],
                          normalize=False,
                          title='Roof-Top Detection: Non-Normalized Confusion Matrix',
                          cmap=plt.cm.Blues)
    # Save the figure
    # file name
    plt_filename = train_output_folder + "non_normalized_confusion_matrix_train_images.jpg"
    plt.savefig(plt_filename)

    # -------------------------------------------------------------------------------
    # Plot Normalized Confusion Matrix
    # -------------------------------------------------------------------------------
    plt.figure()
    plot_confusion_matrix(cnf_matrix_train, classes=['Roof-Top(1)', 'Background(0)'],
                          normalize=True,
                          title='Roof-Top Detection: Normalized Confusion Matrix',
                          cmap=plt.cm.Greens)
    # Save the figure
    # file name
    plt_filename = train_output_folder + "normalized_confusion_matrix_train_images.jpg"
    plt.savefig(plt_filename)

    # -------------------------------------------------------------------------------
    # 5.3.1.2: Compute the Classification Report using the Training Images:
    # -------------------------------------------------------------------------------
    # Print the Classification Report for the Training Images:
    # Set the number of decimal places
    np.set_printoptions(precision=3)
    # Compute the classification report
    classification_report_train = classification_report(y_train_true, y_train_pred, output_dict=True)

    print('-------------------------------------------------------------------------')
    print(f"The Training Images Classification Report:")
    print('-------------------------------------------------------------------------')
    print(classification_report_train)
    print('-------------------------------------------------------------------------\n')

    # -------------------------------------------------------------------------------
    # Save the Classification Report to CSV file:
    # -------------------------------------------------------------------------------
    clsf_report_train = pd.DataFrame(classification_report_train).transpose()
    csv_file_name = train_output_folder + 'classification_report_train_images.csv'
    clsf_report_train.to_csv(csv_file_name, index=True)

    # -------------------------------------------------------------------------------
    # 5.3.2: Performance Evaluation using the Validation Images:
    # -------------------------------------------------------------------------------
    # 5.3.2.1: Compute the Confusion Matrix using the validation Images:
    # -------------------------------------------------------------------------------
    # Get the true and predicted labels for the Validation Images:
    # -------------------------------------------------------------------------------
    # Reshape the validation images true masks
    y_valid_true = np.squeeze(y_valid.reshape(-1, ))
    # Reshape the validation images predicted masks
    y_valid_pred = np.squeeze(valid_preds.reshape(-1, ))

    # -------------------------------------------------------------------------------
    # Compute the Confusion Matrix
    # -------------------------------------------------------------------------------
    # Compute confusion matrix
    cnf_matrix_valid = confusion_matrix(y_valid_true, y_valid_pred)

    # -------------------------------------------------------------------------------
    # Plot the Confusion Matrix
    # -------------------------------------------------------------------------------
    # Plot Non-Normalized Confusion Matrix
    # -------------------------------------------------------------------------------
    plt.figure()
    plot_confusion_matrix(cnf_matrix_valid, classes=['Roof-Top(1)', 'Background(0)'],
                          normalize=False,
                          title='Roof-Top Detection: Non-Normalized Confusion Matrix',
                          cmap=plt.cm.Blues)
    # Save the figure
    # file name
    plt_filename = train_output_folder + "non_normalized_confusion_matrix_valid_images.jpg"
    plt.savefig(plt_filename)

    # -------------------------------------------------------------------------------
    # Plot Normalized Confusion Matrix
    # -------------------------------------------------------------------------------
    plt.figure()
    plot_confusion_matrix(cnf_matrix_valid, classes=['Roof-Top(1)', 'Background(0)'],
                          normalize=True,
                          title='Roof-Top Detection: Normalized Confusion Matrix',
                          cmap=plt.cm.Greens)
    # Save the figure
    # file name
    plt_filename = train_output_folder + "normalized_confusion_matrix_valid_images.jpg"
    plt.savefig(plt_filename)

    # -------------------------------------------------------------------------------
    #  5.3.2.2: Compute the Classification Report using the Validation Images:
    # -------------------------------------------------------------------------------
    # Print the Classification Report for the validation Images:
    # -------------------------------------------------------------------------------
    # Set the number of decimal places
    np.set_printoptions(precision=3)
    # Compute the classification report
    classification_report_valid = classification_report(y_valid_true, y_valid_pred, output_dict=True)

    print('-------------------------------------------------------------------------')
    print(f"The Validation Images Classification Report:")
    print('-------------------------------------------------------------------------')
    print(classification_report_valid)
    print('-------------------------------------------------------------------------\n')

    # -------------------------------------------------------------------------------
    # Save the Classification Report to CSV file:
    # -------------------------------------------------------------------------------
    clsf_report_valid = pd.DataFrame(classification_report_valid).transpose()
    csv_file_name = train_output_folder + 'classification_report_valid_images.csv'
    clsf_report_valid.to_csv(csv_file_name, index=True)

    # -------------------------------------------------------------------------------
    # Step 6. Generate the Model Prediction for the full Input Image and Evaluate Performance:
    # -------------------------------------------------------------------------------
    # - Merge the model predictions of the image partition 100 sub-images to obtain the model prediction
    #   of the input image: image.tif.
    # - Evaluate the model performance by comparing the model prediction to the ground-truth mask: labels.tif.
    # -------------------------------------------------------------------------------
    # 6.1: Get the 100 sub-images of the ```image.tif``` image partition and their associated labels:
    # -------------------------------------------------------------------------------
    # The 100 images have multiple of 4 indices in features matrix: X
    # -------------------------------------------------------------------------------
    partition_images = X[0:-1:4]
    # Get their labels
    partition_images_labels = y[0:-1:4]
    # Get the number of partition images
    num_partition_images = partition_images.shape[0]
    # Display a message
    print(f"The input image partition has {num_partition_images} images.")

    # -------------------------------------------------------------------------------
    # 6.2: Generate the model predictions of the partition images and visualize sample results:
    # -------------------------------------------------------------------------------
    # Compute the model predictions for the image partition sub-images:
    partition_images_preds = model.predict(partition_images, verbose=2)
    # Apply a threshold to convert the mask to binary:
    # - values > 0.5 are assumed to belong to the mask
    partition_images_preds = (partition_images_preds > 0.5).astype(np.uint8)

    # -------------------------------------------------------------------------------
    # - Visualize 10 partition sub-images and their associated
    #   ground-truth masks as well as their predicted masks:
    # -------------------------------------------------------------------------------
    # Set the plot file name
    plt_filename = train_output_folder + "model_preds_10_sub_images.jpg"
    # Call the function to visualize and save the sample results
    visualize_sample_results(partition_images, partition_images_labels, partition_images_preds, 10, 3, plt_filename)

    # -------------------------------------------------------------------------------
    # 6.3: Merge the model predictions of the partition images together to get
    #      the model prediction for the entire input image: image.tif
    # -------------------------------------------------------------------------------
    # Call the functionality to generate the full-image prediction labels.
    pred_labels = generate_full_image_predictions(partition_images_preds, img_labels.shape[0], img_labels.shape[1])

    # -------------------------------------------------------------------------------
    # 6.4: Visualize the model predictions for the entire input image: image.tif
    #      and compare it to the  ground-truth mask: labels.tif.
    # -------------------------------------------------------------------------------
    # Save pred_labels mask
    # -------------------------------------------------------------------------------
    # - Set the figure file name
    filename = train_output_folder + "model_preds_full_input_image.jpg"
    cv2.imwrite(filename, 255 * pred_labels)

    # -------------------------------------------------------------------------------
    # Save the ground-truth mask: labels.tif
    # -------------------------------------------------------------------------------
    # - Set the figure file name
    filename = train_output_folder + "labels.jpg"
    cv2.imwrite(filename, img_labels)

    # -------------------------------------------------------------------------------
    # 6.5: Evaluate the model performance using the entire image (image.tif) and its
    #      associated ground-truth (labels.tif).
    # -------------------------------------------------------------------------------
    # 6.5.1: Compute the Confusion Matrix using the entire image (image.tif) and its
    #        associated ground-truth (labels.tif).
    # -------------------------------------------------------------------------------
    # Get the true and predicted labels for the entire image:
    # - Reshape the entire image true mask
    img_labels = img_labels > 0
    y_image_true = np.squeeze(img_labels.reshape(-1, ))
    # Reshape the entire image predicted mask
    y_image_pred = np.squeeze(pred_labels.reshape(-1, ))

    # Compute the Confusion Matrix
    cnf_matrix_image = confusion_matrix(y_image_true, y_image_pred)

    # -------------------------------------------------------------------------------
    # Plot the Confusion Matrix
    # -------------------------------------------------------------------------------
    # Plot Non-Normalized Confusion Matrix
    # -------------------------------------------------------------------------------
    plt.figure()
    plot_confusion_matrix(cnf_matrix_image, classes=['Roof-Top(1)', 'Background(0)'],
                          normalize=False,
                          title='Roof-Top Detection: Non-Normalized Confusion Matrix',
                          cmap=plt.cm.Blues)

    # Save the figure
    # file name
    plt_filename = train_output_folder + "non_normalized_confusion_matrix_input_image.jpg"
    plt.savefig(plt_filename)

    # -------------------------------------------------------------------------------
    # Plot Normalized Confusion Matrix
    # -------------------------------------------------------------------------------
    plt.figure()
    plot_confusion_matrix(cnf_matrix_image, classes=['Roof-Top(1)', 'Background(0)'],
                          normalize=True,
                          title='Roof-Top Detection: Normalized Confusion Matrix',
                          cmap=plt.cm.Greens)

    # Save the figure
    # file name
    plt_filename = train_output_folder + "normalized_confusion_matrix_input_image.jpg"
    plt.savefig(plt_filename)

    # -------------------------------------------------------------------------------
    # 6.5.2: Compute the Classification Report using the entire image (image.tif)
    #        and its associated ground-truth (labels.tif).
    # -------------------------------------------------------------------------------
    # Print the Classification Report for the Validation Images:
    # -------------------------------------------------------------------------------
    # Set the number of decimal places
    np.set_printoptions(precision=3)
    # Compute the classification report
    classification_report_image = classification_report(y_image_true, y_image_pred, output_dict=True)

    print('-------------------------------------------------------------------------')
    print(f"The Entire Image Classification Report:")
    print('-------------------------------------------------------------------------')
    print(classification_report_image)
    print('-------------------------------------------------------------------------\n')

    # -------------------------------------------------------------------------------
    # Save the Classification Report to CSV file:
    # -------------------------------------------------------------------------------
    clsf_report_valid = pd.DataFrame(classification_report_image).transpose()
    csv_file_name = train_output_folder + 'classification_report_input_image.csv'
    clsf_report_valid.to_csv(csv_file_name, index=True)

    # -------------------------------------------------------------------------------
    # 6.6: End of successful model training:
    # -------------------------------------------------------------------------------
    # Display a successful end of execution message
    # Current time
    now = datetime.datetime.now()
    # Display a message
    print("-------------------------------------------------------------------------------")
    print('Model trained successfully on: ' + str(now.strftime("%Y-%m-%d %H:%M:%S") + ", Good-bye!"))
    print("-------------------------------------------------------------------------------")


###############################################################################
# Execute the main() function:
###############################################################################
if __name__ == "__main__":
    main()

