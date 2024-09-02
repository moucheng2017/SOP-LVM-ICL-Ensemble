#!/bin/bash

# The config file template to be used for the experiment is: /home/moucheng/projects/screen_action_labels/code/action-labelling/configs/wonderbread_icl.yml
CONFIG_TEMPLATE="/home/moucheng/projects/screen_action_labels/code/action-labelling/configs/wonderbread_icl.yml"

# The training data to be used are in the folder: /home/moucheng/projects/screen_action_labels/data/split/Wonderbread/1723811870.4719887/training_data_bs10_sd42
TRAINING_DATA_FOLDER="/home/moucheng/projects/screen_action_labels/data/split/Wonderbread/1723811870.4719887/training_data_bs8_sd42"

# List all txt files in the training data folder
TRAINING_FILES=($(ls $TRAINING_DATA_FOLDER/*.txt))

# In the above List of txt files, the files are called "batch_0.txt", "batch_1.txt", "batch_2.txt", etc.
# Get a new training_batches list containing user specified training batches:
Training_batchs=(3 5 7 9)

# Get the training data files corresponding to the training batches
TRAINING_FILES=()
for batch in "${Training_batchs[@]}"; do
    TRAINING_FILES+=("$TRAINING_DATA_FOLDER/batch_$batch.txt")
done

# # Iterate over each training data file from the first one:
# for TRAIN_FILE in "${TRAINING_FILES[@]}"; do
# Iterate over each training data file from the second one:
for TRAIN_FILE in "${TRAINING_FILES[@]}"; do
    # Create a new config file for each training data file
    NEW_CONFIG_FILE="${CONFIG_TEMPLATE%.yml}_$(basename $TRAIN_FILE .txt).yml"
    cp $CONFIG_TEMPLATE $NEW_CONFIG_FILE
    
    # Replace the train_screenshots_txt in the new config file with the current training data file
    sed -i "s|train_screenshots_txt:.*|train_screenshots_txt: $TRAIN_FILE|" $NEW_CONFIG_FILE
    
    # Run the experiment with the new config file
    # Assuming the experiment command is `run_experiment`
    python main.py --config $NEW_CONFIG_FILE

    # if experiment failed, just stop the whole process:
    if [ $? -ne 0 ]; then
        echo "Experiment failed. Exiting..."
        exit 1
    fi
done