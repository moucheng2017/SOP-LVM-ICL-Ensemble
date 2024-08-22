#!/bin/bash

# Define the command to run the experiment
experiment_cmd="python main.py --config configs/wonderbread.yml"

# Run the experiment in a loop until it finishes successfully
while true; do
  # Run the experiment command
  $experiment_cmd
  
  # Check the exit status of the command
  if [ $? -eq 0 ]; then
    # Exit status 0 means the command was successful
    echo "Experiment finished successfully."
    break
  else
    # Exit status not 0 means the command failed
    echo "Experiment failed. Retrying in 30 seconds..."
    sleep 30
  fi
done
