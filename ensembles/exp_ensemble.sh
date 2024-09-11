# A bash script to run a function with a config from a list of configs one by one until it finishes successfully.

# The experiment command to run:
experiment_cmd="python ensemble_gpt.py"

# The list of configs to run the experiment with:
configs=(
  "../configs_ensemble/exp2.yml"
  "../configs_ensemble/exp3.yml"
  "../configs_ensemble/exp4.yml"
  "../configs_ensemble/exp1.yml"
)

# Run the experiment with each config in the list
for config in "${configs[@]}"; do
  # Run the experiment command with the current config
  $experiment_cmd --config $config
  
  # Check the exit status of the command
  if [ $? -eq 0 ]; then
    # Exit status 0 means the command was successful
    echo "Experiment finished successfully."
  else
    # Exit status not 0 means the command failed
    echo "Experiment failed."
  fi
done