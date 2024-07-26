# Use LLM to clean up the format of the saved predictions:
import os
import time
import hashlib
from openai import OpenAI
from tqdm import tqdm
from helpers import save_config, load_config, read_frames
from helpers import read_labels, majority_vote, prediction_template
from helpers import read_paths_from_txt, check_videos_paths
import yaml
from pathlib import Path
import argparse
import random

# =============================================================================
# Use gpt3.5 to clean up the format of the saved predictions.
# 1. Read the template format.
# 2. Trim and change the prediction to follow the format of the template.
# 3. Save the clean version of prediction in the same folder as the prediction.
# =============================================================================
template_path = 'path/to/gt/txt'
prediction_path = 'path/to/prediction/txt'
client = OpenAI(api_key='dummy_key')
model_name = 'gpt-3.5-turbo'

# load the template:
with open(template_path, 'r') as f:
    template = f.read()
template = template.split('\n')  
template = '\n'.join(template[1:]) 
template = template.split('\n')  
template = [line for line in template if line.strip() != '']
template = '\n'.join(template)

# load the prediction:
with open(prediction_path, 'r') as f:
    prediction = f.read()

prompt = [{
    "role": "system",
    "content": '''
    Your job is to make the prediction follow the format of the template. Do not change the content of the prediction.
    '''
}]

content_example = "The following is the template:\n"
content_example += template 
content_example += '\n'
example = {
    "role": "user",
    "content": content_example
}
prompt.append(example)

content_prediction = "The following is the prediction:\n"
content_prediction += prediction
prediction = {
    "role": "user",
    "content": content_prediction
}
prompt.append(prediction)

params = {
    "model": model_name,
    "messages": prompt,
    "max_tokens": 500
}
result = client.chat.completions.create(**params)
prediction = result.choices[0].message.content
print(prediction)
# save the cleaned prediction in the same folder as the prediction:
prediction_path = prediction_path.replace('predictions.txt', 'cleaned_predictions.txt')
with open(prediction_path, 'w') as f:
    f.write(prediction)
