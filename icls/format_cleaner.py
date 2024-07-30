# Use LLM to clean up the format of the saved predictions:
import os
import time
import hashlib
from openai import OpenAI
from tqdm import tqdm
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
template_path = '/home/moucheng/data/Wonderbread/gold_demos/14 @ 2024-01-05-02-12-26/SOP - 14 @ 2024-01-05-02-12-26.txt'
prediction_path = '/home/moucheng/results/1721919139/Wonderbread/gold_demos/69 @ 2024-01-05-04-25-33/label_prediction.txt'
client = OpenAI(api_key='sk-proj-ZxH9n4f7EHjWlCBo0bdjT3BlbkFJzpfDqdqNHisk1b56DZoM')
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
    Your job is to clean up the prediction text. Do not change the content of the prediction. 
    Remove empty lines. 
    Remove any lines which are not enumerated. 
    Remove symbols like "-", "**", etc.  
    Merge the lines that are split into multiple lines for each enumerated item.
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
question = "Please make the prediction follow the format of the template if necessary:"
content_prediction += '\n' + question
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
