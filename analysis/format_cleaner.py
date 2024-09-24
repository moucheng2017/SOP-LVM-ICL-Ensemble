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

# A lot of times, the SOPs are generated in a different format than the template, this is to clean it.

# =============================================================================
# Use gpt3.5 to clean up the format of the saved predictions.
# 1. Read the template format.
# 2. Trim and change the prediction to follow the format of the template.
# 3. Save the clean version of prediction in the same folder as the prediction.
# =============================================================================
# change here:
template_path = '/home/moucheng/data/Wonderbread/gold_demos/14 @ 2024-01-05-02-12-26/SOP - 14 @ 2024-01-05-02-12-26.txt'
template_path2 = '/home/moucheng/data/Wonderbread/gold_demos/1 @ 2023-12-25-15-44-04/SOP - 1 @ 2023-12-25-15-44-04.txt'
template_path3 = '/home/moucheng/data/Wonderbread/gold_demos/63 @ 2024-01-05-04-12-52/SOP - 63 @ 2024-01-05-04-12-52.txt'
client = OpenAI(api_key='some-key')
model_name = 'gpt-3.5-turbo'
prediction_folder = '/home/moucheng/results/1721919139/Wonderbread/gold_demos'

## code:
prediction_folder_videos = os.listdir(prediction_folder)
prediction_folder_videos = [os.path.join(prediction_folder, video) for video in prediction_folder_videos]
for i, video in tqdm(enumerate(prediction_folder_videos)):
    # prediction_path = '/home/moucheng/results/1721919139/Wonderbread/gold_demos/69 @ 2024-01-05-04-25-33/label_prediction.txt'
    prediction_path = os.path.join(video, 'label_prediction.txt')
    # load the template:
    with open(template_path, 'r') as f:
        template = f.read()
    template = template.split('\n')  
    template = '\n'.join(template[1:]) 
    template = template.split('\n')  
    template = [line for line in template if line.strip() != '']
    template = '\n'.join(template)

    # load the template2:
    with open(template_path2, 'r') as f:
        template2 = f.read()
    template2 = template2.split('\n')
    template2 = '\n'.join(template2[1:])
    template2 = template2.split('\n')
    template2 = [line for line in template2 if line.strip() != '']
    template2 = '\n'.join(template2)

    # load the template3:
    with open(template_path3, 'r') as f:
        template3 = f.read()
    template3 = template3.split('\n')
    template3 = '\n'.join(template3[1:])
    template3 = template3.split('\n')
    template3 = [line for line in template3 if line.strip() != '']
    template3 = '\n'.join(template3)

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

    content_example2 = "The following is the template2:\n"
    content_example2 += template2
    content_example2 += '\n'
    example2 = {
        "role": "user",
        "content": content_example2
    }
    prompt.append(example2)

    content_example3 = "The following is the template3:\n"
    content_example3 += template3
    content_example3 += '\n'
    example3 = {
        "role": "user",
        "content": content_example3
    }
    prompt.append(example3)

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

    # Use majority voting to get the final prediction:
    predictions = []
    for i in range(5):
        result = client.chat.completions.create(**params)
        prediction = result.choices[0].message.content
        predictions.append(prediction)
        time.sleep(2)
    prediction = max(set(predictions), key=predictions.count)
    print(prediction)
    # save the cleaned prediction in the same folder as the prediction:
    # prediction_path = prediction_path.replace('predictions.txt', 'cleaned_predictions.txt')
    with open(prediction_path, 'w') as f:
        f.write(prediction)
