import os
import time
from openai import OpenAI
from tqdm import tqdm
from helpers import save_config, load_config, read_frames
from helpers import read_labels, majority_vote, prediction_template
from helpers import read_paths_from_txt, check_videos_paths
import yaml
from pathlib import Path
import argparse
import random
import base64
import cv2
import sys

def main_gpt(args):
    config = load_config(args.config)
    print('Configurations:')
    print(config)

    if config.get("api_key") == None:
        print('Please export the OpenAI API key as an environment variable: export API_KEY=your_dummy_api_key')
        return
    else:
        API_KEY = config['api_key']
    
    client = OpenAI(api_key=API_KEY)
    
    train_screenshots = config['train_screenshots_txt']
    test_screenshots = config['test_screenshots_txt']

    train_screenshots_paths = read_paths_from_txt(train_screenshots)
    test_screenshots_paths = read_paths_from_txt(test_screenshots)
    
    resize = config['image_resize']
    save_base_path = Path(config['save_path'])
    
    # Save the used config
    if config.get("resume_testing_path") and config["resume_testing_path"] != None:
        current_save_path = Path(config["resume_testing_path"])
        save_config(config, current_save_path)
    elif config.get("resume_testing_path") and config["resume_testing_path"] == False:
        current_save_path = Path(config["resume_testing_path"])
        save_config(config, current_save_path)
    else:
        timestamp = str(int(time.time()))
        current_save_path = save_base_path / timestamp
        current_save_path.mkdir(parents=True, exist_ok=True)
        save_config(config, current_save_path)

    train_videos_paths = [path.rsplit('/', 1)[0] for path in train_screenshots_paths]
    test_videos_paths = [path.rsplit('/', 1)[0] for path in test_screenshots_paths]

    if config['debug_mode'] and config['debug_mode'] == True:
        print('Debug mode is on, only testing the last video.')
        test_videos_paths = test_videos_paths[-1:]
        assert len(test_videos_paths) == 1
        # train_videos_paths = train_videos_paths[:1]
    else:
        print('Testing on all videos.')

    prompt = [{
        "role": "system",
        "content": config['prompts']['system']
    }]

    if config.get("in_context_learning") and config["in_context_learning"] == True: 
        print('Using in-context learning..')
        effective_train_videos_number = config['effective_train_videos_number']
        number_train_videos = len(train_videos_paths)
        if number_train_videos > effective_train_videos_number:
            train_videos_paths_ = random.sample(train_videos_paths, effective_train_videos_number)
        else:
            train_videos_paths_ = train_videos_paths
        
        with open(current_save_path / 'train_videos_paths.txt', 'w') as f:
            for item in train_videos_paths_:
                f.write("%s\n" % item)
        
        for video in train_videos_paths_:
            frames, number_frames = read_frames(video, resize)
 
            # Add the training start prompt to the prompt:
            prompt.append(
                {
                    "role": "user",
                    "content": config['prompts']['training_example']
                }
            )

            # Add the frames to the prompt:
            images = []
            for j in range(number_frames):
                images.append(
                    {"type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{frames[j]}"}
                    }
                )
            prompt.append(
                {"role": "user",
                "content": images}
            )

            # Add the labels to the prompt:
            labels = read_labels(video)
            # remove first line:
            labels = labels.split('\n') 
            labels = '\n'.join(labels[1:])
            labels = labels.split('\n') 
            # remove empty lines: 
            labels = [line for line in labels if line.strip() != '']
            labels = '\n'.join(labels)
            prompt.append({
                "role": "user",
                "content": labels
            })

        print('In-context learning completed..\n')
    
    print('Testing started..\n')


    for video in tqdm(test_videos_paths, desc="Testing videos"):
        prompt_test_index = 0
        frames, number_frames = read_frames(video, resize)

        # Add the system prompt to the prompt:
        prompt.append(
            {
                "role": "user",
                "content": config['prompts']['testing_example'].format(number_frames=number_frames)
            }
        )
        prompt_test_index += 1

        # Add the frames to the prompt:
        images = []
        for j in range(number_frames):
            images.append(
                {"type": "image_url",
                 "image_url": {
                     "url": f"data:image/png;base64,{frames[j]}"}
                 }
            )
        prompt.append({
            "role": "user",
            "content": images
        })
        prompt_test_index += 1

        # Add the question to the prompt:
        prompt.append({
            "role": "user",
            "content": config['prompts']['question']
        })
        prompt_test_index += 1

        predictions = []
        num_inferences = config['majority_voting_candidates']
        
        for n in range(num_inferences):
            params = {
                "model": config['model_name'],
                "messages": prompt,
                "max_tokens": int(config['max_tokens']),
                "temperature": config['temperature_majority_voting'],
                "top_p": config['top_p_majority_voting']
            }
            result = client.chat.completions.create(**params)
            prediction = result.choices[0].message.content
            predictions.append(prediction)
            time.sleep(2)
        
        if num_inferences > 1:
            print('Using majority voting to select the final prediction..')
            initial_prediction = majority_vote(predictions)
        else:
            initial_prediction = predictions[0]

        if config['use_self_reflect'] and config['use_self_reflect'] == True:
            print('Using self-reflection to refine the prediction..')
            reflection_prompt = prompt.copy()
            reflection_prompt.append({
                "role": "user",
                "content": config['prompts']['reflection'].format(initial_prediction=initial_prediction)
            })
            
            reflection_params = {
                "model": config['model_name'],
                "messages": reflection_prompt,
                "max_tokens": int(config['max_tokens']),
                "temperature": config['temperature_self_reflect'],
                "top_p": config['top_p_self_reflect']
            }
            
            reflection_result = client.chat.completions.create(**reflection_params)
            final_prediction = reflection_result.choices[0].message.content
        else:
            final_prediction = initial_prediction
        
        video_name = video.split('/')[-3:]
        video_name = '/'.join(video_name)
        video_save_path = Path(current_save_path) / video_name
        video_save_path.mkdir(parents=True, exist_ok=True)
        print('Testing on video: ', video)
        save_path = video_save_path / 'label_prediction.txt'
        with open(save_path, 'w') as f:
            f.write(final_prediction)
        
        for i in range(prompt_test_index):
            prompt.pop()

        time.sleep(10)


    print('Testing completed..\n')


