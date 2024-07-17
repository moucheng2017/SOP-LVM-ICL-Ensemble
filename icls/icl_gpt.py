import os
import time
import hashlib
from openai import OpenAI
from tqdm import tqdm
from helpers import save_config, load_config, read_frames
from helpers import read_labels, majority_vote, prediction_template
import yaml
from pathlib import Path
import argparse

def main_gpt(args):
    config = load_config(args.config)
    print('Configurations:')
    print(config)
    client = OpenAI(api_key=config['api_key'])
    
    train_data = config['train_data_path']
    test_data = config['test_data_path']
    resize = config['image_resize']
    save_base_path = Path(config['save_path'])
    
    timestamp = str(int(time.time()))
    current_save_path = save_base_path / timestamp
    current_save_path.mkdir(parents=True, exist_ok=True)
    
    # Save the used config
    save_config(config, current_save_path)
    
    all_train_videos = os.listdir(train_data)
    all_test_videos = os.listdir(test_data)
    train_videos_paths = [os.path.join(train_data, video) for video in all_train_videos]
    if config['debug_mode'] and config['debug_mode'] == True:
        print('Debug mode is on, only testing the last video.')
        # test_videos_paths = [os.path.join(test_data, video) for video in all_test_videos[-2:]]
        test_videos_paths = [os.path.join(test_data, video) for video in all_test_videos[-1:]]
    else:
        print('Testing on all videos.')
        test_videos_paths = [os.path.join(test_data, video) for video in all_test_videos]

    prompt = [{
        "role": "system",
        "content": config['prompts']['system']
    }]

    for video in train_videos_paths:
        frames, number_frames = read_frames(video, resize)
        labels = read_labels(video)
        if labels:
            contents = []
            contents.append(config['prompts']['training_example'].format(number_frames=number_frames))
            for j in range(number_frames):
                contents.append({"image": frames[j], "resize": tuple(resize)})
                contents.append(labels.split('\n')[j])
            frames_labels = {
                "role": "user",
                "content": contents
            }
            prompt.append(frames_labels)

    # print('Testing started..\n')
    for video in tqdm(test_videos_paths, desc="Testing videos"):
        frames, number_frames = read_frames(video, resize)
        contents = []
        contents.append(config['prompts']['testing_example'].format(number_frames=number_frames))
        for j in range(number_frames):
            contents.append({"image": frames[j], "resize": tuple(resize)})
        prediction_template_ = prediction_template(num_frames=number_frames)
        question = config['prompts']['question'] 
        question += "---START FORMAT TEMPLATE---\n"
        question += prediction_template_
        question += "\n---END FORMAT TEMPLATE---\n"
        question += "Do not deviate from the above format."
        # print(question)
        contents.append(question)
        frames_labels = {
            "role": "user",
            "content": contents
        }
        prompt.append(frames_labels)
        
        predictions = []
        num_inferences = config['majority_voting_candidates']
        
        for n in range(num_inferences):
            params = {
                "model": config['model_name'],
                "messages": prompt,
                "max_tokens": config['max_tokens'],
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
                "max_tokens": config['max_tokens'],
                "temperature": config['temperature_self_reflect'],
                "top_p": config['top_p_self_reflect']
            }
            
            reflection_result = client.chat.completions.create(**reflection_params)
            final_prediction = reflection_result.choices[0].message.content
        else:
            final_prediction = initial_prediction
        
        # Make the video_save_path from the video name, the last three elements of the video path
        video_name = video.split('/')[-3:]
        # concatenate the elements of the list to form a string separeted by '/':
        video_name = '/'.join(video_name)
        video_save_path = current_save_path / video_name
        video_save_path.mkdir(parents=True, exist_ok=True)
        # print('video save path: ', video_save_path)
        
        save_path = video_save_path / 'label_prediction.txt'
        with open(save_path, 'w') as f:
            f.write(final_prediction)
            print('The prediction is saved at: ', save_path)
            print('The prediction is: \n', final_prediction)
        
        prompt.pop()
        time.sleep(10)

    print('Testing completed..\n')

