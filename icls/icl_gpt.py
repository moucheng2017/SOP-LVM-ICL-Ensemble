import os
import time
import openai
# from openai import OpenAI
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
import traceback

def openai_completion(client, params) -> str:
    try:
        response = client.chat.completions.create(
            **params,
            stream=False
        )
    except openai.RateLimitError:
        print("Rate limit exceeded -- waiting 25 min before retrying")
        time.sleep(1500)
        return openai_completion(client, params)
    except openai.APIError as e:
        traceback.print_exc()
        print(f"OpenAI API error: {e}")
        if 'Error code: 500' in str(e) or 'Error code: 503' in str(e): # 500, 503 are openai internal server errors
            time.sleep(1800)
            print('Retrying after 30 minutes..')
            return openai_completion(client, params)
        else:
            raise e
    except Exception as e:
        traceback.print_exc()
        print(f"Unknown error: {e}")
        raise e
    return response.choices[0].message.content

def main_gpt(args):
    config = load_config(args.config)
    print('Configurations:')
    print(config)

    if config.get("api_key") == None:
        print('Please export the OpenAI API key as an environment variable: export API_KEY=your_dummy_api_key')
        return
    else:
        API_KEY = config['api_key']
    
    # errors: tuple = (openai.RateLimitError,),
    
    client = openai.OpenAI(api_key=API_KEY)
    
    train_screenshots = config['train_screenshots_txt']
    test_screenshots = config['test_screenshots_txt']

    # # check if the file name in train_screenshots is screenshots.txt
    # assert train_screenshots.split('/')[-1] == 'screenshots.txt'
    # assert test_screenshots.split('/')[-1] == 'screenshots.txt'

    train_screenshots_paths = read_paths_from_txt(train_screenshots)
    test_screenshots_paths = read_paths_from_txt(test_screenshots)
    
    resize = config['image_resize']
    save_base_path = Path(config['save_path'])

    train_videos_paths = [path.rsplit('/', 1)[0] for path in train_screenshots_paths]
    test_videos_paths = [path.rsplit('/', 1)[0] for path in test_screenshots_paths]
    
    # Save the used config
    # if config.get("resume_testing_path") and config["resume_testing_path"] != None and config["resume_testing_path"] != False:
    if config["resume_testing_path"] != None and config["resume_testing_path"] != False:
        print('Resuming testing..')
        current_save_path = config["resume_testing_path"]
        tested_videos = os.listdir(current_save_path)
        print('Number of tested videos: ', len(tested_videos)) 
        all_test_videos = [video.rsplit('/', 1)[1] for video in test_videos_paths]
        print('Number of all test videos: ', len(all_test_videos))
        test_videos_parent_path = Path(test_videos_paths[0]).parent
        untested_videos = [video for video in all_test_videos if video not in tested_videos]
        test_videos_paths = [os.path.join(test_videos_parent_path, video) for video in untested_videos]
        print('Number of untested videos: ', len(test_videos_paths))
        current_save_path = '/'.join(current_save_path.split('/')[:-2])
        print('Current save path is: ', current_save_path)
        current_save_path = Path(current_save_path)
        save_config(config, current_save_path)

    else:
        timestamp = str(int(time.time()))
        current_save_path = save_base_path / timestamp
        current_save_path.mkdir(parents=True, exist_ok=True)
        save_config(config, current_save_path)

    if config['debug_mode'] and config['debug_mode'] == True:
        print('Debug mode is on, only testing the last video.')
        test_videos_paths = test_videos_paths[-1:]
        assert len(test_videos_paths) == 1
    else:
        pass
        # print('Testing on all videos.')

    prompt = [{
        "role": "system",
        "content": config['prompts']['system']
    }]

    if config["in_context_learning"] == True: 
        print('Using in-context learning..')
        if config["resume_testing_path"] != None and config["resume_testing_path"] != False:
            print('Resume testing using the previous training videos..')
            train_video_path_txt = current_save_path / 'train_videos_paths.txt'
            with open(train_video_path_txt, 'r') as f:
                train_videos_paths_ = f.readlines()
            train_videos_paths_ = [video.strip() for video in train_videos_paths_]
            print('Number of training videos from last train: ', len(train_videos_paths_))
        else:
            effective_train_videos_number = config['effective_train_videos_number']
            print('Effective number of training videos: ', effective_train_videos_number)
            number_train_videos = len(train_videos_paths)
            if number_train_videos > effective_train_videos_number:
                train_videos_paths_ = random.sample(train_videos_paths, effective_train_videos_number)
            else:
                train_videos_paths_ = train_videos_paths
        
            with open(current_save_path / 'train_videos_paths.txt', 'w') as f:
                for item in train_videos_paths_:
                    f.write("%s\n" % item)

        prompt.append({
            "role": "user",
            "content": """
            You are given the following sequences of screenshots and their SOP labels. 
            Each sequence is sourced from a demonstration of the workflow. 
            Each sequence is presented in chronological order.
            """
        })
        
        num_train_frames = 0

        for video in train_videos_paths_:
            frames, number_frames = read_frames(video, resize) 
            print(f'Number of frames in the training video {video}: {number_frames}')
            num_train_frames += number_frames
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
                        "url": f"data:image/png;base64,{frames[j]}",
                        "detail": "high"}
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
            labels_header = "The above screenshots have SOP labels as follows:\n"
            labels = labels_header + labels
            prompt.append({
                "role": "user",
                "content": labels
            })

        print(f'In-context learning completed after reading {num_train_frames} training frames.')
    
    print('Testing started..')
    testing_videos_number = len(test_videos_paths)
    print('Number of testing videos: ', testing_videos_number)
    testing_time_start = time.time()

    for video in tqdm(test_videos_paths, desc="Testing videos"):
        prompt_test_index = 0
        frames, number_frames = read_frames(video, resize)

        if number_frames > 20:
            print(f'Number of frames in the testing video {video}: {number_frames}. Too many frames, downsampling to 20 frames.')
            frames = [frames[i] for i in range(0, len(frames), len(frames)//20)]
            number_frames = 20
        else:
            print(f'Number of frames in the testing video {video}: {number_frames}')
        
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
                    "url": f"data:image/png;base64,{frames[j]}",
                    "detail": "high"}
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

        params = {
            "model": config['model_name'],
            "messages": prompt,
            "max_tokens": int(config['max_tokens']),
            "temperature": config['temperature'],
            "top_p": config['top_p'],
            "timeout": config['timeout']
        }
        
        for i in range(num_inferences):
            prediction = openai_completion(client, params)
            predictions.append(prediction)
            time.sleep(60)

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
                "temperature": config['temperature_reflect'],
                "top_p": config['top_p_reflect']
            }
            
            final_prediction = openai_completion(client, reflection_params)
            time.sleep(60)
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
            
    testing_time_end = time.time()
    testing_time = testing_time_end - testing_time_start
    print('Testing time: ', testing_time)
    # save the testing time information as a txt file in the save path
    with open(current_save_path / 'testing_time.txt', 'w') as f:
        f.write(str(testing_time))
    print('Testing completed..\n')


