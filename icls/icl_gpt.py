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
    # train_labels = config['train_labels_txt']
    # test_labels = config['test_labels_txt']

    train_screenshots_paths = read_paths_from_txt(train_screenshots)
    test_screenshots_paths = read_paths_from_txt(test_screenshots)
    # train_labels_paths = read_paths_from_txt(train_labels)

    # check_videos_paths(train_screenshots_paths, train_labels_paths)
    resize = config['image_resize']
    save_base_path = Path(config['save_path'])
    
    if config["resume_testing"] == True:
        assert config.get("resume_testing_path") != None

    # Save the used config
    if config.get("resume_testing") and config["resume_testing"] == True:
        pass
    else:
        timestamp = str(int(time.time()))
        current_save_path = save_base_path / timestamp
        current_save_path.mkdir(parents=True, exist_ok=True)
        save_config(config, current_save_path)

    # get train_data and test_data:
    # iterate over the train_screenshots_paths, for each path, remove the last part after the last '/':
    train_videos_paths = [path.rsplit('/', 1)[0] for path in train_screenshots_paths]
    test_videos_paths = [path.rsplit('/', 1)[0] for path in test_screenshots_paths]
    # print('train_data: ', train_videos_paths)
    # print('test_data: ', test_videos_paths)
    
    if config['debug_mode'] and config['debug_mode'] == True:
        print('Debug mode is on, only testing the last video.')
        # test_videos_paths = [os.path.join(test_data, video) for video in all_test_videos[-2:]]
        test_videos_paths = test_videos_paths[-1:]
        assert len(test_videos_paths) == 1
        # only use the first two videos from train_data:
        train_videos_paths = train_videos_paths[:2]
    else:
        print('Testing on all videos.')
        pass

    prompt = [{
        "role": "system",
        "content": config['prompts']['system']
    }]

    # use in context learning if config["in_context_learning"] exists and is true
    if config.get("in_context_learning") and config["in_context_learning"] == True: 
        print('Using in-context learning..')
        # check how many videos in train_videos_paths
        number_train_videos = len(train_videos_paths)
        if number_train_videos > 5:
            # randomly select 5 videos from train_videos_paths
            train_videos_paths_ = random.sample(train_videos_paths, 5)
        else:
            train_videos_paths_ = train_videos_paths
        
        for video in train_videos_paths_:
            frames, number_frames = read_frames(video, resize)
            labels = read_labels(video)
            # print(labels)
            if labels:
                contents = []
                contents.append(config['prompts']['training_example'].format(number_frames=number_frames))
                if len(labels.split('\n')) != number_frames:
                    # set a flag to see if the labels are not aligned with the frames:
                    # frames_labels_aligned = False
                    for j in range(number_frames):
                        contents.append({"image": frames[j], "resize": tuple(resize)})
                    transition_text = "The label of the SOP for the above screenshots are:"
                    contents.append(transition_text)
                    # remove the first line of labels
                    labels = labels.split('\n')  
                    labels = '\n'.join(labels[1:]) 
                    # remove empty lines
                    labels = labels.split('\n')  
                    labels = [line for line in labels if line.strip() != '']
                    labels = '\n'.join(labels)
                    # print(labels)
                    contents.append(labels)
                else:
                    for j in range(number_frames):
                        contents.append({"image": frames[j], "resize": tuple(resize)})
                        contents.append(labels.split('\n')[j])
                frames_labels = {
                    "role": "user",
                    "content": contents
                }
                prompt.append(frames_labels)
        print('In-context learning completed..\n')
    print('Testing started..\n')

    # ================ testing resume ====================
    # If config["resume"] exists and is true, current_save_path should be the path to the folder containing the last saved prediction
    if config.get("resume_testing") and config["resume_testing"] == True:
        print('Resuming testing..')
        # get the last saved prediction
        current_save_path = config["resume_testing_path"]
        # list all tested videos, the folders in the current_save_path:
        tested_videos = os.listdir(current_save_path)
        print('Number of tested videos: ', len(tested_videos))  
        # each item is the path to the video folder in all_test_videos, keep only the last part after the last '/' which is the video name:
        all_test_videos = [video.rsplit('/', 1)[1] for video in test_videos_paths]
        print('Number of all test videos: ', len(all_test_videos))
        test_videos_parent_path = Path(test_videos_paths[0]).parent
        # get all untested_videos:
        untested_videos = [video for video in all_test_videos if video not in tested_videos]
        # complete the path for each video in untested_videos:
        test_videos_paths = [os.path.join(test_videos_parent_path, video) for video in untested_videos]
        # print the number of untested videos:
        print('Number of untested videos: ', len(test_videos_paths))
        # remove the last part after the last '/' from the current_save_path:
        current_save_path = '/'.join(current_save_path.split('/')[:-2])
        print('Current save path is: ', current_save_path)

    
    for video in tqdm(test_videos_paths, desc="Testing videos"):

        # check if the variable train_videos_paths_ exists, if so, save it to the video_save_path:
        if 'train_videos_paths_' in locals():
            with open(video_save_path / 'train_videos_paths.txt', 'w') as f:
                for item in train_videos_paths_:
                    f.write("%s\n" % item)
            # print('The train_videos_paths_ is saved at: ', video_save_path / 'train_videos_paths.txt')

        prompt_test_index = 0
        frames, number_frames = read_frames(video, resize)
        contents = []
        contents.append(config['prompts']['testing_example'].format(number_frames=number_frames))
        
        for j in range(number_frames):
            contents.append({"image": frames[j], "resize": tuple(resize)})
        # prediction_template_ = prediction_template(num_frames=number_frames)
        question = config['prompts']['question'] 
        # question += "---START FORMAT TEMPLATE---\n"
        # question += prediction_template_
        # question += "\n---END FORMAT TEMPLATE---\n"
        # question += "Do not deviate from the above format."
        # print(question)
        contents.append(question)
        frames_labels = {
            "role": "user",
            "content": contents
        }
        prompt.append(frames_labels)
        prompt_test_index += 1
        
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
        
        video_name = video.split('/')[-3:]
        video_name = '/'.join(video_name)
        video_save_path = Path(current_save_path) / video_name
        video_save_path.mkdir(parents=True, exist_ok=True)
        print('Testing on video: ', video)
        save_path = video_save_path / 'label_prediction.txt'
        with open(save_path, 'w') as f:
            f.write(final_prediction)
            # print('The prediction is saved at: ', save_path)
            # print('The prediction is: \n', final_prediction)
        
        # remove testing prompts:
        prompt.pop()
        time.sleep(10)

    print('Testing completed..\n')


