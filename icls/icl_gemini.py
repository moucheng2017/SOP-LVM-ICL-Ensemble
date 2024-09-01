import os
import time
from google.generativeai.types import GenerationConfigType
from tqdm import tqdm
from helpers import save_config, load_config, read_frames, get_screenshots
from helpers import read_labels, majority_vote, prediction_template
from helpers import read_paths_from_txt, check_videos_paths
import yaml
from pathlib import Path
import argparse
import random
import base64
import cv2
import sys
import google.generativeai as genai
from google.generativeai.types.generation_types import GenerationConfig
from IPython.display import Image
from IPython.core.display import HTML

def main_gemini(args):
    config = load_config(args.config)
    print('Configurations:')
    print(config)

    if config.get("api_key") == None:
        print('Please export the Gemini API key as an environment variable: export GEMINI_API_KEY=your_dummy_api_key')
        return
    else:
        API_KEY = config['api_key']
    
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
    prompt = []
    # prompt = [{
    #     "role": "system",
    #     "parts": str(config['prompts']['system'])
    # }]

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
            "parts": """
            You are given the following sequences of screenshots and their SOP labels. 
            Each sequence is sourced from a demonstration of the workflow. 
            Each sequence is presented in chronological order.
            """
        })
        
        num_train_frames = 0

        for video in train_videos_paths_:
            frames, number_frames = read_frames(video, resize) # NOTE: This is not the method recommended by the Gemini documentation
            print(f'Number of frames in the training video {video}: {number_frames}')
            num_train_frames += number_frames
            # Add the training start prompt to the prompt:
            prompt.append(
                {
                    "role": "user",
                    "parts": str(config['prompts']['training_example'])
                }
            )

            # Add the frames to the prompt:
            # images = []
            # for j in range(number_frames):
            #     images.append(
            #         {"type": "image_url",
            #         "image_url": {
            #             "url": f"data:image/png;base64,{frames[j]}",
            #             "detail": "high"}
            #         }
            #     )
            # prompt.append(
            #     {"role": "user",
            #     "parts": str(images)}
            # )

            # Add the labels to the prompt:
            labels = read_labels(video)
            # remove first line:
            labels = labels.split('\n') 
            labels = '\n'.join(labels[1:])
            labels = labels.split('\n') 
            # remove empty lines: 
            labels = [line for line in labels if line.strip() != '']
            labels = '\n'.join(labels)
            labels_header = "the above screenshots have sop labels as follows:\n"
            labels = labels_header + labels
            prompt.append({
                "role": "user",
                "parts": str(labels)
            })

        print(f'in-context learning completed after reading {num_train_frames} training frames.')
    
    print('testing started..')
    testing_videos_number = len(test_videos_paths)
    print('number of testing videos: ', testing_videos_number)
    testing_time_start = time.time()

    # test_videos_paths = ["data/demos/debug_demos/494 @ 2024-01-07-17-31-39/"]
    for video in tqdm(test_videos_paths, desc="testing videos"):
        prompt_test_index = 0
        frames, number_frames = read_frames(video, resize)

        if number_frames > 50:
            print(f'number of frames in the testing video {video}: {number_frames}. too many frames, skipping..')
            pass
        else:
            print(f'number of frames in the testing video {video}: {number_frames}')
            # add the system prompt to the prompt:
            prompt.append(
                {
                    "role": "user",
                    "parts": str(config['prompts']['testing_example'].format(number_frames=number_frames))
                }
            )
            prompt_test_index += 1

            # add the frames to the prompt:
            # for j in range(number_frames):
                # images.append(frames[j]

            screenshots, screenshots_folder_path = get_screenshots(video)

            # note: may be worth investigasting if converting list to string is what gemini wants for multiple images
            # prompt.append({
            #     "role": "user",
            #     "parts": str(images[:9]) ## control the number of images added to the input
            # })
            prompt_test_index += 1

            # add the question to the prompt:
            prompt.append({
                "role": "user",
                "parts": str(config['prompts']['question'])
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

                # Set up Gemini model
                genai.configure(api_key=API_KEY)
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=str(config['prompts']['system'])
)
                except:
                    breakpoint()

                images = []
                for i, img in enumerate(screenshots):
                    img_path = screenshots_folder_path + "/" + img
                    print(f">> Uploading: {img_path} at {-(i+1)}") # insert images into prompt
                    # f = {"role":"user", "parts":  [Image(url=img_path)] }#genai.upload_file(img_path)}#
                    f = {"role":"user", "parts":  [genai.upload_file(img_path)] }
                    prompt.insert(-(i+1), f)

                # prompt = prompt[1:]
                chat = model.start_chat(history=prompt)
                print("Message:",prompt[-1]["parts"])
                response = chat.send_message(str(prompt[-1]["parts"]))
                # conf = GenerationConfig(temperature=config["temperature"])
                # response = model.generate_parts(prompt)
                print("Response:", response.text)

                # result = client.chat.completions.create(**params)
                # prediction = result.choices[0].message.parts
                # predictions.append(prediction)
                predictions.append(response.text)
                time.sleep(1)
            
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
                    "parts": str(config['prompts']['reflection'].format(initial_prediction=initial_prediction))
                })
                
                reflection_params = {
                    "model": config['model_name'],
                    "messages": reflection_prompt,
                    "max_tokens": int(config['max_tokens']),
                    "temperature": config['temperature_self_reflect'],
                    "top_p": config['top_p_self_reflect']
                }
                
                reflection_result = client.chat.completions.create(**reflection_params)
                final_prediction = reflection_result.choices[0].message.parts
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
            
            # save the prompt for comparisons:
            # prompt_save_path = video_save_path / 'prompt.txt' 
            # with open(prompt_save_path, 'w') as f:
            #     yaml.dump(prompt, f)

            time.sleep(5)

    testing_time_end = time.time()
    testing_time = testing_time_end - testing_time_start
    print('Testing time: ', testing_time)
    # save the testing time information as a txt file in the save path
    with open(current_save_path / 'testing_time.txt', 'w') as f:
        f.write(str(testing_time))
    print('Testing completed..\n')


