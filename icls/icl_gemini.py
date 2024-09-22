import os
import time
from tqdm import tqdm
from helpers import save_config, load_config, read_frames, get_screenshots
from helpers import read_labels, majority_vote, prediction_template
from helpers import read_paths_from_txt, check_videos_paths
from pathlib import Path
import random
import google.generativeai as genai
from google.generativeai import GenerationConfig

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

    train_screenshots_paths = read_paths_from_txt(train_screenshots)
    test_screenshots_paths = read_paths_from_txt(test_screenshots)
    
    # resize = config['image_resize']
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

    prompt = []
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(config['model_name'], 
                                    system_instruction=str(config['prompts']['system']))
    
    # Configurations for model inference and safety settings, very important to set up, otherwise the server just randomly block your requests
    generation_config = GenerationConfig(
    temperature=config['temperature'],
    top_p=config['top_p_majority_voting'],
    top_k=32,
    candidate_count=1,
    max_output_tokens=config['max_tokens'])
    safe = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]
    
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
            # Add the training start prompt to the prompt:
            prompt.append(
                {
                    "role": "user",
                    "parts": str(config['prompts']['training_example'])
                }
            )

            # get screenshots and labels:
            screenshots, screenshots_folder_path = get_screenshots(video)
            num_train_frames += len(screenshots)

            for i, img in enumerate(screenshots):
                img_path = screenshots_folder_path + "/" + img
                print(f">> Uploading: {img_path} at {-(i+1)}") # insert images into prompt
                # f = {"role":"user", "parts":  [Image(url=img_path)] }#genai.upload_file(img_path)}#
                f = {"role":"user", "parts":  [genai.upload_file(img_path)] }
                prompt.insert(-(i+1), f)
                
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
    
    else:
        pass
    
    print('testing started..')
    testing_videos_number = len(test_videos_paths)
    print('number of testing videos: ', testing_videos_number)
    testing_time_start = time.time()
    skips = []
    iteration = 0

    for video in tqdm(test_videos_paths, desc="testing videos"):
        iteration += 1
        prompt_test_index = 0

        prompt.append(
            {
                "role": "user",
                "parts": str(config['prompts']['testing_example'])
            }
        )
        prompt_test_index += 1

        screenshots, screenshots_folder_path = get_screenshots(video)
        if len(screenshots) > 20:
            num_screenshots = len(screenshots)
            print(f"Too many screenshots{num_screenshots}, downsampling to 20 screenshots.")
            screenshots = [screenshots[i] for i in range(0, len(screenshots), len(screenshots)//20)]

        # add the question to the prompt:
        prompt.append({
            "role": "user",
            "parts": str(config['prompts']['question'])
        })
        prompt_test_index += 1

        predictions = []
        num_inferences = config['majority_voting_candidates']
        
        for n in range(num_inferences):
            for try_time in range(5): # Try 5 times
                try:                    
                    for i, img in enumerate(screenshots):
                        img_path = screenshots_folder_path + "/" + img
                        print(f">> Uploading: {img_path} at {-(i+1)}") # insert images into prompt
                        display_name = video.split('/')[-1] + "_" + img
                        image = genai.upload_file(path=img_path, display_name=display_name)
                        f = {"role":"user", "parts":  [image] }
                        prompt.insert(-(i+1), f)
                        prompt_test_index += 1

                    # chat = model.start_chat(history=prompt)
                    print("Message:",prompt[-1]["parts"])
                    response = model.generate_content(prompt,
                                                      generation_config=generation_config,
                                                      safety_settings=safe,
                                                      request_options={"timeout": 2000})
                    break
                except Exception as e:
                    print("ERROR:", e)
                    wait_time = 30*try_time**2
                    print(f"retrying for {i+1} time and waiting for {wait_time} seconds")
                    time.sleep(wait_time)
                    if try_time == 4:
                        print("!! ERROR HAS OCCURED !!: CONTINUING")
                        skips.append({"prompt": prompt, "error": e, "test_num": iteration })

            print("Response:", response.text)
            predictions.append(response.text)
            time.sleep(20)
        
        if num_inferences > 1:
            print('Using majority voting to select the final prediction..')
            # TODO: to be tested
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
                        
            for try_time in range(10): # Try 5 times
                try:                    
                    response = model.generate_content(reflection_prompt,
                                                      generation_config=generation_config,
                                                      safety_settings=safe,
                                                      request_options={"timeout": 2000})
                    break
                except Exception as e:
                    print("ERROR:", e)
                    wait_time = 30*try_time**2
                    print(f"retrying for {i+1} time and waiting for {wait_time} seconds")
                    time.sleep(wait_time)
                    if try_time == 4:
                        print("!! ERROR HAS OCCURED !!: CONTINUING")
                        skips.append({"prompt": prompt, "error": e, "test_num": iteration })

            final_prediction = response.text
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
        
        time.sleep(5)

    testing_time_end = time.time()
    testing_time = testing_time_end - testing_time_start
    print('Testing time: ', testing_time)
    # save the testing time information as a txt file in the save path
    with open(current_save_path / 'testing_time.txt', 'w') as f:
        f.write(str(testing_time))
    print('Testing completed..\n')
    print(f"skipped questions: {skips}")
    with open(current_save_path / 'debug_info.txt', 'w') as f:
        f.write(str(skips))


