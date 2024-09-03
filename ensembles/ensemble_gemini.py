import os
import time
import google.generativeai as genai
import sys
sys.path.append("..")
from tqdm import tqdm
from helpers import save_config, load_config, read_frames
from helpers import read_labels, majority_vote, prediction_template
from helpers import read_paths_from_txt, check_videos_paths
from pathlib import Path
import argparse
import traceback

# (TODO): This whole file needs to be updated

# Ensemble predctions as priors for Gemini (1.5-flash):
# 1. For each video, read the frames first
# 2. For each video, read all of available predictions as pseudo labels
# 3. For each video, prompt GPT4o-mini with the frames and pseudo labels
# 4. Get the final prediction

def preprocess_pseudo_labels(pseudo_labels):
    # pseudo labels are neumrated list of actions in txt:
    pseudo_labels = pseudo_labels.split('\n')
    # remove lines with empty strings:
    pseudo_labels = [line for line in pseudo_labels if line.strip() != ""]
    # remove lines which does not start with a number:
    # pseudo_labels = [label for label in pseudo_labels if label[0].isdigit()]
    # assert if it is empty:
    assert len(pseudo_labels) > 0
    return '\n'.join(pseudo_labels)

# def gemini_completion(client, params) -> str:
#     try:
#         response = client.chat.completions.create(
#             **params,
#             stream=False
#         )
#     except openai.RateLimitError:
#         print("Rate limit exceeded -- waiting 25 min before retrying")
#         time.sleep(1500)
#         return openai_completion(client, params)
#     except openai.APIError as e:
#         traceback.print_exc()
#         print(f"OpenAI API error: {e}")
#         if 'Error code: 500' in str(e) or 'Error code: 503' in str(e): # 500, 503 are openai internal server errors
#             time.sleep(1800)
#             print('Retrying after 30 minutes..')
#             return openai_completion(client, params)
#         else:
#             raise e
#     except Exception as e:
#         traceback.print_exc()
#         print(f"Unknown error: {e}")
#         raise e
#     return response.choices[0].message.content


def main(args):
    config = load_config(args.config)
    print('Configurations:')
    print(config)

    if config.get("api_key") == None:
        print('Please export the OpenAI API key as an environment variable: export API_KEY=your_dummy_api_key')
        return
    else:
        API_KEY = config['api_key']

    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=str(config['prompts']['system']))
    
    test_screenshots = config['test_screenshots_txt']
    test_screenshots_paths = read_paths_from_txt(test_screenshots)

    # config['pseudo_labels_path_list'] = pseudo_labels_path_list
    pseudo_labels_path_list = config['pseudo_labels_path_list']
    pseudo_labels_number = len(pseudo_labels_path_list)
    print('Number of pseudo labels: ', pseudo_labels_number)
    
    resize = config['image_resize']
    save_base_path = Path(config['save_path'])

    test_videos_paths = [path.rsplit('/', 1)[0] for path in test_screenshots_paths]
    
    # Save the used config
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

    prompt = [{
        "role": "system",
        "content": config['prompts']['system']
    }]

    print('Ensemble Learning from Pseudo Labels started..')
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

        # Now add pseudo labels to the prompt:
        for i, pseudo_labels_path in enumerate(pseudo_labels_path_list):
            pseudo_labels_video_path = Path(pseudo_labels_path) / video.split('/')[-1]
            pseudo_labels = pseudo_labels_video_path / 'label_prediction.txt'
            pseudo_labels = Path(pseudo_labels).read_text()
            # print i-th pseudo labels:
            print(f'Pseudo labels {i+1}: {pseudo_labels}')
            pseudo_labels = preprocess_pseudo_labels(pseudo_labels)
            pseudo_labels = "The following is a pseudo label of actions for the above frames that you can use as a prior reference:" + '\n' + pseudo_labels
            prompt.append({
                "role": "user",
                "content": pseudo_labels
            })
            prompt_test_index += 1

        # Add the question to the prompt:
        prompt.append({
            "role": "user",
            "content": config['prompts']['question']
        })
        prompt_test_index += 1

        params = {
            "model": config['model_name'],
            "messages": prompt,
            "max_tokens": int(config['max_tokens']),
            "temperature": config['temperature'],
            "top_p": config['top_p'],
            "timeout": config['timeout']
        }

        '''
        (TODO):
        prediction = openai_completion(client, params)
        video_name = video.split('/')[-3:]
        video_name = '/'.join(video_name)
        video_save_path = Path(current_save_path) / video_name
        video_save_path.mkdir(parents=True, exist_ok=True)
        print('Testing on video: ', video)
        save_path = video_save_path / 'label_prediction.txt'
        with open(save_path, 'w') as f:
            f.write(prediction)
        
        for i in range(prompt_test_index):
            prompt.pop()
        '''
        
        time.sleep(60)

    testing_time_end = time.time()
    testing_time = testing_time_end - testing_time_start
    print('Testing time: ', testing_time)
    # save the testing time information as a txt file in the save path
    with open(current_save_path / 'testing_time.txt', 'w') as f:
        f.write(str(testing_time))
    print('Testing completed..\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()    
    config = load_config(args.config)
    main(args)

    