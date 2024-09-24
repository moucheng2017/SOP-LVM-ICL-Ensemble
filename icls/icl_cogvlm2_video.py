import base64
import os
import pathlib
import random
import time
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm  # type: ignore
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

import helpers


def _get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_torch_type():
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch_type = torch.bfloat16
    else:
        torch_type = torch.float16
    return torch_type


def load_cogvlm2(config) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    quant = config["quant"]
    assert quant in [4, 8]
    model_name = config["model_name"] or "THUDM/cogvlm2-video-llama3-chat"
    assert "cog" in model_name

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=_get_torch_type(),
        low_cpu_mem_usage=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=quant == 4,
            load_in_8bit=quant == 8,
        ),
        trust_remote_code=True,
        device_map="auto",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"], trust_remote_code=True
    )
    return model, tokenizer


def openai_messages_to_cogvlm2(messages):
    """Modified from https://github.com/THUDM/CogVLM/blob/main/openai_demo/openai_api.py"""
    formatted_history = []
    image_list = []
    last_user_query = ""

    for i, message in enumerate(messages):
        role = message["role"]
        content = message["content"]

        if isinstance(content, list):  # text
            text_content = " ".join(
                item.text for item in content if item["type"] == "text"
            )
        else:
            text_content = content

        if isinstance(content, list):  # image
            for item in content:
                if item["type"] == "image_url":
                    image_url = item["image_url"]["url"]
                    if image_url.startswith("data:image/png;base64,"):
                        base64_encoded_image = image_url.split(
                            "data:image/png;base64,"
                        )[1]
                        image_data = base64.b64decode(base64_encoded_image)
                        image = Image.open(BytesIO(image_data)).convert("RGB")
                        image_list.append(image)

        if role == "user" or role == "system":
            if i == len(messages) - 1:
                last_user_query = text_content
            else:
                formatted_history.append((text_content, ""))
        elif role == "assistant":
            if formatted_history:
                if formatted_history[-1][1] != "":
                    assert (
                        False
                    ), f"the last query is answered. answer again. {formatted_history[-1][0]}, {formatted_history[-1][1]}, {text_content}"
                formatted_history[-1] = (formatted_history[-1][0], text_content)
            else:
                assert False, "assistant reply before user"
        else:
            assert False, f"unrecognized role: {role}"

    return last_user_query, formatted_history, image_list


def inference_cogagent(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages,
    max_tokens,
    temperature,
    top_p,
    repetition_penalty,
):
    query, history, image_list = openai_messages_to_cogvlm2(messages)
    DEVICE = _get_device()
    torch_type = _get_torch_type()
    video = np.array(image_list)
    video_tensor = torch.tensor(
        video.transpose((3, 0, 1, 2)), dtype=torch.float32
    )  # channel first

    input_by_model = model.build_conversation_input_ids(
        tokenizer, query=query, history=history, images=[video_tensor]
    )
    inputs = {
        "input_ids": input_by_model["input_ids"].unsqueeze(0).to(DEVICE),
        "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).to(DEVICE),
        "attention_mask": input_by_model["attention_mask"].unsqueeze(0).to(DEVICE),
        "images": [[input_by_model["images"][0].to(DEVICE).to(torch_type)]],
    }

    gen_kwargs = {
        "max_length": int(max_tokens),  # TODO maybe max_new_tokens
        "pad_token_id": 128002,
        "top_k": 1,
        "temperature": temperature,
        "top_p": top_p if temperature > 1e-5 else 0,
        "do_sample": True if temperature > 1e-5 else False,
        "repetition_penalty": repetition_penalty,
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("</s>")[0]

    return response


def main_cogvlm2_video(args):
    config = helpers.load_config(args.config)
    print("Configurations:")
    print(config)

    # Load model
    model, tokeinzer = load_cogvlm2(config)

    train_screenshots = config["train_screenshots_txt"]
    test_screenshots = config["test_screenshots_txt"]

    train_screenshots_paths = helpers.read_paths_from_txt(train_screenshots)
    test_screenshots_paths = helpers.read_paths_from_txt(test_screenshots)

    resize = config["image_resize"]
    save_base_path = pathlib.Path(config["save_path"])

    train_videos_paths = [path.rsplit("/", 1)[0] for path in train_screenshots_paths]
    test_videos_paths = [path.rsplit("/", 1)[0] for path in test_screenshots_paths]

    if "resume_testing_path" in config and config["resume_testing_path"]:
        print("Resuming testing..")
        current_save_path = config["resume_testing_path"]
        tested_videos = os.listdir(current_save_path)
        print("Number of tested videos: ", len(tested_videos))
        all_test_videos = [video.rsplit("/", 1)[1] for video in test_videos_paths]
        print("Number of all test videos: ", len(all_test_videos))
        test_videos_parent_path = pathlib.Path(test_videos_paths[0]).parent
        untested_videos = [
            video for video in all_test_videos if video not in tested_videos
        ]
        test_videos_paths = [
            os.path.join(test_videos_parent_path, video) for video in untested_videos
        ]
        print("Number of untested videos: ", len(test_videos_paths))
        current_save_path = "/".join(current_save_path.split("/")[:-2])
        print("Current save path is: ", current_save_path)
        current_save_path = pathlib.Path(current_save_path)
        helpers.save_config(config, current_save_path)
    else:
        timestamp = str(int(time.time()))
        current_save_path = save_base_path / timestamp
        current_save_path.mkdir(parents=True, exist_ok=True)
        helpers.save_config(config, current_save_path)

    if config["debug_mode"] and config["debug_mode"]:
        print("Debug mode is on, only testing the last video.")
        test_videos_paths = test_videos_paths[-1:]
        assert len(test_videos_paths) == 1
    else:
        pass

    prompt = [{"role": "system", "content": config["prompts"]["system"]}]

    if config["in_context_learning"]:
        print("Using in-context learning..")
        random.seed(config["seed"])

        if "resume_testing_path" in config and config["resume_testing_path"]:
            print("Resume testing using the previous training videos..")
            train_video_path_txt = current_save_path / "train_videos_paths.txt"
            with open(train_video_path_txt, "r") as f:
                train_videos_paths_ = f.readlines()
            train_videos_paths_ = [video.strip() for video in train_videos_paths_]
            print(
                "Number of training videos from last train: ", len(train_videos_paths_)
            )
        else:
            effective_train_videos_number = config["effective_train_videos_number"]
            print(
                "Effective number of training videos: ", effective_train_videos_number
            )
            number_train_videos = len(train_videos_paths)
            if number_train_videos > effective_train_videos_number:
                train_videos_paths_ = random.sample(
                    train_videos_paths, effective_train_videos_number
                )
            else:
                train_videos_paths_ = train_videos_paths

            with open(current_save_path / "train_videos_paths.txt", "w") as f:
                for item in train_videos_paths_:
                    f.write("%s\n" % item)

        prompt.append(
            {
                "role": "user",
                "content": """
            You are given the following sequences of screenshots and their SOP labels. 
            Each sequence is sourced from a demonstration of the workflow. 
            Each sequence is presented in chronological order.
            """,
            }
        )

        num_train_frames = 0

        for video in train_videos_paths_:
            frames, number_frames = helpers.read_frames(video, resize)
            print(f"Number of frames in the training video {video}: {number_frames}")
            num_train_frames += number_frames
            # Add the training start prompt to the prompt:
            prompt.append(
                {"role": "user", "content": config["prompts"]["training_example"]}
            )

            # Add the frames to the prompt:
            images = []
            for j in range(number_frames):
                images.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{frames[j]}",
                            "detail": "high",
                        },
                    }
                )
            prompt.append({"role": "user", "content": images})

            # Add the labels to the prompt:
            labels = helpers.read_labels(video)
            # remove first line:
            labels = labels.split("\n")
            labels = "\n".join(labels[1:])
            labels = labels.split("\n")
            # remove empty lines:
            labels = [line for line in labels if line.strip() != ""]
            labels = "\n".join(labels)
            labels_header = "The above screenshots have SOP labels as follows:\n"
            labels = labels_header + labels
            prompt.append({"role": "user", "content": labels})

        print(
            f"In-context learning completed after reading {num_train_frames} training frames."
        )

    print("Testing started..")
    testing_videos_number = len(test_videos_paths)
    print("Number of testing videos: ", testing_videos_number)
    testing_time_start = time.time()

    for video in tqdm(test_videos_paths, desc="Testing videos"):
        prompt_test_index = 0
        frames, number_frames = helpers.read_frames(video, resize)

        if number_frames > 50:
            print(
                f"Number of frames in the testing video {video}: {number_frames}. Too many frames, skipping.."
            )
            pass
        else:
            print(f"Number of frames in the testing video {video}: {number_frames}")
            # Add the system prompt to the prompt:
            prompt.append(
                {
                    "role": "user",
                    "content": config["prompts"]["testing_example"].format(
                        number_frames=number_frames
                    ),
                }
            )
            prompt_test_index += 1

            # Add the frames to the prompt:
            images = []
            for j in range(number_frames):
                images.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{frames[j]}",
                            "detail": "high",
                        },
                    }
                )
            prompt.append({"role": "user", "content": images})
            prompt_test_index += 1

            # Add the question to the prompt:
            prompt.append({"role": "user", "content": config["prompts"]["question"]})
            prompt_test_index += 1

            predictions = []
            num_inferences = config["majority_voting_candidates"]

            for _ in range(num_inferences):
                params = {
                    "max_tokens": int(config["max_tokens"]),
                    "temperature": config["temperature"],
                    "top_p": config["top_p"],
                    "repetition_penalty": config["repetition_penalty"],
                }
                prediction = inference_cogagent(model, tokeinzer, prompt, **params)
                predictions.append(prediction)
                time.sleep(1)

            if num_inferences > 1:
                print("Using majority voting to select the final prediction..")
                initial_prediction = helpers.majority_vote(predictions)
            else:
                initial_prediction = predictions[0]

            if config["use_self_reflect"] and config["use_self_reflect"]:
                print("Using self-reflection to refine the prediction..")
                reflection_prompt = prompt.copy()
                reflection_prompt.append(
                    {
                        "role": "user",
                        "content": config["prompts"]["reflection"].format(
                            initial_prediction=initial_prediction
                        ),
                    }
                )

                reflection_params = {
                    "max_tokens": int(config["max_tokens"]),
                    "temperature": config["temperature_reflect"],
                    "top_p": config["top_p_reflect"],
                    "repetition_penalty": config["repetition_penalty_reflect"],
                }
                final_prediction = inference_cogagent(
                    model, tokeinzer, reflection_prompt, **reflection_params
                )
            else:
                final_prediction = initial_prediction

            video_name = video.split("/")[-3:]
            video_name = "/".join(video_name)
            video_save_path = pathlib.Path(current_save_path) / video_name
            video_save_path.mkdir(parents=True, exist_ok=True)
            print("Testing on video: ", video)
            save_path = video_save_path / "label_prediction.txt"
            with open(save_path, "w") as f:
                f.write(final_prediction)

            for _ in range(prompt_test_index):
                prompt.pop()

            time.sleep(5)

    testing_time_end = time.time()
    testing_time = testing_time_end - testing_time_start
    print("Testing time: ", testing_time)
    # save the testing time information as a txt file in the save path
    with open(current_save_path / "testing_time.txt", "w") as f:
        f.write(str(testing_time))
    print("Testing completed..\n")
