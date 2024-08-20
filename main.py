import argparse

from helpers import load_config

# todo: add the following imports:
# from icl_gemini import main_gemini
# from icl_claude import main_claude
# from icl_llama import main_llama
from icls.icl_cogvlm2_video import main_cogvlm2_video
from icls.icl_gpt import main_gpt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    config = load_config(args.config)
    if "gpt" in config["model_name"] or "GPT" in config["model_name"]:
        main_gpt(args)

    if "cog" in config["model_name"]:
        main_cogvlm2_video(args)
