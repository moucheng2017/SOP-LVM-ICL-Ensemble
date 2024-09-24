import argparse

from helpers import load_config
from icls.icl_gemini import main_gemini
from icls.icl_cogvlm2_video import main_cogvlm2_video
from icls.icl_gpt import main_gpt
from icls.icl_phi3_5 import main_phi3_5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    config = load_config(args.config)
    
    if "gpt" in config["model_name"].lower():
        main_gpt(args)

    elif "cog" in config["model_name"].lower():
        main_cogvlm2_video(args)
    
    elif 'gemini' in config['model_name']:
        main_gemini(args)
    
    elif "phi" in config["model_name"].lower():
        main_phi3_5(args)
    
    else:
        raise ValueError(f"Model {config['model_name']} not implemented.")
