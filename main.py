from helpers import load_config
import argparse
from icls.icl_gpt import main_gpt
# todo: add the following imports:
from icls.icl_gemini import main_gemini
# from icl_claude import main_claude
# from icl_llama import main_llama
from icls.icl_cogagent import main_cogagent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    
    config = load_config(args.config)
    if 'gpt' in config['model_name'] or 'GPT' in config['model_name']:
        main_gpt(args)
    
    if 'cog' in config['model_name']:
        main_cogagent(args)

    if 'gemini' in config['model_name']:
        main_gemini(args)
    
