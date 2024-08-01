"""
demonstration-collection/experiments/eval/eval.py

This script is utilized to perform automatic evaluation of the generated SOPs
through comparison with the gold standard SOPs. Various evaluation metrics are
calculated and logged in a CSV file.
"""

import concurrent.futures
import random
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import os
from tqdm import tqdm
from metrics import PairwiseComparison


# PREPROCESSING ===============================================================
def preprocess_sop(sop: str) -> List[str]:
    """
    Given a SOP, this function preprocesses the SOP and returns a list of
    sentences.

    Args:
      sop (str): The SOP to be preprocessed

    Returns:
      List[str]: A list of sentences
    """
    # Split lines by newline character
    lines = sop.split("\n")

    # Remove empty lines
    lines = [line for line in lines if line.strip() != ""]

    # For each line, find the first instance of "." and keep everything after that
    lines = [line[line.find(".") + 1 :] for line in lines]

    # Remove any leading or trailing whitespace
    lines = [line.strip() for line in lines]

    # Remove line if it is empty
    lines = [line for line in lines if len(line) > 0]

    return lines


# LLM INFRA ===============================================================


def _evaluate_sops(sop_dict):
    """Helper function to evaluate a single pair of SOPs. Çalled in parallel in `evaluate_sops`."""

    # Preprocess the SOPs
    pred_sop = preprocess_sop(sop_dict["pred_sop"])
    gold_sop = preprocess_sop(sop_dict["gold_sop"])

    # Create a cache_id for saving prompt eval results
    cache_id: str = (
        # sop_dict["experiment_name"]
        # sop_dict[""]
        "_"
        + (
            sop_dict["demo_name"] + "_" + sop_dict["ablation"]
            if "demo_name" in sop_dict and "ablation" in sop_dict
            else str(random.randint(0, 1000000))
        )
    )
    del sop_dict["experiment_name"]

    # Create a PairwiseComparison
    pair_comp = PairwiseComparison(
        pred_sop=pred_sop, ref_sop=gold_sop, cache_id=cache_id
    )

    # Add metrics
    metrics: Dict[str, Any] = {
        "precision": pair_comp.precision(),
        "recall": pair_comp.recall(),
        "ordering": pair_comp.ordering(),
        "n_lines_pred_sop": len(pred_sop),
        "n_lines_gold_sop": len(gold_sop),
    }

    # Add all other key value pairs from the input dict
    metrics.update(sop_dict)

    return metrics


def evaluate_sops(
    list_of_sops: List[Dict[str, str]], experiment_name: str, n_threads: int = 10
) -> pd.DataFrame:
    """
    Given an input list of dictionaries containing generated SOPs and corresponding
    references, this function evaluates the generated SOPs and returns a list of
    dictionaries containing the evaluation metrics.

    Args:
      list_of_sops (List[Dict[str, str]]): A list of dictionaries containing
        generated SOPs and corresponding references.
        Each dictionary contains the following required items:
        - "pred_sop" (str): The generated SOP
        - "gold_sop" (str): The gold standard SOP
        And the following optional items:
        - "id" (str): The id for this unique demo pair

    Returns:
        pd.DataFrame: A pandas dataframe containing the columns:
        - "id" (str): The id for this unique demo pair
        - "precision" (float): The precision score
        - "recall" (float): The recall score
        - "ordering" (float): The ordering score
        - all other key value pairs from the input dict
    """
    # Iterate over each SOP dict
    results: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [
            executor.submit(
                _evaluate_sops, sop_dict | {"experiment_name": experiment_name}
            )
            for idx, sop_dict in enumerate(list_of_sops)
        ]
        with tqdm(total=len(list_of_sops), desc="Evaluating SOPs") as pbar:
            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                pbar.update(1)
    
    return pd.DataFrame(results)


def add_gold_sops(df_all_results: pd.DataFrame, path_to_demos_dir: str) -> pd.DataFrame:
    """Add Gold SOPs to the `df_all_results` dataframe.
    `path_to_demos_dir` is the path to the directory containing all the demos."""
    gold_sops: List[str] = []
    for idx, row in df_all_results.iterrows():
        # Collect the current demo name
        demo_name: str = row["demo_name"]

        # Ensure the folder exists
        if not os.path.exists(os.path.join(path_to_demos_dir, demo_name)):
            print(f"Error: `{demo_name}` does not exist")
            continue

        # Collect the contents of the folder
        folder_contents = os.listdir(os.path.join(path_to_demos_dir, demo_name))

        # Ensure exactly one file starts with "SOP"
        sop_files = [f for f in folder_contents if f.startswith("SOP")]
        if len(sop_files) != 1:
            print(f"Error: `{demo_name}` has {len(sop_files)} SOP files")
            continue

        # Create gold sop path file location
        gold_sop_file_path: str = os.path.join(
            path_to_demos_dir, demo_name, sop_files[0]
        )

        # Read in the contents of the gold SOP file as string
        with open(gold_sop_file_path, "r") as f:
            gold_sop: str = f.read().strip()

        # If first line doesn't start with a number, then remove b/c it's a title rather than a step
        if not gold_sop.split("\n")[0][0].isdigit():
            gold_sop = "\n".join(gold_sop.split("\n")[1:])

        # Save the gold SOP string to the df_all_results dataframe
        gold_sops.append(gold_sop)
    df_all_results["gold_sop"] = gold_sops
    return df_all_results


if __name__ == "__main__":

    # Create list to store collected SOP pairs
    sops = []

    # Debug to see how it works:
    with open("/home/moucheng/data/Wonderbread/gold_demos/0 @ 2023-12-25-15-10-58/SOP - 0 @ 2023-12-25-15-10-58.txt", "r") as f:
        rank_1 = f.read()
    
    # Remove the first line from each SOP
    rank_1 = "\n".join(rank_1.split("\n")[1:])

    # Add an instance with rank_1 as the pred_sop and rank_1 as the gold_sop
    sops.append(
        {
            "pred_sop": rank_1,
            "gold_sop": rank_1,
            "cache_id": "task_name_SOP_1_vs_2",
        }
    )

    # Evaluate the SOPs
    results = evaluate_sops(list_of_sops=sops, 
                            experiment_name='dummy_debug',
                            n_threads=1)
    
    # print the data frame in results in a nice way:
    # print(results.to_string())
    save_path = "dummy_debug_eval_results.csv"
    results.to_csv(save_path, index=False)
