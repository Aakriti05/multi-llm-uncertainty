import os
import argparse
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Any

from global_utils.eval_utils import get_eval_data, compute_accuracy
from metrics.utils import transform_metrics

def filter_completions(completions: List[Any],
                       num_completions: int,
                       eval_type: str = "debate",
                       num_models: int|None = None,
                       num_completions_per_model: int|None = None):
    if num_completions == -1:
        return completions
    else:
        if eval_type == "debate" or eval_type == "baseline-1":
            return completions[:num_completions]
        elif eval_type == "baseline-2":
            assert num_models is not None and num_completions_per_model is not None, "num_models and num_completions_per_model must be provided for baseline-2"
            assert len(completions) == num_models * num_completions_per_model, "length of completions must be equal to num_models * num_completions_per_model"
            return [completions[i*num_completions_per_model + j] for i in range(num_models) for j in range(num_completions)]
        else:
            raise ValueError(f"Invalid eval type: {eval_type}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_folder_path", type=str, default="gsm/results/")
    parser.add_argument("--eval_type", type=str, default="debate", help="debate or baseline-1 or baseline-2")
    parser.add_argument("--random_choice", action="store_true")
    parser.add_argument("--metric_only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--dataset_name", type=str, default="gsm")
    parser.add_argument("--num_completions", type=int, default=-1)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.eval_type == "debate":
        print("Evaluating debate type...")
        print("Only considering zero indexed completion for each round and model...")
    elif args.eval_type == "baseline-1":
        print("Evaluating baseline-1 type...")
        print("Considering all completions from first round for each model...")
    elif args.eval_type == "baseline-2":
        print("Evaluating baseline-2 type...")
        print("Considering all completions from each round together for all models...")
    else:
        raise ValueError(f"Invalid eval type: {args.eval_type}")

    # Get the data
    print("Loading data from json files...")
    gt_answers, completions, metrics = get_eval_data(json_folder_path=args.json_folder_path,
                                                     eval_type=args.eval_type,
                                                     calibration=args.calibration)
    num_models = int([x for x in os.path.basename(os.path.dirname(args.json_folder_path)).split("_") if x.isdigit()][0])
    num_completions_per_model = int([x for x in os.path.basename(os.path.dirname(args.json_folder_path)).split("_") if x.isdigit()][2])

    if args.num_completions != -1:
        print(f"Using {args.num_completions} completions for each model...")
    else:
        print(f"Using all completions...")

    accuracies = []
    for question_idx, gt_answer in tqdm(enumerate(gt_answers), total=len(gt_answers), desc="Computing accuracies", unit="question"):
        if gt_answer is None:
            print(f"Question {question_idx} has no gt answer. Skipping...")
            continue
        for round_idx, round_completions in enumerate(completions[question_idx]):
            metric_list = metrics[question_idx][round_idx]
            try:
                metric_list = [transform_metrics(metric_dict) for metric_dict in metric_list]
            except Exception as e:
                print(f"Error in transforming metrics for question {question_idx}, round {round_idx}: {e}")
                print("Skipping this round...")
                continue
            round_completions = filter_completions(completions=round_completions,
                                                   num_completions=args.num_completions,
                                                   eval_type=args.eval_type,
                                                   num_models=num_models,
                                                   num_completions_per_model=num_completions_per_model)
            metric_list = filter_completions(completions=metric_list,
                                             num_completions=args.num_completions,
                                             eval_type=args.eval_type,
                                             num_models=num_models,
                                             num_completions_per_model=num_completions_per_model)
            for metric_name in metric_list[0].keys():
                try:
                    answer_correct, gt, pred, pred_answers, prob = compute_accuracy(gt=gt_answer,
                                                                                    pred_solutions=round_completions,
                                                                                    prob=[metric_dict[metric_name] for metric_dict in metric_list],
                                                                                    random_choice=args.random_choice,
                                                                                    log_prob_only=args.metric_only,
                                                                                    data=args.dataset_name)
                except Exception as e:
                    print(f"Error in computing accuracy for question {question_idx}, round {round_idx}, metric {metric_name}: {e}")
                    print("Skipping this metric...")
                    continue
                accuracies.append({
                    "question_idx": question_idx,
                    "round_idx": round_idx,
                    "metric_name": metric_name,
                    "answer_correct": answer_correct,
                    "gt": gt,
                    "pred": pred,
                    "pred_answers": pred_answers,
                    "prob": prob,
                })
    accuracy_df = pd.DataFrame(accuracies)
    accuracy_df.to_csv(os.path.join(args.json_folder_path, f"accuracy_question_level_{args.calibration}_{args.random_choice}_{args.metric_only}_{args.eval_type}_{args.num_completions}.csv"),
                       index=False,
                       header=True)
    accuracy_df["answer_correct"] = accuracy_df["answer_correct"].apply(lambda x: abs(int(x)))
    accuracy_df = accuracy_df[accuracy_df["answer_correct"] != 2]
    accuracy_df = accuracy_df[["metric_name", "round_idx", "answer_correct", "question_idx"]].drop_duplicates()
    metric_accuracy_df = accuracy_df.groupby(["metric_name", "round_idx"]).mean("answer_correct").reset_index().drop(columns=["question_idx"])
    metric_accuracy_df.rename(columns={"metric_name": "metric"},
                              inplace=True)
    metric_accuracy_df.to_csv(os.path.join(args.json_folder_path, f"accuracy_metric_level_{args.calibration}_{args.random_choice}_{args.metric_only}_{args.eval_type}_{args.num_completions}.csv"),
                              index=False,
                              header=True)

if __name__ == "__main__":
    main()