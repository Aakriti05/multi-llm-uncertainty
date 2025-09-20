import re
import random
random.seed(0)

from collections import Counter
from typing import List, Optional

from global_utils.json_utils import load_all_jsons
from metrics.utils import average_metrics

def get_eval_data(json_folder_path: str,
                  eval_type: str = "debate",
                  calibration: bool = False):
    jsons = load_all_jsons(json_folder_path)
    if len(jsons) == 0:
        print(f"No jsons found in {json_folder_path}. Returning empty lists.")
        return [], [], []
    num_rounds = max([int(json_name.split("_")[0]) for json_name in jsons.keys()]) + 1
    num_models = max([int(json_name.split("_")[1]) for json_name in jsons.keys()]) + 1
    num_questions = max([int(json_name.split("_")[2]) for json_name in jsons.keys()]) + 1
    # if len(jsons) != num_rounds * num_models * num_questions:
    #     print(f"Number of jsons ({len(jsons)}) does not match the expected number of jsons ({num_rounds * num_models * num_questions}).",
    #           "Returning empty lists.")
    #     return [], [], []
    if eval_type == "baseline-1":
        sample_data = next(iter(jsons.values()))
        num_completions = len(sample_data["completions"])
        gt_answers = [None for _ in range(num_questions)]
        completions = [[[None for _ in range(num_completions)] for _ in range(num_models)] for _ in range(num_questions)]
        metrics = [[[None for _ in range(num_completions)] for _ in range(num_models)] for _ in range(num_questions)]
    elif eval_type == "debate":
        gt_answers = [None for _ in range(num_questions)]
        completions = [[[None for _ in range(num_models)] for _ in range(num_rounds)] for _ in range(num_questions)]
        metrics = [[[None for _ in range(num_models)] for _ in range(num_rounds)] for _ in range(num_questions)]
    elif eval_type == "baseline-2":
        num_completions = len(next(iter(jsons.values()))["completions"])
        gt_answers = [None for _ in range(num_questions)]
        completions = [[[None for _ in range(num_models*num_completions)] for _ in range(num_rounds)] for _ in range(num_questions)]
        metrics = [[[None for _ in range(num_models*num_completions)] for _ in range(num_rounds)] for _ in range(num_questions)]
    else:
        raise ValueError(f"Invalid eval_type: {eval_type}")
    for json_name, json_data in jsons.items():
        round_idx, model_idx, question_idx = [int(x) for x in json_name.split("_")]
        gt_answers[question_idx] = json_data["answer"]
        if eval_type == "debate":
            completions[question_idx][round_idx][model_idx] = json_data["completions"][0]
            if calibration:
                metrics[question_idx][round_idx][model_idx] = average_metrics([model_metrics[0] for model_metrics in json_data["metrics"]])
            else:
                metrics[question_idx][round_idx][model_idx] = json_data["metrics"][model_idx][0]
        elif eval_type == "baseline-1":
            if round_idx == 0:
                completions[question_idx][model_idx] = json_data["completions"]
                if calibration:
                    averaged_metrics = []
                    for i in range(len(json_data["completions"])):
                        avg_metric = average_metrics([model_metrics[i] for model_metrics in json_data["metrics"]])
                        averaged_metrics.append(avg_metric) 
                    metrics[question_idx][model_idx] = averaged_metrics
                else:
                    metrics[question_idx][model_idx] = json_data["metrics"][model_idx]
        elif eval_type == "baseline-2":
            if calibration:
                averaged_metrics = []
                for i in range(len(json_data["completions"])):
                    avg_metric = average_metrics([model_metrics[i] for model_metrics in json_data["metrics"]])
                    averaged_metrics.append(avg_metric)
            for completion_idx, completion in enumerate(json_data["completions"]):
                completions[question_idx][round_idx][model_idx*num_completions + completion_idx] = completion
                if calibration:
                    metrics[question_idx][round_idx][model_idx*num_completions + completion_idx] = averaged_metrics[completion_idx]
                else:
                    metrics[question_idx][round_idx][model_idx*num_completions + completion_idx] = json_data["metrics"][model_idx][completion_idx]
    return gt_answers, completions, metrics

def parse_bullets(sentence):
    bullets_preprocess = sentence.split("\n")
    bullets = []

    for bullet in bullets_preprocess:
        try:
            idx = bullet.find(next(filter(str.isalpha, bullet)))
        except:
            continue

        bullet = bullet[idx:]

        if len(bullet) != 0:
            bullets.append(bullet)

    return bullets

def parse_yes_no(string):
    """
    Parses a string containing "yes" or "no" and returns a boolean value.

    Args:
        string (str): The string to parse.

    Returns:
        bool: True if the string contains "yes", False if the string contains "no".

    Raises:
        ValueError: If the input string does not contain "yes" or "no".
    """
    if "yes" in string.lower():
        return True
    elif "no" in string.lower():
        return False
    else:
        return None


def solve_math_problems(input_str):
    pattern = r"\d+\.?\d*"

    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1]

    return None


def parse_answer(input_str):
    pattern = r"\{([0-9.,$]*)\}"
    matches = re.findall(pattern, input_str)

    solution = None

    for match_str in matches[::-1]:
        solution = re.sub(r"[^0-9.]", "", match_str)
        if solution:
            break

    return solution


def solve_math_problems_mcq(input_str):
    # pattern = r"\d+\.?\d*"
    # matches = re.findall(pattern, input_str)

    pattern = r"[ABCD]"       # match a single uppercase A, B, C, or D
    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1]

    return None

def parse_answer_mcq(input_str):
    # pattern = r'\((\w)\)'
    # matches = re.findall(pattern, input_str)

    pattern = r"\\boxed\{[^}]*?([ABCD])[^}]*?\}"
    matches = re.findall(pattern, input_str)

    solution = None

    for match_str in matches[::-1]:
        solution = match_str.upper()
        if solution:
            break

    return solution


def parse_mcq_answer(input_str):
    if input_str is None:
        return None

    solution_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    solution = None

    solution = solution_map.get(input_str)

    return solution

# def parse_mcq_answer(input_str):
#     pattern = r"[A-D]"
#     matches = re.findall(pattern, input_str)

#     solution_map = {"A": 0, "B": 1, "C": 2, "D": 3}
#     solution = None

#     for match_str in matches[::-1]:
#         solution = solution_map.get(match_str.strip())
#         if solution is not None:
#             break

#     return solution


def compute_accuracy(gt, pred_solutions, prob, random_choice, log_prob_only, data="mmlu"):
    if data == "mmlu" or data=="ARC":
        answers =  gt#solve_math_problems_mcq(gt)
    else:
        answers = solve_math_problems(gt)

    if answers is None:
        return -2, None, None, None, None


    if type(pred_solutions) == list:
        pred_answers = []

        for pred_solution in pred_solutions:
            if data == "mmlu" or data=="ARC":
                pred_answer = parse_answer_mcq(pred_solution)
            else:
                pred_answer = parse_answer(pred_solution)

            if pred_answer is None:
                if data == "mmlu" or data=="ARC":
                    pred_answer = solve_math_problems_mcq(pred_solution)
                else:
                    pred_answer = solve_math_problems(pred_solution)
            if data == "mmlu" or data=="ARC":
                pred_answer = parse_mcq_answer(pred_answer)
            pred_answers.append(pred_answer)

        pred_answer = most_frequent(pred_answers, prob, random_choice, log_prob_only)
    else:
        pred_answer = parse_answer(pred_solutions)
        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solutions)

    if pred_answer is None:
        return -1, answers, pred_answer, pred_answers, prob

    if float(answers) == float(pred_answer):
        return 1, answers, pred_answer, pred_answers, prob
    else:
        return 0, answers, pred_answer, pred_answers, prob

def most_frequent_new(
    items: List[Optional[str]],
    probs: List[float],
    random_choice: bool = False,
    log_prob_only: bool = False
) -> Optional[str]:
    """
    Select an element from `items` according to these rules:

    1. If log_prob_only is True:
       - Return the non-None/non-empty item whose corresponding probability in `probs` is highest.
    2. Otherwise:
       - Exclude None and "" from consideration.
       - Compute the frequency of each remaining item.
       - Let max_count = highest frequency, and let
         candidates = [item for item, count in counts.items() if count == max_count].
       - If there's exactly one candidate and max_count > 1, return it.
       - Otherwise (tie for frequency, or all items unique):
         a) If random_choice is True:
            - Return a random choice from `candidates`.
         b) Else:
            - Return the candidate whose **highest** associated probability is largest.
    - If no valid item exists at any point, returns None.

    Time complexity is O(n) for counting + O(n log n) for any needed sorting, but in typical use
    this will be dominated by the linear pass to build counts.
    """
    # Pair items with their indices for easy lookups
    n = len(items)
    if len(probs) != n:
        raise ValueError("Length of items and probs must match")

    # 1) Handle log-prob-only mode
    if log_prob_only:
        # Filter to indices of valid items
        valid = [(probs[i], i) for i in range(n) if items[i] not in (None, "")]
        if not valid:
            return None
        # Pick the index with the highest probability
        _, best_idx = max(valid, key=lambda x: x[0])
        return items[best_idx]

    # 2) Build frequency counts of non-null, non-empty items
    valid_items = [x for x in items if x not in (None, "")]
    if not valid_items:
        return None

    counts = Counter(valid_items)
    max_count = max(counts.values())
    candidates = [val for val, cnt in counts.items() if cnt == max_count]

    # 3) If there's a clear winner with frequency > 1, return it immediately
    if max_count > 1 and len(candidates) == 1:
        return candidates[0]

    # 4) Tie-breaking (including the all-unique case where max_count == 1)
    if random_choice:
        # Return a random candidate
        return random.choice(candidates)

    # 5) Deterministic tie-break: pick candidate with highest prob
    #    For each candidate, find the maximum probability across all its positions
    best_val: Optional[str] = None
    best_prob = float("-inf")
    for val in candidates:
        # look at all positions where items[i] == val
        for i, it in enumerate(items):
            if it == val and probs[i] > best_prob:
                best_prob = probs[i]
                best_val = val

    return best_val

def most_frequent(List, prob, random_choice, log_prob_only):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    if counter == 1:
        if random_choice:
            while True:
                num = random.choice(List)  # Select a random element from the list
                try:
                    float(num)
                    break
                except:
                    pass
        else:
            prob_with_indices = [(p, i) for i, p in enumerate(prob)]
            prob_with_indices.sort(
                reverse=True, key=lambda x: x[0]
            )  # Sort by probability, descending

            # Select the element with the max probability that is not None
            for p, idx in prob_with_indices:
                if (List[idx] is not None) and (List[idx] != ""):
                    num = List[idx]
                    break
                #import pdb; pdb.set_trace()
                print("Its none only",  List[idx], List, prob_with_indices)

    if log_prob_only:
        prob_with_indices = [(p, i) for i, p in enumerate(prob)]
        prob_with_indices.sort(
            reverse=True, key=lambda x: x[0]
        )  # Sort by probability, descending

        # Select the element with the max probability that is not None
        for p, idx in prob_with_indices:
            if List[idx] is not None:
                num = List[idx]
                break

    return num