import argparse
from eval_gsm import *
import json
import numpy as np
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import copy
from datasets import load_dataset
# from generate_metrics import compute_logits_and_targets

SEED = 0
huggingface_token = os.getenv("HF_TOKEN")


def construct_message(agents, question, idx):
    if len(agents) == 0:
        return {
            "role": "user",
            "content": "Can you double check that your answer is correct. Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{{answer}}.",
        }

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        if idx == None:
            for msg in agent:
                if msg.get("role") == "assistant":
                    agent_response = msg["content"]
                    prefix_string += f"\n\nOne agent solution: ```{agent_response}```"
        else:
            agent_response = agent[idx]["content"]
            prefix_string += f"\n\nOne agent solution: ```{agent_response}```"

    prefix_string = (
        prefix_string
        + """\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? \n The original math problem is {}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.""".format(
            question
        )
    )
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    return {"role": "assistant", "content": completion}


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def load_model_and_tokenizer(model_name, device_id):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        # attn_implementation="flash_attention_2",
        trust_remote_code=True,
        use_cache=True,
        cache_dir="./cache",
        use_auth_token=huggingface_token,
    ).to(device_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_auth_token=huggingface_token, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer



def load_dataset_all(data_name="mmlu", subset="abstract_algebra"):
    if data_name == "mmlu":
        print("Loading MMLU dataset with subset", subset)
        dataset = load_dataset("cais/mmlu", subset)
    elif data_name == "ARC":
        print("Loading ARC dataset")
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    return dataset


def generate_chat_response(model,
                           tokenizer,
                           text,
                           device,
                           length_normalize_log_likelihood=False):
    try:
        batch_size = 9
        completions = []
        log_likelihoods = []
        tokens_list = []
        num_batches = (args.num_samples + batch_size - 1) // batch_size
        input_ids = tokenizer([text], return_tensors="pt").to(device)
        prompt_length = input_ids["input_ids"].shape[1]
        eos_token_id = tokenizer.eos_token_id

        for b in range(num_batches):
            current_batch = min(batch_size, args.num_samples - b * batch_size)
            with torch.no_grad():
                outputs = model.generate(
                    **input_ids,
                    max_new_tokens=1024,
                    do_sample=True,  # Enable sampling (rather than greedy search)
                    temperature=0.7,  # Controls randomness: lower = less random
                    top_p=0.9,  # Nucleus (top-p) sampling
                    top_k=50,  # Restrict to top-k tokens
                    repetition_penalty=1.3,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                    num_return_sequences=current_batch,
                    use_cache=True,  # Enable caching for faster generation
                )

            all_logits = torch.stack(
                outputs.scores, dim=1
            )  # Shape: (batch_size, sequence_length, vocab_size)

            all_probs = F.log_softmax(
                all_logits, dim=-1
            )  # Convert to log probabilities

            for i in range(current_batch):
                token_ids = outputs.sequences[i][prompt_length:]
                completion = tokenizer.decode(token_ids, skip_special_tokens=True)
                completions.append(completion)

                eos_positions = (token_ids == eos_token_id).nonzero(as_tuple=True)[0]
                actual_length = (
                    eos_positions[0].item()
                    if eos_positions.numel() > 0
                    else len(token_ids)
                )
                if token_ids.size(0) < 1024 and actual_length < token_ids.size(0):
                    actual_length += 1

                log_probs = all_probs[
                    i, torch.arange(actual_length), token_ids[:actual_length]
                ]
                if length_normalize_log_likelihood:
                    log_likelihood = log_probs.mean().item()
                else:
                    log_likelihood = log_probs.sum().item()
                tokens_list.append(token_ids[:actual_length].detach().cpu().numpy().tolist())
                log_likelihoods.append(log_likelihood)

    except Exception as e:
        completion = "API FAILED"
        log_likelihood = None
        print(f"From agent {i}, round {round}: {completion} \nError: {str(e)}")

    result = {
        "completions": completions,
        "log_likelihoods": log_likelihoods,
        "tokens_list": tokens_list
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple argparse example")
    parser.add_argument("--num_models", type=int, default=3, help="Number of models")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds")
    parser.add_argument("--dataset_path", type=str, default="data/test.json", help="Dataset path or subset incase of mmlu")
    parser.add_argument("--dataset_name", type=str, default="gsm", help="Dataset name")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--selfref",
        type=bool,
        default=False,
        help="Include self-reference (default: False)",
    )
    parser.add_argument(
        "--max_prob_element",
        type=bool,
        default=False,
        help="Include self-reference (default: False)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples per model per round",
    )
    parser.add_argument(
        "--unique_response_only",
        type=bool,
        default=False,
        help="Passing only unique responses for a paritcular agent in future round of debate",
    )
    args = parser.parse_args()
    print(args)

    agents = args.num_models
    rounds = args.rounds
    if args.selfref:
        print("SELF REFERENCE")
    elif args.max_prob_element:
        print("MAX prob answer being transferred")
    random.seed(SEED)

    model_names = [
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Ministral-8B-Instruct-2410",
        "meta-llama/Llama-3.1-8B-Instruct",
    ][: args.num_models]
    if args.device == "cuda":
        #devices = [f"cuda:{i}" for i in range(len(model_names))][: args.num_models]
        # devices = ["cuda:3"]
        devices = ["cuda:0", "cuda:0", "cuda:0"][: args.num_models]
    else:
        devices = (["cpu"] * len(model_names))[: args.num_models]
    models_tokenizers = [
        load_model_and_tokenizer(model_name, device)
        for model_name, device in zip(model_names, devices)
    ]

    start_time = time.time()

    generated_description = {}

    if args.dataset_name == "gsm" :
        questions = read_jsonl(args.dataset_path)
        save_name = args.dataset_name
        random.shuffle(questions)
    elif args.dataset_name == "mmlu" :
        questions = load_dataset_all(data_name=args.dataset_name, subset=args.dataset_path)["test"].shuffle(seed=SEED)
        save_name = args.dataset_name + "_" + args.dataset_path
    elif args.dataset_name == "ARC" :
        questions = load_dataset_all(data_name=args.dataset_name)["test"].shuffle(seed=SEED)
        save_name = args.dataset_name

        
    questions_no = 0

    log_probs_file = f"prob_{save_name}_{agents}_{rounds}_{args.num_samples}_diff_LLM_allprob.txt"
    output_file = f"{save_name}_{agents}_{rounds}_{args.num_samples}_diff_LLM_allprob.json"
    pred_ans_file = f"pred_{save_name}_{agents}_{rounds}_{args.num_samples}_diff_LLM_allprob.txt"
    all_results_json = f"{save_name}_{agents}_{rounds}_{args.num_samples}_diff_LLM_allresults"
    os.makedirs(all_results_json, exist_ok=True)

    all_results = {}
    for file in os.listdir(all_results_json):
        if file.endswith(".json"):
            with open(os.path.join(all_results_json, file)) as f:
                all_results[(int(x) for x in file.split(".")[0].split("_"))] = json.load(f)

    if os.path.exists(log_probs_file):
        with open(log_probs_file) as f:
            questions_processed = len(f.readlines())
        if os.path.exists(output_file):
            with open(output_file) as f:
                generated_description = json.load(f)
    else:
        questions_processed = -1
    for data in tqdm(questions):
        if questions_no < questions_processed:
            questions_no += 1
            print("Question Num: ", questions_no)
            continue
        
        question = data["question"]
        if args.dataset_name == "ARC":
            answer = data["choices"]["label"].index(data["answerKey"])
            gt = data["answerKey"]
            options = data["choices"]["text"]
        if args.dataset_name == "mmlu":
            gt = data["answer"]
            answer = gt #data["choices"].index(data["answer"])
            options = data["choices"]
        if args.dataset_name == "gsm":
            answer = data["answer"]
            gt = solve_math_problems(answer)

        if args.dataset_name == "mmlu":
            agent_contexts = [
                [
                    {
                        "role": "user",
                        #Can you answer the following question as accurately as possible? {}: A) {}, B) {}, C) {}, D) {} Explain your answer, putting the answer in the form (X) at the end of your response.".format(question, a, b, c, d)

                        "content": """Can you answer the following question as accurately as possible? {} Options: A) {}, B) {}, C) {}, D) {}. Explain your reasoning step by step. At the end, give the final answer as a single letter (A, B, C, or D) in the form \\boxed{{<letter>}}.""".format(
                            question, options[0], options[1], options[2], options[3]
                        ),
                    }
                ]
                for agent in range(agents)
            ]
        elif args.dataset_name == "gsm":
            agent_contexts = [
                [
                    {
                        "role": "user",
                        "content": """Can you solve the following math problem? {} Step-by-step explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. """.format(
                            question
                        ),
                    }
                ]
                for agent in range(agents)
            ]

        elif args.dataset_name == "ARC":
            letters = ["A", "B", "C", "D", "E", "F"]          # labels in ARC order
            option_str = ", ".join(
                f"{letters[i]}) {text}" for i, text in enumerate(options)
            )
            allowed = ", ".join(letters[: len(options)])

            agent_contexts = [
                [
                    {
                        "role": "user",
                        "content": (
                            f"Can you answer the following question? {question} "
                            f"Options: {option_str}. "
                            f"At the end, give the final answer as a single letter "
                            f"({allowed}) in the form \\boxed{{<letter>}}."
                        ),
                    }
                ]
                for agent in range(agents)
            ]

        all_probabilities = []
        all_pred_ans = []
        unique_answers = set()

        for round in range(rounds):
            probability = []
            pred_ans = []
            for i, agent_context in enumerate(agent_contexts):
                if round != 0:
                    if args.selfref:
                        agent_contexts_other = agent_contexts
                    elif args.max_prob_element:
                        agent_contexts_other = [
                            agent_contexts[np.argmax(all_probabilities[round - 1])]
                        ]
                    else:
                        agent_contexts_other = (
                            agent_contexts[:i] + agent_contexts[i + 1 :]
                        )

                    if args.unique_response_only:
                        idx = None
                    else:
                        idx = 2 * round - 1
                    message = construct_message(agent_contexts_other, question, idx)
                    agent_context.append(message)

                unique_samples_for_this_agent = []
                model, tokenizer = models_tokenizers[i]

                # if args.history:
                text = tokenizer.apply_chat_template(
                    agent_context, tokenize=False, add_generation_prompt=True
                )
                result = generate_chat_response(
                    model, tokenizer, text, device=devices[i], length_normalize_log_likelihood=True
                )
                completion = result["completions"]
                log_prob_sum = result["log_likelihoods"]
                result["prompt"] = text
                result["agent_context"] = agent_context
                result["question"] = question
                result["answer"] = answer
                result["gt"] = gt
                result["num_samples"] = args.num_samples
                result[i] = {}
                result[i]["log_likelihoods"] = result["log_likelihoods"]
                result[i]["tokens_list"] = result["tokens_list"]
                for key in ["log_likelihoods", "tokens_list"]:
                    del result[key]
                if args.dataset_name == "mmlu" or args.dataset_name == "ARC":
                    result["options"] = options
                all_results[(round, i, questions_no)] = result
                json.dump(result, open(os.path.join(all_results_json, f"{round}_{i}_{questions_no}.json"), "w"))


                parsed_values = []
                for sample_idx in range(args.num_samples):
                    if args.dataset_name == "gsm":
                        parsed_str = solve_math_problems(completion[sample_idx])
                    elif args.dataset_name == "mmlu" or  args.dataset_name == "ARC":
                        print(completion[sample_idx])
                        parsed_str = parse_answer_mcq(completion[sample_idx])
                        print("parsed_str pre: ", parsed_str)
                        if parsed_str is None:
                            parsed_str = solve_math_problems_mcq(completion[sample_idx])
                        parsed_str = parse_mcq_answer(parsed_str)

                        print("parsed_str post: ", parsed_str)
                        #parsed_str = parse_mcq_answer(completion[sample_idx])
                    try:
                        parsed_float = float(parsed_str)
                    except (ValueError, TypeError):
                        parsed_float = None

                    ans_key = (parsed_str, parsed_float)
                    parsed_values.append(ans_key)
                    pred_ans.append(parsed_str)
                    probability.append(log_prob_sum[sample_idx])
                    if args.unique_response_only:
                        if ans_key in unique_answers:  # unique_answers_for_this_agent:
                            continue
                    unique_answers.add(ans_key)
                    assistant_message = construct_assistant_message(
                        completion[sample_idx]
                    )
                    unique_samples_for_this_agent.append(assistant_message)

                agent_context.extend(unique_samples_for_this_agent)
                all_results[(round, i, questions_no)]["parsed_values"] = parsed_values
            print(
                "Round: ",
                round,
                "prob: ",
                probability,
                "gt: ",
                gt,
                "pred ans: ",
                pred_ans,
            )
            tmp_pro = copy.deepcopy(probability)
            all_probabilities.append(tmp_pro)
            tmp_pred_ans = copy.deepcopy(pred_ans)
            all_pred_ans.append(tmp_pred_ans)

        # for i in range(agents):
        #     prompt_list = []
        #     completion_list = []
        #     id_list = []
        #     for key, result in sorted(all_results.items(), key=lambda x: x[0]):
        #         if key[0] != rounds - 1 or key[1] == i:
        #             print(f"Skipping {key} for agent {i}")
        #             continue
        #         for completion in result["completions"]:
        #             prompt_list.append(result["prompt"])
        #             completion_list.append(completion)
        #             id_list.append(key)
        #     for model_idx, (model, tokenizer), key in enumerate(zip(models_tokenizers, id_list)):
        #         if model_idx != i:
        #             continue
        #         logits_list, targets_list, log_likelihoods = compute_logits_and_targets(
        #             prompt_list,
        #             completion_list,
        #             model,
        #             tokenizer,
        #             batch_size=16,
        #             length_normalize_log_likelihood=True
        #         )
        #         result = all_results[key]
        #         # result[i]["logits_list"] = logits_list
        #         result[i]["tokens_list"] = targets_list
        #         result[i]["log_likelihoods"] = log_likelihoods
        #         all_results[key] = result
        #         json.dump(result, open(os.path.join(all_results_json, f"{round}_{i}_{questions_no}.json"), "w"))
        
        with open(
            log_probs_file,
            "a",
        ) as prob_file:
            with open(log_probs_file, "a") as prob_file:
                prob_file.write(str(all_probabilities) + "\n")

        with open(pred_ans_file, "a") as ans_file:
            ans_file.write(str(all_pred_ans) + "\n")

        generated_description[question] = [agent_contexts, answer]
        questions_no += 1
        with open(
            output_file,
            "w",
        ) as json_file:
            json.dump(generated_description, json_file)
