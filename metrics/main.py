import os
import argparse
import torch
import concurrent.futures
import gc

from metrics.utils import model_tokenizer_generator, get_prompt_completion_pairs
from metrics.inference import compute_metrics
from global_utils.json_utils import load_all_jsons, write_all_jsons

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_models", type=int, default=3)
    parser.add_argument("--results_folder", type=str, default="/work/hdd/bcwu/agrawal5/Debate_uncern/gsm/results")
    parser.add_argument("--replace_in_place", action="store_true")
    parser.add_argument("--use_single_gpu", action="store_true")
    parser.add_argument("--no_length_normalize_log_likelihood", action="store_false", dest="length_normalize_log_likelihood")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=24)
    args = parser.parse_args()

    if not args.replace_in_place:
        output_folder = os.path.join(args.results_folder, "calibrated")
        os.makedirs(output_folder, exist_ok=True)
    else:
        output_folder = args.results_folder

    model_names = [
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Ministral-8B-Instruct-2410",
        "meta-llama/Llama-3.1-8B-Instruct",
    ][:args.num_models]
    if args.device == "cuda":
        if args.use_single_gpu:
            devices = ["cuda:0" for _ in range(len(model_names))]
        else:
            devices = [f"cuda:{i}" for i in range(len(model_names))]
    else:
        devices = (["cpu"] * len(model_names))[: args.num_models]

    json_list = get_prompt_completion_pairs(results_folder=args.results_folder)

    metrics = [None for _ in range(len(model_names))]
    if args.use_single_gpu:
        # generate metrics for each model sequentially
        for model_idx, (model, tokenizer) in enumerate(model_tokenizer_generator(model_names, devices)):
            metrics[model_idx] = compute_metrics(json_list,
                                                 model,
                                                 tokenizer,
                                                 batch_size=args.batch_size,
                                                 device=devices[model_idx],
                                                 length_normalize_log_likelihood=args.length_normalize_log_likelihood,
                                                 num_workers=args.num_workers)
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()
    else:
        # generate metrics for each model in parallel across GPUs
        # kick off one thread per model
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(model_names)) as executor:
            future_to_idx = {}
            for model_idx, (model, tokenizer) in enumerate(model_tokenizer_generator(model_names, devices)):
                # submit compute_metrics; it should internally set num_workers = total_cores/num_models
                future = executor.submit(
                    compute_metrics,
                    json_list,
                    model,
                    tokenizer,
                    batch_size=args.batch_size,
                    device=devices[model_idx],
                    length_normalize_log_likelihood=args.length_normalize_log_likelihood,
                    num_workers=(args.num_workers // len(model_names))
                )
                future_to_idx[future] = model_idx

            # collect results as they finish
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    metrics[idx] = future.result()
                except Exception as e:
                    print(f"[!] Error computing metrics for {model_names[idx]}: {e}")
                    metrics[idx] = None

    generated_jsons = load_all_jsons(args.results_folder)
    for generated_json in generated_jsons.values():
        generated_json["metrics"] = [[None for _ in range(len(generated_json["completions"]))] for _ in range(len(model_names))]
    for model_idx, model_metrics in enumerate(metrics):
        for (file_key, completion_idx), completion_metrics in model_metrics.items():
            generated_jsons[file_key]["metrics"][model_idx][completion_idx] = completion_metrics
    print(f"Writing to {output_folder}")
    write_all_jsons(generated_jsons, output_folder)

if __name__ == "__main__":
    main()