import argparse
import json
import math
import os
import random

import numpy as np
import torch


def get_difficulty_safe(item):
    """Safely gets and converts difficulty, handling errors."""
    difficulty = item.get("difficulty")
    if difficulty is None:
        return None
    try:
        difficulty_float = float(difficulty)
        return (
            difficulty_float
            if not (math.isnan(difficulty_float) or math.isinf(difficulty_float))
            else None
        )
    except (ValueError, TypeError):
        return None


def save_json(data_list, output_path, description):
    """Helper function to save a list to a JSON file."""
    if data_list and isinstance(data_list[0], dict) and "data" in data_list[0]:
        data_to_save = [item["data"] for item in data_list]
    else:
        data_to_save = data_list
    if not data_to_save:
        return

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=4, ensure_ascii=False)
        print(f"save to: {output_path}")


def random_sample(data_list, k, output_path, description):
    """Helper: Randomly samples k items, NO sorting afterwards."""
    if not isinstance(data_list, list):
        print(f"Error ({description})")
        return

    n = len(data_list)
    k = min(n, k)
    sampled = data_list if k >= n else random.sample(data_list, k)
    save_json(
        sampled,
        output_path,
        f"{description} (random sample: {len(sampled)})",
    )


def difficulty_sorted_sample(data_list, k, output_path, description):
    """Helper: Sorts list by difficulty descending, samples k items using torch.linspace."""
    if not data_list or k <= 0:
        return
    n = len(data_list)
    actual_k = min(n, k)
    sorted_list = sorted(data_list, key=lambda x: x["difficulty_float"], reverse=True)
    sampled = []
    if actual_k >= n:
        sampled = sorted_list
    else:
        indices = torch.linspace(0, n - 1, steps=actual_k).round().long()
        indices = torch.clamp(indices, 0, n - 1)
        unique_indices = torch.unique(indices)
        sampled = [sorted_list[i] for i in unique_indices]
    save_json(
        sampled,
        output_path,
        f"{description}",
    )


def gaussian_sample(data_list, k, output_path, description, center=0.3, std_dev=0.2):
    """Samples k items based on a Gaussian distribution centered on 'center'."""
    if not data_list or k <= 0:
        return

    n = len(data_list)
    actual_k = min(n, k)

    if actual_k == 0:
        return

    difficulties = [item["difficulty_float"] / 100.0 for item in data_list]

    probs = np.exp(-((np.array(difficulties) - center) ** 2) / (2 * std_dev**2))
    probs /= np.sum(probs)  # Normalize to sum to 1

    try:
        sampled = [data_list[i] for i in np.random.choice(n, k, False, p=probs)]
        save_json(
            sampled,
            output_path,
            f"{description} (gaussian,mean: {center}, var:{std_dev})",
        )
    except ValueError as e:
        print(f"{e}")


def process_ddata(input_json_path, output_prefix, task, k=2500):
    try:
        with open(input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as E:
        print(f"{E}")
        return

    valid_items = []
    for item in data:
        d = get_difficulty_safe(item)
        if isinstance(item, dict) and d is not None:
            valid_items.append(
                {"difficulty_float": d, "p_value": d / 100.0, "data": item}
            )
    if len(valid_items) == 0:
        return
    print(f"valid data: {len(valid_items)}条 (original: {len(data)}条)")

    if task == "0070_all":
        subset = [item for item in valid_items if 0 < item["p_value"] <= 0.7]
        difficulty_sorted_sample(
            subset,
            k,
            f"{output_prefix}_0070_all.json",
            "(0 < p <= 0.7)",
        )

    elif task == "gaussian_03":
        subset = [item for item in valid_items if item["p_value"] > 0]
        gaussian_sample(
            subset,
            k,
            f"{output_prefix}_gaussian_03.json",
            "gaussian: 0.3 center, 0.2 variance",
        )
    elif task == "random_sample":
        random_sample(valid_items, k, f"{output_prefix}_random.json", "random_sample")

    print("\n finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json")
    parser.add_argument(
        "-o",
        "--output_prefix",
        default="",
    )
    parser.add_argument("-t", "--task", default="")
    parser.add_argument(
        "-k",
        "--k_dynamic_total",
        default=2500,
    )
    args = parser.parse_args()
    if not args.output_prefix:
        args.output_prefix = args.input_json[:-5]
    print(f"prefix: {args.output_prefix}")
    args.k_dynamic_total = int(args.k_dynamic_total)
    process_ddata(args.input_json, args.output_prefix, args.task, args.k_dynamic_total)
