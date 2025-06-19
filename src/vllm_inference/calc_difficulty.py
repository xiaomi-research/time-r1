import argparse
import json
import os
import re

import numpy as np
from data.data_loader import load_tvgbench_filter


def compute_IoU(pred, gt):
    """Compute the IoU given predicted and ground truth windows."""
    assert isinstance(pred, list) and isinstance(gt, list)
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    if not pred_is_list:
        pred = [pred]
    if not gt_is_list:
        gt = [gt]
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(0.0, union_right - union_left)
    overlap = 1.0 * inter / union
    if not gt_is_list:
        overlap = overlap[:, 0]
    if not pred_is_list:
        overlap = overlap[0]
    return overlap


def calc_difficulty(pred, gt):
    if None in pred:
        return 0.0
    return compute_IoU(pred, gt) * 100.0


def extract_answer_force(output_string):
    number_pattern = r"\d+(?:\.\d+)?"
    matches = re.findall(number_pattern, output_string)
    output = [float(num) for num in matches[:2]]
    if len(output) == 2:
        return output
    return [None, None]


def load_new_data(data_dir):
    data = {}
    for file in os.listdir(data_dir):
        if "jsonl" not in file:
            continue
        file_path = os.path.join(data_dir, file)
        for line in open(file_path):
            tmp = json.loads(line)
            if (
                None in tmp["pred"]
            ):  # The model output may not follow the rules but still be correct.
                tmp["pred"] = extract_answer_force(tmp["output_text"])
            data[tmp["qid"]] = {
                "difficulty": calc_difficulty(tmp["pred"], tmp["target"]),
                "pred": tmp["pred"],
            }

    return data


def calc_score(difficulty_data_dict):
    data = list(difficulty_data_dict.values())
    for i in [30.0, 50.0, 70.0]:
        cnt = len([itm for itm in data if itm["difficulty"] > i])
        score = round(cnt / len(difficulty_data_dict) * 100, 1)
        print(score)


def main(input_dir=None, split="p03c_v2_top_2500", output_dir=None):
    original_data = load_tvgbench_filter(split=split)
    difficulty_data_dict = load_new_data(f"./{input_dir}")
    # steps = input_dir.split('_')[-1]

    print(len(difficulty_data_dict))
    calc_score(difficulty_data_dict)

    new_data = []
    for itm in original_data:
        if itm["qid"] in difficulty_data_dict:
            itm["difficulty"] = difficulty_data_dict[itm["qid"]]["difficulty"]
            itm["pred"] = difficulty_data_dict[itm["qid"]]["pred"]
            new_data.append(itm)

    if len(new_data) != len(original_data):
        print("Not All!! Attention!!")

    if not os.path.exists(f"{output_dir}/{input_dir}"):
        os.makedirs(f"{output_dir}/{input_dir}")
    path_name = f"{output_dir}/{input_dir}/train_v4_cloud.json"

    with open(path_name, "w") as f:
        json.dump(new_data, f)

    print(len(new_data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")  # Updated description
    parser.add_argument("--input", help="logging file")
    parser.add_argument("--split", help="annotation loading")
    parser.add_argument("--output_dir")
    args = parser.parse_args()
    main(input_dir=args.input, split=args.split, output_dir=args.output_dir)
