import argparse
import json
import os
import random
import re

import numpy as np
import requests
from data.data_loader import *

random.seed(42)


def find_dataset_results(dataset_name, split, model_name):
    """
    logs/eval/{model_name}/{dataset_name}
    e.g., /logs/eval/kl_cot_gaussian_03_iouv2_2500/tvgbench
    """
    data_dirs = []
    eval_root = f"./logs/eval/{model_name}"
    for data_dir in os.listdir(eval_root):
        if dataset_name in data_dir:
            data_dirs.append(os.path.join(eval_root, data_dir))
    return sorted(data_dirs)


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluation for training-free video temporal grounding (Single GPU Version)"
    )
    parser.add_argument(
        "--dataset",
        default=[
            "charades",
            "activitynet",
            "mvbench",
            "tvgbench",
            "videomme",
            "tempcompass",
            "egoschema",
        ],
        help="Specify the dataset.",
        choices=[
            "charades",
            "activitynet",
            "mvbench",
            "videomme",
            "tvgbench",
            "videomme",
            "egoschema",
            "tempcompass",
        ],
        nargs="+",
    )
    parser.add_argument("--split", type=str, default="test", help="dataset type")
    parser.add_argument(
        "--model_name",
        type=str,
        default="kl_cot_gaussian_03_iouv2_2500",
        help="model name",
    )
    return parser.parse_args()


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


def mcq_is_correct(pred, gt):
    gt = chr(gt + ord("A"))
    matches = re.findall(r"\(([A-Z])\)", pred)
    if matches:
        return int(matches[-1] == gt)
    return int(pred[0] == gt)


def load_scored_data(data_dir, datasetname):
    data = {}
    cnt = 0
    for file in os.listdir(data_dir):
        if "jsonl" not in file:
            continue
        file_path = os.path.join(data_dir, file)
        for line in open(file_path):
            tmp = json.loads(line)
            cnt += 1
            if datasetname in ["activitynet", "charades", "tvgbench"]:
                score = 0.0
                if None not in tmp["pred"]:
                    score = compute_IoU(tmp["pred"], tmp["target"])
            else:
                if tmp["pred"] is not None:
                    score = int(tmp["pred"] == tmp["target"])
                else:
                    score = mcq_is_correct(tmp["output_text"], tmp["target"])
            data[tmp["qid"]] = score
    return data


def calc_score(difficulty_data_dict, datasetname):
    data = list(difficulty_data_dict.values())
    if datasetname in ["activitynet", "charades", "tvgbench"]:
        scores = {}
        scores["mIoU"] = np.mean([itm for itm in data]) * 100
        for i in [0.3, 0.5, 0.7]:
            cnt = len([itm for itm in data if itm > i])
            score = cnt / len(difficulty_data_dict) * 100.0
            scores[i] = score
        scores["avg"] = sum(scores.values()) / len(scores)
    else:
        correct = sum([itm for itm in data])
        scores = {
            "correct": correct,
            "total": len(data),
            "avg": round(correct / len(data) * 100, 2),
        }
    return scores


def upload_json_to_server(
    data, api_url="https://validation-server.onrender.com/api/upload/"
):
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url=api_url, headers=headers, json=data)
        response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            return {"status": "success", "response_text": response.text}

    except requests.exceptions.RequestException as e:
        return {
            "status": "error",
            "message": str(e),
            "details": f"Failed to upload data to {api_url}",
        }


def eval_egoschema_online(data_dir, original_data):
    qid_to_vid = {}
    for itm in original_data:
        qid, vid = itm["qid"], itm["video"].split("/")[-1].split(".")[0]
        qid_to_vid[qid] = vid

    data = {}
    for file in os.listdir(data_dir):
        if "jsonl" not in file:
            continue
        file_path = os.path.join(data_dir, file)
        for line in open(file_path):
            tmp = json.loads(line)
            matches = re.findall(r"\(([A-Z])\)", tmp["output_text"])
            if matches:
                pred = ord(matches[-1]) - ord("A")
            else:
                pred = ord(random.choice(["A", "B", "C", "D", "E"])) - ord("A")
            data[qid_to_vid[tmp["qid"]]] = pred

    return upload_json_to_server(data)


def main(args):
    for dataset in args.dataset:
        if dataset == "charades":
            load_func = load_charades
        if dataset == "activitynet":
            load_func = load_activitynet
        if dataset == "mvbench":
            load_func = load_mvbench
        if dataset == "videomme":
            load_func = load_videomme
        if dataset == "tvgbench":
            load_func = load_tvgbench
        if dataset == "egoschema":
            load_func = load_egoschema
        if dataset == "tempcompass":
            load_func = load_tempcompass

            for split in ["multi-choice"]:
                original_data = load_func(split)
                print(f"==========={dataset} {split}===========")
                print(f"Original data length: {len(original_data)}")
                for data_dir in find_dataset_results(
                    dataset, args.split, args.model_name
                ):

                    print(f"data_dir: {data_dir}")
                    if "captioning" in data_dir:
                        continue
                    difficulty_data_dict = load_scored_data(data_dir, dataset)
                    if len(difficulty_data_dict) == 0:
                        continue
                    print(f"len(difficulty_data_dict): {len(difficulty_data_dict)}")
                    for k, v in calc_score(difficulty_data_dict, dataset).items():
                        print(v)
                    with open(data_dir + "/scores.json", "w") as f:
                        json.dump(
                            calc_score(difficulty_data_dict, dataset), f, indent=4
                        )
            continue

        original_data = None
        if dataset == "egoschema":
            original_data = load_func()

        print(f"==========={dataset}===========")

        if original_data is not None:
            print(f"Original data length: {len(original_data)}")
        for data_dir in find_dataset_results(dataset, args.split, args.model_name):

            print(f"data_dir: {data_dir}")
            if dataset == "egoschema":
                results_ego = eval_egoschema_online(data_dir, original_data)
                print(results_ego)
                with open(data_dir + "/scores.json", "w") as f:
                    json.dump(results_ego, f, indent=4)
                continue

            difficulty_data_dict = load_scored_data(data_dir, dataset)
            if len(difficulty_data_dict) == 0:
                continue
            print(f"len(difficulty_data_dict): {len(difficulty_data_dict)}")
            for k, v in calc_score(difficulty_data_dict, dataset).items():
                print(f"IoU R1@ {k}: {v}")
            with open(data_dir + "/scores.json", "w") as f:
                json.dump(calc_score(difficulty_data_dict, dataset), f, indent=4)


if __name__ == "__main__":
    args = get_args()
    main(args)
