import json
import os

import datasets
import pandas as pd


def load_activitynet(split="test"):
    data_root = "./dataset/activitynet"
    data_path = f"{data_root}/annotations/sentence_temporal_grounding/{split}.json"
    data = json.load(open(data_path))
    qid, conv_data = 0, []

    for video_id, meta_data in data.items():
        video_path = None
        for ext in ["mp4", "mkv", "webm"]:
            tmp = os.path.join(f"{data_root}/videos", f"{video_id}.{ext}")
            if os.path.exists(tmp):
                video_path = tmp
                break
        assert video_path is not None

        for i in range(len(meta_data["timestamps"])):
            conv_data.append(
                {
                    "video": video_path,
                    "duration": meta_data["duration"],
                    "timestamp": meta_data["timestamps"][i],
                    "sentence": meta_data["sentences"][i].strip(),
                    "qid": f"activitynet_{qid}",
                }
            )
            qid += 1

    return conv_data


def load_charades(split="test"):
    data_root = "./dataset/charades"
    data_path = f"{data_root}/Charades_anno/Charades_sta_{split}.json"
    if not os.path.exists(data_path):
        data = {}
        old_data_path = f"{data_root}/Charades_anno/Charades_sta_{split}.txt"
        data_csv = f"{data_root}/Charades_anno/Charades_v1_{split}.csv"
        df = pd.read_csv(data_csv)
        video_to_duration = dict(zip(df["id"], df["length"]))

        for line in open(old_data_path):
            if line.strip() == "":
                continue
            meta_data, sentence = line.split("##")
            video_id, start, end = meta_data.split(" ")
            if video_id not in data:
                data[video_id] = {
                    "duration": video_to_duration[video_id],
                    "timestamps": [],
                    "sentences": [],
                }
            data[video_id]["timestamps"].append([float(start), float(end)])
            data[video_id]["sentences"].append(sentence)
        with open(data_path, "w") as f:
            json.dump(data, f)
    else:
        data = json.load(open(data_path))

    qid, conv_data = 0, []
    for video_id, meta_data in data.items():
        video_path = os.path.join(f"{data_root}/Charades_v1", f"{video_id}.mp4")
        for i in range(len(meta_data["timestamps"])):
            conv_data.append(
                {
                    "video": video_path,
                    "duration": meta_data["duration"],
                    "timestamp": meta_data["timestamps"][i],
                    "sentence": meta_data["sentences"][i].strip(),
                    "qid": f"charades_{qid}",
                }
            )
            qid += 1

    return conv_data


def load_tvgbench_filter(split):
    data_path = split
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    qid, conv_data = 0, []
    for meta_data in data:
        video = meta_data["video"]
        duration = meta_data["duration"]
        timestamps = meta_data["timestamp"]
        sentences = meta_data["sentence"]
        qid = meta_data["qid"]
        pred = meta_data["pred"]
        video_start = meta_data["video_start"]
        video_end = meta_data["video_end"]
        conv_data.append(
            {
                "video": video,
                "duration": duration,
                "timestamp": timestamps,
                "pred": pred,
                "sentence": sentences,
                "qid": qid,
                "video_start": video_start,
                "video_end": video_end,
            }
        )

    return conv_data


def load_tvgbench(split="default"):
    """
    Load JSON data in TVGBench format.

    Args:
        data_path (str): Path to the JSON file in TVGBench format.

    Returns:
        list: A list containing processed data, where each element is a dictionary
            in the format {'video': str, 'duration': float, 'timestamp': list[float, float], 'sentence': str, 'qid': str}.
            Returns an empty list if the file does not exist or cannot be parsed.
    """
    data_path = "./dataset/timer1/annotations/tvgbench.json"

    with open(data_path, "r") as f:
        raw_data = json.load(f)

    qid_counter = 0
    conv_data = []

    for item in raw_data:

        video_path = item["path"]

        if not os.path.exists(video_path):
            continue

        duration_str = item["duration"]
        answer_str = item["answer"]
        question_str = item["question"]
        start = item["start"]
        end = item["end"]
        duration = duration_str

        parts = answer_str.split("-")

        start_time = float(parts[0])
        end_time = float(parts[1])
        timestamp = [start_time, end_time]

        sentence = question_str

        if "source" in item and isinstance(item["source"], str):
            source_filename = os.path.basename(item["source"])
            source_prefix = (
                os.path.splitext(source_filename)[0].replace(".", "_").replace("-", "_")
            )

        qid_str = f"{source_prefix}_{qid_counter}"
        qid_counter += 1

        conv_data.append(
            {
                "video": video_path,
                "duration": duration,
                "timestamp": timestamp,
                "sentence": sentence,
                "qid": qid_str,
                "start": start,
                "end": end,
            }
        )

    return conv_data


def load_videomme(split="default"):
    if split in ["test", "train"]:
        split = "default"
    assert split in ["short", "medium", "long", "default"]
    data_root = "./dataset/videomme"
    data_path = f"{data_root}/videomme"

    conv_data = []
    data = datasets.load_dataset(
        "parquet", split="test", data_dir=data_path, streaming=True
    )
    for itm in data:
        if split == "default" or itm["duration"] == split:
            video_path = os.path.join(f"{data_root}/data", itm["videoID"] + ".mp4")
            conv_data.append(
                {
                    "video": video_path,
                    "question": itm["question"],
                    "options": [op[2:].strip() for op in itm["options"]],
                    "answer": ord(itm["answer"]) - ord("A"),
                    "duration": None,
                    "qid": f'videomme_{itm["question_id"]}',
                }
            )

    return conv_data


def load_egoschema(split="default"):
    if split in ["test", "train"]:
        split = "default"
    assert split in ["default", "subset"]
    data_root = "./dataset/egoschema"
    if split == "subset":
        data_path = f"{data_root}/Subset"
    else:
        data_path = f"{data_root}/MC"

    conv_data = []
    data = datasets.load_dataset(
        "parquet", split="test", data_dir=data_path, streaming=True
    )
    for itm in data:
        video_path = os.path.join(f"{data_root}/videos", itm["video_idx"] + ".mp4")
        conv_data.append(
            {
                "video": video_path,
                "question": itm["question"],
                "options": [op[2:].strip() for op in itm["option"]],
                "answer": itm["answer"],
                "duration": None,
                "qid": f'egoschema_{itm["question_idx"]}',
            }
        )

    return conv_data


def load_tempcompass(split="default"):
    if split in ["test", "train", "default"]:
        split = "multi-choice"
    assert split in ["multi-choice", "captioning", "caption_matching", "yes_no"]
    data_root = "./dataset/tempcompass"
    data_path = f"{data_root}/questions/{split}.json"

    conv_data = []
    for key, value in json.load(open(data_path)).items():
        video_path = os.path.join(f"{data_root}/videos", key + ".mp4")
        for dim in value.keys():
            for idx, itm in enumerate(value[dim]):
                question, options, answer = itm["question"], [], itm["answer"]
                if split == "yes_no":
                    options = ["yes", "no"]
                    answer = options.index(answer)
                if split == "caption_matching":
                    tmp = question.split("\n")
                    question, options, answer = (
                        tmp[0],
                        [],
                        ":".join(answer.split(":")[1:]).strip(),
                    )
                    for i in range(1, len(tmp)):
                        option = ":".join(tmp[i].split(":")[1:]).strip()
                        options.append(option)
                    answer = options.index(answer)
                if split == "multi-choice":
                    tmp = question.split("\n")
                    question, options, answer = tmp[0], [], ord(answer[0]) - ord("A")
                    for i in range(1, len(tmp)):
                        options.append(tmp[i][2:].strip())

                conv_data.append(
                    {
                        "video": video_path,
                        "question": question,
                        "options": options,
                        "answer": answer,
                        "duration": None,
                        "qid": f"tempcompass|{split}|{key}|{dim}|{idx}",
                    }
                )

    return conv_data


def load_mvbench(split="default"):
    data_root = "./dataset/mvbench"
    data_path = f"{data_root}/json"

    DATASET_CONFIG = {
        "action_sequence": f"{data_root}/video/star/Charades_v1_480/",
        "action_prediction": f"{data_root}/video/star/Charades_v1_480/",
        "action_antonym": f"{data_root}/video/ssv2_video/",
        "fine_grained_action": f"{data_root}/video/Moments_in_Time_Raw/videos/",
        "unexpected_action": f"{data_root}/video/FunQA_test/test/",
        "object_existence": f"{data_root}/video/clevrer/video_validation/",
        "object_interaction": f"{data_root}/video/star/Charades_v1_480/",
        "object_shuffle": f"{data_root}/video/perception/videos/",
        "moving_direction": f"{data_root}/video/clevrer/video_validation/",
        "action_localization": f"{data_root}/video/sta/sta_video/",
        "scene_transition": f"{data_root}/video/scene_qa/video/",
        "action_count": f"{data_root}/video/perception/videos/",
        "moving_count": f"{data_root}/video/clevrer/video_validation/",
        "moving_attribute": f"{data_root}/video/clevrer/video_validation/",
        "state_change": f"{data_root}/video/perception/videos/",
        "fine_grained_pose": f"{data_root}/video/nturgbd/",
        "character_order": f"{data_root}/video/perception/videos/",
        "egocentric_navigation": f"{data_root}/video/vlnqa/",
        "episodic_reasoning": f"{data_root}/video/tvqa/output_videos/",
        "counterfactual_inference": f"{data_root}/video/clevrer/video_validation/",
    }

    conv_data = []
    for file_name in os.listdir(data_path):
        data_type = file_name.split(".")[0]
        data = json.load(open(os.path.join(data_path, file_name)))
        for qid, itm in enumerate(data):
            video_name = itm["video"]
            video_path = os.path.join(DATASET_CONFIG[data_type], video_name)
            conv_data.append(
                {
                    "video": video_path,
                    "question": itm["question"],
                    "options": itm["candidates"],
                    "answer": itm["candidates"].index(itm["answer"]),
                    "duration": None,
                    "qid": f"mvbench|{data_type}|{qid}",
                }
            )
            if "start" in itm and "end" in itm:
                video_name = (
                    itm["video"].split(".mp4")[0]
                    + "_"
                    + str(itm["start"]).replace(".", "-")
                    + "_"
                    + str(itm["end"]).replace(".", "-")
                    + ".mp4"
                )
                video_path = os.path.join(
                    DATASET_CONFIG[data_type], "split", video_name
                )
                conv_data[-1]["video"] = video_path
            else:
                if "start" in itm:
                    conv_data[-1]["video_start"] = itm["start"]
                if "end" in itm:
                    conv_data[-1]["video_end"] = itm["end"]

    return conv_data


def _extract_qid(itm):
    vtype, vid, question = (
        None,
        itm["video"].split("/")[-1].split(".")[0],
        itm["sentence"],
    )
    video_path = itm["video"].lower()
    if "cosmo" in video_path or "howto100m" in video_path:
        vtype = "cosmo"
    if "queryd" in video_path:
        vtype = "queryd"
    if "vtime" in video_path:
        vtype = "internvid-vtime"
        if ":" in vid:
            vid = vid.split(":")[0][:-3]
    if "didemo" in video_path:
        vtype = "didemo"
    if "yt_temporal_videos" in video_path:
        vtype = "yt-temporal"

    return f"my|{vtype}|{vid}|{question}"
