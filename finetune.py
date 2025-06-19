# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
from datasets import Dataset
from rouge_score import rouge_scorer
from src.time_r1 import TimeR1_Trainer_ft
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config


@dataclass
class MY_GRPOConfig(GRPOConfig):
    fix_vit: bool = field(
        default=False,
        metadata={"help": "Whether to fix the ViT model"},
    )

    slide_window: bool = field(
        default=False,
        metadata={"help": "Whether to use slide window"},
    )

    prompt_type: str = field(
        default="v1",
        metadata={"help": "Prompt type. Possible values: 'v1', 'v2', 'v3'"},
    )

    use_grpo: bool = field(
        default=False,
        metadata={"help": "Whether to use GRPO"},
    )


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'iou', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["iou", "format"],
        metadata={"help": "List of reward functions. Possible values: 'iou', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

    train_data_path: str = field(
        default="./dataset/finetune/charades/Charades/charades_annotation/train.json",
        metadata={"help": "Path to the training data JSON file."},
    )

    eval_data_path: str = field(
        default="./dataset/finetune/charades/Charades/charades_annotation/val.json",
        metadata={"help": "Path to the evaluation data JSON file."},
    )

    video_folder: str = field(
        default="./dataset/finetune/charades/Charades/Charades_v1",  # Replace with your actual video folder path
        metadata={"help": "Path to the folder containing video files."},
    )

    is_curriculum_learning: bool = field(
        default=False,
        metadata={"help": "Whether to use curriculum learning."},
    )
    preprocessed_data_path: Optional[str] = (
        field(  # Add preprocessed_data_path argument
            default="",
            metadata={
                "help": "Path to the preprocessed dataset directory. If provided, load preprocessed data instead of raw videos."
            },
        )
    )


def parse_timestamp_output(output_string):
    """Parses timestamp output, similar to the example code."""
    # 1. Find all <answer>...</answer> blocks.
    answer_matches = re.findall(r"<answer>(.*?)</answer>", output_string, re.DOTALL)

    if not answer_matches:
        return None  # No <answer> tags found.

    # 2. Use the content of the *last* <answer> block.
    last_answer_content = answer_matches[-1]
    print("last_answer_content:", last_answer_content)

    matches = re.findall(
        r"(\d+\.?\d*) (to|and) (\d+\.?\d*)", last_answer_content, re.IGNORECASE
    )
    if not matches:
        return None
    last_match = matches[-1]
    start_time = float(last_match[0])
    end_time = float(last_match[2])
    return start_time, end_time


def iou_timestamp_reward(
    completions, solution, **kwargs
):  # Modified reward function name and arguments
    """Reward function that calculates IoU between predicted and ground truth timestamps."""
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(completions, solution):  # Added video_durations

        reward = 0.0
        parsed_times = parse_timestamp_output(content)
        start_time, end_time = 0, 0
        gt_start, gt_end = sol
        s, e = gt_start, gt_end
        if parsed_times:
            start_time, end_time = parsed_times
            from_number = start_time
            to_number = end_time

            intersection = max(0, min(to_number, e) - max(from_number, s))
            union = max(to_number, e) - min(from_number, s)
            if union > 0:
                iou = intersection / union  # 0.1 0.3

            reward = iou

        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"Content: {content}\n")
                f.write(f"pred second: {str(start_time)}, {str(end_time)}\n")
                f.write(f"gt second: {str(gt_start)}, {str(gt_end)}\n")
                f.write(
                    f"------------- {current_time} IoU reward: {reward} -------------\n"
                )  # Modified log message

    return rewards


def iou_timestamp_reward_v2(
    completions, solution, **kwargs
):  # Modified reward function name and arguments
    """Reward function that calculates IoU between predicted and ground truth timestamps."""
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    durations = kwargs.get("durations")
    for content, sol, duration in zip(
        completions, solution, durations
    ):  # Added video_durations

        reward = 0.0
        parsed_times = parse_timestamp_output(content)
        start_time, end_time = 0, 0
        gt_start, gt_end = sol
        s, e = gt_start, gt_end
        if parsed_times:
            start_time, end_time = parsed_times
            from_number = start_time
            to_number = end_time

            intersection = max(0, min(to_number, e) - max(from_number, s))
            union = max(to_number, e) - min(from_number, s)
            if union > 0:
                iou = intersection / union  # 0.1 0.3

            gt_start_norm = 1.0 * s / duration
            gt_end_norm = 1.0 * e / duration
            pred_start_norm = 1.0 * start_time / duration
            pred_end_norm = 1.0 * end_time / duration
            reward = (
                iou
                * (1 - abs(gt_start_norm - pred_start_norm))
                * (1 - abs(gt_end_norm - pred_end_norm))
            )

        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"Content: {content}\n")
                f.write(f"pred second: {str(start_time)}, {str(end_time)}\n")
                f.write(f"gt second: {str(gt_start)}, {str(gt_end)}\n")
                f.write(
                    f"------------- {current_time} IoU reward: {reward} -------------\n"
                )  # Modified log message

    return rewards


def mqa_answer_reward(
    completions, solution, task_type, **kwargs
):  # Modified reward function name and arguments
    """Reward function that calculates IoU between predicted and ground truth timestamps."""

    def extract_characters_regex(s):
        s = s.strip()
        answer_prefixes = [
            "The best answer is",
            "The correct answer is",
            "The answer is",
            "The answer",
            "The best option is",
            "The correct option is",
            "Best answer:" "Best option:",
        ]
        for answer_prefix in answer_prefixes:
            s = s.replace(answer_prefix, "")

        if len(s.split()) > 10 and not re.search("[ABCDEFG]", s):
            return ""

        matches = re.search(r"[ABCDEFG]", s)
        if matches is None:
            return ""
        return matches[0]

    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol, task in zip(completions, solution, task_type):
        if task != "mqa":
            rewards.append(None)
            continue

        reward = 0.0

        pattern_answer = r"<answer>(.*?)</answer>"

        match_answer = re.search(pattern_answer, content, re.DOTALL)

        answer = ""
        if match_answer:
            answer = match_answer.group(1)
            if extract_characters_regex(answer) == extract_characters_regex(sol):
                reward = 1.0
        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"Content: {content}\n")
                f.write(f"pred: {answer}\n")
                f.write(f"gt: {sol}\n")
                f.write(
                    f"------------- {current_time} ACC reward: {reward} -------------\n"
                )  # Modified log message

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    matches = [re.fullmatch(pattern, content.strip()) for content in completions]
    print("matches:", matches)
    return [1.0 if match else 0.0 for match in matches]


def extract_think_content(completion: str) -> Optional[str]:
    """
    Extract the content within the last <think>...</think> block from the completion.
    Use findall to get all matches, then take the last one.
    """
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    matches = think_pattern.findall(completion)
    if matches:
        return matches[-1].strip()
    return None


def reward_timestep_pair(
    completions: List[str],
    weight: float = 0.2,
    max_count: int = 1,
    **kwargs, 
) -> List[float]:
    rewards = []
    pair_pattern = re.compile(
        r"<timestep>\s*(\d+\.?\d*)\s+to\s+(\d+\.?\d*)\s*</timestep>",
        re.IGNORECASE | re.DOTALL,
    )

    for completion in completions:
        score = 0.0
        think_content = extract_think_content(completion)

        if think_content:
            pair_matches = pair_pattern.findall(think_content)
            pair_count = len(pair_matches)
            capped_count = min(pair_count, max_count)
            score = weight * capped_count
        else:
            score = 0.0

        rewards.append(max(0.0, score))

    print("reward_timestep_pair", rewards)
    return rewards


def reward_think_length(
    completions: List[str],
    weight: float = 0.001,
    max_length: int = 500,
    **kwargs,
) -> List[float]:
    rewards = []
    for completion in completions:
        score = 0.0
        think_content = extract_think_content(completion)

        if think_content:
            think_length = len(think_content)
            capped_length = min(think_length, max_length)
            score = weight * capped_length
        else:
            score = 0.0

        rewards.append(max(0.0, score))

    return rewards


DEFAULT_STRUCTURE_KEYWORDS = [
    "analyze",
    "compare",
    "deduce",
    "however",
    "therefore",
    "because",
    "step",
    "observe",
    "notice",
    "identify",
    "wait",
]


def reward_keyword_usage(
    completions: List[str],
    keywords: Optional[List[str]] = None,
    weight: float = 0.1,
    max_count: int = 2,
    **kwargs,
) -> List[float]:
    if keywords is None:
        keywords = DEFAULT_STRUCTURE_KEYWORDS
    rewards = []

    for completion in completions:
        score = 0.0
        think_content = extract_think_content(completion)

        if think_content:
            content_lower = think_content.lower()
            keyword_count = sum(1 for word in keywords if word in content_lower)
            capped_count = min(keyword_count, max_count)
            score = weight * capped_count
        else:
            score = 0.0

        rewards.append(max(0.0, score))

    return rewards


def reward_paragraph_structure(
    completions: List[str],
    weight: float = 0.05, 
    max_paragraphs: int = 2, 
    **kwargs,
) -> List[float]:
    rewards = []
    for completion in completions:
        score = 0.0
        think_content = extract_think_content(completion)

        if think_content:
            paragraphs = [p for p in think_content.split("\n") if p.strip()]
            capped_paragraphs = min(len(paragraphs), max_paragraphs)
            score = weight * capped_paragraphs
        else:
            score = 0.0

        rewards.append(max(0.0, score))

    return rewards


def diversity_reward_func(completions, num_generations=8, **kwargs):
    if not completions:
        return []

    batch_size = len(completions) // num_generations
    diversity_rewards = []
    scorer = rouge_scorer.RougeScorer(
        ["rougeL"], use_stemmer=True
    )

    for i in range(batch_size):
        group_start_idx = i * num_generations
        group_end_idx = (i + 1) * num_generations
        current_group_completions = completions[group_start_idx:group_end_idx]

        group_rewards = np.zeros(num_generations)
        for j in range(num_generations):
            total_dissimilarity = 0
            count = 0
            for k in range(num_generations):
                if j == k:
                    continue
                try:
                    # rouge_score expects strings, handle potential non-string content if necessary
                    score = scorer.score(
                        str(current_group_completions[j]),
                        str(current_group_completions[k]),
                    )["rougeL"].fmeasure
                    total_dissimilarity += 1.0 - score
                    count += 1
                except Exception as e:
                    print(
                        f"Warning: Error calculating ROUGE score: {e}. Skipping pair."
                    )

            if count > 0:
                group_rewards[j] = total_dissimilarity / count
            else:  # Handle case with only one generation or all others failed
                group_rewards[j] = 0.0

        diversity_rewards.extend(group_rewards.tolist())

    print("diversity_rewards", diversity_rewards)
    return diversity_rewards


def load_json_dataset_tg(
    train_data_path, is_curriculum_learning=False, preprocessed_data_path=None
):
    def create_dataset_from_json(file_path, split_name):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        examples = []
        for item in tqdm(data, desc=f"Processing {split_name} items"):
            video_path = item.get(
                "video"
            )  # .replace("images-ks3-starfs", "images-starfs")
            timestamps = item.get("timestamp")
            sentence = item.get("sentence")
            duration = item.get("duration")
            video_start = item.get("video_start")
            video_end = item.get("video_end")

            sentence = sentence.strip().lower()
            if sentence.endswith("."):
                sentence = sentence[:-1]

            if not os.path.isfile(video_path):
                continue

            example = {
                "task_type": "tg",
                "problem": sentence,
                "choices": "",
                "solution": (
                    float(timestamps[0]),
                    float(timestamps[1]),
                ), 
                "video_path": video_path, 
                "durations": duration,
                "video_start": video_start,
                "video_end": video_end,
                "preprocessed_path": "",
            }
            examples.append(example)

        if not examples:
            return None

        print("is_curriculum_learning:", is_curriculum_learning)
        if not is_curriculum_learning:
            random.shuffle(examples)

        for i, ex in enumerate(examples[:5]):
            print(f"  sample {i+1}: {ex}")

        dataset = Dataset.from_list(examples)

        def __getitem__(self, idx):
            example = dataset[idx]
            return example

        from types import MethodType

        dataset.__getitem__ = MethodType(__getitem__, dataset)

        return dataset

    train_dataset = create_dataset_from_json(train_data_path, "train")

    return train_dataset


def load_json_dataset(
    train_data_path, video_folder, preprocessed_data_path=None
):  # Modified to accept preprocessed_data_path
    def create_dataset_from_json(file_path, split_name):
        with open(file_path, "r") as f:
            data = json.load(f)
        examples = []
        for video_id, video_data in tqdm(data.items()):
            for sentence_id, (timestamps, sentence) in enumerate(
                zip(video_data["timestamps"], video_data["sentences"])
            ):
                sentence = sentence.strip().lower()
                if sentence.endswith("."):
                    sentence = sentence[:-1]
                video_filename_base = video_id
                video_path = None
                for ext in ["mp4", "mkv", "webm"]:
                    candidate_path = os.path.join(
                        video_folder, f"{video_filename_base}.{ext}"
                    )
                    if os.path.isfile(candidate_path):
                        video_path = candidate_path
                        break
                example = {
                    "problem": sentence,
                    "solution": (timestamps[0], timestamps[1]),
                    "video_path": video_path,
                    "durations": video_data["duration"],
                    "video_start": None,
                    "video_end": None,
                    "preprocessed_path": "",  # Initialize preprocessed_path as None
                }
                example["preprocessed_path"] = os.path.join(
                    preprocessed_data_path, video_id
                )
                if not os.path.exists(example["preprocessed_path"]):
                    print(
                        f"Warning: Preprocessed path not found for video_id: {video_id}"
                    )

                examples.append(example)

        random.shuffle(examples)
        print(len(examples))
        print(examples[:5])
        dataset = Dataset.from_list(examples)

        def __getitem__(
            self, idx
        ):  # Define getitem within the scope where dataset is available
            example = dataset[idx]

            # return example
            data_to_return = {
                k: v for k, v in example.items()
            }  # Create a copy to avoid modifying original dataset

            if isinstance(example["preprocessed_path"], list):
                data_to_return["video_inputs"] = [
                    torch.load(
                        os.path.join(example["preprocessed_path"][0], "video_inputs.pt")
                    )
                ]
                with open(
                    os.path.join(example["preprocessed_path"][0], "video_kwargs.json"),
                    "r",
                ) as f:
                    data_to_return["video_kwargs"] = [json.load(f)]
            else:
                data_to_return["video_inputs"] = [
                    torch.load(
                        os.path.join(example["preprocessed_path"], "video_inputs.pt")
                    )
                ]
                with open(
                    os.path.join(example["preprocessed_path"], "video_kwargs.json"), "r"
                ) as f:
                    data_to_return["video_kwargs"] = [json.load(f)]
            data_to_return["use_preprocessed"] = [
                True
            ]  # Flag to indicate preprocessed data is used

            return data_to_return

        dataset.__getitem__ = __getitem__.__get__(
            dataset, Dataset
        )  # Bind getitem to the dataset

        return dataset

    train_dataset = create_dataset_from_json(train_data_path, "train")
    return train_dataset


class SaveEpochEndCallback(TrainerCallback):
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero:
            trainer = kwargs.get("trainer")
            if trainer is None:
                return

            epoch_checkpoint_dir = os.path.join(
                args.output_dir, f"epoch-{int(state.epoch)}"
            )

            print(
                f"\n{'='*20} Callback: Saving model checkpoint at end of epoch {int(state.epoch)} to {epoch_checkpoint_dir} {'='*20}\n"
            )
            # 调用 trainer 的 save_model 方法
            trainer.save_model(epoch_checkpoint_dir)


def set_global_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def main(script_args, training_args, model_args):

    set_global_seed(42)

    # Get reward functions
    print(script_args.reward_funcs)
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    metric_funcs = list(metric_funcs_registry.values())

    print(reward_funcs)
    print(metric_funcs)
    print("dataset_tg")
    dataset = load_json_dataset(
        script_args.train_data_path,
        script_args.video_folder,
        script_args.preprocessed_data_path,
    )

    print(len(dataset))

    print(dataset.__getitem__(10).keys())

    trainer_cls = TimeR1_Trainer_ft
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        metric_funcs=metric_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        callbacks=[SaveEpochEndCallback()],
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


reward_funcs_registry = {
    "iou": iou_timestamp_reward,  # Modified registry to use iou_timestamp_reward
    "iou_v2": iou_timestamp_reward_v2,
    "format": format_reward,
}

metric_funcs_registry = {
    "reward_timestep_pair": reward_timestep_pair,
    "reward_think_length": reward_think_length,
    "reward_keyword_usage": reward_keyword_usage,
    "reward_paragraph_structure": reward_paragraph_structure,
    # "diversity_reward_func": diversity_reward_func,
}

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, MY_GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
