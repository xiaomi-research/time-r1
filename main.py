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
import math
import os
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
from datasets import Dataset
from deepspeed.runtime.fp16.loss_scaler import LossScaler
from deepspeed.runtime.zero.config import ZeroStageEnum
from rouge_score import rouge_scorer
from src.time_r1 import TimeR1_Trainer
from tqdm import tqdm
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config

torch.serialization.add_safe_globals([ZeroStageEnum])
torch.serialization.add_safe_globals([LossScaler])


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
    max_window_layers: int = field(
        default=2, metadata={"help": "sliding window layers bottom"}
    )
    sliding_window_length: int = field(
        default=4096, metadata={"help": "sliding window length"}
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

    is_early_stopping: bool = field(
        default=False,
        metadata={"help": "Whether to use early stopping"},
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
                iou = intersection / union

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


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    matches = [re.fullmatch(pattern, content.strip()) for content in completions]
    print("matches:", matches)
    return [1.0 if match else 0.0 for match in matches]


def extract_think_content(completion: str) -> Optional[str]:
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
                    # Handle potential errors gracefully, e.g., assign neutral dissimilarity

            if count > 0:
                group_rewards[j] = total_dissimilarity / count
            else:  # Handle case with only one generation or all others failed
                group_rewards[j] = 0.0

        diversity_rewards.extend(group_rewards.tolist())

    print("diversity_rewards", diversity_rewards)
    return diversity_rewards


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


def load_json_dataset_tg(
    train_data_path, is_curriculum_learning=False, preprocessed_data_path=None
):  # 移除了 video_folder 参数

    def create_dataset_from_json(file_path, split_name):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        examples = []
        for item in tqdm(data, desc=f"Processing {split_name} items"):
            video_path = item.get("video")
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
            print(f"  sample: {i+1}: {ex}")

        dataset = Dataset.from_list(examples)

        def __getitem__(self, idx):
            example = dataset[idx]
            return example

        from types import MethodType

        dataset.__getitem__ = MethodType(__getitem__, dataset)
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
            trainer.save_model(epoch_checkpoint_dir)


class StopAfterNEpochsCallback(TrainerCallback):
    def __init__(self, num_epochs_to_train=1):
        super().__init__()
        self.num_epochs_to_train = num_epochs_to_train
        print(
            f"Callback initialized: Training will stop after {self.num_epochs_to_train} completed epoch(s)."
        )

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.epoch >= self.num_epochs_to_train:
            print(
                f"Epoch {state.epoch:.0f} completed. Stopping training as per StopAfterNEpochsCallback (target: {self.num_epochs_to_train} epoch(s))."
            )
            control.should_training_stop = True


def set_global_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def main(script_args, training_args, model_args):

    set_global_seed(42)

    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    metric_funcs = list(metric_funcs_registry.values())

    dataset = load_json_dataset_tg(
        script_args.train_data_path,
        script_args.is_curriculum_learning,
    )

    trainer_cls = (
        TimeR1_Trainer
    )
    print("using: ", trainer_cls)

    callbacks_list = []
    if script_args.is_early_stopping:
        callbacks_list.append(StopAfterNEpochsCallback())

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
        callbacks=callbacks_list,
    )

    # Train and push the model to the Hub
    # trainer.train()
    if training_args.resume_from_checkpoint is not None:
        trainer_state_path = os.path.join(
            training_args.resume_from_checkpoint, "trainer_state.json"
        )
        if os.path.exists(trainer_state_path):
            print(f"Loading trainer state from: {trainer_state_path}")
            with open(trainer_state_path, "r") as f:
                trainer_state = json.load(f)
            resumed_global_step = trainer_state.get("global_step", 0)

        num_micro_batches_per_epoch_per_gpu = len(trainer.get_train_dataloader())
        max_step = math.ceil(
            trainer.args.num_train_epochs
            * num_micro_batches_per_epoch_per_gpu
            / trainer.args.gradient_accumulation_steps
        )
        trainer.args.max_steps = resumed_global_step + max_step

        if hasattr(trainer, "state") and hasattr(trainer.state, "max_steps"):
            trainer.state.max_steps = max_step
        else:
            print(
                "Warning: trainer.state.max_steps not found or state not fully initialized. Relying on trainer.args.max_steps."
            )


        print(
            f"Resuming training from checkpoint: {training_args.resume_from_checkpoint}"
        )
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, MY_GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
