import json
import os
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from dataclasses import dataclass
from multiprocessing import Manager
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import ProcessorMixin

from src.utils import process_vision_info_v3

from .config import *
from .data_loader import *


@dataclass
class MultiModalDataCollator:
    processor: "ProcessorMixin"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        utils, together_feature = {}, {}
        for feature in features:
            for key in feature["inputs"].keys():
                if feature["inputs"][key] is None:
                    continue
                if key not in together_feature:
                    together_feature[key] = []
                if isinstance(feature["inputs"][key], list):
                    together_feature[key] += feature["inputs"][key]
                else:
                    together_feature[key].append(feature["inputs"][key])

            for key in feature.keys():
                if key == "inputs":
                    continue
                if key not in utils:
                    utils[key] = []
                utils[key].append(feature[key])

        inputs = self.processor(
            text=together_feature["text"],
            images=together_feature.get("images", None),
            videos=together_feature.get("videos", None),
            padding="longest",
            truncation=True,
            return_tensors="pt",
            padding_side="left",
            do_rescale=False,
            **(
                {"fps": together_feature["fps"]}
                if together_feature["fps"] is not None
                else {}
            ),
        )
        for key in utils.keys():
            utils[key] = np.array(utils[key], dtype=object)

        return {"inputs": inputs, **utils}


@dataclass
class vllmMultiModalDataCollator:

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        raw_prompt_ids, multi_modal_data, mm_processor_kwargs = [], [], []

        utils = {}
        for feature in features:
            raw_prompt_ids.append(feature["inputs"]["raw_prompt_ids"])
            multi_modal_data.append(feature["inputs"]["multi_modal_data"])
            mm_processor_kwargs.append(feature["inputs"]["mm_processor_kwargs"])
            for key in feature.keys():
                if key == "inputs":
                    continue
                if key not in utils:
                    utils[key] = []
                utils[key].append(feature[key])

        raw_prompt_ids = np.array(raw_prompt_ids, dtype=object)
        multi_modal_data = np.array(multi_modal_data, dtype=object)
        mm_processor_kwargs = np.array(mm_processor_kwargs, dtype=object)
        for key in utils.keys():
            utils[key] = np.array(utils[key], dtype=object)

        return {
            "inputs": {
                "raw_prompt_ids": raw_prompt_ids,
                "multi_modal_data": multi_modal_data,
                "mm_processor_kwargs": mm_processor_kwargs,
            },
            **utils,
        }


class LimitedSizeSharedDict(MutableMapping):
    def __init__(self, max_size=8):
        self.manager = Manager()
        self._data = self.manager.dict()
        self._order = self.manager.list()
        self.max_size = max_size
        self.lock = self.manager.Lock()

    def __setitem__(self, key, value):
        with self.lock:
            if key not in self._data:
                if len(self._data) >= self.max_size:
                    oldest_key = self._order.pop(0)
                    del self._data[oldest_key]
                self._data[key] = value
                self._order.append(key)

    def __getitem__(self, key):
        with self.lock:
            return self._data[key]

    def __delitem__(self, key):
        with self.lock:
            del self._data[key]
            self._order.remove(key)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __contains__(self, key):
        return key in self._data

    def get(self, key, default=None):
        try:
            return self._data[key]
        except KeyError:
            return default


class BaseDataset(Dataset, ABC):
    def __init__(
        self,
        processor,
        sys_prompt="You are a helpful assistant.",
        min_pixels=None,
        total_pixels=None,
        max_frames=None,
        fps=None,
        cache_size=0,
    ):
        self.sys_prompt = sys_prompt
        self.min_pixels = min_pixels
        self.total_pixels = total_pixels
        self.max_frames = max_frames
        self.fps = fps

        if cache_size == 0:
            self.use_video_cache = False
        else:
            print("Use Video Cache!")
            self.use_video_cache = True
            self.video_cache = LimitedSizeSharedDict(max_size=cache_size)

        self.processor = processor
        print(self.default_ele())

    @staticmethod
    def _load_data(dataset_names, split):
        data = []
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        for name in dataset_names:
            if name == "charades":
                print("Use charades")
                data += load_charades(split=split)
            if name == "activitynet":
                print("Use activitynet")
                data += load_activitynet(split=split)
            if name == "tvgbench":
                print("Use tvgbench")
                data += load_tvgbench(split=split)
            if name == "tvgbench_filter":
                print("Use tvgbench_filter")
                data += load_tvgbench_filter(split=split)

            if name == "videomme":
                print("Use videomme")
                data += load_videomme(split=split)
            if name == "mvbench":
                print("Use mvbench")
                data += load_mvbench(split=split)
            if name == "egoschema":
                print("Use egoschema")
                data += load_egoschema(split=split)
            if name == "tempcompass":
                print("Use tempcompass")
                if len(data) > 0 and split == "captioning":
                    assert False
                data += load_tempcompass(split=split)

        return data

    @staticmethod
    def _split_data(data, curr_idx, total_idx):
        data_len = (len(data) + total_idx - 1) // total_idx
        st = curr_idx * data_len
        ed = min(len(data), st + data_len)
        print("Split data in:", st, ed)
        print("Total data len:", ed - st)
        return data[st:ed]

    def _save_video_to_cache(self, video_path, video_ele, video):
        if self.use_video_cache:
            key = video_path + json.dumps(video_ele)
            self.video_cache[key] = video

    def _load_video_from_cache(self, video_path, video_ele):
        if self.use_video_cache:
            key = video_path + json.dumps(video_ele)
            return self.video_cache.get(key, None)
        else:
            return None

    @staticmethod
    def _load_video_from_prepared(video_path, video_dirs):
        video_id = video_path.split("/")[-1].split(".")[0]
        for video_dir in video_dirs:
            video_prepared_path = os.path.join(video_dir, video_id + ".pt")
            if os.path.exists(video_prepared_path):
                return torch.load(video_prepared_path)
        return None

    @abstractmethod
    def __len__(self):
        pass

    def default_ele(self):
        ele = {}
        if self.min_pixels is not None:
            ele["min_pixels"] = self.min_pixels
        if self.total_pixels is not None:
            ele["total_pixels"] = self.total_pixels
        if self.max_frames is not None:
            ele["max_frames"] = self.max_frames
        if self.fps is not None:
            ele["fps"] = self.fps
        return ele

    @abstractmethod
    def _preprocess(self, itm):
        pass


class MultipleChoiceQADataset(BaseDataset):
    def __init__(
        self,
        processor,
        curr_idx=0,
        total_idx=1,
        split="train",
        prompt=DEFAULT_MCQ_PROMPT,
        already_finished=set([]),
        dataset_names=["charades"],
        use_prepared_video=False,
        use_r1_thinking_prompt=False,
        prompt_type="r1",
        use_nothink=False,
        **kwargs,
    ):
        super().__init__(processor, **kwargs)
        self.prompt = prompt
        if isinstance(prompt, dict):
            if use_r1_thinking_prompt:
                self.prompt = prompt[prompt_type]
            else:
                self.prompt = prompt["default"]

        # video/question/options/answer/qid
        self.data = self._load_data(dataset_names, split=split)
        self.data = self._split_data(self.data, curr_idx, total_idx)
        self.data = [itm for itm in self.data if itm["qid"] not in already_finished]

        self.use_prepared_video = use_prepared_video
        if use_prepared_video:
            self.prepared_video_dirs = []
            for dataname in dataset_names:
                self.prepared_video_dirs.append(
                    os.path.join(
                        "./video_cache", f"{dataname}_test_{self.total_pixels//28//28}"
                    )
                )

        self.use_answer_prompt = True
        self.answer_prompt = "Best Option: ("

    def __len__(self):
        return len(self.data)

    def _choice_id_to_letter(self, offset):
        return chr(ord("A") + offset)

    def _build_user_prompt(self, itm):
        num_choices = len(itm["options"])
        options_prompt = []
        for i in range(num_choices):
            options_prompt.append(
                f'({self._choice_id_to_letter(i)}) {itm["options"][i]}'
            )
        options_prompt = "\n".join(options_prompt)

        return self.prompt.format(itm["question"], options_prompt)

    def _build_options(self, itm):
        return [self._choice_id_to_letter(i) for i in range(len(itm["options"]))]

    def _preprocess(self, itm):
        ele = self.default_ele()
        if "video_start" in itm and itm["video_start"] is not None:
            ele["video_start"] = itm["video_start"]
        if "video_end" in itm and itm["video_end"] is not None:
            ele["video_end"] = itm["video_end"]

        messages = [
            {"role": "system", "content": []},
            {"role": "user", "content": []},
        ]
        messages[0]["content"].append({"type": "text", "text": self.sys_prompt})
        messages[1]["content"].append({"type": "video", "video": itm["video"], **ele})
        user_prompt = self._build_user_prompt(itm)
        messages[1]["content"].append(
            {
                "type": "text",
                "text": user_prompt,
            }
        )

        tmp = self._load_video_from_cache(itm["video"], ele)
        if tmp is not None:
            video_inputs, utils = tmp
        else:
            if self.use_prepared_video:
                tmp = self._load_video_from_prepared(
                    itm["video"], self.prepared_video_dirs
                )
            if tmp is not None:
                video_inputs, utils = tmp["video"], {"fps": tmp["fps"]}
            else:
                _, video_inputs, utils = process_vision_info_v3(
                    messages, return_video_kwargs=True
                )
                self._save_video_to_cache(itm["video"], ele, (video_inputs, utils))

        if self.use_answer_prompt:
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": self.answer_prompt,
                        }
                    ],
                }
            )
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            text = "<|im_end|>".join(text.split("<|im_end|>")[:-1])
        else:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return {"text": text, "videos": video_inputs, "fps": utils["fps"]}

    def __getitem__(self, idx):
        inputs = self._preprocess(self.data[idx])
        return {
            "inputs": inputs,
            "options": self._build_options(self.data[idx]),
            "answer": self.data[idx]["answer"],
            "qid": self.data[idx]["qid"],
            "video_paths": self.data[idx]["video"],
        }


class vllmMultipleChoiceQADataset(MultipleChoiceQADataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_answer_prompt = False

    def __getitem__(self, idx):
        inputs = self._preprocess(self.data[idx])

        multi_modal_data = {}
        if "images" in inputs and inputs["images"] is not None:
            multi_modal_data["image"] = inputs["images"]
        if "videos" in inputs and inputs["videos"] is not None:
            multi_modal_data["video"] = inputs["videos"]

        return {
            "inputs": {
                "raw_prompt_ids": self.processor.tokenizer.encode(
                    inputs["text"], add_special_tokens=False
                ),
                "multi_modal_data": multi_modal_data,
                "mm_processor_kwargs": (
                    {"fps": inputs["fps"]} if inputs["fps"] is not None else {}
                ),
            },
            "options": self._build_options(self.data[idx]),
            "answer": self.data[idx]["answer"],
            "duration": self.data[idx]["duration"],
            "qid": self.data[idx]["qid"],
            "video_paths": self.data[idx]["video"],
        }


class vllmMultipleChoiceQADataset_nothink(MultipleChoiceQADataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_answer_prompt = False

    def __getitem__(self, idx):
        inputs = self._preprocess(self.data[idx])
        inputs["text"] = inputs["text"] + "<think>\n</think>\n"

        multi_modal_data = {}
        if "images" in inputs and inputs["images"] is not None:
            multi_modal_data["image"] = inputs["images"]
        if "videos" in inputs and inputs["videos"] is not None:
            multi_modal_data["video"] = inputs["videos"]

        return {
            "inputs": {
                "raw_prompt_ids": self.processor.tokenizer.encode(
                    inputs["text"], add_special_tokens=False
                ),
                "multi_modal_data": multi_modal_data,
                "mm_processor_kwargs": (
                    {"fps": inputs["fps"]} if inputs["fps"] is not None else {}
                ),
            },
            "options": self._build_options(self.data[idx]),
            "answer": self.data[idx]["answer"],
            "duration": self.data[idx]["duration"],
            "qid": self.data[idx]["qid"],
            "video_paths": self.data[idx]["video"],
        }


class TemporalGroundingDataset(BaseDataset):
    def __init__(
        self,
        processor,
        curr_idx=0,
        total_idx=1,
        split="train",
        prompt=DEFAULT_TG_PROMPT,
        already_finished=set([]),
        dataset_names=["charades"],
        use_prepared_video=False,
        use_r1_thinking_prompt=False,
        prompt_type="r1",
        use_nothink=False,
        **kwargs,
    ):
        super().__init__(processor, **kwargs)
        self.prompt = prompt
        if isinstance(prompt, dict):
            if use_r1_thinking_prompt:
                self.prompt = prompt[prompt_type]
            else:
                self.prompt = prompt["default"]

        # video/sentence/timestamp/qid
        self.data = self._load_data(dataset_names, split=split)
        self.data = self._split_data(self.data, curr_idx, total_idx)
        self.data = [itm for itm in self.data if itm["qid"] not in already_finished]

        self.use_prepared_video = use_prepared_video
        if use_prepared_video:
            self.prepared_video_dirs = []
            for dataname in dataset_names:
                self.prepared_video_dirs.append(
                    os.path.join(
                        "./video_cache", f"{dataname}_test_{self.total_pixels//28//28}"
                    )
                )

    def __len__(self):
        return len(self.data)

    def _preprocess(self, itm):
        ele = self.default_ele()
        if "video_start" in itm and itm["video_start"] is not None:
            ele["video_start"] = itm["video_start"]
        if "video_end" in itm and itm["video_end"] is not None:
            ele["video_end"] = itm["video_end"]

        messages = [
            {"role": "system", "content": []},
            {"role": "user", "content": []},
        ]
        messages[0]["content"].append({"type": "text", "text": self.sys_prompt})
        messages[1]["content"].append({"type": "video", "video": itm["video"], **ele})
        messages[1]["content"].append(
            {
                "type": "text",
                "text": self.prompt.format(itm["sentence"]),
            }
        )

        tmp = self._load_video_from_cache(itm["video"], ele)
        if tmp is not None:
            video_inputs, utils = tmp
        else:
            if self.use_prepared_video:
                tmp = self._load_video_from_prepared(
                    itm["video"], self.prepared_video_dirs
                )
            if tmp is not None:
                video_inputs, utils = tmp["video"], {"fps": tmp["fps"]}
            else:
                _, video_inputs, utils = process_vision_info_v3(
                    messages, return_video_kwargs=True
                )
                self._save_video_to_cache(itm["video"], ele, (video_inputs, utils))

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return {"text": text, "videos": video_inputs, "fps": utils["fps"]}

    def __getitem__(self, idx):
        inputs = self._preprocess(self.data[idx])
        return {
            "inputs": inputs,
            "timestamps": self.data[idx]["timestamp"],
            "qid": self.data[idx]["qid"],
            "duration": self.data[idx]["duration"],
            "video_paths": self.data[idx]["video"],
        }


class vllmTemporalGroundingDataset(TemporalGroundingDataset):
    def __getitem__(self, idx):
        inputs = self._preprocess(self.data[idx])

        multi_modal_data = {}
        if "images" in inputs and inputs["images"] is not None:
            multi_modal_data["image"] = inputs["images"]
        if "videos" in inputs and inputs["videos"] is not None:
            multi_modal_data["video"] = inputs["videos"]

        return {
            "inputs": {
                "raw_prompt_ids": self.processor.tokenizer.encode(
                    inputs["text"], add_special_tokens=False
                ),
                "multi_modal_data": multi_modal_data,
                "mm_processor_kwargs": (
                    {"fps": inputs["fps"]} if inputs["fps"] is not None else {}
                ),
            },
            "timestamps": self.data[idx]["timestamp"],
            "qid": self.data[idx]["qid"],
            "duration": self.data[idx]["duration"],
            "video_paths": self.data[idx]["video"],
        }


class vllmTemporalGroundingDataset_nothink(TemporalGroundingDataset):
    def __getitem__(self, idx):
        inputs = self._preprocess(self.data[idx])
        inputs["text"] = inputs["text"] + "<think>\n</think>\n"

        multi_modal_data = {}
        if "images" in inputs and inputs["images"] is not None:
            multi_modal_data["image"] = inputs["images"]
        if "videos" in inputs and inputs["videos"] is not None:
            multi_modal_data["video"] = inputs["videos"]

        return {
            "inputs": {
                "raw_prompt_ids": self.processor.tokenizer.encode(
                    inputs["text"], add_special_tokens=False
                ),
                "multi_modal_data": multi_modal_data,
                "mm_processor_kwargs": (
                    {"fps": inputs["fps"]} if inputs["fps"] is not None else {}
                ),
            },
            "timestamps": self.data[idx]["timestamp"],
            "qid": self.data[idx]["qid"],
            "duration": self.data[idx]["duration"],
            "video_paths": self.data[idx]["video"],
        }


def build_dataloader(
    processor,
    datatype,
    batch_size=1,
    num_workers=8,
    already_finished=set([]),
    curr_idx=0,
    total_idx=1,
    split="train",
    dataset_names=["charades"],
    use_prepared_video=False,
    sys_prompt="You are a helpful assistant.",
    prompt=None,
    min_pixels=16 * 28 * 28,
    total_pixels=3584 * 28 * 28,
    use_r1_thinking_prompt=False,
    prompt_type="r1",
    use_nothink=False,
    use_huggingface=False,
    return_probs=False,
):
    if use_huggingface or (datatype == "mcq" and split == "train"):
        collate_fn = MultiModalDataCollator(processor)
    else:
        collate_fn = vllmMultiModalDataCollator()

    kwargs = {
        "min_pixels": min_pixels,
        "total_pixels": total_pixels,
        "already_finished": already_finished,
        "split": split,
        "curr_idx": curr_idx,
        "total_idx": total_idx,
        "dataset_names": dataset_names,
        "use_prepared_video": use_prepared_video,
        "sys_prompt": sys_prompt,
        "use_r1_thinking_prompt": use_r1_thinking_prompt,
        "prompt_type": prompt_type,
        "use_nothink": use_nothink,
    }
    if prompt is not None:
        kwargs["prompt"] = prompt

    if datatype == "tg":
        if use_huggingface:
            data = TemporalGroundingDataset(processor, **kwargs)
        else:
            if use_nothink:
                data = vllmTemporalGroundingDataset_nothink(processor, **kwargs)
            else:
                data = vllmTemporalGroundingDataset(processor, **kwargs)
    elif datatype == "mcq" and not return_probs:
        if use_huggingface:
            data = MultipleChoiceQADataset(processor, **kwargs)
        else:
            if use_nothink:
                data = vllmMultipleChoiceQADataset_nothink(processor, **kwargs)
            else:
                data = vllmMultipleChoiceQADataset(processor, **kwargs)
    else:
        data = MultipleChoiceQADataset(processor, **kwargs)

    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        prefetch_factor=2,
        pin_memory=True,
    )
    return dataloader
