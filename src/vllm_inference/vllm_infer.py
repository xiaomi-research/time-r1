# Copyright 2025 the LlamaFactory team.
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

import copy
import re
from typing import Any, Optional

import torch
from transformers.utils import (
    is_torch_cuda_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)
from vllm import LLM, SamplingParams


def get_device_count() -> int:
    r"""Get the number of available GPU or NPU devices."""
    if is_torch_xpu_available():
        return torch.xpu.device_count()
    elif is_torch_npu_available():
        return torch.npu.device_count()
    elif is_torch_cuda_available():
        return torch.cuda.device_count()
    else:
        return 0


class vllmWrapper:
    def __init__(self, args):
        pipeline_parallel_size = args.pipeline_parallel_size
        if pipeline_parallel_size > get_device_count():
            raise ValueError(
                "Pipeline parallel size should be smaller than the number of gpus."
            )

        engine_args = {
            "model": args.model_base,
            "tensor_parallel_size": (get_device_count() // pipeline_parallel_size) or 1,
            "pipeline_parallel_size": pipeline_parallel_size,
            "max_model_len": args.total_pixels // 28 // 28 + 4096 + args.max_new_tokens,
            "max_num_batched_tokens": 8192,
            "gpu_memory_utilization": 0.95,
            "disable_mm_preprocessor_cache": True,  # Otherwise, once the cache hits, the number of FPS won't match the number of videos, causing a bug. Our speed bottleneck isn't here.
        }
        engine_args["limit_mm_per_prompt"] = {"image": 0, "video": 1, "audio": 0}
        self.model = LLM(**engine_args)

        self.tokenizer = self.model.get_tokenizer()

    def find_answer_token_last_occurrence(self, text: str):
        answer_token = "<answer>"
        idx = text.rfind(answer_token)
        return idx

    @staticmethod
    def extract_timestamps(response):
        number_pattern = r"\d+(?:\.\d+)?"
        matches = re.findall(number_pattern, response)
        output = [float(num) for num in matches[-2:]]
        if len(output) == 2:
            return output[0], output[1]
        return None, None

    @torch.no_grad()
    def generate(
        self,
        inputs: dict[str, Any],
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = -1,
        max_new_tokens: int = 128,
        repetition_penalty: float = 1.0,
        seed: Optional[int] = None,
        answer_prompt: Optional[str] = None,  # only r1 model needed
    ):
        r"""Perform batch generation using vLLM engine, which supports tensor parallelism.

        Usage: python vllm_infer.py --model_name_or_path meta-llama/Llama-2-7b-hf --template llama --dataset alpaca_en_demo
        """
        vllm_inputs = []
        for raw_prompt_ids, multi_modal_data, mm_processor_kwargs in zip(
            inputs["raw_prompt_ids"],
            inputs["multi_modal_data"],
            inputs["mm_processor_kwargs"],
        ):
            vllm_inputs.append(
                {
                    "prompt_token_ids": list(raw_prompt_ids),
                    "multi_modal_data": multi_modal_data,
                    "mm_processor_kwargs": mm_processor_kwargs,
                }
            )

        sampling_params = SamplingParams(
            repetition_penalty=repetition_penalty or 1.0,  # repetition_penalty must > 0
            temperature=temperature,
            top_p=top_p or 1.0,  # top_p must > 0
            top_k=top_k or -1,  # top_k must > 0
            stop=None,
            stop_token_ids=[151645, 151643],  # generate_config.json
            max_tokens=max_new_tokens,
            include_stop_str_in_output=True,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
            seed=seed,
        )
        sampling_params = [
            copy.deepcopy(sampling_params) for _ in range(len(vllm_inputs))
        ]

        results = self.model.generate(
            prompts=vllm_inputs, sampling_params=sampling_params
        )
        preds = [result.outputs[0].text for result in results]

        # For MCQ questions, to obtain option letters, we follow the MVBench approach by adding an answer prompt to force the model to print the option in the specified position.
        # This uses string matching, as input_ids vary too much otherwise.
        if answer_prompt is not None:
            indices = [self.find_answer_token_last_occurrence(text) for text in preds]
            continue_vllm_inputs, continue_sampling_params = [], []
            for i, (raw_prompt_ids, multi_modal_data, mm_processor_kwargs) in enumerate(
                zip(
                    inputs["raw_prompt_ids"],
                    inputs["multi_modal_data"],
                    inputs["mm_processor_kwargs"],
                )
            ):
                if indices[i] == -1:
                    continue
                new_token_ids = self.tokenizer.encode(
                    preds[i][: indices[i]] + "<answer>\n" + answer_prompt,
                    add_special_tokens=False,
                )
                continue_vllm_inputs.append(
                    {
                        "prompt_token_ids": list(raw_prompt_ids) + list(new_token_ids),
                        "multi_modal_data": multi_modal_data,
                        "mm_processor_kwargs": mm_processor_kwargs,
                    }
                )
                sampling_params[i].max_tokens = 16
                continue_sampling_params.append(sampling_params[i])
            results = self.model.generate(
                prompts=continue_vllm_inputs, sampling_params=continue_sampling_params
            )

            continue_cnt = 0
            for i in range(len(preds)):
                if indices[i] == -1:
                    continue
                preds[i] = (
                    preds[i][: indices[i]]
                    + "<answer>\n"
                    + answer_prompt
                    + results[continue_cnt].outputs[0].text
                )
                continue_cnt += 1

        return preds
