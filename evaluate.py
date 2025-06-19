import argparse
import json
import os
import re
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from src.vllm_inference.data import build_dataloader
from src.vllm_inference.vllm_infer import vllmWrapper


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluation for training-free video temporal grounding (Single GPU Version)"
    )
    parser.add_argument(
        "--datatype",
        default="tg",
        type=str,
        help="Specify the dataset.",
        choices=["tg", "mcq"],
    )
    parser.add_argument(
        "--model_base", type=str, default="../pretrained_models/Qwen2.5-VL-7B-Instruct"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="GPU device to use"
    )
    parser.add_argument(
        "--pipeline_parallel_size", type=int, default=1, help="GPU nodes"
    )
    parser.add_argument("--split", type=str, default="train", help="dataset type")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--curr_idx", type=int, default=0, help="数据分片")
    parser.add_argument("--total_idx", type=int, default=1, help="数据分片")
    parser.add_argument(
        "--total_pixels", type=int, default=3584 * 28 * 28, help="total_pixels"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=str,
        help="dataset names",
        choices=[
            "charades",
            "activitynet",
            "videomme",
            "mvbench",
            "tvgbench_filter",
            "tvgbench",
            "egoschema",
            "tempcompass",
        ],
    )
    parser.add_argument(
        "--use_r1_thinking_prompt", action="store_true", help="On R1 SHOUD BE TRUE!"
    )
    parser.add_argument(
        "--use_vllm_inference", action="store_true"
    )
    parser.add_argument("--prompt_type", type=str, default="r1", help="Prompt type")
    parser.add_argument(
        "--use_nothink", action="store_true", help="Use no think prompt"
    )
    parser.add_argument(
        "--use_prepared_video",
        action="store_true",
        help="Use video cache in ./video_cache",
    )
    return parser.parse_args()


def build_model(args):
    processor = AutoProcessor.from_pretrained(args.model_base, use_fast=True)
    if args.datatype in ["tg"]:
        processor.tokenizer.padding_side = "left"

    if (args.datatype == "tg" or (args.datatype == "mcq" and args.split != "train")) and args.use_vllm_inference:
        # vllm inference
        model = vllmWrapper(args)
    else:
        # transformers inference
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_base,
            torch_dtype="auto",
            device_map=args.device,
            attn_implementation="flash_attention_2",
        )
        model.eval()

    return model, processor


@torch.no_grad()
def inference(model, inputs):
    for key in inputs.keys():
        if not isinstance(inputs[key], torch.Tensor):
            continue
        inputs[key] = inputs[key].to(model.device)

    logits = model(**inputs).logits
    bsz, seq_len, _ = logits.shape
    if "attention_mask" in inputs:
        pred_token_indices = torch.sum(inputs["attention_mask"], dim=-1) - 1
    else:
        pred_token_indices = torch.full((bsz,), seq_len - 1, device=logits.device)

    pred_token_logits = logits[
        torch.arange(bsz, device=logits.device), pred_token_indices, :
    ]

    return pred_token_logits


def extract_answer(output_string, datatype):
    if datatype == "tg":
        matches = re.findall(r"(\d+\.?\d*) (to|and) (\d+\.?\d*)", output_string)
        if not matches:
            answer_match = re.search(r"<answer>(.*?)</answer>", output_string)
            if answer_match:
                answer_content = answer_match.group(1).strip()
                answer_matches = re.findall(
                    r"(\d+\.?\d*) (to|and) (\d+\.?\d*)", answer_content
                )
                if answer_matches:
                    last_match = answer_matches[-1]
                    return [float(last_match[0]), float(last_match[2])]
            return [None, None]

        last_match = matches[-1]
        start_time_str = last_match[0]
        end_time_str = last_match[2]

        try:
            start_time = float(start_time_str)
            end_time = float(end_time_str)
            return [start_time, end_time]
        except ValueError:
            return [None, None]

    if datatype == "mcq":
        matches = re.findall(r"\(([A-Z])\)", output_string)
        if matches:
            return ord(matches[-1]) - ord("A")
        return None


@torch.no_grad()
def calc_prob(logits, options_token_ids):
    bsz = logits.shape[0]
    probs = []
    for i in range(bsz):
        logit = logits[i, options_token_ids]
        probs.append(F.softmax(logit, dim=1))
    return probs


@torch.no_grad()
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir, f"{args.datatype}_{args.curr_idx}_{args.total_idx}.jsonl"
    )

    already_finished = set([])
    f = open(output_file, "a+")
    try:
        with open(output_file, "r") as g:
            for line in g:
                old_data = json.loads(line)
                already_finished.add(old_data["qid"])
    except Exception as e:
        print(e)

    model, processor = build_model(args)

    dataloader_args = {
        "batch_size": args.batch_size,
        "already_finished": already_finished,
        "curr_idx": args.curr_idx,
        "total_idx": args.total_idx,
        "split": args.split,
        "num_workers": min(8, args.batch_size),
        "dataset_names": args.datasets,
        "use_prepared_video": args.use_prepared_video,
        "total_pixels": args.total_pixels,
        "use_r1_thinking_prompt": args.use_r1_thinking_prompt,
        "prompt_type": args.prompt_type,
        "use_nothink": args.use_nothink,
    }

    dataloader = build_dataloader(processor, args.datatype, **dataloader_args)

    program_start_time = time.perf_counter()

    for batch_itm in tqdm(dataloader):
        if args.datatype == "tg":
            output_texts = model.generate(
                batch_itm["inputs"],
                max_new_tokens=args.max_new_tokens,
            )
            targets = batch_itm["timestamps"]

            for i in range(len(targets)):
                pred = extract_answer(output_texts[i], args.datatype)
                f.write(
                    json.dumps(
                        {
                            "qid": batch_itm["qid"][i],
                            "pred": pred,
                            "target": list(targets[i]),
                            "duration": (
                                None
                                if "duration" not in batch_itm
                                else batch_itm["duration"][i]
                            ),
                            "output_text": output_texts[i],
                        }
                    )
                    + "\n"
                )
                f.flush()
        elif args.datatype == "mcq" and args.split != "train":
            output_texts = model.generate(
                batch_itm["inputs"],
                max_new_tokens=args.max_new_tokens,
                answer_prompt=dataloader.dataset.answer_prompt,
            )
            targets = batch_itm["answer"]

            for i in range(len(targets)):
                pred = extract_answer(output_texts[i], args.datatype)
                f.write(
                    json.dumps(
                        {
                            "qid": batch_itm["qid"][i],
                            "pred": None,
                            "target": targets[i],
                            "duration": (
                                None
                                if "duration" not in batch_itm
                                else batch_itm["duration"][i]
                            ),
                            "output_text": output_texts[i],
                        }
                    )
                    + "\n"
                )
                f.flush()
        else:
            logits = inference(model, batch_itm["inputs"])
            options_token_ids = [
                [processor.tokenizer.vocab[word] for word in word_list]
                for word_list in batch_itm["options"]
            ]
            probs = calc_prob(logits, options_token_ids)

            for i in range(len(logits)):
                f.write(
                    json.dumps(
                        {
                            "qid": batch_itm["qid"][i],
                            "pred": probs[i].argmax().item(),
                            "target": batch_itm["answer"][i],
                            "duration": (
                                None
                                if "duration" not in batch_itm
                                else batch_itm["duration"][i]
                            ),
                            "probs": probs[i].cpu().tolist(),
                        }
                    )
                    + "\n"
                )
                f.flush()

    # --- END TOTAL TIME & CALCULATIONS ---
    program_end_time = time.perf_counter()
    total_program_duration = program_end_time - program_start_time

    print("\n--- Timing Summary ---")
    print(f"Total program execution time: {total_program_duration:.2f} seconds")

    output_filename = f"{args.output_dir}/timing_summary_vllm.txt"

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("\n--- Timing Summary ---\n")
        f.write(f"Total program execution time: {total_program_duration:.2f} seconds\n")
        f.write("Another line of summary using write.\n")


if __name__ == "__main__":
    from src.vllm_inference.utils import monkey_patch

    monkey_patch()
    args = get_args()
    if "mvbench" in args.datasets \
        or "videomme" in args.datasets \
        or "tempcompass" in args.datasets:
        args.datatype = "mcq"
    elif (
        "tvgbench" in args.datasets \
        or "tvgbench_filter" in args.datasets \
        or "charades" in args.datasets \
        or "activitynet" in args.datasets \
    ):
        args.datatype = "tg"
    else:
        raise ValueError("Unsupported dataset type. Please check your datasets.")
    main(args)
