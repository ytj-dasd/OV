"""
This script is designed for academic inference benchmarking with the `GLM-4.1V-9B-Thinking` model,
particularly in multi-modal settings involving video inputs. It ensures robust handling of structured
reasoning outputs such as `<think>...</think><answer>...</answer>`.

Core features:
- Automatically detects whether the model output includes a complete reasoning block.
- If the model reaches the `first_max_tokens` limit without emitting `</think>` or `<answer>`,
  it forcefully appends `</think><answer>` and re-generates to ensure complete output.
- Accepts video input via `video_url`, pointing to a local video file to simulate real multi-modal inputs.
- Designed for the Hugging Face `transformers` inference pipeline. If using `vLLM`, similar logic must be adapted manually
  to support forced continuation and special token handling across generation rounds.

Arguments:
- `--model_path`: Path to the model. Defaults to `THUDM/GLM-4.1V-9B-Thinking`.
- `--video_path`: Path to the input video file (required).
- `--prompt`: Text prompt for the model (required).
- `--first_max_tokens` / `--force_max_tokens`: Maximum tokens for initial generation and forced continuation.
- `--temperature`: Generation temperature (default: 0.1 for stability).
"""

import argparse

import torch
from transformers import AutoProcessor, Glm4vForConditionalGeneration


class RobustInference:
    def __init__(self, model_path):
        self.model_path = model_path

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.special_tokens = {
            "think_start": self.processor.tokenizer.convert_tokens_to_ids("<think>"),
            "think_end": self.processor.tokenizer.convert_tokens_to_ids("</think>"),
            "answer_start": self.processor.tokenizer.convert_tokens_to_ids("<answer>"),
            "answer_end": self.processor.tokenizer.convert_tokens_to_ids("</answer>"),
        }

        self.model = Glm4vForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def generate_with_force_completion(
        self,
        messages,
        first_max_tokens=8192,
        force_max_tokens=8192,
        temperature=0.1,
        do_sample=True,
    ):
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)
        inputs.pop("token_type_ids", None)
        input_length = inputs["input_ids"].shape[1]

        first_generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=first_max_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )

        first_output_ids = first_generated_ids[0][input_length:]

        needs_completion = self._check_needs_completion_by_tokens(
            first_output_ids, first_max_tokens
        )

        if not needs_completion:
            final_output = self.processor.decode(
                first_output_ids, skip_special_tokens=False
            )
            return {
                "output_text": final_output,
                "complete": True,
                "reason": "first_generation_complete",
                "input_length": input_length,
                "first_generation_length": len(first_output_ids),
                "generation_rounds": 1,
            }

        force_input_ids = self._prepare_force_input(
            inputs["input_ids"], first_output_ids
        )

        force_inputs = {
            "input_ids": force_input_ids.to(self.model.device),
            "attention_mask": torch.ones_like(force_input_ids).to(self.model.device),
        }

        if "pixel_values" in inputs:
            force_inputs["pixel_values"] = inputs["pixel_values"]
        if "video_metadata" in inputs:
            force_inputs["video_metadata"] = inputs["video_metadata"]

        second_generated_ids = self.model.generate(
            **force_inputs,
            max_new_tokens=force_max_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )

        second_output_ids = second_generated_ids[0][force_input_ids.shape[1] :]

        added_tokens = force_input_ids[0][input_length + len(first_output_ids) :]
        complete_output_ids = torch.cat(
            [first_output_ids, added_tokens, second_output_ids], dim=0
        )
        complete_output_text = self.processor.decode(
            complete_output_ids, skip_special_tokens=False
        )

        return {
            "output_text": complete_output_text,
            "complete": (
                self.special_tokens["answer_end"] in complete_output_ids.tolist()
            ),
            "reason": "force_completion_success",
            "input_length": input_length,
            "first_generation_length": len(first_output_ids),
            "second_generation_length": len(second_output_ids),
            "total_generation_length": len(complete_output_ids),
            "generation_rounds": 2,
        }

    def _check_needs_completion_by_tokens(self, output_token_ids, max_tokens):
        token_list = output_token_ids.tolist()

        reached_max = len(token_list) >= max_tokens
        has_answer_end = self.special_tokens["answer_end"] in token_list
        has_think_start = self.special_tokens["think_start"] in token_list
        has_think_end = self.special_tokens["think_end"] in token_list

        if has_answer_end:
            return False

        if reached_max:
            return True

        if has_think_start and not has_think_end:
            return True

        return False

    def _prepare_force_input(self, original_input_ids, first_output_ids):
        first_output_list = first_output_ids.tolist()

        has_think_end = self.special_tokens["think_end"] in first_output_list
        has_answer_start = self.special_tokens["answer_start"] in first_output_list

        tokens_to_add = []

        if not has_think_end:
            tokens_to_add.extend(
                [self.special_tokens["think_end"], self.special_tokens["answer_start"]]
            )
        elif not has_answer_start:
            tokens_to_add.append(self.special_tokens["answer_start"])

        if tokens_to_add:
            additional_tokens = (
                torch.tensor(tokens_to_add).unsqueeze(0).to(self.model.device)
            )
            force_input_ids = torch.cat(
                [original_input_ids, first_output_ids.unsqueeze(0), additional_tokens],
                dim=1,
            )
        else:
            force_input_ids = torch.cat(
                [original_input_ids, first_output_ids.unsqueeze(0)], dim=1
            )

        return force_input_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video inference script")
    parser.add_argument(
        "--model_path",
        type=str,
        default="zai-org/GLM-4.1V-9B-Thinking",
        help="Model path",
    )
    parser.add_argument("--video_path", type=str, required=True, help="Video file path")
    parser.add_argument(
        "--first_max_tokens", type=int, default=8192, help="First generation max tokens"
    )
    parser.add_argument(
        "--force_max_tokens", type=int, default=8192, help="Force completion max tokens"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Generation temperature"
    )
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text")

    args = parser.parse_args()

    runner = RobustInference(args.model_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "url": args.video_path,
                },
                {
                    "type": "text",
                    "text": args.prompt,
                },
            ],
        }
    ]

    result = runner.generate_with_force_completion(
        messages=messages,
        first_max_tokens=args.first_max_tokens,
        force_max_tokens=args.force_max_tokens,
        temperature=args.temperature,
    )

    print("=" * 50)
    print("Inference Results:")
    print(f"Complete: {result['complete']}")
    print(f"Reason: {result['reason']}")
    print(f"Generation rounds: {result['generation_rounds']}")
    print(f"Input length: {result['input_length']}")
    if "first_generation_length" in result:
        print(f"First generation length: {result['first_generation_length']}")
    if "second_generation_length" in result:
        print(f"Second generation length: {result['second_generation_length']}")
    print("=" * 50)
    print(result["output_text"])
