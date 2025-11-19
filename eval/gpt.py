import argparse
import json
import random
from openai import OpenAI
import os
import dotenv
from tqdm import tqdm

dotenv.load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GPT model performance on BAS dataset")
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=True,
        help="GPT model name for evaluation (e.g., 'gpt-4o-mini').",
    )
    parser.add_argument(
        "-i",
        "--input_jsonl",
        type=str,
        required=True,
        help="Path to the input JSONL file for evaluation.",
    )
    parser.add_argument(
        "-o",
        "--output_jsonl",
        type=str,
        default="./eval_results.jsonl",
        help="Path to save evaluation results.",
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=-1,
        help="Number of samples to evaluate.",
    )
    parser.add_argument(
        "--few_shot",
        type=int,
        default=0,
        help="Number of few-shot examples to provide.",
    )
    parser.add_argument(
        "--few_shot_path",
        type=str,
        default="",
        help="Path to few-shot examples JSONL file.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    inputs = []
    
    with open(args.input_jsonl, 'r') as f:
        for line in f:
            inputs.append(json.loads(line))
    client = OpenAI()

    outputs = []
    if args.num_samples > 0:
        inputs = inputs[: args.num_samples]
    if args.few_shot > 0 and args.few_shot_path:
        with open(args.few_shot_path, 'r') as f:
            few_shot_examples = [json.loads(line) for line in f]
    for input in tqdm(inputs):
        prefix = ""
        if args.few_shot > 0 and args.few_shot_path:
            random_examples = random.sample(few_shot_examples, args.few_shot)
            for example in random_examples:
                if example["label"] == 1:
                    answer = "Yes"
                    explanation = "This is true because the set blocks all backdoor paths."
                elif example["label"] == 0:
                    answer = "No"
                    if "neg_type" not in example["meta"]:
                        explanation = "This is false because conditioning on a descendant violates the backdoor criterion."
                    elif example["meta"]["neg_type"] == "collider":
                        explanation = "This is false because conditioning on a collider opens a backdoor path."
                    elif example["meta"]["neg_type"] == "near_miss":
                        explanation = "This is false because the set does not fully block the backdoor path."
                    else: # It is conditioning on a descendant
                        explanation = "This is false because conditioning on a descendant violates the backdoor criterion."
                prefix += f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}\nQuestion: Is the hypothesis true under the premise? Concisely explain and answer 'Yes' or 'No'. Put the final answer inside <answer></answer> tags.\nExplanation: {explanation} Final Answer: <answer>{answer}</answer>\n\n"
        prompt = f"{prefix}\nPremise: {input['premise']}\nHypothesis: {input['hypothesis']}\nQuestion: Is the hypothesis true under the premise? Concisely explain and answer 'Yes' or 'No'. Put the final answer inside <answer></answer> tags."
        # print(prompt)
        response = client.responses.create(
            model=args.model_path,
            input=prompt,
            max_output_tokens=512,
            temperature=0.0
        )
        output = input.copy()
        output["response"] = response.output[0].content[0].text
        output.pop("graph", None)
        outputs.append(output)

    with open(args.output_jsonl, 'w') as f:
        for output in outputs:
            f.write(json.dumps(output) + "\n")