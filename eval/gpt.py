import argparse
import json
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
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    inputs = []
    
    with open(args.input_jsonl, 'r') as f:
        for line in f:
            inputs.append(json.loads(line))
    client = OpenAI()

    outputs = []
    for input in tqdm(inputs[:10]):
        prompt = f"Premise: {input['premise']}\nHypothesis: {input['hypothesis']}\nQuestion: Is the hypothesis true under the premise? Concisely explain and answer 'Yes' or 'No'. Put the final answer inside <answer></answer> tags."
        response = client.responses.create(
            model=args.model_path,
            input=prompt,
            max_tokens=512,
            temperature=0.0
        )
        output = input.copy()
        output["response"] = response.output[0].content[0].text
        output.pop("graph", None)
        outputs.append(output)

    with open(args.output_jsonl, 'w') as f:
        for output in outputs:
            f.write(json.dumps(output) + "\n")