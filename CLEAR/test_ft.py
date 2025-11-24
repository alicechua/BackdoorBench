from unsloth import FastLanguageModel
from peft import PeftModel
import argparse
import json
import fix_prompt
import os


parser = argparse.ArgumentParser()
parser.add_argument('--model_checkpoint_path', default=None, type=str, required=True)
parser.add_argument('--data_file', default=None, type=str, help="A file that contains instructions (one instruction per line)")
parser.add_argument('--output_file', default="./output.jsonl", type=str, help="Output file.")
# Four prompts: basic, one-shot-IcL, three-shot-IcL and definition-guided
parser.add_argument('--prompt', default="basic", type=str, help='Choose prompt style.')
# Two criteria: normal and definition-proficiency
parser.add_argument('--criterion', default="normal", type=str,  help='Choose evaluation criterion.')
args = parser.parse_args()


# Initialize Model

def load_from_checkpoint(max_seq_length: int = 2048, checkpoint_dir: str = args.model_checkpoint_path):
    print("Loading base model (Llama 3.1 8B 4-bit) ...")
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        max_seq_length = max_seq_length,
        load_in_4bit   = True,
        dtype          = None,  # let Unsloth pick
    )

    print(f"Loading LoRA adapter from {checkpoint_dir} ...")
    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_dir,
        is_trainable = False,   # inference only
    )

    # Optimize for inference (e.g. sets requires_grad=False, maybe hooks, etc.)
    model = FastLanguageModel.for_inference(model)
    model.eval()

    return model, tokenizer

print("Initializing model...", flush=True)
model, tokenizer = load_from_checkpoint(max_seq_length=2048, checkpoint_dir=args.model_checkpoint_path)
print("Finish initialization model.", flush=True)

# Read data
data_list = []
with open(args.data_file, "r") as f:
    for line in f:
        item = json.loads(line)
        # B3: Correct utilization of causal definitions.
        if args.criterion == 'definition-proficiency' and any(question_type in item["question_type"] for question_type in ["HM", "YN", "MC", "EX"]):
            data_list.append(item)
        # B1, B2 and B4
        elif args.criterion == 'normal':
            data_list.append(item)

# Inference
print("Start Inference...", flush=True)
os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
f_out = open(args.output_file, 'w')
for index, data in enumerate(data_list):
    if args.prompt == 'basic':
        example = data["description"] + '\n' + data["question"]
    # 1/3-shot IcL and definition-guided prompt are only used in B3
    else:
        prompt_functions = {
        'one-shot-IcL': fix_prompt.add_1example,
        'three-shot-IcL': fix_prompt.add_3example,
        'definition-guided': fix_prompt.add_def,
        }
        prompt_function = prompt_functions.get(args.prompt)
        example = prompt_function(data) + '\n' + data["description"] + '\n' + data["question"]

    input_ids = tokenizer(example, return_tensors="pt").input_ids.to(model.device)
    generation_output = model.generate(
        input_ids,
        max_new_tokens=300,
    )
    response = tokenizer.decode(generation_output[0], skip_special_tokens=True)
    prompt_length = len(
        tokenizer.decode(
        input_ids[0],
        skip_special_tokens=True,
        )
    )
    response = response[prompt_length:]

    print(f"======={index}=======", flush=True)
    print(f"query: {example}\n", flush=True)
    print(f"response: {response}\n", flush=True)

    data["model_response"] = response
    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
    f_out.flush()
f_out.close()