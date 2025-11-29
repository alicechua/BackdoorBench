from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from peft import PeftModel
import argparse
import json
import fix_prompt
import os
import torch
import torch.nn as nn

INFIX_MARKER = "<INFIX>"

class SoftPromptWrapper(nn.Module):
    def __init__(self, base_model, tokenizer, n_tokens=10):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.n_tokens = n_tokens
        self.config = base_model.config
        
        embed_dim = base_model.config.hidden_size
        self.soft_prompt = nn.Parameter(torch.randn(n_tokens, embed_dim) * 0.01)
        
        self.infix_marker = INFIX_MARKER
        
    @property
    def device(self):
        return self.base_model.device
    
    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()



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
    print("Loading base model (Llama 3.1 8B 16-bit) ...")
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = "unsloth/Meta-Llama-3.1-8B-Instruct",
        max_seq_length = max_seq_length,
        load_in_4bit   = False,
        dtype          = torch.float16 if torch.cuda.is_available() else None,
    )
    
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    
    # Add INFIX_MARKER if needed
    if INFIX_MARKER not in tokenizer.get_vocab():
        tokenizer.add_tokens([INFIX_MARKER])
        base_model.resize_token_embeddings(len(tokenizer))
    
    lora_adapter_dir = os.path.join(checkpoint_dir, "lora_adapters")
    print(f"Loading LoRA adapter from {lora_adapter_dir} ...")
    model = PeftModel.from_pretrained(
        base_model,
        lora_adapter_dir,
        is_trainable = False,   # inference only
    )
        
    # Wrap with SoftPromptWrapper
    print("Wrapping with SoftPromptWrapper...")
    model = SoftPromptWrapper(model, tokenizer, n_tokens=10)
    
    # Load soft prompt
    sp_path = os.path.join(checkpoint_dir, "soft_prompt.pt")
    if not os.path.exists(sp_path):
        # Try parent directory
        sp_path = os.path.join(os.path.dirname(checkpoint_dir.rstrip("/")), "soft_prompt.pt")
        
    if os.path.exists(sp_path):
        print(f"Loading soft prompt from {sp_path} ...")
        sp_data = torch.load(sp_path, map_location=model.device)
        model.soft_prompt.data = sp_data['soft_prompt'].to(model.device)
    else:
        print(f"Warning: soft_prompt.pt not found at {sp_path}")

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

    # Apply chat template
    messages = [
        {"role": "user", "content": INFIX_MARKER + example},
    ]
    example = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    input_ids = tokenizer(example, return_tensors="pt").input_ids.to(model.device)
    
    # Manual generation loop
    with torch.no_grad():
        embed_layer = model.get_input_embeddings()
        input_embeds = embed_layer(input_ids)
        
        marker_token_id = tokenizer.encode(INFIX_MARKER, add_special_tokens=False)[0]
        marker_positions = (input_ids[0] == marker_token_id).nonzero(as_tuple=True)[0]
        
        if len(marker_positions) > 0:
            marker_pos = marker_positions[0].item()
            before_embeds = input_embeds[0, :marker_pos]
            after_embeds = input_embeds[0, marker_pos+1:]
            
            input_embeds = torch.cat([
                before_embeds.unsqueeze(0),
                model.soft_prompt.to(input_embeds.device).unsqueeze(0),
                after_embeds.unsqueeze(0)
            ], dim=1)
            
        generated_ids = []
        current_embeds = input_embeds
        
        for _ in range(300):
            outputs = model.base_model(inputs_embeds=current_embeds)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            
            if next_token_id.item() == tokenizer.eos_token_id:
                break
                
            generated_ids.append(next_token_id.item())
            
            next_token_embed = model.get_input_embeddings()(next_token_id.unsqueeze(0))
            current_embeds = torch.cat([current_embeds, next_token_embed], dim=1)
            
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).split('assistant\n\n')[-1].strip()

    print(f"======={index}=======", flush=True)
    print(f"query: {example}\n", flush=True)
    print(f"response: {response}\n", flush=True)

    data["model_response"] = response
    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
    f_out.flush()
f_out.close()