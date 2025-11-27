import sys
import json
import re
import os
import time
from openai import OpenAI
import dotenv
from tqdm import tqdm

dotenv.load_dotenv()
client = OpenAI()

def openai_api(input_text, model_name="gpt-4o-mini", retries=3):
  retry_cnt = 0
  backoff_time = 5
  while retry_cnt < retries:
    try:
      response = client.chat.completions.create(
          model=model_name, 
          messages=[
              {"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": input_text}
          ]
      )
      return response.choices[0].message.content
    except:
      time.sleep(backoff_time)
      backoff_time *= 1.5
      retry_cnt += 1
  return None

def extract_answer(response_text, question_type):
    if question_type in ["YN", "EX"]:
        prompt = f"Please extract the final answer from the following text. The answer should be either 'Yes' or 'No'. \n\nText: {response_text}\n\nAnswer:"
        res = openai_api(prompt)
        if res:
            if "yes" in res.lower(): return "Yes."
            if "no" in res.lower(): return "No."
        return "Unknown."
    elif question_type == "MC":
        prompt = f"Please extract the final answer from the following text. The answer should be one of the options: A, B, C, or D. Return only the letter.\n\nText: {response_text}\n\nAnswer:"
        res = openai_api(prompt)
        if res:
            match = re.search(r'[ABCD]', res.upper())
            if match: return match.group(0)
        return "unknown"
    elif question_type == "HM":
        prompt = f"Please extract the final numerical answer from the following text. Return only the number.\n\nText: {response_text}\n\nAnswer:"
        res = openai_api(prompt)
        if res:
            match = re.search(r'-?\d+', res)
            if match: return int(match.group(0))
        return -1
    return None

input_filename, question_type = sys.argv[1], sys.argv[2]
lines = open(input_filename, encoding='utf-8').readlines()

is_yn = (question_type == "YN" or question_type == "EX")
is_mc = (question_type == "MC")
is_hm = (question_type == "HM")
assert is_yn or is_mc or is_hm


result_dict = dict()
result_dict["total"] = {
    "tp": 0,
    "fn": 0,
    "fp": 0,
    "tn": 0,
    "correct": 0,
    "wrong": 0
}

task_types = [
    "node",
    "edge",
    "2_node_relation",
    "3_node_relation",
    "path",
    "cycle",
    "topological",
    "blocked_path",
    "d-separation",
    "markov_equivalent",
    "markov_blanket",
    "directed_path",
    "backdoor_path",
    "c-component",
    "c-tree",
    "c-forest",
    "maximal_root_set",
    "backdoor_adjustment_set",
    "frontdoor_adjustment_set",
    "identification",
]

for line in tqdm(lines):
    obj = json.loads(line)
    if question_type != obj["question_type"]: continue
    task_type = obj["task_type"]
    if is_yn:
        assert obj["answer"] in ["Yes.", "No."], obj["answer"]
        pred = extract_answer(obj["model_response"], "YN")
        
        if task_type not in result_dict:
            result_dict[task_type] = {
                "tp": 0,
                "fn": 0,
                "fp": 0,
                "tn": 0,
                "correct": 0,
                "wrong": 0
            }
        for key in ["total", task_type]:
            if obj["answer"] == "Yes.":
                if pred == "Yes.":
                    result_dict[key]["tp"] += 1
                    result_dict[key]["correct"] += 1
                else:
                    result_dict[key]["fn"] += 1
                    result_dict[key]["wrong"] += 1
            else:
                if pred == "No.":
                    result_dict[key]["tn"] += 1
                    result_dict[key]["correct"] += 1
                else:
                    result_dict[key]["fp"] += 1
                    result_dict[key]["wrong"] += 1
    elif is_mc:
        assert obj["answer"] in ["A", "B", "C", "D"]
        pred = extract_answer(obj["model_response"], "MC")
        
        if task_type not in result_dict:
            result_dict[task_type] = {
                "correct": 0,
                "wrong": 0
            }
        for key in ["total", task_type]:
            if obj["answer"] == pred:
                result_dict[key]["correct"] += 1
            else:
                result_dict[key]["wrong"] += 1
    elif is_hm:
        gt = int(obj["answer"])
        dt = extract_answer(obj["model_response"], "HM")
        
        if task_type not in result_dict:
            result_dict[task_type] = {
                "correct": 0,
                "wrong": 0
            }
        for key in ["total", task_type]:
            if gt == dt:
                result_dict[key]["correct"] += 1
            else:
                result_dict[key]["wrong"] += 1        

for key in task_types + ['total']:
    if key not in result_dict:
        continue
    # Accuracy
    correct = result_dict[key]["correct"]
    wrong = result_dict[key]["wrong"]
    total = correct + wrong
    acc = correct / total if total > 0 else 0
    result_dict[key]["accuracy"] = acc * 100
    result_dict[key]["acc_str"] = f"{correct}/{total}"

    if is_yn:
        # Precision
        tp = result_dict[key].get("tp", 0)
        fp = result_dict[key].get("fp", 0)
        prec_denom = tp + fp
        prec = tp / prec_denom if prec_denom > 0 else 0
        result_dict[key]["precision"] = prec * 100
        result_dict[key]["prec_str"] = f"{tp}/{prec_denom}"

        # Recall
        fn = result_dict[key].get("fn", 0)
        rec_denom = tp + fn
        rec = tp / rec_denom if rec_denom > 0 else 0
        result_dict[key]["recall"] = rec * 100
        result_dict[key]["rec_str"] = f"{tp}/{rec_denom}"

        # F1
        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0
        result_dict[key]["f1"] = f1 * 100

for task_type in task_types + ["total"]:
    if task_type not in result_dict:
        continue
    res = result_dict[task_type]
    print(f"Task: {task_type}")
    print(f"  Accuracy: {res['accuracy']:.2f}% ({res['acc_str']})")
    if is_yn:
        print(f"  Precision: {res['precision']:.2f}% ({res['prec_str']})")
        print(f"  Recall: {res['recall']:.2f}% ({res['rec_str']})")
        print(f"  F1: {res['f1']:.2f}")