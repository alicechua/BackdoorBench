import os
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Score evaluation results against ground truth")
    parser.add_argument(
        "-i",
        "--input_jsonl",
        type=str,
        required=True,
        help="Path to the input JSONL file with model responses.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    inputs = []
    with open(args.input_jsonl, 'r') as f:
        for line in f:
            inputs.append(json.loads(line))
    total = 0
    correct = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Calculate accuracy and F1 score
    for input in inputs:
        gt_label = input.get('label', None)
        response = input.get('response', '')
        
        predicted_label = None
        if '<answer>' in response and '</answer>' in response:
            answer = response.split('<answer>')[1].split('</answer>')[0].strip().lower()
            if answer == 'yes':
                predicted_label = 1
            elif answer == 'no':
                predicted_label = 0
        
        if predicted_label is not None:
            if predicted_label == gt_label:
                correct += 1
            
            # F1 calculation (assuming 1 is positive class)
            if predicted_label == 1 and gt_label == 1:
                true_positives += 1
            elif predicted_label == 1 and gt_label == 0:
                false_positives += 1
            elif predicted_label == 0 and gt_label == 1:
                false_negatives += 1
        
        total += 1

    # Calculate metrics
    accuracy = correct / total if total > 0 else 0.0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")