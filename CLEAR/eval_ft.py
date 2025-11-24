import json
from pathlib import Path
from collections import Counter
import re

IN_PATH = Path("CLEAR_results/CLEAR_results/BAS_llama_cft.jsonl")   # change if needed

MARKER = "### Answer:\n"

def extract_answer(model_resp: str | None) -> str | None:
    """Strip everything before '### Answer:\\n' and return the first line after it (for YN)."""
    if not model_resp:
        return None

    idx = model_resp.find(MARKER)
    if idx != -1:
        # Take everything after the marker
        ans = model_resp[idx + len(MARKER):]
    else:
        # Fallback: use whole response
        ans = model_resp

    ans = ans.strip()
    if not ans:
        return None

    # Often you only want the first line / token like "Yes" or "No"
    first_line = ans.splitlines()[0].strip()
    return first_line


def yn_to_label(text: str | None) -> int | None:
    """Map 'Yes'/'No' (with variations) to 1/0."""
    if not text:
        return None
    t = text.strip().lower()
    if t.startswith("yes"):
        return 1
    if t.startswith("no"):
        return 0
    return None


def evaluate_yn(jsonl_path: Path):
    y_true = []
    y_pred = []

    with jsonl_path.open() as f:
        for line in f:
            ex = json.loads(line)

            # 1) Only evaluate YN questions
            if ex.get("question_type") != "YN":
                continue

            gold_label = yn_to_label(ex.get("answer", ""))
            pred_raw = ex.get("model_response", "")

            # 2) Strip model response after "### Answer:\\n"
            pred_str = extract_answer(pred_raw)
            print("YN raw prediction:", repr(pred_str))
            pred_label = yn_to_label(pred_str)

            # Skip examples we can't parse
            if gold_label is None or pred_label is None:
                continue

            y_true.append(gold_label)
            y_pred.append(pred_label)

    # --- Simple metrics ---
    assert len(y_true) == len(y_pred)
    n = len(y_true)
    if n == 0:
        print("No usable YN examples found.")
        return

    # accuracy
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    acc = correct / n

    # precision/recall/F1 for positive class (label=1, i.e. Yes)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"# YN examples evaluated: {n}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")


# ===== NEW: MC handling =====

def extract_mc_answer(model_resp: str | None) -> str | None:
    """
    Robustly extract an MC option letter (A/B/C/D) from the model response.

    Strategy:
      - Split into lines.
      - For each line containing "answer" (case-insensitive):
          * Try to match "Answer: X" or "### Answer: X" on the same line.
          * If the line ends with "Answer:" / "### Answer:", look at the next
            non-empty line and take a single-letter A–D if present.
      - Return the LAST such answer we see (usually the final answer in the convo).
    """
    if not model_resp:
        return None

    lines = model_resp.splitlines()
    answers: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        lower = line.lower()

        if "answer" in lower:
            # 1) Same-line pattern: "Answer: B", "### Answer: C", etc.
            m = re.search(r"answer\s*[:\-]\s*([A-D])\b", line, flags=re.IGNORECASE)
            if m:
                answers.append(m.group(1).upper())
            else:
                # 2) Line that ends with "Answer:" or "### Answer:"
                if re.search(r"answer\s*[:\-]?\s*$", line, flags=re.IGNORECASE):
                    j = i + 1
                    # Skip blank lines
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    if j < len(lines):
                        next_line = lines[j].strip()
                        m2 = re.match(r"^([A-D])\b", next_line)
                        if m2:
                            answers.append(m2.group(1).upper())
                            i = j  # jump to the line we consumed
        i += 1

    if not answers:
        return None
    print("Extracted MC answers found:", answers)
    # Heuristic: use the LAST answer – usually the final one the model gives.
    return answers[0]

def evaluate_mc(jsonl_path: Path):
    y_true = []
    y_pred = []

    with jsonl_path.open() as f:
        for line in f:
            ex = json.loads(line)

            if ex.get("question_type") != "MC":
                continue

            gold = ex.get("answer", "")
            if not gold:
                continue

            gold = gold.strip().upper()
            gold = gold[0] if gold else None
            if gold not in "ABCD":
                continue

            pred_raw = ex.get("model_response", "")
            pred = extract_mc_answer(pred_raw)

            print("MC raw prediction:", repr(pred), "| full response:", repr(pred_raw[:120]), "...")

            if pred is None:
                continue

            pred = pred.strip().upper()[0]
            if pred not in "ABCD":
                continue

            y_true.append(gold)
            y_pred.append(pred)

    n = len(y_true)
    if n == 0:
        print("No usable MC examples found.")
        return

    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    acc = correct / n

    dist_true = Counter(y_true)
    dist_pred = Counter(y_pred)

    print(f"# MC examples evaluated: {n}")
    print(f"Accuracy: {acc:.4f}")
    print("Gold distribution:", dict(dist_true))
    print("Pred distribution:", dict(dist_pred))


if __name__ == "__main__":
    # Run both; you can comment out one if you only care about MC.
    print("=== Evaluating YN ===")
    evaluate_yn(IN_PATH)

    print("\n=== Evaluating MC ===")
    evaluate_mc(IN_PATH)