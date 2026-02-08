import json
import re
from typing import List, Dict, Optional

PRED_FILE = 'china7.json'
SINGLE_GT_FILE = 'single.json'
SCALE_GT_FILE = 'scale.json'

def load_json_or_jsonl(filepath: str) -> List[Dict]:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    if not content:
        return []
    if content.startswith('[') and content.endswith(']'):
        try:
            return json.loads(content)
        except Exception:
            pass
    data = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            try:
                data.append(json.loads(line))
            except Exception:
                continue
    return data

def clean_response_prefix(response: str) -> str:
    if not isinstance(response, str):
        return ""
    cleaned = response.lstrip()
    while cleaned and cleaned[0] in '(<\\/[{"\'*':
        cleaned = cleaned[1:].lstrip()
    return cleaned

def extract_letter_in_parentheses(text: str, valid_letters: str = "ABCDE") -> Optional[str]:
    if not isinstance(text, str):
        return None
    match = re.search(r'[\(（]([A-ZＡ-Ｚ])[\)）]', text)
    if match:
        char = match.group(1)
        if '\uff21' <= char <= '\uff3a':
            char = chr(ord(char) - 0xFEE0)
        char = char.upper()
        if char in valid_letters:
            return char
    return None

def extract_single_answer(resp: str) -> Optional[str]:
    ans = extract_letter_in_parentheses(resp, valid_letters="ABCDE")
    if ans is not None:
        return ans
    cleaned = clean_response_prefix(resp)
    match = re.search(r'[A-EＡ-Ｅ]', cleaned)
    if match:
        char = match.group(0)
        if '\uff21' <= char <= '\uff3a':
            char = chr(ord(char) - 0xFEE0)
        return char.upper()
    return None

def extract_scale_answer(resp: str) -> str:
    if not isinstance(resp, str):
        return ""
    letter = extract_letter_in_parentheses(resp, valid_letters="ABCDEFGHIJ")
    if letter is not None:
        return str(ord(letter) - ord('A') + 1)
    letter_match = re.search(r'[A-JＡ-Ｊ]', resp)
    if letter_match:
        char = letter_match.group(0)
        if '\uff21' <= char <= '\uff3a':
            char = chr(ord(char) - 0xFEE0)
        char = char.upper()
        if 'A' <= char <= 'J':
            return str(ord(char) - ord('A') + 1)
    digit_match = re.search(r'\b([1-9]|10)\b', resp)
    if digit_match:
        return digit_match.group(1)
    return resp

def main():
    pred_data = load_json_or_jsonl(PRED_FILE)
    single_gt = load_json_or_jsonl(SINGLE_GT_FILE)
    scale_gt = load_json_or_jsonl(SCALE_GT_FILE)

    print(f"  predict data: {len(pred_data)} ")
    print(f"  single label: {len(single_gt)} ")
    print(f"  scale label: {len(scale_gt)} ")

    single_preds = []
    scale_pred_strs = []

    for item in pred_data:
        if not isinstance(item, dict) or "messages" not in item or "response" not in item:
            continue

        messages = item.get("messages", [])
        is_scale = False
        if isinstance(messages, list) and len(messages) >= 2:
            user_msg = messages[-2] if messages[-1].get("role") == "assistant" else messages[-1]
            content = user_msg.get("content", "") if isinstance(user_msg, dict) else ""
            if "(J) 10" in str(content):
                is_scale = True

        resp = item["response"]
        if is_scale:
            ans_str = extract_scale_answer(resp)
            scale_pred_strs.append(ans_str)
        else:
            ans = extract_single_answer(resp)
            single_preds.append(ans)

    # ========== single question evaluation==========
    total_single = len(single_gt)
    min_single_len = min(len(single_preds), total_single)
    correct = 0
    for i in range(min_single_len):
        gt_ans = str(single_gt[i].get("output", "")).strip().upper()
        pred_ans = single_preds[i]
        if isinstance(pred_ans, str) and pred_ans == gt_ans:
            correct += 1

    single_acc = correct / total_single if total_single > 0 else 0
    print(f"\nsingle question evaluation:")
    print(f"  correct: {correct} / {total_single}")
    print(f"  acc: {single_acc:.4f}")

    # ========== scale question evaluation ==========
    total_scale = len(scale_gt)
    min_scale_len = min(len(scale_pred_strs), total_scale)
    total_score = 0.0

    for i in range(min_scale_len):
        try:
            gold_val = int(scale_gt[i]['output'])
            if not (1 <= gold_val <= 10):
                gold_val = None
        except (KeyError, ValueError, TypeError):
            gold_val = None

        if gold_val is None:
            total_score += 0.0
            continue

        pred_str = scale_pred_strs[i]
        pred_val = None
        try:
            pred_val = int(pred_str)
        except (ValueError, TypeError):
            if isinstance(pred_str, str):
                letter = extract_letter_in_parentheses(pred_str, valid_letters="ABCDEFGHIJ")
                if letter:
                    pred_val = ord(letter) - ord('A') + 1
                else:
                    letter_match = re.search(r'[A-JＡ-Ｊ]', pred_str)
                    if letter_match:
                        char = letter_match.group(0)
                        if '\uff21' <= char <= '\uff3a':
                            char = chr(ord(char) - 0xFEE0)
                        char = char.upper()
                        if 'A' <= char <= 'J':
                            pred_val = ord(char) - ord('A') + 1

        if pred_val is None or not (1 <= pred_val <= 10):
            score = 0.0
        else:
            abs_error = abs(gold_val - pred_val)
            score = max(0.0, 1.0 - abs_error / 5.0)
        total_score += score

    scale_acc = total_score / total_scale if total_scale > 0 else 0
    print(f"\nscale question evaluation:")
    print(f"  total_score: {total_score:.2f} / {total_scale}")
    print(f"  scale_acc: {scale_acc:.4f}")

    # ========== Overall score ==========
    total_items = total_single + total_scale
    total_correct_score = correct + total_score
    final_score = total_correct_score / total_items if total_items > 0 else 0

    print(f"\n🎯 overall score:")
    print(f"  total_single correct: {correct}")
    print(f"  total_scale correct: {total_score:.2f}")
    print(f"  questions amount: {total_items}")
    print(f"  overall = ({correct} + {total_score:.2f}) / {total_items} = {final_score:.4f}")

if __name__ == "__main__":
    main()