
import os
import json
import time
import dashscope
from dashscope import Generation


DASHSCOPE_API_KEY = ""
INPUT_FILE = "event.json"
OUTPUT_FILE = "event_impact.json"


dashscope.api_key = DASHSCOPE_API_KEY

MODEL_NAME = "qwen3-14b"


def analyze_influence(item):
    question_input = item.get("input", "")
    notes_list = item.get("notes", [])
    current_output = item.get("output", "").strip()

    notes_str = "\n".join(f"- {note}" for note in notes_list) if notes_list else "- 无相关事件"

    messages = [
            {"role": "system",
             "content": "You are a sociology expert specializing in analyzing how historical events influence public response tendencies to specific questions."},
            {"role": "user", "content": f"""Please analyze the following:

    [Question Context]
    {question_input}

    [Relevant Historical Events]
    {notes_str}


    - Only analyze how the answer changes reflect shifts in response tendencies (e.g., trust, support, vigilance) potentially caused by these events.
    - Keep your response concise, within 80 words.
    """}
    ]

    try:
        response = Generation.call(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.4,
            top_p=0.8,
            max_tokens=256,
            result_format='message',
            enable_thinking=False
        )
        if response.status_code == 200:
            reason = response.output.choices[0].message.content.strip()
            return reason
        else:
            return f"[API ERROR] status_code: {response.status_code}, message: {response.message}"
    except Exception as e:
        return f"[EXCEPTION] {str(e)}"


if __name__ == "__main__":
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"total locad {len(data)} data")

    for i, item in enumerate(data):
        reason = analyze_influence(item)
        item["reason"] = reason
        preview = reason[:120] + "..." if len(reason) > 120 else reason
        print("→", preview)
        print("-" * 60)

        if i < len(data) - 1:
            time.sleep(0.3)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)