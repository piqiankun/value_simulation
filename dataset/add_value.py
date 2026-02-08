import json
import re


RANGES = [
    (1, 45, "SOCIAL VALUES, ATTITUDES & STEREOTYPES"),
    (46, 56, "HAPPINESS AND WELL-BEING"),
    (57, 105, "SOCIAL CAPITAL, TRUST & ORGANIZATIONAL MEMBERSHIP"),
    (106, 111, "ECONOMIC VALUES"),
    (112, 120, "CORRUPTION"),
    (121, 130, "MIGRATION"),
    (131, 151, "SECURITY"),
    (152, 157, "POSTMATERIALIST INDEX"),
    (158, 163, "SCIENCE & TECHNOLOGY"),
    (164, 175, "RELIGIOUS VALUES"),
    (176, 198, "ETHICAL VALUES AND NORMS"),
    (199, 234, "POLITICAL INTEREST & POLITICAL PARTICIPATION"),
    (235, 259, "POLITICAL CULTURE & POLITICAL REGIMES"),
]


def get_value_from_qid(qid: str) -> str:

    if not isinstance(qid, str):
        return "OTHER"


    match = re.search(r'Q(\d+)', qid, re.IGNORECASE)
    if not match:
        return "OTHER"

    try:
        num = int(match.group(1))
    except ValueError:
        return "OTHER"

    for start, end, category in RANGES:
        if start <= num <= end:
            return category

    return "OTHER"



INPUT_FILE = "output.json"
OUTPUT_FILE = "output_data_with_value.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


updated_data = []
for item in data:
    qid = item.get("question_id")
    item["value"] = get_value_from_qid(qid)
    updated_data.append(item)


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(updated_data, f, ensure_ascii=False, indent=2)
