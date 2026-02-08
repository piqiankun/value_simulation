# -*- coding: utf-8 -*-
import os
import json
import time
from typing import List, Dict

# 🔥 必须在 import torch 之前设置！指定使用的 GPU（例如只用第 0 块）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ==============================
# 配置区 —— 请根据你的实际情况修改
# ==============================
MODEL_PATH = "/home/dataica401/PersonalFile/kanghankun/lijinliang/LLMS/GLM/GLM-4-9B-0414"  # ←←← 替换为你的本地模型路径
# 例如：
# MODEL_PATH = "/home/models/Llama-3-8B-Instruct"
# MODEL_PATH = "/home/models/Qwen2-7B-Instruct"
# MODEL_PATH = "/home/models/glm-4-9b-chat"

VALUE_LABELS = [
    "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "HAPPINESS AND WELL-BEING",
    "SOCIAL CAPITAL, TRUST & ORGANIZATIONAL MEMBERSHIP",
    "ECONOMIC VALUES",
    "CORRUPTION",
    "MIGRATION",
    "SECURITY",
    "POSTMATERIALIST INDEX",
    "SCIENCE & TECHNOLOGY",
    "RELIGIOUS VALUES",
    "ETHICAL VALUES AND NORMS",
    "POLITICAL INTEREST & POLITICAL PARTICIPATION",
    "POLITICAL CULTURE & POLITICAL REGIMES"
]

# ✅中文
# SYSTEM_PROMPT = (
#     "你是一位资深政治学与社会学研究者，熟悉世界价值观调查（World Values Survey）理论框架。"
#     "请根据用户提供的英文价值观类别名称，结合当代中国社会治理与公共舆论语境，"
#     "撰写一段严谨、中立、简短的中文解释。"
#     "\n\n具体要求："
#     "\n1. 内容必须包含：该价值观的核心内涵、在公共事件或政策中的典型体现、民众态度或制度关联；"
#     "\n3. 使用规范学术用语（如“社会信任”“政治效能感”“制度合法性”等）,控制在100字以内；"
#     "\n4. 禁止使用“指...”开头，避免定义式套话，应直接展开论述；"
#     "\n5. 不要编号、不要标题、不要引号，仅输出纯文本段落。"
# )
# ✅英文
SYSTEM_PROMPT = (
    "You are a senior researcher in political science and sociology, deeply familiar with the World Values Survey (WVS) theoretical framework. "
    "Based on the provided English value category name, generate a concise, neutral, and informative explanation in english that reflects contemporary United States governance and public discourse contexts."
    "\n\nSpecific requirements:"
    "\n1. The explanation must include: core conceptual meaning, typical manifestations in public events or policies, and connections to public attitudes or institutional arrangements;"
    "\n2. Strictly limit the response to within 80 english characters;"
    "\n3. Use formal academic terminology (e.g., 'social trust', 'political efficacy', 'institutional legitimacy');"
    "\n4. Do NOT start with phrases like 'refers to...' or 'means...'; avoid definitional clichés—begin directly with substantive analysis;"
    "\n5. Output only plain text: no numbering, no headings, no quotation marks."
)

# ==============================
# 初始化模型和 tokenizer
# ==============================
print("正在加载本地大模型和 tokenizer...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"目标设备: {device}")


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto"  # 自动分配到可用 GPU
)
model.eval()

# 创建文本生成 pipeline
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto"
)


def generate_value_explanation(label: str, retry=2) -> str:
    """调用本地 LLM 生成解释，支持重试"""
    user_prompt = f"Please explain the following value category: {label}"

    # 构造对话消息（适配不同模型的 chat template）
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    # 使用 tokenizer 的 chat template 格式化 prompt
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        # 如果不支持 apply_chat_template，回退到手动拼接（适用于部分旧模型）
        prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{user_prompt}\n<|assistant|>\n"

    for attempt in range(retry + 1):
        try:
            outputs = llm_pipeline(
                prompt,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

            generated_text = outputs[0]["generated_text"]
            # 提取模型生成部分（去掉 prompt）
            if generated_text.startswith(prompt):
                response = generated_text[len(prompt):].strip()
            else:
                response = generated_text.strip()

            # 后处理
            response = response.replace("\n", " ").strip()
            char_len = len(response)

            # 过滤过短或无效输出
            if char_len < 50:
                print(f"⚠️ {label} 输出过短 ({char_len}字)，重试中...")
                time.sleep(2)
                continue

            # 截断超过 80 字的部分（可选）


            return response

        except Exception as e:
            print(f"⚠️ 异常 ({label}, 尝试 {attempt + 1}): {e}")
            time.sleep(3)

    return ""


def main():
    print("🚀 调用本地大模型生成 13 类价值观的中文解释（50–80字）...\n")

    results: Dict[str, str] = {}

    for i, label in enumerate(VALUE_LABELS, 1):
        print(f"[{i}/{len(VALUE_LABELS)}] 正在生成：{label}")
        explanation = generate_value_explanation(label)
        results[label] = explanation or "（生成失败）"
        time.sleep(1.0)  # 避免显存峰值过高

    # 保存结果
    output_file = "wvs_values_chinese_explanations_local_llm.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 完成！结果已保存至：{output_file}\n")

    valid_count = sum(1 for v in results.values() if len(v) >= 50)
    print(f"📊 成功生成：{valid_count} / {len(VALUE_LABELS)} 项")

    print("\n🔍 预览前两项:")
    for label, desc in list(results.items())[:2]:
        print(f"\n{label}\n→ {desc} （{len(desc)}字）")


if __name__ == "__main__":
    main()