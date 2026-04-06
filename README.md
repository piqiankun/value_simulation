## Beyond Static Snapshots: Dynamic Modeling and Forecasting of Group-Level Value Evolution with Large Language Models

## 1. Instruction
This repository contains the code and resources for the main paper titled "Beyond Static Snapshots: Dynamic Modeling and Forecasting of Group-Level Value Evolution with Large Language Models". The project focuses on adapting large language models (LLMs) to dynamic simulate survey response distributions across diverse groups populations.

## 2. Installation
Our fine-tuning and inference processes are implemented based on the ms-swift framework.
```
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e
```

## 3. Data preparation
The data folder includes all intermediate and processed files involved in the data pipelines for the Chinese and U.S. datasets,
including gold labels, initial data, groups construction,questions construction, and group–question matching.

```php
data/
│
├── china/
│   ├── 0_groundtruth
│       ├──seen questions
│            ├──single.json
│            ├──scale.json
│       ├──unseen questions
│            ├──single.json
│            ├──scale.json
│   ├── 1_initial data
│       ├──china_wvs5_initial.xlsx
│       ├──china_wvs6_initial.xlsx
│       ├──china_wvs7_initial.xlsx
│   ├── 2_groups
│       ├──wvs5_28groups.xlsx
│       ├──wvs6_28groups.xlsx
│       ├──wvs7_28groups.xlsx
│   ├── 3_questions
│       ├──seen questions
│            ├──wvs5.json
│            ├──wvs6.json
│            ├──wvs7.json
│       ├──unseen questions
│            ├──wvs6.json
│            ├──wvs7.json
│   ├── 4_group and questions matching
│       ├──seen questions
│            ├──wvs5-6.json
│            ├──wvs6-7.json
│       ├──unseen questions
│            ├──wvs6-7.json
└── united states
     ...
```


## 4. Fine-tuning
```
bash lora_sft.sh
```
## 5. Inference
```
bash infer.sh
```

## 6.Evaluation
```
python evaluation.py
```

