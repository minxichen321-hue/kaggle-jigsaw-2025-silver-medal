import numpy as np
import pandas as pd
from datasets import Dataset
import os
seed = 22

base_model_path = "/kaggle/input/qwen-3/transformers/4b-base/1"
pretrain_lora_path = None
lora_path = "/kaggle/working/pseudo_lora"
use_gptq = "gptq" in base_model_path

positive = "Yes"
negative = "No"
judge_words = "Violation:"
system_prompt = '''You are given a comment from reddit and a rule. 
Your task is to classify whether the comment violates the rule. 
Only respond Yes/No.'''

frac = 0.05
use_train = True

import kagglehub

deterministic = kagglehub.package_import('wasupandceacar/deterministic').deterministic
deterministic.init_all(seed)


IS_KAGGLE_SUBMISSION = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))

def build_prompt(row):
    return f"""{system_prompt}
Subreddit: r/{row["subreddit"]}
Rule: {row["rule"]}
Examples:
1) {row["positive_example"]}
{judge_words} Yes
2) {row["negative_example"]}
{judge_words} No
Comment: {row["body"]}
{judge_words}"""

def get_df():
    merge = list()
    if use_train:
        train_dataset = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
        train_df = train_dataset[["body", "rule", "subreddit", "rule_violation",
                                "positive_example_1", "positive_example_2", 
                                "negative_example_1", "negative_example_2"]].copy()
        train_df["positive_example"] = np.where(np.random.rand(len(train_df)) < 0.5, train_df["positive_example_1"], train_df["positive_example_2"])
        train_df["negative_example"] = np.where(np.random.rand(len(train_df)) < 0.5, train_df["negative_example_1"], train_df["negative_example_2"])
        train_df.drop(columns=["positive_example_1", "positive_example_2", "negative_example_1", "negative_example_2"], inplace=True)
        merge.append(train_df)
    test_dataset = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
    test_dataset = test_dataset.groupby('rule', group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=seed)).reset_index(drop=True)
    print(f"Select {len(test_dataset)} test data")
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            sub_dataset = test_dataset[["rule", "subreddit", "positive_example_1", "positive_example_2", "negative_example_1", "negative_example_2"]].copy()
            body_col = f"{violation_type}_example_{i}"
            other_positive_col = f"{violation_type}_example_{3-i}"
            sub_dataset["body"] = sub_dataset[body_col]
            sub_dataset[f"{violation_type}_example"] = sub_dataset[other_positive_col]
            anti_violation_type = "negative" if violation_type == "positive" else "positive"
            sub_dataset[f"{anti_violation_type}_example"] = np.where(np.random.rand(len(sub_dataset)) < 0.5, sub_dataset[f"{anti_violation_type}_example_1"], sub_dataset[f"{anti_violation_type}_example_2"])
            sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
            sub_dataset.drop(columns=["positive_example_1", "positive_example_2", "negative_example_1", "negative_example_2"], inplace=True)
            merge.append(sub_dataset)
    return pd.concat(merge, axis=0).drop_duplicates(ignore_index=True)

def build_dataset(df):
    df["prompt"] = df.apply(build_prompt, axis=1)
    columns = ["prompt"]
    if "rule_violation" in df:
        df["completion"] = df["rule_violation"].map({
            1: positive,
            0: negative,})
        columns.append("completion")
    if IS_KAGGLE_SUBMISSION:
        dataset = Dataset.from_pandas(df[columns])
    else:
        dataset = Dataset.from_pandas(df[columns].head(64))
    return dataset



import torch
import pandas as pd
from trl import SFTTrainer, SFTConfig
from peft import PeftModel, LoraConfig, get_peft_model
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_torch_bf16_gpu_available

def main():
    train_dataset = build_dataset(get_df())
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    
    training_args = SFTConfig(
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        weight_decay=0.20,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=is_torch_bf16_gpu_available(),
        fp16=not is_torch_bf16_gpu_available(),
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="no",
        report_to="none",
        completion_only_loss=True,
        packing=False,
        remove_unused_columns=False,
    )

    if use_gptq:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="balanced_low_0",
            trust_remote_code=True,
            use_cache=False,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,     
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ),
            device_map="balanced_low_0",
            trust_remote_code=True,
            use_cache=False,
        )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    if pretrain_lora_path:
        model = PeftModel.from_pretrained(model, pretrain_lora_path)
        model = model.merge_and_unload()

    if len(train_dataset) > 0:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            peft_config=lora_config,
        )
        trainer.train()
        trainer.save_model(lora_path)
    else:
        peft_model = get_peft_model(model, lora_config)
        peft_model.save_pretrained(lora_path)
        tokenizer.save_pretrained(lora_path)

if __name__ == "__main__":
    main()
