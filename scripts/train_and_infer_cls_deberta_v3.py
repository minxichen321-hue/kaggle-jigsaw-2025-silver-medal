import pandas as pd
import re

# from rule_augmenter import build_synthetic_rule_map

def url_to_semantics(text: str) -> str:
    if not isinstance(text, str):
        return ""

    url_pattern = r'https?://[^\s/$.?#].[^\s]*'
    urls = re.findall(url_pattern, text)
    
    if not urls:
        return "" 

    all_semantics = []
    seen_semantics = set()

    for url in urls:
        url_lower = url.lower()
        
        domain_match = re.search(r"(?:https?://)?([a-z0-9\-\.]+)\.[a-z]{2,}", url_lower)
        if domain_match:
            full_domain = domain_match.group(1)
            parts = full_domain.split('.')
            for part in parts:
                if part and part not in seen_semantics and len(part) > 3: # Avoid short parts like 'www'
                    all_semantics.append(f"domain:{part}")
                    seen_semantics.add(part)

        # 2. Extract path parts
        path = re.sub(r"^(?:https?://)?[a-z0-9\.-]+\.[a-z]{2,}/?", "", url_lower)
        path_parts = [p for p in re.split(r'[/_.-]+', path) if p and p.isalnum()] # Split by common delimiters

        for part in path_parts:
            # Clean up potential file extensions or query params
            part_clean = re.sub(r"\.(html?|php|asp|jsp)$|#.*|\?.*", "", part)
            if part_clean and part_clean not in seen_semantics and len(part_clean) > 3:
                all_semantics.append(f"path:{part_clean}")
                seen_semantics.add(part_clean)

    if not all_semantics:
        return ""

    return f"\nURL Keywords: {' '.join(all_semantics)}"


def get_dataframe_to_train(data_path):
    train_dataset = pd.read_csv(f"{data_path}/train.csv") 
    test_dataset = pd.read_csv(f"{data_path}/test.csv")

    flatten = []

    flatten.append(train_dataset[["body", "rule", "subreddit","rule_violation"]].copy())

    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col_name = f"{violation_type}_example_{i}"
            
            if col_name in train_dataset.columns:
                sub_dataset = train_dataset[[col_name, "rule", "subreddit"]].copy()
                sub_dataset = sub_dataset.rename(columns={col_name: "body"})
                sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
                
                sub_dataset.dropna(subset=['body'], inplace=True)
                sub_dataset = sub_dataset[sub_dataset['body'].str.strip().str.len() > 0]
                
                if not sub_dataset.empty:
                    flatten.append(sub_dataset)
    
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col_name = f"{violation_type}_example_{i}"
            
            if col_name in test_dataset.columns:
                sub_dataset = test_dataset[[col_name, "rule", "subreddit"]].copy()
                sub_dataset = sub_dataset.rename(columns={col_name: "body"})
                sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
                
                sub_dataset.dropna(subset=['body'], inplace=True)
                sub_dataset = sub_dataset[sub_dataset['body'].str.strip().str.len() > 0]
                
                if not sub_dataset.empty:
                    flatten.append(sub_dataset)
    
    dataframe = pd.concat(flatten, axis=0)
    dataframe = dataframe.drop_duplicates(subset=['body', 'rule', 'subreddit'], ignore_index=True)
    dataframe.drop_duplicates(subset=['body','rule'],keep='first',inplace=True)
    
    return dataframe.sample(frac=1, random_state=42).reset_index(drop=True)

import os
import pandas as pd
import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split 
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    get_scheduler
)

from copy import deepcopy
from contextlib import contextmanager
import torch.nn as nn

#from utils import get_dataframe_to_train, url_to_semantics

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CFG:
    model_name_or_path = "/kaggle/input/huggingfacedebertav3variants/deberta-v3-base"
    data_path = "/kaggle/input/jigsaw-agile-community-rules/"
    output_dir = "./save"
  
    EPOCHS = 3
    LEARNING_RATE = 2e-5  
    
    MAX_LENGTH = 512
    BATCH_SIZE = 8

class JigsawDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from transformers.modeling_outputs import SequenceClassifierOutput

class ModelWithAttentionHead(nn.Module):
    def __init__(self, model_name_or_path, num_labels=2, dropout_rate=0.1):
        super().__init__()
        
        # 先加载配置并更新
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.config.output_hidden_states = True
        self.config.hidden_dropout_prob = 0.0
        self.config.return_dict = True
        self.config.num_labels = num_labels  # 添加这行
        
        # 使用更新后的配置加载模型
        self.backbone = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        
        # 获取隐藏层大小
        hidden_size = self.config.hidden_size
        self.num_labels = num_labels
        
        # CNN层
        self.attention = nn.Sequential(            
            nn.Linear(self.config.hidden_size, 1024),            
            nn.Tanh(),                       
            nn.Linear(1024, 1),
            nn.Softmax(dim=1)
        )        

        self.regressor = nn.Linear(self.config.hidden_size, self.config.num_labels)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Enables gradient checkpointing for the backbone model.
        This forwards the call to the underlying Hugging Face model.
        """

        # DeBERTa-v3/v2 使用不同的 gradient_checkpointing_kwargs
        if "deberta-v2" in self.config.model_type or "deberta-v3" in self.config.model_type:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # 获取backbone输出
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )
        
        # 获取隐藏层状态
        hs = outputs.hidden_states
        
        # 检查是否有足够的层
        if hs is None or len(hs) < 4:
            raise ValueError(f"Not enough hidden states. Got {len(hs) if hs else 0} layers.")
        
        last_layer_hidden_states = torch.mean(torch.stack([hs[-1]*2.2, hs[-2]]), 0)
        weights = self.attention(last_layer_hidden_states)
        context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)
        #logits = self.regressor(context_vector)
        logits1 = self.regressor(self.dropout1(context_vector))
        logits2 = self.regressor(self.dropout2(context_vector))
        logits3 = self.regressor(self.dropout3(context_vector))
        logits4 = self.regressor(self.dropout4(context_vector))
        logits5 = self.regressor(self.dropout5(context_vector))

        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5

        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.10)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        # 返回 SequenceClassifierOutput 对象
        if labels is not None:
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}
            
class AdvancedTrainer(Trainer):
    def __init__(self, *args, alpha=0.3, epsilon=1.0, 
                 use_ema=True, ema_decay=0.99, ema_start_ratio=0.8, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha, self.epsilon, self.emb_name, self.backup = alpha, epsilon, 'word_embeddings', {}
        self.use_ema, self.ema_decay, self.ema_start_ratio = use_ema, ema_decay, ema_start_ratio
        self.ema_model, self.backup_weights = None, None

    def _create_ema_model(self): self.ema_model = deepcopy(self.model)
    def _update_ema(self, model):
        if not self.use_ema or self.ema_model is None: return
        with torch.no_grad():
            for ema, param in zip(self.ema_model.parameters(), model.parameters()):
                ema.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    @contextmanager
    def _swap_ema_weights(self):
        if self.use_ema and self.ema_model:
            self.backup_weights = {n: p.data.clone() for n, p in self.model.named_parameters()}
            for n, p in self.model.named_parameters(): p.data.copy_(self.ema_model.state_dict()[n])
            try: yield
            finally:
                for n, p in self.model.named_parameters(): p.data.copy_(self.backup_weights[n])
                self.backup_weights = None
        else: yield
    
    def training_step(self, model, inputs, *args, **kwargs):
        loss = super().training_step(model, inputs, *args, **kwargs)
        if self.use_ema:
            start_step = int(self.state.max_steps * self.ema_start_ratio)
            if self.state.global_step >= start_step:
                if self.ema_model is None:
                    self._create_ema_model(); print(f"\nINFO: EMA starting at step {self.state.global_step}.")
                self._update_ema(model)
        return loss
    
    def evaluate(self, *args, **kwargs):
        with self._swap_ema_weights(): return super().evaluate(*args, **kwargs)
    def predict(self, *args, **kwargs):
        with self._swap_ema_weights(): return super().predict(*args, **kwargs)

    def _save_restore_embeddings(self, restore=False):
        for n, p in self.model.named_parameters():
            if p.requires_grad and self.emb_name in n:
                if not restore: self.backup[n] = p.data.clone()
                else: p.data = self.backup[n]
    def _attack_embeddings(self, step_size):
        for n, p in self.model.named_parameters():
            if p.requires_grad and self.emb_name in n and p.grad is not None:
                norm = torch.norm(p.grad); p.data.add_(step_size * p.grad / (norm + 1e-8))
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs['loss']; loss = loss.mean() if loss.ndim > 0 else loss
        if model.training:
            self.accelerator.backward(loss, retain_graph=True)
            self._save_restore_embeddings(restore=False)
            self._attack_embeddings(self.epsilon)
            for _ in range(2):
                with self.compute_loss_context_manager(): adv_out = model(**inputs)
                adv_loss = adv_out['loss']; adv_loss = adv_loss.mean() if adv_loss.ndim > 0 else adv_loss
                self.accelerator.backward(adv_loss)
                self._attack_embeddings(0.3)
            with self.compute_loss_context_manager(): final_adv_out = model(**inputs)
            adv_loss = final_adv_out['loss']; adv_loss = adv_loss.mean() if adv_loss.ndim > 0 else adv_loss
            total_loss = loss + self.alpha * adv_loss
            self._save_restore_embeddings(restore=True)
            model.zero_grad()
        else: total_loss = loss
        return (total_loss, outputs) if return_outputs else total_loss



def main():
    seed_everything(42)
    training_data_df = get_dataframe_to_train(CFG.data_path)
    # training_data_df, valid_df = train_test_split(full_df,test_size=0.2,stratify=full_df['rule'],random_state=42)
    print(f"Training dataset (from examples only) size: {len(training_data_df)}")

    test_df_for_prediction = pd.read_csv(f"{CFG.data_path}/test.csv")
    
    training_data_df['body_with_url'] = training_data_df['body'].apply(lambda x: x + url_to_semantics(x))
    training_data_df['input_text'] = training_data_df['rule'] + "[SEP]" + training_data_df['body_with_url']
    if len(test_df_for_prediction)==10:
        training_data_df = training_data_df.head(128)
        #CFG.EPOCHS=1
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name_or_path)
    train_encodings = tokenizer(training_data_df['input_text'].tolist(), truncation=True, padding=True, max_length=CFG.MAX_LENGTH)
    train_labels = training_data_df['rule_violation'].tolist()
    train_dataset = JigsawDataset(train_encodings, train_labels)

    model = ModelWithAttentionHead(CFG.model_name_or_path, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir=CFG.output_dir,
        num_train_epochs=CFG.EPOCHS,
        learning_rate=CFG.LEARNING_RATE,
        per_device_train_batch_size=CFG.BATCH_SIZE,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type='cosine',
        report_to="none",
        save_strategy="no", 
        logging_steps=10,
        #fp16=True,
        #gradient_checkpointing=True,
        
    )
    
    trainer = AdvancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    trainer.train()

    test_df_for_prediction['body_with_url'] = test_df_for_prediction['body'].apply(lambda x: x + url_to_semantics(x))
    test_df_for_prediction['input_text'] = test_df_for_prediction['rule'] + "[SEP]" + test_df_for_prediction['body_with_url']
    
    test_encodings = tokenizer(test_df_for_prediction['input_text'].tolist(), truncation=True, padding=True, max_length=CFG.MAX_LENGTH)
    test_dataset = JigsawDataset(test_encodings)
    
    predictions = trainer.predict(test_dataset)
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()

    submission_df = pd.DataFrame({
        "row_id": test_df_for_prediction["row_id"],
        "rule_violation": probs
    })
    submission_df.to_csv("sub_deberta_v3.csv", index=False)

if __name__ == "__main__":
    main()
