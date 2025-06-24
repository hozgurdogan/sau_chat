import json
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer
)
from peft import get_peft_model, LoraConfig, TaskType
import torch

# Veri yolu
json_path = "yeniSoruCevap.json"

# Veriyi oku


with open(json_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Soru-Cevap birleştir
texts = [
    f"Soru: {entry['question'].strip()}\nCevap: {entry['answer'].strip()}"
    for entry in raw_data if entry.get("question") and entry.get("answer")
]

# Dataset oluştur
dataset = Dataset.from_dict({"text": texts})

# Model ve tokenizer yükle
model_id = "Trendyol/Trendyol-LLM-8b-chat-v2.0"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

# LoRA ile model sarmala
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    inference_mode=False
)
model = get_peft_model(model, peft_config)

# Tokenizasyon fonksiyonu
def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize, batched=True)

# Eğitim ayarları
output_dir = "./trendyol-finetuned"
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,  # Daha fazla GPU varsa bu artırılabilir
    gradient_accumulation_steps=8,
    num_train_epochs=10,  # Daha fazla epoch
    learning_rate=1e-4,  # Daha düşük, daha stabil
    warmup_steps=10,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    logging_steps=5,
    save_steps=50,
    save_total_limit=3,
    fp16=True,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# Eğitimi başlat
train_result = trainer.train()

# Modeli ve tokenizer’ı kaydet
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Metrikleri kaydet
metrics_path = os.path.join(output_dir, "train_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(train_result.metrics, f, indent=4)




from transformers import pipeline

# Modeli değerlendirme moduna al
model.eval()

# pipeline oluştur (device parametresi olmadan!)
chat_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Test prompt
prompt = "Soru: ÇAP ve Yandal aynı anda yapılabilir mi?\nCevap:"
outputs = chat_pipeline(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)

# Çıktıyı yazdır
print("Modelin cevabı:\n", outputs[0]["generated_text"])
