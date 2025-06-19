import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Yol bilgileri
base_model_id = "Trendyol/Trendyol-LLM-8b-chat-v2.0"
adapter_path = "./trendyol-finetuned"
merged_output_path = "./yeniEgitilmisTrendyolLlama3"

# Tokenizer zaten aynı
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

# Base model (GPU kullanımı için device_map="auto")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Adapter ağırlıklarını yükle
peft_model = PeftModel.from_pretrained(base_model, adapter_path)

# LoRA katmanlarını birleştir
merged_model = peft_model.merge_and_unload()

# GPU'ya taşı (isteğe bağlı ama önerilir)
merged_model = merged_model.to("cuda")

# Tek model olarak kaydet
merged_model.save_pretrained(merged_output_path)
tokenizer.save_pretrained(merged_output_path)

print(f"Model başarıyla birleştirildi ve '{merged_output_path}' klasörüne kaydedildi.")
