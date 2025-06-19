import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Model klasörü
model_path = "./yeniEgitilmisTrendyolLlama3"

# Tokenizer ve model yükle
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Model değerlendirme moduna alınır
model.eval()

# Pipeline (device verilmeden)
chat_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Soru
prompt = "Soru: Yatay geçiş başvurusu ne zaman yapılır?\nCevap:"
output = chat_pipeline(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)

# Sonuç
print("Modelin cevabı:\n", output[0]["generated_text"])
