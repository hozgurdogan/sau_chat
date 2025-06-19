# run_chatbot.py
#!/usr/bin/env python
# filepath: d:\gitClones\sau_chat\run_chatbot.py

import os
import sys
import argparse
from typing import List, Dict, Any # Bu satır aslında kullanılmıyor gibi, ama kalsın
import torch # MODIFIED: Added torch
# REMOVED: from llama_cpp import Llama
from transformers import AutoModelForCausalLM, AutoTokenizer # MODIFIED: Import transformers
from vector_db_helpers import retrieve_relevant_context

def main():
    """
    HuggingFace Transformers formatında bir dil modelini kullanarak konsol tabanlı bir chatbot çalıştırır.
    Sorguları vektör veritabanında arar ve ilgili bağlamı kullanarak yanıt üretir.
    """
 #   parser = argparse.ArgumentParser(description="SAÜChat - HuggingFace Llama 3 tabanlı Sakarya Üniversitesi Yönetmelik Chatbotu")
    # MODIFIED: Model path artık bir klasör yolu
   # parser.add_argument("--model_path", type=str, default="./yeniEgitilmisTrendyolLlama3",
#                        help="HuggingFace formatındaki model klasörünün yolu")
    parser = argparse.ArgumentParser(description="SAÜChat - HuggingFace Llama 3 tabanlı Sakarya Üniversitesi Yönetmelik Chatbotu")
    # MODIFIED: Model path artık bir klasör yolu ve varsayılan değer Google Drive adresi olarak güncellendi
    parser.add_argument("--model_path", type=str,
                        default="/content/drive/MyDrive/HasanProje/sau_chat/yeniEgitilmisTrendyolLlama3", # <--- DEĞİŞİKLİK BURADA
                        help="HuggingFace formatındaki model klasörünün yolu (varsayılan: Google Drive)")
    
    parser.add_argument("--db_path", type=str, default="vector_db",
                        help="FAISS vektör veritabanının yolu")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Her sorgu için döndürülecek en alakalı belge sayısı")
    # MODIFIED: max_tokens -> max_new_tokens
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Yanıt için üretilecek maksimum yeni token sayısı")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Model yanıtlarının sıcaklık değeri (yaratıcılık seviyesi)")
    # MODIFIED: Add context length argument (optional, tokenizer'dan da alınabilir)
    parser.add_argument("--max_context_length", type=int, default=4096,
                        help="Modelin işleyebileceği maksimum toplam token sayısı (prompt + yanıt).")

    args = parser.parse_args()

    # MODIFIED: Model yolu klasör olarak kontrol edilecek
    if not os.path.isdir(args.model_path):
        print(f"Hata: Model klasörü bulunamadı: {args.model_path}")
        print("Lütfen doğru model klasör yolunu belirtin.")
        sys.exit(1)

    if not os.path.exists(args.db_path): # Bu kontrol aynı kalıyor
        print(f"Hata: Vektör veritabanı bulunamadı: {args.db_path}")
        print("Lütfen doğru veritabanı yolunu belirtin veya veritabanını oluşturun")
        sys.exit(1)

    print(f"HuggingFace modeli yükleniyor: {args.model_path}")
    print("Bu işlem GPU'nuzun hızına bağlı olarak biraz zaman alabilir...")

    try:
        # MODIFIED: HuggingFace modelini ve tokenizer'ını yükle
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map="auto", # Otomatik GPU veya CPU ataması
            trust_remote_code=True
        )
        
        # Tokenizer'dan max_length almak daha dinamik olabilir:
        # effective_max_context_length = tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else args.max_context_length
        effective_max_context_length = args.max_context_length


        print("Model başarıyla yüklendi!")
        if hasattr(model, 'hf_device_map'):
             print(f"Model cihaz haritası: {model.hf_device_map}")
        else:
            try:
                device = next(model.parameters()).device
                print(f"Model şuraya yüklendi: {device}")
            except Exception:
                 print("Modelin cihazı belirlenemedi.")

        print("SAÜChat'e hoş geldiniz! Üniversite yönerge ve yönetmelikleri hakkında sorular sorabilirsiniz.")
        print("Çıkmak için 'q', 'quit' veya 'exit' yazabilirsiniz.\n")

        while True:
            user_input = input("\n\033[1;34mSoru: \033[0m")

            if user_input.lower() in ['q', 'quit', 'exit', 'çıkış']:
                print("Görüşmek üzere!")
                break

            print("İlgili bilgiler aranıyor...")
            retrieval_result = retrieve_relevant_context(user_input, args.db_path, args.top_k, return_sources=False) # Sadece text lazım
            context = retrieval_result.get("text", "Bağlam alınamadı.")


            # MODIFIED: Prompt Llama 3 Instruct formatına uygun hale getirildi
            system_prompt_template = """Sen Sakarya Üniversitesi'nin bilgi asistanısın. Yukarıdaki bağlamda verilen bilgileri kullanarak öğrencilere yardımcı oluyorsun. Sadece bağlamda bulunan bilgileri kullan, uydurma bilgi verme. Eğer cevabı bilmiyorsan veya bağlamda yeterli bilgi yoksa, dürüstçe bilmediğini söyle."""

            if "Hata oluştu:" in context or "Veritabanı yüklenemedi" in context or "Sorguya uygun sonuç bulunamadı" in context:
                system_prompt_without_context = "Sen Sakarya Üniversitesi'nin yardımsever bir bilgi asistanısın. Kullanıcının sorusu hakkında veritabanında yeterli bilgi bulunmadığını veya bilgi alınırken bir sorun oluştuğunu nazikçe açıkla."
                messages = [
                    {"role": "system", "content": system_prompt_without_context},
                    {"role": "user", "content": user_input}
                ]
            else:
                full_system_prompt = f"{system_prompt_template}\n\n<CONTEXT>\n{context}\n</CONTEXT>"
                messages = [
                    {"role": "system", "content": full_system_prompt},
                    {"role": "user", "content": user_input}
                ]
            
            prompt_for_model = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = tokenizer(prompt_for_model, return_tensors="pt", truncation=True, max_length=effective_max_context_length - args.max_new_tokens)
            
            print("Yanıt oluşturuluyor...")

            # MODIFIED: Modelden yanıt al (transformers)
            generation_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature if args.temperature > 0 else None,
                "do_sample": True if args.temperature > 0 else False,
                "pad_token_id": tokenizer.eos_token_id,
                # "top_p": 0.95, # İsteğe bağlı, LlamaCPP'deki gibi
                # "repetition_penalty": 1.1, # generate'de de var
            }
            if args.temperature == 0:
                generation_kwargs.pop("temperature", None)
                generation_kwargs["do_sample"] = False

            with torch.no_grad():
                # Girişleri modelin olduğu cihaza gönder
                output_sequences = model.generate(
                    input_ids=inputs["input_ids"].to(model.device if hasattr(model, 'device') else "cuda" if torch.cuda.is_available() else "cpu"),
                    attention_mask=inputs["attention_mask"].to(model.device if hasattr(model, 'device') else "cuda" if torch.cuda.is_available() else "cpu"),
                    **generation_kwargs
                )
            
            generated_ids = output_sequences[0][inputs["input_ids"].shape[1]:]
            response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            print("\n\033[1;32mYanıt:\033[0m", response_text)

    except Exception as e:
        print(f"\nHata oluştu: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()