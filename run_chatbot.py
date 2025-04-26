#!/usr/bin/env python
# filepath: d:\gitClones\sau_chat\run_chatbot.py

import os
import sys
import argparse
from typing import List, Dict, Any
from llama_cpp import Llama
from vector_db_helpers import retrieve_relevant_context

def main():
    """
    GGUF formatında LLaMA modelini kullanarak konsol tabanlı bir chatbot çalıştırır.
    Sorguları vektör veritabanında arar ve ilgili bağlamı kullanarak yanıt üretir.
    """
    parser = argparse.ArgumentParser(description="SAÜChat - LLaMA 3 tabanlı Sakarya Üniversitesi Yönetmelik Chatbotu")
    parser.add_argument("--model_path", type=str, default="D:\models\gguf\llama-3-GGUF\llama3-8B-trendyol-rag-merged-Q8_0.gguf", 
                        help="GGUF formatındaki model dosyasının yolu")
    parser.add_argument("--db_path", type=str, default="vector_db",
                        help="FAISS vektör veritabanının yolu")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Her sorgu için döndürülecek en alakalı belge sayısı")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Yanıt için maksimum token sayısı")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Model yanıtlarının sıcaklık değeri (yaratıcılık seviyesi)")
    
    args = parser.parse_args()
    
    # Model dosyasının varlığını kontrol et
    if not os.path.exists(args.model_path):
        print(f"Hata: Model dosyası bulunamadı: {args.model_path}")
        print("Lütfen doğru model yolunu belirtin veya modeli indirin")
        sys.exit(1)
    
    # Vektör veritabanının varlığını kontrol et
    if not os.path.exists(args.db_path):
        print(f"Hata: Vektör veritabanı bulunamadı: {args.db_path}")
        print("Lütfen doğru veritabanı yolunu belirtin veya veritabanını oluşturun")
        sys.exit(1)
    
    print(f"Model yükleniyor: {args.model_path}")
    print("Bu işlem birkaç dakika sürebilir...")
    
    try:
        # GGUF formatındaki modeli yükle
        model = Llama(
            model_path=args.model_path,
            n_ctx=2048,  # Bağlam penceresi boyutu
            n_gpu_layers=-1,  # Mümkünse tüm katmanları GPU'ya yükle
            n_threads=os.cpu_count(),  # Kullanılabilir tüm CPU çekirdeklerini kullan
        )
        
        print("Model başarıyla yüklendi!")
        print("SAÜChat'e hoş geldiniz! Üniversite yönerge ve yönetmelikleri hakkında sorular sorabilirsiniz.")
        print("Çıkmak için 'q', 'quit' veya 'exit' yazabilirsiniz.\n")
        
        while True:
            user_input = input("\n\033[1;34mSoru: \033[0m")
            
            # Çıkış kontrolü
            if user_input.lower() in ['q', 'quit', 'exit', 'çıkış']:
                print("Görüşmek üzere!")
                break
            
            # Vektör veritabanından ilgili bilgileri getir
            print("İlgili bilgiler aranıyor...")
            relevant_context = retrieve_relevant_context(user_input, args.db_path, args.top_k)
            
            if not relevant_context:
                print("\033[1;31mBu konuda bilgi bulunamadı.\033[0m")
                continue
            
            # Prompt oluştur
            prompt = f"""<CONTEXT>
{relevant_context}
</CONTEXT>

Sen Sakarya Üniversitesi'nin bilgi asistanısın. Yukarıdaki bağlamda verilen bilgileri kullanarak öğrencilere
yardımcı oluyorsun. Sadece bağlamda bulunan bilgileri kullan, uydurma bilgi verme. 
Eğer cevabı bilmiyorsan veya bağlamda yeterli bilgi yoksa, dürüstçe bilmediğini söyle.

Soru: {user_input}

Yanıt:"""
            
            print("Yanıt oluşturuluyor...")
            
            # Modelden yanıt al
            response = model.create_completion(
                prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=0.95,
                repeat_penalty=1.1,
                stop=["Soru:", "<CONTEXT>"]
            )
            
            # Yanıtı göster
            print("\n\033[1;32mYanıt:\033[0m", response['choices'][0]['text'].strip())
            
    except Exception as e:
        print(f"\nHata oluştu: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()