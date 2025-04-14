"""
FAISS Vektör Veritabanı İşleyici

Bu betik, işlenmiş metin parçalarını alır ve FAISS vektör veritabanı oluşturur veya mevcut bir veritabanını günceller.
Metinleri vektör temsillerine dönüştürmek için HuggingFace'ten alınan önceden eğitilmiş bir model kullanır.

Ana İşlevleri:
1. Metin parçalarını vektörlere dönüştürme
2. Yeni bir FAISS vektör veritabanı oluşturma
3. Mevcut bir FAISS veritabanına yeni verileri ekleme

Kullanım:
1. Doğrudan çalıştırma:
   python faiss_index_process.py --input_file processed_chunks/processed_chunks.json --db_path vector_db --create_new
   python faiss_index_process.py --input_file processed_chunks/processed_chunks.json --db_path vector_db

2. Fonksiyon olarak:
   from faiss_index_process import create_faiss_index, add_to_faiss_index
   
   # Yeni bir veritabanı oluşturmak için:
   create_faiss_index("processed_chunks.json", "vector_db")
   
   # Mevcut bir veritabanına veri eklemek için:
   add_to_faiss_index("processed_chunks.json", "vector_db")
"""

import os
import json
import argparse
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Varsayılan vektör modeli (Türkçe dili için uygun bir model seçildi)
DEFAULT_MODEL = "dbmdz/bert-base-turkish-cased"

def load_documents(input_file: str) -> List[Dict[str, Any]]:
    """
    İşlenmiş metin parçalarını JSON dosyasından yükler.
    
    Args:
        input_file (str): İşlenmiş metin parçalarını içeren JSON dosyasının yolu
        
    Returns:
        List[Dict[str, Any]]: Metin parçası sözlüklerinin listesi
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        # Metin içeriğinin ve ID'lerin var olduğunu kontrol et
        valid_docs = []
        for doc in documents:
            if "text" in doc and "id" in doc:
                valid_docs.append(doc)
            else:
                print(f"Uyarı: Geçersiz belge formatı: {doc}")
        
        print(f"Toplam {len(valid_docs)} geçerli belge yüklendi.")
        return valid_docs
    
    except Exception as e:
        print(f"Belgeleri yüklerken hata oluştu: {str(e)}")
        return []

def get_embeddings(texts: List[str], model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """
    Metin listesi için vektör embeddingleri oluşturur.
    
    Args:
        texts (List[str]): Vektörlere dönüştürülecek metinlerin listesi
        model_name (str): Kullanılacak model adı
        
    Returns:
        np.ndarray: Oluşturulan vektör embeddinglerinin dizisi
    """
    try:
        # Modeli yükle
        print(f"{model_name} modeli yükleniyor...")
        model = SentenceTransformer(model_name)
        print("Model yüklendi.")
        
        # Metinleri vektörlere dönüştür ve ilerlemeyi göster
        print("Metinleri vektörlere dönüştürme işlemi başlatılıyor...")
        embeddings = []
        
        # İlerleme çubuğu ile vektörleştirme
        for text in tqdm(texts, desc="Metinleri Vektörleştirme"):
            embedding = model.encode(text)
            embeddings.append(embedding)
        
        # NumPy dizisine dönüştür
        embeddings_array = np.array(embeddings).astype('float32')
        
        print(f"Toplam {len(embeddings)} metin vektöre dönüştürüldü.")
        return embeddings_array
    
    except Exception as e:
        print(f"Vektörleri oluştururken hata meydana geldi: {str(e)}")
        return np.array([])

def create_faiss_index(input_file: str, db_path: str, model_name: str = DEFAULT_MODEL) -> bool:
    """
    Yeni bir FAISS vektör veritabanı oluşturur.
    
    Args:
        input_file (str): İşlenmiş metin parçalarını içeren JSON dosyasının yolu
        db_path (str): Oluşturulacak veritabanının kaydedileceği klasör
        model_name (str): Vektörleştirmede kullanılacak model adı
        
    Returns:
        bool: İşlem başarılıysa True, değilse False
    """
    try:
        # Klasörü oluştur (yoksa)
        os.makedirs(db_path, exist_ok=True)
        
        # Belgeleri yükle
        documents = load_documents(input_file)
        if not documents:
            print("İşlenecek belge bulunamadı.")
            return False
        
        # Metin ve ID'leri ayır
        texts = [doc["text"] for doc in documents]
        ids = [doc["id"] for doc in documents]
        
        # Metinleri vektörlere dönüştür
        embeddings = get_embeddings(texts, model_name)
        if len(embeddings) == 0:
            print("Vektör oluşturma işlemi başarısız oldu.")
            return False
        
        # FAISS indeksi oluştur
        vector_dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(vector_dimension)
        
        # Metinleri indekse ekle
        index.add(embeddings)
        
        # Indeksi kaydet
        faiss.write_index(index, os.path.join(db_path, "index.faiss"))
        
        # Metin içeriklerini ve meta verileri ayrı bir dosyaya kaydet
        with open(os.path.join(db_path, "documents.json"), 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        
        # ID'leri ayrı bir dosyaya kaydet
        with open(os.path.join(db_path, "ids.json"), 'w', encoding='utf-8') as f:
            json.dump(ids, f, ensure_ascii=False)
        
        print(f"FAISS veritabanı başarıyla oluşturuldu ve {db_path} klasörüne kaydedildi.")
        print(f"Toplam {len(documents)} belge indekslendi.")
        
        return True
    
    except Exception as e:
        print(f"Veritabanı oluşturulurken hata meydana geldi: {str(e)}")
        return False

def add_to_faiss_index(input_file: str, db_path: str, model_name: str = DEFAULT_MODEL) -> bool:
    """
    Mevcut bir FAISS vektör veritabanına yeni belgeleri ekler.
    
    Args:
        input_file (str): İşlenmiş metin parçalarını içeren JSON dosyasının yolu
        db_path (str): Mevcut veritabanının klasör yolu
        model_name (str): Vektörleştirmede kullanılacak model adı
        
    Returns:
        bool: İşlem başarılıysa True, değilse False
    """
    try:
        # Veritabanının var olup olmadığını kontrol et
        index_path = os.path.join(db_path, "index.faiss")
        documents_path = os.path.join(db_path, "documents.json")
        ids_path = os.path.join(db_path, "ids.json")
        
        if not (os.path.exists(index_path) and os.path.exists(documents_path) and os.path.exists(ids_path)):
            print("Veritabanı dosyaları bulunamadı. Yeni bir veritabanı oluşturmayı deneyin.")
            return False
        
        # Mevcut indeksi yükle
        index = faiss.read_index(index_path)
        
        # Mevcut belgeleri yükle
        with open(documents_path, 'r', encoding='utf-8') as f:
            existing_documents = json.load(f)
        
        # Mevcut ID'leri yükle
        with open(ids_path, 'r', encoding='utf-8') as f:
            existing_ids = json.load(f)
        
        # Yeni belgeleri yükle
        new_documents = load_documents(input_file)
        if not new_documents:
            print("Eklenecek yeni belge bulunamadı.")
            return False
        
        # Yeni metinleri ve ID'leri ayır
        new_texts = [doc["text"] for doc in new_documents]
        new_ids = [doc["id"] for doc in new_documents]
        
        # ID çakışmalarını kontrol et
        duplicate_ids = set(new_ids).intersection(set(existing_ids))
        if duplicate_ids:
            print(f"Uyarı: {len(duplicate_ids)} adet çakışan ID bulundu. Bu belgeler güncelleniyor.")
            
            # Çakışan ID'leri filtreleyerek yeni benzersiz belgeler oluştur
            filtered_texts = []
            filtered_documents = []
            filtered_ids = []
            
            for i, doc_id in enumerate(new_ids):
                if doc_id not in duplicate_ids:
                    filtered_texts.append(new_texts[i])
                    filtered_documents.append(new_documents[i])
                    filtered_ids.append(doc_id)
            
            new_texts = filtered_texts
            new_documents = filtered_documents
            new_ids = filtered_ids
        
        # Yeni metinleri vektörlere dönüştür
        if new_texts:
            new_embeddings = get_embeddings(new_texts, model_name)
            if len(new_embeddings) == 0:
                print("Yeni vektörler oluşturulamadı.")
                return False
            
            # Yeni vektörleri indekse ekle
            index.add(new_embeddings)
            
            # Birleştirilmiş belgeleri ve ID'leri oluştur
            all_documents = existing_documents + new_documents
            all_ids = existing_ids + new_ids
            
            # Güncellenmiş indeksi kaydet
            faiss.write_index(index, index_path)
            
            # Güncellenmiş belgeleri kaydet
            with open(documents_path, 'w', encoding='utf-8') as f:
                json.dump(all_documents, f, ensure_ascii=False, indent=2)
            
            # Güncellenmiş ID'leri kaydet
            with open(ids_path, 'w', encoding='utf-8') as f:
                json.dump(all_ids, f, ensure_ascii=False)
            
            print(f"FAISS veritabanı başarıyla güncellendi.")
            print(f"Toplam {len(new_documents)} yeni belge eklendi.")
            print(f"Veritabanında artık toplam {len(all_documents)} belge bulunmaktadır.")
        else:
            print("Eklenecek yeni belge kalmadı (hepsi zaten mevcut).")
        
        return True
    
    except Exception as e:
        print(f"Veritabanı güncellenirken hata meydana geldi: {str(e)}")
        return False

def main():
    """
    Ana program işlevi. Komut satırı argümanlarını işler ve uygun fonksiyonu çağırır.
    """
    parser = argparse.ArgumentParser(description="FAISS vektör veritabanı oluşturma veya güncelleme")
    parser.add_argument("--input_file", required=True, help="İşlenmiş metin parçalarını içeren JSON dosyasının yolu")
    parser.add_argument("--db_path", required=True, help="Veritabanı klasörünün yolu")
    parser.add_argument("--create_new", action="store_true", help="Yeni bir veritabanı oluştur (varsa üzerine yaz)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Kullanılacak vektörleştirme modeli")
    
    args = parser.parse_args()
    
    # Input dosyasının var olup olmadığını kontrol et
    if not os.path.exists(args.input_file):
        print(f"Hata: {args.input_file} dosyası bulunamadı.")
        return
    
    # Veritabanı klasörünün durumunu kontrol et
    db_exists = os.path.exists(os.path.join(args.db_path, "index.faiss"))
    
    # İşlemi seç ve başlat
    if args.create_new or not db_exists:
        print("Yeni bir FAISS veritabanı oluşturuluyor...")
        success = create_faiss_index(args.input_file, args.db_path, args.model)
    else:
        print("Mevcut FAISS veritabanı güncelleniyor...")
        success = add_to_faiss_index(args.input_file, args.db_path, args.model)
    
    if success:
        print("İşlem başarıyla tamamlandı.")
    else:
        print("İşlem başarısız oldu.")

if __name__ == "__main__":
    main()