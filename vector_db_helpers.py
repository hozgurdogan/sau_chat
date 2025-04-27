# "Ekle-sil işlemi nasıl yapabilirim?"


"""
Vektör Veritabanı Yardımcı Fonksiyonları

Bu modül, FAISS vektör veritabanından benzer metinleri sorgulama ve getirme işlemlerini gerçekleştirir.
Kullanıcı sorusunu vektöre dönüştürür ve veritabanındaki en benzer metinleri bulur.

Ana İşlevleri:
1. Veritabanını yükleme
2. Kullanıcı sorgularını vektöre dönüştürme
3. En benzer metinleri bulma ve getirme

Kullanım:
1. Doğrudan çalıştırma:
   python vector_db_helpers.py --query "Sorgunuz buraya" --db_path vector_db

2. Fonksiyon olarak:
   from vector_db_helpers import load_vector_db, query_vector_db
   
   # Veritabanını yükle
   index, documents, ids = load_vector_db("vector_db")
   
   # Sorgu yap
   results = query_vector_db("Sorgunuz buraya", index, documents, ids)
"""

import os
import json
import argparse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple

# Varsayılan vektör modeli (Türkçe dili için uygun bir model)
DEFAULT_MODEL = "dbmdz/bert-base-turkish-cased"

def load_vector_db(db_path: str) -> Tuple[Any, List[Dict[str, Any]], List[str]]:
    """
    FAISS vektör veritabanını yükler.
    
    Args:
        db_path (str): Veritabanı klasör yolu
        
    Returns:
        Tuple: (FAISS indeksi, belge listesi, ID listesi)
    """
    try:
        # Dosya yollarını belirle
        index_path = os.path.join(db_path, "index.faiss")
        documents_path = os.path.join(db_path, "documents.json")
        ids_path = os.path.join(db_path, "ids.json")
        
        # Dosyaların var olup olmadığını kontrol et
        if not (os.path.exists(index_path) and os.path.exists(documents_path) and os.path.exists(ids_path)):
            print(f"Hata: Veritabanı dosyaları bulunamadı: {db_path}")
            return None, [], []
        
        # FAISS indeksini yükle
        index = faiss.read_index(index_path)
        
        # Belgeleri yükle
        with open(documents_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        # ID'leri yükle
        with open(ids_path, 'r', encoding='utf-8') as f:
            ids = json.load(f)
        
        print(f"Veritabanı başarıyla yüklendi: {len(documents)} belge, {index.ntotal} vektör.")
        return index, documents, ids
    
    except Exception as e:
        print(f"Veritabanı yüklenirken hata oluştu: {str(e)}")
        return None, [], []

def query_vector_db(query: str, 
                   index: Any, 
                   documents: List[Dict[str, Any]], 
                   ids: List[str], 
                   top_k: int = 5, 
                   model_name: str = DEFAULT_MODEL) -> List[Dict[str, Any]]:
    """
    Vektör veritabanında sorgu yapar ve en benzer belgeleri getirir.
    
    Args:
        query (str): Kullanıcı sorusu/sorgusu
        index (Any): FAISS indeksi
        documents (List[Dict]): Belge listesi
        ids (List[str]): ID listesi
        top_k (int): Kaç sonuç getirileceği
        model_name (str): Vektörleştirme için kullanılacak model adı
        
    Returns:
        List[Dict[str, Any]]: En benzer belgelerin listesi
    """
    try:
        # Sorguyu vektöre dönüştür
        model = SentenceTransformer(model_name)
        query_vector = model.encode([query])
        
        # En benzer belgeleri bul
        distances, indices = index.search(query_vector.astype('float32'), top_k)
        
        # Sonuçları hazırla
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(documents) and idx >= 0:
                # Bulunan belge bilgilerini al
                document = documents[idx]
                score = float(1 - distances[0][i] / 100)  # Benzerlik skorunu 0-1 aralığına normalize et
                
                # Sonuç sözlüğünü oluştur
                result = {
                    "text": document["text"],
                    "source": document.get("source", ""),
                    "metadata": document.get("metadata", {}),
                    "score": score
                }
                results.append(result)
        
        return results
    
    except Exception as e:
        print(f"Sorgu sırasında hata oluştu: {str(e)}")
        return []

# def retrieve_relevant_context(query: str, db_path: str, top_k: int = 5) -> str:
#     """
#     Verilen sorguya en benzer içerikleri getirir ve tek bir metin olarak birleştirir.
    
#     Args:
#         query (str): Kullanıcı sorusu/sorgusu
#         db_path (str): Veritabanı klasör yolu
#         top_k (int): Kaç sonuç getirileceği
        
#     Returns:
#         str: Birleştirilmiş ve ilgili içeriklerden oluşan metin
#     """
#     try:
#         # Veritabanını yükle
#         index, documents, ids = load_vector_db(db_path)
#         if not index:
#             return "Veritabanı yüklenemedi."
        
#         # Sorgu yap
#         results = query_vector_db(query, index, documents, ids, top_k=top_k)
        
#         if not results:
#             return "Sorguya uygun sonuç bulunamadı."
        
#         # Sonuçları birleştir ve kaynaklarıyla birlikte formatla
#         combined_text = ""
#         for i, result in enumerate(results):
#             source_info = f"Kaynak: {result['source']}" if result.get('source') else ""
#             combined_text += f"\n\n--- BÖLÜM {i+1} {source_info} ---\n\n"
#             combined_text += result["text"]
        
#         return combined_text
    
#     except Exception as e:
#         return f"Hata oluştu: {str(e)}"

# def retrieve_relevant_context(query: str, db_path: str, top_k: int = 5, return_sources: bool = False) -> str:
#     """
#     Verilen sorguya en benzer içerikleri getirir ve tek bir metin olarak birleştirir.
    
#     Args:
#         query (str): Kullanıcı sorusu/sorgusu
#         db_path (str): Veritabanı klasör yolu
#         top_k (int): Kaç sonuç getirileceği
#         return_sources (bool): Eğer True ise, kaynak bilgilerini de içeren bir sözlük döndürür
        
#     Returns:
#         Union[str, Dict[str, Any]]: Birleştirilmiş metin veya metin ve kaynakları içeren sözlük
#     """
#     try:
#         # Veritabanını yükle
#         index, documents, ids = load_vector_db(db_path)
#         if not index:
#             return "Veritabanı yüklenemedi." if not return_sources else {"text": "Veritabanı yüklenemedi.", "sources": []}
        
#         # Sorgu yap
#         results = query_vector_db(query, index, documents, ids, top_k=top_k)
        
#         if not results:
#             return "Sorguya uygun sonuç bulunamadı." if not return_sources else {"text": "Sorguya uygun sonuç bulunamadı.", "sources": []}
        
#         # Sonuçları birleştir ve kaynaklarıyla birlikte formatla
#         combined_text = ""
#         sources = []
        
#         for i, result in enumerate(results):
#             source_info = f"Kaynak: {result['source']}" if result.get('source') else ""
#             if source_info and result.get('source') not in sources:
#                 sources.append(result.get('source'))
                
#             combined_text += f"\n\n--- BÖLÜM {i+1} {source_info} ---\n\n"
#             combined_text += result["text"]
        
#         if return_sources:
#             return {
#                 "text": combined_text,
#                 "sources": sources
#             }
        
#         return combined_text
    
#     except Exception as e:
#         error_msg = f"Hata oluştu: {str(e)}"
#         if return_sources:
#             return {"text": error_msg, "sources": []}
#         return error_msg


def retrieve_relevant_context(query: str, db_path: str, top_k: int = 5, return_sources: bool = False) -> str:
    """
    Verilen sorguya en benzer içerikleri getirir ve tek bir metin olarak birleştirir.
    
    Args:
        query (str): Kullanıcı sorusu/sorgusu
        db_path (str): Veritabanı klasör yolu
        top_k (int): Kaç sonuç getirileceği
        return_sources (bool): Eğer True ise, kaynak bilgilerini de içeren bir sözlük döndürür
        
    Returns:
        Union[str, Dict[str, Any]]: Birleştirilmiş metin veya metin ve kaynakları içeren sözlük
    """
    try:
        # Veritabanını yükle
        index, documents, ids = load_vector_db(db_path)
        
        if index is None or documents is None or ids is None:
            error_msg = "Veritabanı yüklenemedi. Lütfen veritabanı dosyalarını kontrol edin."
            return {"text": error_msg, "sources": []} if return_sources else error_msg
        
        # Sorgu yap
        results = query_vector_db(query, index, documents, ids, top_k=top_k)
        
        if not results:
            no_results_msg = "Sorguya uygun sonuç bulunamadı."
            return {"text": no_results_msg, "sources": []} if return_sources else no_results_msg
        
        # Sonuçları birleştir ve kaynaklarıyla birlikte formatla
        combined_text = ""
        sources = set()  # Kaynakları tekrarsız bir şekilde tutmak için set kullanıyoruz
        
        for i, result in enumerate(results):
            source_info = f"Kaynak: {result.get('source', 'Bilinmiyor')}"
            
            # Kaynağı yalnızca bir kez ekle
            sources.add(result.get('source', 'Bilinmiyor'))
            
            # Metni ve kaynağı birleştir
            combined_text += f"\n\n--- BÖLÜM {i+1} {source_info} ---\n\n"
            combined_text += result.get("text", "Metin bulunamadı.")
        
        if return_sources:
            # Kaynakları listeye dönüştür ve döndür
            return {
                "text": combined_text,
                "sources": list(sources)
            }
        
        return combined_text
    
    except Exception as e:
        # Hata durumunda uygun mesaj döndür
        error_msg = f"Hata oluştu: {str(e)}"
        return {"text": error_msg, "sources": []} if return_sources else error_msg


def main():
    """
    Ana program işlevi. Komut satırı argümanlarını işler ve uygun fonksiyonu çağırır.
    """
    parser = argparse.ArgumentParser(description="Vektör veritabanında sorgu yapma")
    parser.add_argument("--query", required=True, help="Sorgu metni")
    parser.add_argument("--db_path", default="vector_db", help="Veritabanı klasörünün yolu")
    parser.add_argument("--top_k", type=int, default=5, help="Kaç sonuç getirileceği")
    
    args = parser.parse_args()
    
    # Sorgu yap ve sonuçları getir
    context = retrieve_relevant_context(args.query, args.db_path, args.top_k)
    
    print("\n=== SORGU SONUÇLARI ===\n")
    print(context)

if __name__ == "__main__":
    main()