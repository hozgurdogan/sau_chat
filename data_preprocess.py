"""
Metin Temizleme ve Chunking Aracı

Bu betik, PDF dosyalarından çıkarılan metinleri temizler ve FAISS veritabanına 
yerleştirmek üzere uygun büyüklükte parçalara (chunk) ayırır.

Ana İşlevleri:
1. Metinleri gereksiz karakterlerden ve formatlamalardan temizleme
2. Metinleri anlamlı parçalara bölme (chunking)
3. Bölünmüş parçaları FAISS veritabanı için hazırlama

Kullanım:
1. Doğrudan çalıştırma:
   python data_preprocess.py metin_klasörü [çıktı_klasörü]

2. Fonksiyon olarak:
   from data_preprocess import clean_text, chunk_text, process_text_file, process_directory
   
   # Tek bir metin dosyasını işlemek için:
   chunks = process_text_file("dosya_yolu.txt")
   
   # Bir klasördeki tüm metin dosyalarını işlemek için:
   all_chunks = process_directory("klasör_yolu")
"""

import os
import re
import json
import string
from typing import List, Dict, Tuple, Optional, Any, Union
import nltk
import argparse

# NLTK kaynaklarını indirme (ilk çalıştırmada gerekli)
try:
    nltk.download('punkt', quiet=True)
except LookupError:
    print("NLTK'nın punkt veri setini indirme hatası.")

def clean_text(text: str) -> str:
    """
    Ham metni temizler ve normalleştirir.
    
    İşlemler:
    - Gereksiz boşlukları kaldırır
    - URL'leri, e-posta adreslerini ve özel karakterleri temizler
    - Başlık formatlarını normalleştirir
    - Çoklu noktalama işaretlerini tekli hale getirir
    
    Args:
        text (str): Temizlenecek ham metin
        
    Returns:
        str: Temizlenmiş ve normalleştirilmiş metin
    """
    if not text:
        return ""
    
    # Gereksiz boşlukları temizleme
    text = re.sub(r'\s+', ' ', text)
    
    # URL'leri temizleme
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # E-posta adreslerini temizleme
    text = re.sub(r'\S+@\S+', '', text)
    
    # Başlık formatlarını normalleştirme (ör: "1. Bölüm" -> "Bölüm 1:")
    text = re.sub(r'(\d+)\.\s*([A-Za-zşŞıİçÇöÖüÜğĞ]+)', r'\2 \1:', text)
    
    # Çoklu noktalama işaretlerini tekli hale getirme
    # Bu kısımdaki regex sorununu çözelim
    for punct in string.punctuation:
        if punct in '\\^$.|?*+()[]{}':
            # Özel regex karakterleri için iki kat kaçış karakteri kullanmalıyız
            pattern = '\\\\' + punct + '{2,}'
            try:
                text = re.sub(pattern, punct, text)
            except:
                pass  # Bu karakteri atlayalım
        else:
            pattern = punct + '{2,}'
            text = re.sub(pattern, punct, text)
    
    # Satır başlarını ve sonlarını temizleme
    text = text.strip()
    
    return text

def simple_chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Metni basit bir şekilde parçalara böler (NLTK kullanmadan).
    
    Args:
        text (str): Parçalanacak metin
        chunk_size (int): Her parçanın maksimum karakter sayısı
        chunk_overlap (int): Parçalar arası örtüşme miktarı (karakter sayısı)
        
    Returns:
        List[str]: Parçalara bölünmüş metin listesi
    """
    if not text:
        return []
    
    # Metin çok kısaysa doğrudan döndür
    if len(text) <= chunk_size:
        return [text]
    
    # Metni nokta işaretlerine göre böl
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Cümleye nokta ekleyelim
        sentence = sentence + '.'
        
        # Eğer mevcut cümle chunk_size'dan büyükse, alt parçalara böl
        if len(sentence) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            
            # Uzun cümleyi kelime kelime böl
            words = sentence.split()
            temp_chunk = ""
            
            for word in words:
                if len(temp_chunk) + len(word) + 1 <= chunk_size:
                    if temp_chunk:
                        temp_chunk += " " + word
                    else:
                        temp_chunk = word
                else:
                    chunks.append(temp_chunk)
                    temp_chunk = word
            
            if temp_chunk:
                current_chunk = temp_chunk
        
        # Normal cümleleri ekle
        elif len(current_chunk) + len(sentence) + 1 <= chunk_size:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
    
    # Son chunk'ı da ekle
    if current_chunk:
        chunks.append(current_chunk)
    
    # Örtüşen parçalar oluştur
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            curr_chunk = chunks[i]
            
            # Önceki chunk'ın sonundan overlap kadar al
            if len(prev_chunk) > chunk_overlap:
                overlap_text = prev_chunk[-chunk_overlap:]
                overlapped_chunks.append(overlap_text + " " + curr_chunk)
            else:
                overlapped_chunks.append(curr_chunk)
        
        return overlapped_chunks
    
    return chunks

def convert_chunks_to_dict(chunks: List[str], source_file: str, metadata: Dict = None) -> List[Dict]:
    """
    Metin parçalarını FAISS veritabanına uygun sözlük yapısına dönüştürür.
    
    Args:
        chunks (List[str]): Metin parçaları listesi
        source_file (str): Kaynak dosya adı
        metadata (Dict, optional): Ek meta veri bilgileri
        
    Returns:
        List[Dict]: Her metin parçası için bir sözlük içeren liste
    """
    result = []
    
    if metadata is None:
        metadata = {}
    
    for i, chunk in enumerate(chunks):
        chunk_dict = {
            "id": f"{os.path.basename(source_file)}_{i}",
            "text": chunk,
            "source": source_file,
            "chunk_index": i,
            "metadata": {
                "source_file": source_file,
                "total_chunks": len(chunks),
                **metadata
            }
        }
        result.append(chunk_dict)
    
    return result

def process_text_file(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict]:
    """
    Bir metin dosyasını işler: okur, temizler, parçalar ve sözlük yapısına dönüştürür.
    
    Args:
        file_path (str): İşlenecek metin dosyasının yolu
        chunk_size (int): Her parçanın maksimum karakter sayısı
        chunk_overlap (int): Parçalar arası örtüşme miktarı
        
    Returns:
        List[Dict]: FAISS veritabanı için hazır sözlükler listesi
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()
        
        # Metni temizle
        try:
            cleaned_text = clean_text(text)
        except Exception as e:
            print(f"Metin temizleme hatası ({file_path}): {str(e)}")
            # Regex hatalarında basic bir temizleme yapalım
            cleaned_text = ' '.join(text.split())
        
        # Parçalara böl - NLTK'sız metot kullan
        try:
            text_chunks = simple_chunk_text(cleaned_text, chunk_size, chunk_overlap)
            
            # Chunk'ların minimum boyutunu kontrol et
            valid_chunks = []
            for chunk in text_chunks:
                if len(chunk) > 50:  # En az 50 karakter olsun
                    valid_chunks.append(chunk)
                
            # Sözlük yapısına dönüştür
            result = convert_chunks_to_dict(valid_chunks, file_path)
            
            return result
        except Exception as e:
            print(f"Metin parçalama hatası ({file_path}): {str(e)}")
            return []
    
    except Exception as e:
        print(f"Dosya işlenirken hata oluştu ({file_path}): {str(e)}")
        return []

def process_directory(directory_path: str, output_dir: str = None, 
                     chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, List[Dict]]:
    """
    Bir klasördeki tüm metin dosyalarını işler.
    
    Args:
        directory_path (str): İşlenecek metin dosyalarının bulunduğu klasör
        output_dir (str, optional): İşlenmiş verilerin kaydedileceği klasör
        chunk_size (int): Her parçanın maksimum karakter sayısı
        chunk_overlap (int): Parçalar arası örtüşme miktarı
        
    Returns:
        Dict[str, List[Dict]]: Dosya adı -> parçalar listesi eşlemesi
    """
    if not os.path.exists(directory_path):
        print(f"Hata: {directory_path} klasörü bulunamadı.")
        return {}
    
    results = {}
    processed_files = 0
    total_chunks = 0
    
    # Klasördeki tüm .txt dosyalarını işle
    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            chunks = process_text_file(file_path, chunk_size, chunk_overlap)
            
            if chunks:
                results[filename] = chunks
                processed_files += 1
                total_chunks += len(chunks)
                print(f"{filename} işlendi: {len(chunks)} parça")
    
    # Çıktıyı kaydet (isteğe bağlı)
    if output_dir and results:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Tüm parçaları tek bir JSON dosyasına kaydet
        all_chunks = []
        for chunks_list in results.values():
            all_chunks.extend(chunks_list)
        
        if all_chunks:  # Boş liste kontrolü
            output_file = os.path.join(output_dir, "processed_chunks.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_chunks, f, ensure_ascii=False, indent=2)
            
            print(f"İşlenmiş {len(all_chunks)} parça {output_file} dosyasına kaydedildi.")
    
    print(f"Toplam {processed_files} dosya başarıyla işlendi.")
    print(f"Toplam {total_chunks} metin parçası oluşturuldu.")
    
    return results

def main():
    """
    Ana program işlevi. Komut satırı argümanlarını işler ve uygun fonksiyonu çağırır.
    """
    parser = argparse.ArgumentParser(description="Metin dosyalarını temizle ve parçalara böl")
    parser.add_argument("input_dir", help="İşlenecek metin dosyalarının bulunduğu klasör")
    parser.add_argument("--output_dir", help="İşlenmiş verilerin kaydedileceği klasör")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Parça boyutu (karakter)")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Parçalar arası örtüşme miktarı")
    
    args = parser.parse_args()
    
    # Çıktı klasörü belirtilmemişse varsayılan değer ata
    output_dir = args.output_dir if args.output_dir else "processed_texts"
    
    # İşlemi başlat
    results = process_directory(
        args.input_dir, 
        output_dir, 
        args.chunk_size, 
        args.chunk_overlap
    )

if __name__ == "__main__":
    main()