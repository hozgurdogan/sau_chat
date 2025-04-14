"""
PDF Metin Çıkarma Aracı

Bu betik, PDF dosyalarından metin içeriğini çıkarmak için kullanılır.
PyPDF2 kütüphanesini kullanarak PDF dosyalarını işler ve metin içeriğini çıkarır.

Kullanım:
1. Doğrudan çalıştırma:
   python pdf_text_extract.py dosya_yolu.pdf
   python pdf_text_extract.py klasor_yolu [çıktı_klasörü_yolu]

2. Fonksiyon olarak:
   from pdf_text_extract import extract_text_from_pdf, extract_text_from_folder
   
   # Tek bir PDF'den metin çıkarmak için:
   text = extract_text_from_pdf("dosya_yolu.pdf")
   
   # Bir klasördeki tüm PDF'lerden metin çıkarmak için:
   texts = extract_text_from_folder("klasor_yolu")
"""

import os
import sys
import PyPDF2
from typing import Dict, List, Optional, Union

def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Belirtilen PDF dosyasından tüm metni çıkarır.
    
    Args:
        pdf_path (str): PDF dosyasının yolu
        
    Returns:
        str: PDF'den çıkarılan metin veya hata durumunda None
    """
    if not os.path.exists(pdf_path):
        print(f"Hata: {pdf_path} dosyası bulunamadı.")
        return None
        
    if not pdf_path.lower().endswith('.pdf'):
        print(f"Hata: {pdf_path} bir PDF dosyası değil.")
        return None
    
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n\n"
                
        return text.strip()
    except Exception as e:
        print(f"PDF işlenirken hata oluştu {pdf_path}: {str(e)}")
        return None

def extract_text_from_folder(folder_path: str) -> Dict[str, str]:
    """
    Belirtilen klasördeki tüm PDF dosyalarından metin çıkarır.
    
    Args:
        folder_path (str): PDF dosyalarının bulunduğu klasör yolu
        
    Returns:
        Dict[str, str]: Dosya adı -> metin içeriği şeklinde sözlük
    """
    if not os.path.exists(folder_path):
        print(f"Hata: {folder_path} klasörü bulunamadı.")
        return {}
    
    results = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(file_path)
            if text:
                results[filename] = text
    
    return results

def save_text_to_file(text: str, output_path: str) -> bool:
    """
    Çıkarılan metni bir dosyaya kaydeder.
    
    Args:
        text (str): Kaydedilecek metin
        output_path (str): Çıktı dosyasının yolu
        
    Returns:
        bool: İşlem başarılı ise True, aksi halde False
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(text)
        return True
    except Exception as e:
        print(f"Metin dosyaya kaydedilirken hata oluştu: {str(e)}")
        return False

def main():
    """Ana program fonksiyonu"""
    if len(sys.argv) < 2:
        print("Kullanım: python pdf_text_extract.py <pdf_dosyası_veya_klasör_yolu> [çıktı_dosyası_yolu]")
        return
    
    input_path = sys.argv[1]
    
    # PDF dosyasının mı yoksa klasörün mü işleneceğini belirle
    if os.path.isfile(input_path):
        text = extract_text_from_pdf(input_path)
        if text:
            if len(sys.argv) > 2:
                output_path = sys.argv[2]
            else:
                base_name = os.path.splitext(os.path.basename(input_path))[0]
                output_path = f"{base_name}_metin.txt"
                
            if save_text_to_file(text, output_path):
                print(f"Metin başarıyla çıkarıldı ve {output_path} dosyasına kaydedildi.")
            
    elif os.path.isdir(input_path):
        results = extract_text_from_folder(input_path)
        
        if results:
            output_dir = sys.argv[2] if len(sys.argv) > 2 else "pdf_metinler"
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            for filename, text in results.items():
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_dir, f"{base_name}_metin.txt")
                
                if save_text_to_file(text, output_path):
                    print(f"Metin başarıyla çıkarıldı ve {output_path} dosyasına kaydedildi.")
            
            print(f"Toplam {len(results)} PDF dosyası işlendi.")
        else:
            print("İşlenecek PDF dosyası bulunamadı.")
    else:
        print(f"Hata: {input_path} geçerli bir dosya veya klasör değil.")

if __name__ == "__main__":
    main()