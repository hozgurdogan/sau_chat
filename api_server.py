"""
SAÜChat API Sunucusu

Bu FastAPI uygulaması, SAÜChat sistemi için bir API sunucusu sağlar.
Kullanıcı sorgularını alır, vektör veritabanından ilgili bağlamı çeker,
LLaMA modelini kullanarak yanıt üretir ve sonucu döndürür.
Ayrıca PDF dosyalarını yükleyip vektör veritabanına eklemek için bir endpoint içerir.
"""
import os
import sys
import json
import argparse
import uvicorn
import tempfile
import shutil
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any
from contextlib import asynccontextmanager

# LLaMA modelini yüklemek için llama-cpp-python kütüphanesini import et
try:
    from llama_cpp import Llama, LlamaGrammar
except ImportError:
    print("llama-cpp-python kütüphanesi bulunamadı.")
    print("Lütfen 'pip install llama-cpp-python' komutu ile kurun.")
    sys.exit(1)

# Kendi modüllerimizi import et
try:
    # check_vector_db_exists kaldırıldı, çünkü vector_db_helpers.py içinde tanımlı değil ve kullanılmıyor gibi görünüyor.
    from vector_db_helpers import retrieve_relevant_context, load_vector_db
    from pdf_text_extract import extract_text_from_pdf
    from data_preprocess import convert_chunks_to_dict, clean_text, simple_chunk_text
    from faiss_index_process import add_to_faiss_index, DEFAULT_MODEL as FAISS_MODEL
except ImportError as e:
    print(f"Gerekli modüller yüklenirken hata oluştu: {e}")
    print("Lütfen vector_db_helpers, pdf_text_extract, data_preprocess ve faiss_index_process modüllerinin doğru konumda olduğundan emin olun.")
    sys.exit(1)

# --- Global Değişkenler ve Ayarlar ---
# Model ve veritabanı yolları için argümanları işle
parser = argparse.ArgumentParser(description="SAÜChat API Sunucusu")
parser.add_argument("--model_path", type=str, default="D:\\models\\gguf\\llama-3-GGUF\\llama3-8B-trendyol-rag-merged-Q8_0.gguf",
                    help="GGUF formatındaki LLaMA model dosyasının yolu")
parser.add_argument("--db_path", type=str, default="vector_db",
                    help="FAISS vektör veritabanının bulunduğu klasör")
parser.add_argument("--n_gpu_layers", type=int, default=-1, # Varsayılan olarak tüm katmanları GPU'ya yükle
                    help="GPU'ya yüklenecek katman sayısı (-1: tümü, 0: hiçbiri)")
parser.add_argument("--n_ctx", type=int, default=4096, # Modelin bağlam penceresi
                    help="Modelin maksimum bağlam penceresi boyutu")
args = parser.parse_args()

# Global model ve veritabanı nesneleri
llm: Optional[Llama] = None
faiss_index: Optional[Any] = None
documents: Optional[List[Dict[str, Any]]] = None
ids: Optional[List[str]] = None
is_llm_loaded: bool = False
is_db_loaded: bool = False

# --- FastAPI Uygulama Ömrü Yönetimi ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama başlangıcında modeli ve veritabanını yükler."""
    global llm, faiss_index, documents, ids, is_llm_loaded, is_db_loaded

    # LLaMA modelini yükle
    print(f"LLaMA modeli yükleniyor: {args.model_path}")
    try:
        llm = Llama(
            model_path=args.model_path,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            verbose=True # Model yükleme detaylarını göster
        )
        is_llm_loaded = True
        print("LLaMA modeli başarıyla yüklendi.")
    except Exception as e:
        print(f"LLaMA modeli yüklenirken hata oluştu: {e}")
        llm = None # Hata durumunda None olarak bırak
        is_llm_loaded = False

    # Vektör veritabanını yükle
    print(f"Vektör veritabanı yükleniyor: {args.db_path}")
    try:
        faiss_index, documents, ids = load_vector_db(args.db_path)
        if faiss_index is not None and documents is not None and ids is not None:
            is_db_loaded = True
            print("Vektör veritabanı başarıyla yüklendi.")
        else:
             print("Vektör veritabanı yüklenemedi (dosyalar eksik veya hatalı olabilir).")
             is_db_loaded = False
    except Exception as e:
        print(f"Vektör veritabanı yüklenirken hata oluştu: {e}")
        faiss_index, documents, ids = None, None, None
        is_db_loaded = False

    yield # Uygulama çalışırken burada bekler

    # Uygulama kapanırken kaynakları serbest bırak (isteğe bağlı)
    print("API sunucusu kapatılıyor.")
    llm = None
    faiss_index = documents = ids = None
    is_llm_loaded = is_db_loaded = False


# FastAPI uygulamasını oluştur
app = FastAPI(
    title="SAÜChat API",
    description="Sakarya Üniversitesi Yönetmelik ve Yönergeleri için RAG tabanlı API",
    version="1.1.0",
    lifespan=lifespan # Uygulama ömrü yöneticisini ekle
)

# --- Bağımlılıklar ---
def get_llm():
    """LLaMA modelini döndürür, yüklenmemişse hata verir."""
    if not is_llm_loaded or llm is None:
        raise HTTPException(status_code=503, detail="LLM modeli şu anda kullanılamıyor.")
    return llm

def check_vector_db():
    """Vektör veritabanının yüklenip yüklenmediğini kontrol eder ve yolu döndürür."""
    # Bu fonksiyon sadece yolu döndürmek için kullanılır.
    # Yüklenme kontrolü ilgili endpoint'lerde yapılır.
    return args.db_path

# --- Pydantic Modelleri ---
class ChatQuery(BaseModel):
    query: str = Field(..., description="Kullanıcının sorduğu soru.")
    top_k: int = Field(3, ge=1, le=10, description="Vektör veritabanından alınacak en ilgili belge sayısı.")
    temperature: float = Field(0.1, ge=0.0, le=1.0, description="Modelin yanıt üretirken kullanacağı sıcaklık değeri (yaratıcılık).")
    max_tokens: int = Field(512, ge=50, le=args.n_ctx, description="Modelin üreteceği maksimum token sayısı.")

class ChatResponse(BaseModel):
    model_answer: str = Field(..., description="Modelin ürettiği yanıt.")
    retrieved_context: str = Field(..., description="Yanıtı oluşturmak için kullanılan ilgili metin parçaları.")
    sources: List[str] = Field(..., description="Kullanılan metin parçalarının kaynak dosya adları.")

class HealthStatus(BaseModel):
    status: str
    llm_loaded: bool
    db_loaded: bool
    model_path: str
    db_path: str

class UploadResponse(BaseModel):
    message: str
    processed_files: int
    added_chunks: int
    errors: List[str]


# --- API Endpointleri ---

@app.get("/health", response_model=HealthStatus, summary="API Sağlık Durumu")
async def health_check():
    """API sunucusunun, LLM modelinin ve vektör veritabanının durumunu kontrol eder."""
    return HealthStatus(
        status="ok" if is_llm_loaded and is_db_loaded else "partial_error",
        llm_loaded=is_llm_loaded,
        db_loaded=is_db_loaded,
        model_path=args.model_path,
        db_path=args.db_path
    )

@app.post("/chat", response_model=ChatResponse, summary="Sohbet Sorgusu")
async def chat_endpoint(
    query_data: ChatQuery,
    current_llm: Llama = Depends(get_llm),
    db_path: str = Depends(check_vector_db) # DB yolunu al
):
    """
    Kullanıcı sorgusunu alır, ilgili bağlamı bulur ve LLM ile yanıt üretir.
    """
    # Veritabanının yüklü olup olmadığını burada kontrol et
    if not is_db_loaded or faiss_index is None or documents is None or ids is None:
         raise HTTPException(status_code=503, detail="Vektör veritabanı şu anda kullanılamıyor.")

    try:
        # 1. Vektör veritabanından ilgili bağlamı al
        print(f"Sorgu için ilgili bağlam aranıyor: '{query_data.query}' (top_k={query_data.top_k})")
        context, sources = retrieve_relevant_context(
            query=query_data.query,
            index=faiss_index,
            documents=documents,
            ids=ids,
            top_k=query_data.top_k,
            include_sources=True # Kaynakları da al
        )

        if not context:
            print("İlgili bağlam bulunamadı.")
            context = "İlgili bilgi bulunamadı."
            sources = []

        print(f"Bulunan kaynaklar: {sources}")
        print(f"Oluşturulan bağlam:\n{context[:500]}...") # Bağlamın başını yazdır

        # 2. LLM için prompt oluştur
        prompt = f"""Aşağıdaki Sakarya Üniversitesi yönetmeliklerinden alınan bilgilere dayanarak kullanıcı sorusunu yanıtla. Sadece verilen bilgilerden yararlan, dış bilgi kullanma. Cevabın net ve öz olsun.

Bilgi Kaynakları:
---
{context}
---

Kullanıcı Sorusu: {query_data.query}

Cevap:"""

        print("LLM için prompt oluşturuldu.")

        # 3. LLM ile yanıt üret
        print("LLM ile yanıt üretiliyor...")
        output = current_llm(
            prompt,
            max_tokens=query_data.max_tokens,
            temperature=query_data.temperature,
            stop=["Kullanıcı Sorusu:", "\n\n", "---"], # Yanıtın gereksiz yere uzamasını engelle
            echo=False # Prompt'u yanıtta tekrarlama
        )

        model_answer = output['choices'][0]['text'].strip()
        print(f"Model yanıtı:\n{model_answer}")

        return ChatResponse(
            model_answer=model_answer,
            retrieved_context=context,
            sources=sources
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Sohbet işlenirken hata oluştu: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Sohbet işlenirken beklenmeyen bir hata oluştu: {str(e)}")


# PDF yükleme ve işleme endpoint'i
@app.post("/upload-pdf", response_model=UploadResponse, summary="PDF Dosyalarını Yükle ve İndeksle")
async def upload_pdf(
    files: List[UploadFile] = File(..., description="İndekslenecek PDF dosyaları listesi"),
    db_path: str = Depends(check_vector_db) # Sadece yolu alır
):
    """
    Yüklenen PDF dosyalarını işler ve vektör veritabanına ekler.

    - **files**: Yüklenecek PDF dosyalarının listesi.
    """
    global faiss_index, documents, ids, is_db_loaded # Veritabanını yeniden yüklemek için global erişim

    processed_files_count = 0
    total_added_chunks = 0
    all_chunk_dicts = [] # Tüm dosyalardan gelen chunk'ları biriktir
    errors = []

    # Geçici bir dizin oluştur (tüm dosyalar için tek bir tane yeterli)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_chunks_path = os.path.join(temp_dir, "combined_processed_chunks.json")

        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                errors.append(f"'{file.filename}': Geçersiz dosya türü. Sadece PDF dosyaları kabul edilir.")
                continue # Sonraki dosyaya geç

            temp_pdf_path = os.path.join(temp_dir, file.filename)

            try:
                # 1. PDF dosyasını geçici olarak kaydet
                with open(temp_pdf_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                print(f"PDF geçici olarak kaydedildi: {temp_pdf_path}")

                # 2. PDF'den metin çıkar
                extracted_text = extract_text_from_pdf(temp_pdf_path)
                if not extracted_text:
                    errors.append(f"'{file.filename}': PDF dosyasından metin çıkarılamadı.")
                    continue # Sonraki dosyaya geç
                print(f"'{file.filename}' için metin çıkarıldı.")

                # 3. Metni temizle ve parçalara ayır
                cleaned_text = clean_text(extracted_text)
                chunks = simple_chunk_text(cleaned_text) # Varsayılan chunk_size ve overlap ile
                if not chunks:
                    errors.append(f"'{file.filename}': Metin parçalara ayrılamadı.")
                    continue # Sonraki dosyaya geç

                chunk_dicts = convert_chunks_to_dict(chunks, file.filename) # Kaynak olarak orijinal dosya adını kullan
                print(f"'{file.filename}' için {len(chunk_dicts)} adet metin parçası oluşturuldu.")

                # 4. Bu dosyanın parçalarını genel listeye ekle
                all_chunk_dicts.extend(chunk_dicts)
                processed_files_count += 1

            except Exception as e:
                error_msg = f"'{file.filename}' işlenirken hata oluştu: {str(e)}"
                print(error_msg)
                errors.append(error_msg)
            finally:
                 # Geçici PDF dosyasını sil
                 if os.path.exists(temp_pdf_path):
                     try:
                         os.remove(temp_pdf_path)
                     except OSError as rm_err:
                         print(f"Geçici PDF silinemedi: {rm_err}")


        # Eğer işlenecek geçerli chunk yoksa ve hata varsa, hata döndür
        if not all_chunk_dicts:
             if errors:
                 error_details = "; ".join(errors)
                 raise HTTPException(status_code=400, detail=f"Hiçbir dosya başarıyla işlenemedi. Hatalar: {error_details}")
             else:
                 raise HTTPException(status_code=400, detail="Yüklenen dosyalarda işlenecek içerik bulunamadı veya dosyalar boş.")

        try:
            # 5. Tüm parçaları tek bir geçici JSON dosyasına kaydet
            with open(temp_chunks_path, 'w', encoding='utf-8') as f:
                json.dump(all_chunk_dicts, f, ensure_ascii=False, indent=2)
            print(f"Toplam {len(all_chunk_dicts)} parça geçici JSON'a kaydedildi: {temp_chunks_path}")

            # 6. FAISS veritabanına ekle (tek seferde)
            print("FAISS veritabanına ekleme işlemi başlatılıyor...")
            success = add_to_faiss_index(temp_chunks_path, db_path, model_name=FAISS_MODEL)

            if not success:
                raise HTTPException(status_code=500, detail="Veriler FAISS veritabanına eklenirken genel bir hata oluştu. Detaylar için sunucu loglarına bakın.")

            total_added_chunks = len(all_chunk_dicts) # Başarılı ekleme sonrası sayacı güncelle
            print("Veriler FAISS veritabanına başarıyla eklendi.")

            # 7. Veritabanını belleğe yeniden yükle (önemli!)
            print("Vektör veritabanı belleğe yeniden yükleniyor...")
            try:
                # Önce mevcut global değişkenleri temizle (bellek sızıntısını önlemek için)
                faiss_index = None
                documents = None
                ids = None
                # Yeniden yükle
                faiss_index, documents, ids = load_vector_db(db_path)
                if faiss_index is not None and documents is not None and ids is not None:
                    is_db_loaded = True
                    print("Vektör veritabanı başarıyla yeniden yüklendi.")
                else:
                    is_db_loaded = False # Yükleme başarısız oldu
                    errors.append("Veritabanı eklendikten sonra belleğe yeniden yüklenemedi.")
                    print("HATA: Veritabanı eklendikten sonra belleğe yeniden yüklenemedi.")
            except Exception as reload_e:
                is_db_loaded = False
                errors.append(f"Veritabanı yeniden yüklenirken hata oluştu: {str(reload_e)}")
                print(f"HATA: Veritabanı yeniden yüklenirken hata oluştu: {reload_e}")


            # Başarı mesajını oluştur
            success_message = f"{processed_files_count} PDF dosyası başarıyla işlendi ve toplam {total_added_chunks} parça veritabanına eklendi."
            if errors:
                 final_message = f"{success_message} Ancak bazı sorunlar oluştu."
            else:
                 final_message = success_message


            return UploadResponse(
                message=final_message,
                processed_files=processed_files_count,
                added_chunks=total_added_chunks,
                errors=errors # Hataları döndür
            )

        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            print(f"Veritabanına ekleme veya yeniden yükleme sırasında hata: {str(e)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Veritabanına ekleme veya yeniden yükleme sırasında bir hata oluştu: {str(e)}")
        # finally bloğu 'with tempfile.TemporaryDirectory()' tarafından otomatik olarak halledilir.


# --- Uygulamayı Başlat ---
if __name__ == "__main__":
    # API sunucusunun portunu 8000 olarak ayarla
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0", # Tüm ağ arayüzlerinden erişilebilir yap
        port=8000,
        reload=True, # Kod değişikliklerinde otomatik yeniden başlatma (geliştirme için)
        log_level="info" # Log seviyesini ayarla
    )