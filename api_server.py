"""
SAÜChat API Sunucusu (Refactored)

Bu FastAPI uygulaması, SAÜChat sistemi için bir API sunucusu sağlar.
Kullanıcı sorgularını alır, vektör veritabanından ilgili bağlamı çeker,
LLaMA modelini kullanarak yanıt üretir ve sonucu döndürür.
Ayrıca PDF dosyalarını yükleyip vektör veritabanına eklemek için bir endpoint içerir.
"""
import os
import sys
import json
# import argparse # argparse kaldırıldı
import uvicorn
import tempfile
import shutil
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
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
    from vector_db_helpers import retrieve_relevant_context, load_vector_db
    from pdf_text_extract import extract_text_from_pdf
    from data_preprocess import convert_chunks_to_dict, clean_text, simple_chunk_text
    from faiss_index_process import add_to_faiss_index, DEFAULT_MODEL as FAISS_MODEL
except ImportError as e:
    print(f"Gerekli modüller yüklenirken hata oluştu: {e}")
    print("Lütfen vector_db_helpers, pdf_text_extract, data_preprocess ve faiss_index_process modüllerinin doğru konumda olduğundan emin olun.")
    sys.exit(1)

# --- Konfigürasyon Ayarları (Ortam Değişkenleri veya Varsayılanlar) ---
MODEL_PATH = os.environ.get("MODEL_PATH", "D:\\models\\gguf\\llama-3-GGUF\\llama3-8B-trendyol-rag-merged-Q8_0.gguf")
DB_PATH = os.environ.get("DB_PATH", "vector_db")
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", -1)) # Ortam değişkenleri string döner, int'e çevir
N_CTX = int(os.environ.get("N_CTX", 4096)) # Ortam değişkenleri string döner, int'e çevir

# --- Global Değişkenler ---
llm: Optional[Llama] = None
faiss_index: Optional[Any] = None
documents: Optional[List[Dict[str, Any]]] = None
ids: Optional[List[str]] = None
is_llm_loaded: bool = False
is_db_loaded: bool = False

# --- FastAPI Uygulama Ömrü Yönetimi (Lifespan) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama başlangıcında modeli ve veritabanını yükler."""
    global llm, faiss_index, documents, ids, is_llm_loaded, is_db_loaded

    # LLaMA modelini yükle
    print(f"LLaMA modeli yükleniyor: {MODEL_PATH}") # args.model_path -> MODEL_PATH
    if not os.path.exists(MODEL_PATH): # args.model_path -> MODEL_PATH
        print(f"UYARI: Model dosyası bulunamadı: {MODEL_PATH}") # args.model_path -> MODEL_PATH
        llm = None
        is_llm_loaded = False
    else:
        try:
            llm = Llama(
                model_path=MODEL_PATH, # args.model_path -> MODEL_PATH
                n_ctx=N_CTX,           # args.n_ctx -> N_CTX
                n_gpu_layers=N_GPU_LAYERS, # args.n_gpu_layers -> N_GPU_LAYERS
                verbose=True
            )
            is_llm_loaded = True
            print("LLaMA modeli başarıyla yüklendi.")
        except Exception as e:
            print(f"LLaMA modeli yüklenirken hata oluştu: {e}")
            llm = None
            is_llm_loaded = False

    # Vektör veritabanını yükle
    print(f"Vektör veritabanı yükleniyor: {DB_PATH}") # args.db_path -> DB_PATH
    if not os.path.exists(DB_PATH): # args.db_path -> DB_PATH
         print(f"UYARI: Vektör veritabanı klasörü bulunamadı: {DB_PATH}") # args.db_path -> DB_PATH
         faiss_index, documents, ids = None, None, None
         is_db_loaded = False
    else:
        try:
            faiss_index, documents, ids = load_vector_db(DB_PATH) # args.db_path -> DB_PATH
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

    yield

    print("API sunucusu kapatılıyor.")
    llm = None
    faiss_index = documents = ids = None
    is_llm_loaded = is_db_loaded = False


# FastAPI uygulamasını oluştur
app = FastAPI(
    title="SAÜChat API",
    description="Sakarya Üniversitesi Yönetmelik ve Yönergeleri için RAG tabanlı API",
    version="1.1.1", # Sürüm güncellendi
    lifespan=lifespan
)

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Bağımlılıklar ---
def get_llm():
    """LLaMA modelini döndürür, yüklenmemişse hata verir."""
    if not is_llm_loaded or llm is None:
        raise HTTPException(status_code=503, detail="LLM modeli şu anda kullanılamıyor veya yüklenemedi.")
    return llm

def get_vector_db():
    """Vektör veritabanı bileşenlerini döndürür, yüklenmemişse hata verir."""
    if not is_db_loaded or faiss_index is None or documents is None or ids is None:
         raise HTTPException(status_code=503, detail="Vektör veritabanı şu anda kullanılamıyor veya yüklenemedi.")
    return faiss_index, documents, ids

def get_db_path():
    """Vektör veritabanı yolunu döndürür."""
    return DB_PATH # args.db_path -> DB_PATH

# --- Pydantic Modelleri ---
class ChatQuery(BaseModel):
    query: str = Field(..., min_length=1, description="Kullanıcının sorduğu soru.")
    top_k: int = Field(3, ge=1, le=10, description="Vektör veritabanından alınacak en ilgili belge sayısı.")
    temperature: float = Field(0.1, ge=0.0, le=1.0, description="Modelin yanıt üretirken kullanacağı sıcaklık değeri (yaratıcılık).")
    # max_tokens için N_CTX kullanıldı
    max_tokens: int = Field(512, ge=50, le=N_CTX, description="Modelin üreteceği maksimum token sayısı.")

class ChatResponse(BaseModel):
    model_answer: str = Field(..., description="Modelin ürettiği yanıt.")
    retrieved_context: str = Field(..., description="Yanıtı oluşturmak için kullanılan ilgili metin parçaları.")
    sources: List[str] = Field(..., description="Kullanılan metin parçalarının kaynak dosya adları.")

class HealthStatus(BaseModel):
    status: str
    llm_loaded: bool
    db_loaded: bool
    model_path: Optional[str] = None
    db_path: Optional[str] = None

class UploadResponse(BaseModel):
    message: str
    processed_files: int
    added_chunks: int
    errors: List[str]


# --- API Endpointleri ---

@app.get("/health", response_model=HealthStatus, summary="API Sağlık Durumu")
async def health_check():
    """API sunucusunun, LLM modelinin ve vektör veritabanının durumunu kontrol eder."""
    status_code = "ok"
    if not is_llm_loaded or not is_db_loaded:
        status_code = "partial_error"
    if not is_llm_loaded and not is_db_loaded:
        status_code = "error"

    # MODEL_PATH ve DB_PATH kullanıldı
    model_status_path = MODEL_PATH if os.path.exists(MODEL_PATH) else f"Bulunamadı: {MODEL_PATH}"
    db_status_path = DB_PATH if os.path.exists(DB_PATH) else f"Bulunamadı: {DB_PATH}"

    return HealthStatus(
        status=status_code,
        llm_loaded=is_llm_loaded,
        db_loaded=is_db_loaded,
        model_path=model_status_path,
        db_path=db_status_path
    )

# ... ( /chat ve /upload-pdf endpointleri aynı kalabilir, çünkü artık global konfigürasyon değişkenlerini kullanıyorlar) ...
@app.post("/chat", response_model=ChatResponse, summary="Sohbet Sorgusu")
async def chat_endpoint(
    query_data: ChatQuery,
    current_llm: Llama = Depends(get_llm),
    db_components: tuple = Depends(get_vector_db) # DB bileşenlerini al
):
    """
    Kullanıcı sorgusunu alır, ilgili bağlamı bulur ve LLM ile yanıt üretir.
    """
    current_index, current_documents, current_ids = db_components

    try:
        # 1. Vektör veritabanından ilgili bağlamı al
        print(f"Sorgu için ilgili bağlam aranıyor: '{query_data.query}' (top_k={query_data.top_k})")
        context, sources = retrieve_relevant_context(
            query=query_data.query,
            index=current_index,
            documents=current_documents,
            ids=current_ids,
            top_k=query_data.top_k,
            include_sources=True # Kaynakları da al
        )

        print(f"Bulunan kaynaklar: {sources}")

        # 2. LLM için prompt oluştur (api_server2.py'deki yapıya benzer)
        if not context:
            print("İlgili bağlam bulunamadı.")
            # Bilgi bulunamadığında kullanılacak prompt
            prompt = f"""Sen Sakarya Üniversitesi'nin yardımsever bir bilgi asistanısın.

Kullanıcı Sorusu: {query_data.query}

Bu konuda veritabanımda maalesef yeterli bilgi bulunmuyor. Kullanıcıya bu durumu nazikçe açıkla.

Cevap:"""
            context = "İlgili bilgi bulunamadı." # Yanıtta döndürmek için
            sources = [] # Kaynak yok
        else:
            print(f"Oluşturulan bağlam:\n{context[:500]}...") # Bağlamın başını yazdır
            # Bilgi bulunduğunda kullanılacak prompt
            prompt = f"""<CONTEXT>
{context}
</CONTEXT>

Sen Sakarya Üniversitesi'nin resmi yönetmelikleri konusunda uzman bir asistansın. Yalnızca yukarıdaki <CONTEXT> içinde verilen bilgilere dayanarak kullanıcı sorusunu yanıtla. Cevabın net, doğru ve yalnızca verilen bağlamdaki bilgilerle sınırlı olsun. Eğer cevap bağlamda yoksa veya emin değilsen, bunu belirt.

Kullanıcı Sorusu: {query_data.query}

Cevap:"""

        print("LLM için prompt oluşturuldu.")

        # 3. LLM ile yanıt üret
        print("LLM ile yanıt üretiliyor...")
        output = current_llm(
            prompt,
            max_tokens=query_data.max_tokens,
            temperature=query_data.temperature,
            stop=["Kullanıcı Sorusu:", "\n\n", "---", "<CONTEXT>"], # Durdurma kelimeleri güncellendi
            echo=False # Prompt'u yanıtta tekrarlama
        )

        model_answer = output['choices'][0]['text'].strip()
        print(f"Model yanıtı:\n{model_answer}")

        return ChatResponse(
            model_answer=model_answer,
            retrieved_context=context, # Bulunamasa bile "İlgili bilgi bulunamadı." döner
            sources=sources
        )

    except HTTPException as http_exc:
        # Bağımlılık hatalarını (model/db yüklenemedi) tekrar yükselt
        raise http_exc
    except Exception as e:
        print(f"Sohbet işlenirken hata oluştu: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Sohbet işlenirken beklenmeyen bir hata oluştu: {str(e)}")


# PDF yükleme ve işleme endpoint'i (api_server.py'deki çoklu dosya ve yeniden yükleme mantığı korundu)
@app.post("/upload-pdf", response_model=UploadResponse, summary="PDF Dosyalarını Yükle ve İndeksle")
async def upload_pdf(
    files: List[UploadFile] = File(..., description="İndekslenecek PDF dosyaları listesi"),
    db_path: str = Depends(get_db_path) # Sadece yolu alır
):
    """
    Yüklenen PDF dosyalarını işler ve vektör veritabanına ekler.
    İşlem sonrası veritabanı belleğe yeniden yüklenir.

    - **files**: Yüklenecek PDF dosyalarının listesi.
    """
    global faiss_index, documents, ids, is_db_loaded # Veritabanını yeniden yüklemek için global erişim

    processed_files_count = 0
    total_added_chunks = 0
    all_chunk_dicts = []
    errors = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_chunks_path = os.path.join(temp_dir, "combined_processed_chunks.json")

        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                errors.append(f"'{file.filename}': Geçersiz dosya türü. Sadece PDF dosyaları kabul edilir.")
                continue

            temp_pdf_path = os.path.join(temp_dir, file.filename)

            try:
                with open(temp_pdf_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                print(f"PDF geçici olarak kaydedildi: {temp_pdf_path}")

                extracted_text = extract_text_from_pdf(temp_pdf_path)
                if not extracted_text:
                    errors.append(f"'{file.filename}': PDF dosyasından metin çıkarılamadı.")
                    continue
                print(f"'{file.filename}' için metin çıkarıldı.")

                cleaned_text = clean_text(extracted_text)
                chunks = simple_chunk_text(cleaned_text)
                if not chunks:
                    errors.append(f"'{file.filename}': Metin parçalara ayrılamadı.")
                    continue

                chunk_dicts = convert_chunks_to_dict(chunks, file.filename)
                print(f"'{file.filename}' için {len(chunk_dicts)} adet metin parçası oluşturuldu.")

                all_chunk_dicts.extend(chunk_dicts)
                processed_files_count += 1

            except Exception as e:
                error_msg = f"'{file.filename}' işlenirken hata oluştu: {str(e)}"
                print(error_msg)
                errors.append(error_msg)
            finally:
                 if os.path.exists(temp_pdf_path):
                     try:
                         os.remove(temp_pdf_path)
                     except OSError as rm_err:
                         print(f"Geçici PDF silinemedi: {rm_err}")

        if not all_chunk_dicts:
             if errors:
                 error_details = "; ".join(errors)
                 raise HTTPException(status_code=400, detail=f"Hiçbir dosya başarıyla işlenemedi. Hatalar: {error_details}")
             else:
                 raise HTTPException(status_code=400, detail="Yüklenen dosyalarda işlenecek içerik bulunamadı veya dosyalar boş.")

        try:
            with open(temp_chunks_path, 'w', encoding='utf-8') as f:
                json.dump(all_chunk_dicts, f, ensure_ascii=False, indent=2)
            print(f"Toplam {len(all_chunk_dicts)} parça geçici JSON'a kaydedildi: {temp_chunks_path}")

            print("FAISS veritabanına ekleme işlemi başlatılıyor...")
            # db_path burada Depends'den geliyor
            success = add_to_faiss_index(temp_chunks_path, db_path, model_name=FAISS_MODEL)

            if not success:
                raise HTTPException(status_code=500, detail="Veriler FAISS veritabanına eklenirken genel bir hata oluştu. Detaylar için sunucu loglarına bakın.")

            total_added_chunks = len(all_chunk_dicts)
            print("Veriler FAISS veritabanına başarıyla eklendi.")

            print("Vektör veritabanı belleğe yeniden yükleniyor...")
            try:
                # db_path burada Depends'den geliyor
                faiss_index, documents, ids = load_vector_db(db_path)
                if faiss_index is not None and documents is not None and ids is not None:
                    is_db_loaded = True # Durumu güncelle
                    print("Vektör veritabanı başarıyla yeniden yüklendi.")
                else:
                    is_db_loaded = False
                    errors.append("Veritabanı eklendikten sonra belleğe yeniden yüklenemedi.")
                    print("HATA: Veritabanı eklendikten sonra belleğe yeniden yüklenemedi.")
            except Exception as reload_e:
                is_db_loaded = False
                errors.append(f"Veritabanı yeniden yüklenirken hata oluştu: {str(reload_e)}")
                print(f"HATA: Veritabanı yeniden yüklenirken hata oluştu: {reload_e}")

            success_message = f"{processed_files_count} PDF dosyası başarıyla işlendi ve toplam {total_added_chunks} parça veritabanına eklendi."
            final_message = success_message
            if errors:
                 final_message += " Ancak bazı sorunlar oluştu (detaylar için 'errors' alanına bakın)."

            return UploadResponse(
                message=final_message,
                processed_files=processed_files_count,
                added_chunks=total_added_chunks,
                errors=errors
            )

        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            print(f"Veritabanına ekleme veya yeniden yükleme sırasında hata: {str(e)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Veritabanına ekleme veya yeniden yükleme sırasında bir hata oluştu: {str(e)}")


# --- Uygulamayı Başlat ---
if __name__ == "__main__":
    # argparse kodunu buraya ekleyebiliriz, eğer doğrudan çalıştırma senaryosu için
    # komut satırı argümanları hala isteniyorsa, ancak uvicorn'u etkilemez.
    # Örneğin:
    # local_parser = argparse.ArgumentParser(description="SAÜChat API Sunucusu (Doğrudan Çalıştırma)")
    # local_parser.add_argument("--host", default="0.0.0.0", help="Sunucu adresi")
    # local_parser.add_argument("--port", type=int, default=8000, help="Sunucu portu")
    # local_parser.add_argument("--reload", action="store_true", help="Otomatik yeniden başlatmayı etkinleştir")
    # local_args = local_parser.parse_args()
    #
    # uvicorn.run(
    #     "api_server:app",
    #     host=local_args.host,
    #     port=local_args.port,
    #     reload=local_args.reload,
    #     log_level="info"
    # )
    # Şimdilik basit tutalım:
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True, # Geliştirme için otomatik yeniden başlatma
        log_level="info"
    )