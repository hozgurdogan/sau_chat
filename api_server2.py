#!/usr/bin/env python

import os
import sys
from fastapi import FastAPI, HTTPException, Depends,UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Union
import uvicorn
from llama_cpp import Llama

import tempfile
import shutil
import json

# Kendi modüllerimizi import et
try:
    from vector_db_helpers import retrieve_relevant_context
    from pdf_text_extract import extract_text_from_pdf, save_text_to_file
    from data_preprocess import process_text_file, convert_chunks_to_dict, clean_text, simple_chunk_text
    from faiss_index_process import add_to_faiss_index, DEFAULT_MODEL as FAISS_MODEL
except ImportError as e:
    print(f"Gerekli modüller yüklenirken hata oluştu: {e}")
    print("Lütfen pdf_text_extract, data_preprocess ve faiss_index_process modüllerinin doğru konumda olduğundan emin olun.")
    sys.exit(1)

# FastAPI uygulaması oluştur
app = FastAPI(
    title="SAÜChat API",
    description="Sakarya Üniversitesi yönetmeliklerinden bilgi veren LLM destekli API",
    version="1.0.0"
)

# CORS ayarları (farklı kaynaklardan erişime izin vermek için)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tüm kaynaklara izin ver (production'da sınırlandırılabilir)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global LLM model değişkeni
llm_model = None

# Gelen istek için veri modeli
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Kullanıcının sorgusu")
    top_k: int = Field(default=3, ge=1, le=10, description="Döndürülecek en alakalı belge sayısı")
    temperature: float = Field(default=0.1, ge=0.0, le=1.0, description="Model yaratıcılık seviyesi")
    max_tokens: int = Field(default=512, ge=50, le=2048, description="Maksimum yanıt uzunluğu")

# Yanıt için veri modeli
class ChatResponse(BaseModel):
    retrieved_context: str = Field(description="Vektör veritabanından alınan ilgili bilgiler")
    model_answer: str = Field(description="LLM tarafından oluşturulan yanıt")
    sources: Optional[List[str]] = Field(default=None, description="Bilgi kaynakları (varsa)")

# Uygulama başlangıcında çalışacak olan olay
@app.on_event("startup")
async def startup_event():
    """
    Uygulama başlatıldığında LLM modelini yükler.
    Bu sayede her istekte model tekrar yüklenmez.
    """
    global llm_model
    
    # Model dosyası yolunu kontrol et
    # model_path = "D:/models/gguf/llama-3-GGUF/llama3-8B-trendyol-rag-merged-Q8_0.gguf"
    model_path = "/content/drive/MyDrive/llama_chat/gguf/llama3-8B-trendyol-rag-merged-Q8_0.gguf"
    
    if not os.path.exists(model_path):
        print(f"UYARI: Model dosyası bulunamadı: {model_path}")
        print("Model yüklenemedi. İlk istek geldiğinde tekrar deneyeceğiz.")
        return
    
    try:
        print(f"Model yükleniyor: {model_path}")
        # GGUF formatındaki Llama modelini yükle
        llm_model = Llama(
            model_path=model_path,
            n_ctx=2048,  # Bağlam penceresi boyutu
            n_gpu_layers=-1,  # Tüm katmanları GPU'ya yükle
            n_threads=os.cpu_count(),  # Kullanılabilir tüm CPU çekirdeklerini kullan
        )
        print("Model başarıyla yüklendi ve GPU'ya aktarıldı.")
    except Exception as e:
        print(f"Model yükleme hatası: {str(e)}")
        print("API çalışmaya devam edecek, ancak LLM yanıtları üretilemeyecek.")

# Vektör veritabanını kontrol et
def check_vector_db():
    db_path = "vector_db"
    if not os.path.exists(db_path):
        raise HTTPException(
            status_code=500, 
            detail=f"Vektör veritabanı bulunamadı: {db_path}. Lütfen önce veritabanını oluşturun."
        )
    return db_path

# Model yüklü değilse yüklemeyi dene
def get_llm_model():
    global llm_model
    
    if llm_model is None:
        model_path = "D:/models/gguf/llama-3-GGUF/llama3-8B-trendyol-rag-merged-Q8_0.gguf"
        
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=500,
                detail=f"Model dosyası bulunamadı: {model_path}. LLM özellikleri kullanılamaz."
            )
        
        try:
            llm_model = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_gpu_layers=-1,
                n_threads=os.cpu_count(),
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Model yüklenirken hata oluştu: {str(e)}"
            )
    
    return llm_model

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest, 
    db_path: str = Depends(check_vector_db),
    model: Llama = Depends(get_llm_model)
):
    """
    Kullanıcının sorgusuna göre vektör veritabanından ilgili bilgileri getirir
    ve LLM modelini kullanarak bir yanıt üretir.
    
    - **query**: Kullanıcının sorgusu
    - **top_k**: Döndürülecek en alakalı belge sayısı
    - **temperature**: Model yaratıcılık seviyesi (0.0-1.0)
    - **max_tokens**: Maksimum yanıt uzunluğu
    
    Returns:
        Sorguya ilişkin vektör veritabanından bulunan bilgiler ve model yanıtı
    """
    # Sorgu boş mu kontrol et
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Sorgu boş olamaz. Lütfen geçerli bir soru sorun."
        )
    
    try:
        # Vektör veritabanından ilgili bilgileri getir
        retrieval_result = retrieve_relevant_context(
            query=request.query, 
            db_path=db_path, 
            top_k=request.top_k,
            return_sources=True  # Kaynakları da döndür
        )
        
        # Retrieval sonuçlarını çıkar
        if isinstance(retrieval_result, dict) and "text" in retrieval_result:
            relevant_text = retrieval_result["text"]
            sources = retrieval_result.get("sources", [])
        else:
            relevant_text = retrieval_result
            sources = []
        
        # Relevant context boş mu kontrol et
        if not relevant_text:
            # Eğer ilgili bilgi bulunamadıysa, modele bilgi bulunamadığını belirt
            prompt = f"""Sen Sakarya Üniversitesi'nin bilgi asistanısın.

Soru: {request.query}

Bu konuda veritabanımda yeterli bilgi bulunamadı. Lütfen kullanıcıya bilgi bulunamadığını nazik bir şekilde bildir.

Yanıt:"""
        else:
            # Prompt oluştur
            prompt = f"""<CONTEXT>
{relevant_text}
</CONTEXT>

Sen Sakarya Üniversitesi'nin bilgi asistanısın. Yukarıdaki bağlamda verilen bilgileri kullanarak öğrencilere
yardımcı oluyorsun. Sadece bağlamda bulunan bilgileri kullan, uydurma bilgi verme. 
Eğer cevabı bilmiyorsan veya bağlamda yeterli bilgi yoksa, dürüstçe bilmediğini söyle.

Soru: {request.query}

Yanıt:"""
        
        # Modelden yanıt al
        response = model.create_completion(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=0.95,
            repeat_penalty=1.1,
            stop=["Soru:", "<CONTEXT>"]
        )
        
        model_answer = response['choices'][0]['text'].strip()
        
        # Başarılı cevap döndür
        return ChatResponse(
            retrieved_context=relevant_text,
            model_answer=model_answer,
            sources=sources if sources else None
        )
        
    except Exception as e:
        # Hata durumunda
        raise HTTPException(
            status_code=500,
            detail=f"İşlem sırasında bir hata oluştu: {str(e)}"
        )

# Sunucunun çalışıp çalışmadığını kontrol eden basit bir endpoint
@app.get("/health")
async def health_check():
    """Sunucunun sağlık durumunu kontrol eder"""
    return {
        "status": "online", 
        "message": "SAÜChat API çalışıyor",
        "llm_loaded": llm_model is not None
    }
    
# PDF yükleme ve işleme endpoint'i
@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(..., description="İndekslenecek PDF dosyası"),
    db_path: str = Depends(check_vector_db)
):
    """
    Yüklenen PDF dosyasını işler ve vektör veritabanına ekler.

    - **file**: Yüklenecek PDF dosyası.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Geçersiz dosya türü. Lütfen PDF dosyası yükleyin.")

    # Geçici bir dizin oluştur
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_pdf_path = os.path.join(temp_dir, file.filename)
        temp_txt_path = os.path.join(temp_dir, f"{os.path.splitext(file.filename)[0]}.txt")
        temp_chunks_path = os.path.join(temp_dir, "temp_processed_chunks.json")

        try:
            # 1. PDF dosyasını geçici olarak kaydet
            with open(temp_pdf_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            print(f"PDF geçici olarak kaydedildi: {temp_pdf_path}")

            # 2. PDF'den metin çıkar
            extracted_text = extract_text_from_pdf(temp_pdf_path)
            if not extracted_text:
                raise HTTPException(status_code=500, detail="PDF dosyasından metin çıkarılamadı.")
            print("PDF'den metin çıkarıldı.")

            # 3. Metni temizle ve parçalara ayır (data_preprocess.py'deki mantığı kullanarak)
            cleaned_text = clean_text(extracted_text)
            chunks = simple_chunk_text(cleaned_text) # Varsayılan chunk_size ve overlap ile
            if not chunks:
                 raise HTTPException(status_code=500, detail="Metin parçalara ayrılamadı.")

            chunk_dicts = convert_chunks_to_dict(chunks, file.filename) # Kaynak olarak orijinal dosya adını kullan
            print(f"{len(chunk_dicts)} adet metin parçası oluşturuldu.")

            # 4. Parçaları geçici JSON dosyasına kaydet
            with open(temp_chunks_path, 'w', encoding='utf-8') as f:
                json.dump(chunk_dicts, f, ensure_ascii=False, indent=2)
            print(f"Parçalar geçici JSON'a kaydedildi: {temp_chunks_path}")

            # 5. FAISS veritabanına ekle
            print("FAISS veritabanına ekleme işlemi başlatılıyor...")
            success = add_to_faiss_index(temp_chunks_path, db_path, model_name=FAISS_MODEL)

            if not success:
                raise HTTPException(status_code=500, detail="Veriler FAISS veritabanına eklenirken hata oluştu.")
            print("Veriler FAISS veritabanına başarıyla eklendi.")

            return {"message": f"'{file.filename}' başarıyla işlendi ve veritabanına eklendi.", "added_chunks": len(chunk_dicts)}

        except HTTPException as http_exc:
            # HTTPException'ları doğrudan yükselt
            raise http_exc
        except Exception as e:
            print(f"PDF işleme hatası: {str(e)}")
            raise HTTPException(status_code=500, detail=f"PDF işlenirken bir hata oluştu: {str(e)}")
        finally:
            # Geçici dosyaları temizle (hata olsa bile)
             # shutil.rmtree(temp_dir) # with bloğu bunu otomatik yapar
             print(f"Geçici dizin temizlendi: {temp_dir}")


# Uygulamayı başlat (direkt olarak çalıştırılırsa)
if __name__ == "__main__":
    # API sunucusunun portunu 8000 olarak değiştiriyoruz, çakışmayı önlemek için
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000, # Port 8000 olarak değiştirildi
        reload=True
    )