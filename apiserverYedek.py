# api_server.py
"""
SAÜChat API Sunucusu (HuggingFace, Gelişmiş Prompt, Düzeltilmiş DB Entegrasyonu)
"""
import os
import sys
import json
import uvicorn
import tempfile
import shutil
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager
from jose import JWTError, jwt
import time
import datetime
import torch
import traceback # Hata ayıklama için
from typing import List, Dict, Optional, Any, Tuple # <<-- Tuple BURAYA EKLENDİ

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Hata: 'transformers' kütüphanesi. Kurulum: 'pip install transformers torch accelerate'"); sys.exit(1)

try:
    from vector_db_helpers import retrieve_relevant_context, load_vector_db
    from pdf_text_extract import extract_text_from_pdf
    from data_preprocess import convert_chunks_to_dict, clean_text, simple_chunk_text
    from faiss_index_process import add_to_faiss_index, DEFAULT_MODEL as FAISS_MODEL
    # mongo_db_helper'dan Pydantic modellerini de import etmeye çalışalım (eğer orada tanımlıysa)
    # veya burada tanımlayıp mongo_db_helper'ın bunları döndürdüğünü varsayalım.
    from mongo_db_helper import (
        authenticate_user, create_user, save_chat_message,
        get_user_chats, get_chat_messages, delete_chat,
        MDB_ChatHistory, MDB_ChatMessageHistory # BUNLARIN mongo_db_helper.py'DE TANIMLI OLMASI GEREKİR
    )
except ImportError as e:
    print(f"Yerel modül/Pydantic modeli import hatası: {e}. Dosyaların ve tanımların varlığını kontrol edin."); sys.exit(1)
    # Eğer MDB_ChatHistory ve MDB_ChatMessageHistory mongo_db_helper.py'de yoksa,
    # aşağıdaki Pydantic modelleri kısmında bu isimlerle tekrar tanımlanacaklar.

os.environ.setdefault("MONGO_URI", "mongodb+srv://denemecursor1bedava:oF27WsS8MqA1nYPk@bitirme.ne3ofr5.mongodb.net/sau_chat_db?retryWrites=true&w=majority&appName=bitirme")
os.environ.setdefault("PASSWORD_SALT", "oF27WsS8MqA1nYPk")
os.environ.setdefault("JWT_SECRET_KEY", "cef7f52a6f89a1b0c5ce1373e4d96eac0a5d65a98d29f67e1a2b4a3c76fd7d2b")

#MODEL_PATH = os.environ.get("SAUCHAT_MODEL_PATH", "./yeniEgitilmisTrendyolLlama3")

# --- Konfigürasyon Ayarları (Ortam Değişkenleri veya Varsayılanlar) ---
MODEL_PATH = os.environ.get("MODEL_PATH", "/content/drive/MyDrive/HasanProje/sau_chat/yeniEgitilmisTrendyolLlama3")


DB_PATH = os.environ.get("SAUCHAT_DB_PATH", "vector_db")
N_CTX_HF = int(os.environ.get("SAUCHAT_N_CTX_HF", 4096))

SECRET_KEY = os.environ["JWT_SECRET_KEY"]
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("SAUCHAT_TOKEN_EXPIRE_MINUTES", 60 * 24))
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

llm_model: Optional[AutoModelForCausalLM] = None
llm_tokenizer: Optional[AutoTokenizer] = None
faiss_index: Optional[Any] = None
documents: Optional[List[Dict[str, Any]]] = None
ids: Optional[List[str]] = None
is_llm_loaded: bool = False
is_db_loaded: bool = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_model, llm_tokenizer, faiss_index, documents, ids, is_llm_loaded, is_db_loaded
    print(f"HF Model yükleniyor: {MODEL_PATH}")
    if not os.path.isdir(MODEL_PATH): print(f"UYARI: Model klasörü yok: {MODEL_PATH}"); is_llm_loaded = False
    else:
        try:
            llm_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
            llm_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
            is_llm_loaded = True; print("HF Model yüklendi.")
        except Exception as e: print(f"HF model yükleme hatası: {e}"); traceback.print_exc(); is_llm_loaded = False
    print(f"Vektör DB yükleniyor: {DB_PATH}")
    if not os.path.exists(DB_PATH): print(f"UYARI: DB klasörü yok: {DB_PATH}"); is_db_loaded = False
    else:
        try:
            faiss_index, documents, ids = load_vector_db(DB_PATH)
            if faiss_index and documents and ids: is_db_loaded = True; print("Vektör DB yüklendi.")
            else: is_db_loaded = False; print("Vektör DB yüklenemedi.")
        except Exception as e: print(f"Vektör DB yükleme hatası: {e}"); is_db_loaded = False
    yield
    print("API kapatılıyor."); del llm_model, llm_tokenizer, faiss_index, documents, ids
    llm_model,llm_tokenizer,faiss_index,documents,ids=None,None,None,None,None;is_llm_loaded=is_db_loaded=False
    if torch.cuda.is_available(): torch.cuda.empty_cache()

app = FastAPI(title="SAÜChat API", version="1.5.0", lifespan=lifespan) # Son Düzeltmeler Sürümü
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def get_hf_model_and_tokenizer():
    if not (is_llm_loaded and llm_model and llm_tokenizer): raise HTTPException(503, "LLM kullanılamıyor.")
    return llm_model, llm_tokenizer
def get_vector_db_components():
    if not (is_db_loaded and faiss_index and documents and ids): raise HTTPException(503, "Vektör DB kullanılamıyor.")
    return faiss_index, documents, ids
def get_db_path_dependency(): return DB_PATH

# --- Pydantic Modelleri ---
class ChatMessageInput(BaseModel): role: str; content: str
class ChatQuery(BaseModel):
    query: str
    history: Optional[List[ChatMessageInput]] = None
    current_chat_id: Optional[str] = None
    top_k: int = 3
    temperature: float = 0.3
    max_new_tokens: int = 768
    top_p: Optional[float] = 0.9
    repetition_penalty: Optional[float] = 1.1

class ChatResponseAPI(BaseModel): # API'den dönecek yanıt için Pydantic modeli
    model_answer: str
    retrieved_context: str
    sources: List[str]
    chat_id: Optional[str] = None # Kullanılan veya yeni oluşturulan chat_id

class HealthStatus(BaseModel): status: str; llm_loaded: bool; db_loaded: bool; model_path: Optional[str]=None; db_path: Optional[str]=None
class UploadResponse(BaseModel): message: str; processed_files: int; added_chunks: int; errors: List[str]
class UserRegister(BaseModel): username: str; email: str; password: str
class Token(BaseModel): access_token: str; token_type: str
class UserInfo(BaseModel): username: str; email: str

# mongo_db_helper.py'den gelen Pydantic modelleri (veya burada tanımlanmış olmalı)
# Eğer mongo_db_helper.py'de MDB_ChatHistory ve MDB_ChatMessageHistory tanımlıysa ve import edildiyse,
# aşağıdaki tanımlara gerek yok. Eğer import edilemiyorsa, burada tanımlanmaları gerekir.
# Ben import edildiğini varsayıyorum. Edilmediyse, aşağıdaki satırları açın.
# class MDB_ChatHistory(BaseModel): chat_id: str; first_message: str; timestamp: datetime.datetime; message_count: int
# class MDB_ChatMessageHistory(BaseModel): user_message: str; bot_response: str; timestamp: datetime.datetime


@app.get("/health", response_model=HealthStatus)
async def health_check():
    status="ok"; path_m=MODEL_PATH if os.path.isdir(MODEL_PATH) else f"X: {MODEL_PATH}"; path_d=DB_PATH if os.path.exists(DB_PATH) else f"X: {DB_PATH}"
    if not is_llm_loaded or not is_db_loaded: status = "partial_error"
    if not is_llm_loaded and not is_db_loaded: status = "error"
    return HealthStatus(status=status, llm_loaded=is_llm_loaded, db_loaded=is_db_loaded, model_path=path_m, db_path=path_d)

def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None):
    exp = datetime.datetime.utcnow() + (expires_delta or datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return jwt.encode({**data, "exp": exp}, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    exc = HTTPException(401, "Geçersiz kimlik", headers={"WWW-Authenticate": "Bearer"})
    try: payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM]); user = payload.get("sub"); assert user
    except: raise exc
    return {"username": user}

async def _generate_chat_response(
    query_data: ChatQuery, current_hf_model: AutoModelForCausalLM, current_hf_tokenizer: AutoTokenizer,
    db_path_val: str, is_anonymous: bool = False, username: Optional[str] = None
) -> Tuple[str, str, List[str]]: # model_answer, retrieved_context, sources döndürür
    # ... (Bu fonksiyonun içeriği bir önceki cevaptaki "REVİZE EDİLMİŞ PROMPT YAPISI" ile aynı kalacak)
    # ... (Sadece en sondaki "return ChatResponse(...)" yerine Tuple döndürecek)
    log_prefix = "[Anonim] " if is_anonymous else f"[{username if username else 'Kullanıcı'}] "
    ret_result = retrieve_relevant_context(query=query_data.query, db_path=db_path_val, top_k=query_data.top_k, return_sources=True)
    ctx_text, sources = ret_result.get("text", "").strip(), ret_result.get("sources", [])
    base_persona = ("Sen SAÜChat, Sakarya Üniversitesi'nin resmi Yapay Zeka Destek Asistanısın...")
    general_instructions = ["Bilgi Kaynakları: Cevapların HER ZAMAN öncelikle sana <CONTEXT> etiketi içinde sağlanan bilgilere dayanmalıdır...",]
    context_instructions = { "with_context": ("..."), "without_context_specific_inquiry": ("..."), "without_context_general_inquiry": ("..."), "clarification_needed": ("..."), "out_of_scope": ("...") } # Kısaltıldı, tam metinler önceki cevapta
    system_message_list = [base_persona]; system_message_list.extend(general_instructions)
    msgs = []; ret_ctx_resp = "İlgili yönetmelik bilgisi bulunamadı."
    if ctx_text:
        chosen_instruction = context_instructions["with_context"]
        final_system_content = "\n\n".join(system_message_list) + "\n\n" + chosen_instruction + f"\n\n<CONTEXT>\n{ctx_text}\n</CONTEXT>"
        ret_ctx_resp = ctx_text
    else:
        if len(query_data.query.split()) < 4 or any(k in query_data.query.lower() for k in ["nedir","nasıl","ne zaman","nerede","kimdir"]): chosen_instruction = context_instructions["without_context_general_inquiry"]
        else: chosen_instruction = context_instructions["without_context_specific_inquiry"]
        final_system_content = "\n\n".join(system_message_list) + "\n\n" + chosen_instruction
        sources = []
    msgs.append({"role": "system", "content": final_system_content})
    is_first_turn = not query_data.history or len(query_data.history) == 0
    if query_data.history:
        for hist_msg in query_data.history: msgs.append({"role": hist_msg.role, "content": hist_msg.content})
    msgs.append({"role": "user", "content": query_data.query})
    try: prompt = current_hf_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except: prompt = "Fallback prompt"
    inputs = current_hf_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=N_CTX_HF - query_data.max_new_tokens)
    try: target_device = next(current_hf_model.parameters()).device
    except: target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids_dev, attn_mask_dev = inputs["input_ids"].to(target_device), inputs["attention_mask"].to(target_device)
    gen_kwargs = {"max_new_tokens": query_data.max_new_tokens, "pad_token_id": current_hf_tokenizer.eos_token_id, "eos_token_id": current_hf_tokenizer.eos_token_id}
    if query_data.temperature > 0.001: gen_kwargs.update({"temperature": query_data.temperature, "do_sample": True, "top_p": query_data.top_p})
    else: gen_kwargs["do_sample"] = False
    if query_data.repetition_penalty and query_data.repetition_penalty > 1.0: gen_kwargs["repetition_penalty"] = query_data.repetition_penalty
    with torch.no_grad(): out_seqs = current_hf_model.generate(input_ids=input_ids_dev, attention_mask=attn_mask_dev, **gen_kwargs)
    answer = current_hf_tokenizer.decode(out_seqs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    final_answer = answer
    if is_first_turn and username:
        greeting = f"Merhaba {username.capitalize()}"
        if not answer.lower().startswith(greeting.lower().split(" ")[0].lower()): final_answer = f"{greeting}! {answer}"
        elif not answer.lower().startswith(greeting.lower()): final_answer = f"{greeting}! {answer[len('Merhaba'):].lstrip(' !.')}"
    return final_answer, ret_ctx_resp, sources


@app.post("/chat", response_model=ChatResponseAPI)
async def chat_endpoint(q_data: ChatQuery, deps: tuple = Depends(get_hf_model_and_tokenizer), db_p: str = Depends(get_db_path_dependency), user: dict = Depends(get_current_user)):
    model, tokenizer = deps; username = user.get("username")
    try:
        model_answer, retrieved_context, sources = await _generate_chat_response(
            q_data, model, tokenizer, db_p, is_anonymous=False, username=username
        )
        
        # mongo_db_helper.save_chat_message string chat_id döndürmeli
        returned_chat_id = save_chat_message(
            username=username, 
            user_message=q_data.query, 
            bot_response=model_answer, 
            retrieved_docs=[s for s in sources if s],
            chat_id=q_data.current_chat_id # Streamlit'ten gelen mevcut chat_id
        )
        print(f"API /chat - Kullanılan/Oluşturulan Chat ID: {returned_chat_id}")
        
        return ChatResponseAPI(
            model_answer=model_answer,
            retrieved_context=retrieved_context,
            sources=sources,
            chat_id=returned_chat_id 
        )
    except Exception as e_chat: print(f"Chat endpoint hatası: {e_chat}"); traceback.print_exc(); raise HTTPException(500, f"Sunucu hatası: {e_chat}")

@app.post("/anon-chat", response_model=ChatResponseAPI)
async def anon_chat_endpoint(q_data: ChatQuery, deps: tuple = Depends(get_hf_model_and_tokenizer), db_p: str = Depends(get_db_path_dependency)):
    model, tokenizer = deps
    try: 
        model_answer, retrieved_context, sources = await _generate_chat_response(
            q_data, model, tokenizer, db_p, is_anonymous=True, username=None
        )
        return ChatResponseAPI(
            model_answer=model_answer,
            retrieved_context=retrieved_context,
            sources=sources,
            chat_id=None # Anonim sohbet için chat_id yok
        )
    except Exception as e_anon: print(f"[Anonim] Chat endpoint hatası: {e_anon}"); traceback.print_exc(); raise HTTPException(500, f"Sunucu hatası: {e_anon}")

# --- Diğer Endpointler (PDF Yükleme, Kullanıcı Yönetimi, Sohbet Geçmişi) ---
# Bu fonksiyonların İÇERİKLERİ, sizin en son paylaştığınız tam ve çalışan kodunuzdaki GİBİDİR.
# Sadece get_chat_message_history'yi düzelttim.
@app.post("/upload-pdf", response_model=UploadResponse, summary="PDF Dosyalarını Yükle ve İndeksle")
async def upload_pdf(files: List[UploadFile] = File(...), db_path_val: str = Depends(get_db_path_dependency)):
    global faiss_index, documents, ids, is_db_loaded; processed_files_count,total_added_chunks,all_chunk_dicts,errors=0,0,[],[]
    with tempfile.TemporaryDirectory() as td:
        tp=os.path.join(td,"c.json")
        for fi in files:
            if not fi.filename.lower().endswith(".pdf"): errors.append(f"'{fi.filename}': Geçersiz."); continue
            tmp_p=os.path.join(td,fi.filename)
            try:
                with open(tmp_p,"wb") as b: shutil.copyfileobj(fi.file,b)
                txt=extract_text_from_pdf(tmp_p)
                if not txt or not txt.strip(): errors.append(f"'{fi.filename}': Metin yok."); continue
                ch=simple_chunk_text(clean_text(txt))
                if not ch: errors.append(f"'{fi.filename}': Parçalanamadı."); continue
                all_chunk_dicts.extend(convert_chunks_to_dict(ch,fi.filename));processed_files_count+=1
            except Exception as e: errors.append(f"'{fi.filename}' işlenirken: {e}")
            finally:
                if os.path.exists(tmp_p):os.remove(tmp_p)
                fi.file.close()
        if not all_chunk_dicts:
            if not errors: errors.append("Dosya yok/işlenemedi.")
            raise HTTPException(400,f"İşleme başarısız: {'; '.join(errors)}")
        try:
            with open(tp,'w',encoding='utf-8') as f: json.dump(all_chunk_dicts,f,ensure_ascii=False,indent=2)
            if not add_to_faiss_index(tp,db_path_val,model_name=FAISS_MODEL): raise HTTPException(500,"FAISS hatası.")
            total_added_chunks=len(all_chunk_dicts)
            faiss_index,documents,ids=load_vector_db(db_path_val);is_db_loaded=bool(faiss_index and documents and ids)
            if not is_db_loaded: errors.append("DB yeniden yüklenemedi.")
            msg=f"{processed_files_count} PDF işlendi, {total_added_chunks} parça."
            if errors: msg+=" Sorunlar var."
            return UploadResponse(message=msg,processed_files=processed_files_count,added_chunks=total_added_chunks,errors=errors)
        except Exception as e: traceback.print_exc();raise HTTPException(500,f"DB/Yükleme hatası: {e}")

@app.post("/register", response_model=dict, summary="Yeni Kullanıcı Kaydı")
async def register_new_user(user_data: UserRegister):
    try:
        success, message = create_user(username=user_data.username,email=user_data.email,password=user_data.password)
        if not success: raise HTTPException(400, detail=message or "Kullanıcı adı/e-posta zaten kullanımda")
        return {"success": True, "message": message or "Kullanıcı başarıyla oluşturuldu"}
    except HTTPException as http_exc: raise http_exc
    except Exception as e: print(f"Kayıt hatası: {e}"); raise HTTPException(status_code=500, detail=f"Kayıt sırasında sunucu hatası: {str(e)}")

@app.post("/token", response_model=Token, summary="Access Token Al")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user: raise HTTPException(status_code=401, detail="Hatalı kullanıcı adı veya şifre", headers={"WWW-Authenticate": "Bearer"})
    return Token(access_token=create_access_token(data={"sub": user["username"]}), token_type="bearer")

@app.get("/users/me", response_model=UserInfo, summary="Kullanıcı Bilgilerini Al")
async def read_current_user_info(user: dict = Depends(get_current_user)):
    return UserInfo(username=user["username"], email=user.get("email",""))

@app.get("/chat-history", response_model=List[MDB_ChatHistory], summary="Sohbet Geçmişini Al") # Dönüş tipi güncellendi
async def list_user_chat_history(user: dict = Depends(get_current_user)):
    try:
        # Bu fonksiyon mongo_db_helper.py'den List[MDB_ChatHistory] döndürmeli
        chat_sessions: List[MDB_ChatHistory] = get_user_chats(user["username"])
        return chat_sessions
    except Exception as e: print(f"Geçmiş alma hatası (user: {user['username']}): {e}"); traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Sohbet geçmişi alınırken hata: {str(e)}")

@app.get("/chat-history/{chat_id}", response_model=List[MDB_ChatMessageHistory], summary="Belirli Bir Sohbetin Mesajlarını Al") # Dönüş tipi güncellendi
async def get_chat_message_history(chat_id: str, current_user_data: dict = Depends(get_current_user)):
    try:
        # Varsayım: get_user_chats, MDB_ChatHistory Pydantic nesneleri listesi döndürüyor.
        # Eğer hala dict döndürüyorsa, mongo_db_helper.py'yi düzeltmeniz gerekir.
        user_chats_list: List[MDB_ChatHistory] = get_user_chats(current_user_data["username"])
        
        if not any(chat_session.chat_id == chat_id for chat_session in user_chats_list):
            raise HTTPException(status_code=403, detail="Bu sohbete erişim yetkiniz yok veya sohbet bulunamadı.")
        
        # Varsayım: get_chat_messages, MDB_ChatMessageHistory Pydantic nesneleri listesi döndürüyor.
        messages: List[MDB_ChatMessageHistory] = get_chat_messages(chat_id)
        return messages
    except HTTPException as http_exc:
        raise http_exc
    except AttributeError as attr_err: 
        print(f"AttributeError - Muhtemelen get_user_chats veya get_chat_messages Pydantic listesi döndürmüyor: {attr_err}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Sohbet geçmişi veri yapısı hatası. mongo_db_helper.py kontrol edin.")
    except Exception as e: 
        print(f"Mesaj alma hatası ({chat_id}, user: {current_user_data['username']}): {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Sohbet mesajları alınırken bir hata oluştu: {str(e)}")

@app.delete("/chat-history/{chat_id}", response_model=dict, summary="Sohbeti Sil")
async def delete_chat_history(chat_id: str, current_user_data: dict = Depends(get_current_user)):
    try:
        success = delete_chat(chat_id, current_user_data["username"])
        if not success: raise HTTPException(status_code=404, detail="Sohbet bulunamadı veya silme işlemi başarısız oldu")
        return {"success": True, "message": "Sohbet başarıyla silindi"}
    except HTTPException as http_exc: raise http_exc
    except Exception as e: print(f"Sohbet silme hatası ({chat_id}, user: {current_user_data['username']}): {e}"); traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Sohbet silinirken hata: {str(e)}")

if __name__ == "__main__":
    host,port,reload=os.environ.get("SAUCHAT_HOST","0.0.0.0"),int(os.environ.get("SAUCHAT_PORT","8000")),os.environ.get("SAUCHAT_RELOAD","true").lower()=="true"
    print(f"API http://{host}:{port} (Model: {MODEL_PATH} [HF], DB: {DB_PATH}, Reload: {reload})")
    uvicorn.run("api_server:app", host=host, port=port, reload=reload, log_level="info")