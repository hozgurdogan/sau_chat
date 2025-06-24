# api_server.py
"""
SAÜChat API Sunucusu (HuggingFace, Gelişmiş Prompt, Düzeltilmiş DB Entegrasyonu, Detaylı Debugging)
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
from typing import List, Dict, Optional, Any, Tuple
from contextlib import asynccontextmanager
from jose import JWTError, jwt
import time
import datetime
import torch
import traceback # Hata ayıklama için

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Hata: 'transformers' kütüphanesi. Kurulum: 'pip install transformers torch accelerate'"); sys.exit(1)

try:
    from vector_db_helpers import retrieve_relevant_context, load_vector_db
    from pdf_text_extract import extract_text_from_pdf
    from data_preprocess import convert_chunks_to_dict, clean_text, simple_chunk_text
    from faiss_index_process import add_to_faiss_index, DEFAULT_MODEL as FAISS_MODEL
    from mongo_db_helper import (
        authenticate_user, create_user, save_chat_message,
        get_user_chats, get_chat_messages, delete_chat,
        MDB_ChatHistory, MDB_ChatMessageHistory
    )
except ImportError as e:
    print(f"Yerel modül/Pydantic modeli import hatası: {e}. Dosyaların ve tanımların varlığını kontrol edin."); sys.exit(1)

# Ortam değişkenleri (varsayılanlar veya gerçek değerler)
os.environ.setdefault("MONGO_URI", "mongodb+srv://denemecursor1bedava:oF27WsS8MqA1nYPk@bitirme.ne3ofr5.mongodb.net/sau_chat_db?retryWrites=true&w=majority&appName=bitirme")
os.environ.setdefault("PASSWORD_SALT", "oF27WsS8MqA1nYPk") # Güçlü bir salt kullanın!
os.environ.setdefault("JWT_SECRET_KEY", "cef7f52a6f89a1b0c5ce1373e4d96eac0a5d65a98d29f67e1a2b4a3c76fd7d2b") # Güçlü bir secret key kullanın!

MODEL_PATH = os.environ.get("MODEL_PATH", "/content/drive/MyDrive/HasanProje/sau_chat/yeniEgitilmisTrendyolLlama3")
DB_PATH = os.environ.get("SAUCHAT_DB_PATH", "vector_db")
N_CTX_HF = int(os.environ.get("SAUCHAT_N_CTX_HF", 4096)) # Tokenizer için max_length hesaplamasında kullanılır

SECRET_KEY = os.environ["JWT_SECRET_KEY"]
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("SAUCHAT_TOKEN_EXPIRE_MINUTES", 60 * 24)) # 1 gün
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Global değişkenler (model, tokenizer, db vb.)
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
    if not os.path.isdir(MODEL_PATH):
        print(f"UYARI: Model klasörü bulunamadı: {MODEL_PATH}")
        is_llm_loaded = False
    else:
        try:
            llm_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
            # Modeli float16 olarak yüklemeyi deneyelim (Colab için genellikle daha iyi)
            # Eğer NaN sorunları devam ederse, test için torch_dtype=torch.float32 deneyebilirsiniz.

            llm_model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.float32, # <<<--- float32 OLARAK DEĞİŞTİRDİK
                device_map="auto",
                trust_remote_code=True
            )
            is_llm_loaded = True
            print(f"HF Model yüklendi. (dtype: {llm_model.dtype if llm_model else 'Bilinmiyor'})")
        except Exception as e:
            print(f"HF model yükleme hatası: {e}")
            traceback.print_exc()
            is_llm_loaded = False

    print(f"Vektör DB yükleniyor: {DB_PATH}")
    if not os.path.exists(DB_PATH):
        print(f"UYARI: Vektör DB klasörü bulunamadı: {DB_PATH}")
        is_db_loaded = False
    else:
        try:
            faiss_index, documents, ids = load_vector_db(DB_PATH)
            if faiss_index is not None and documents is not None and ids is not None:
                is_db_loaded = True
                print(f"Vektör DB başarıyla yüklendi: {len(documents)} belge, {len(ids)} ID.")
            else:
                is_db_loaded = False
                print("Vektör DB yüklenemedi (load_vector_db None döndürdü).")
        except Exception as e:
            print(f"Vektör DB yükleme hatası: {e}")
            traceback.print_exc()
            is_db_loaded = False
    yield
    print("API kapatılıyor. Kaynaklar serbest bırakılıyor...")
    del llm_model, llm_tokenizer, faiss_index, documents, ids
    llm_model, llm_tokenizer, faiss_index, documents, ids = None, None, None, None, None
    is_llm_loaded = is_db_loaded = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache temizlendi.")

app = FastAPI(title="SAÜChat API", version="1.5.1-debug", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Bağımlılıklar (Dependencies) ---
def get_hf_model_and_tokenizer_dependency():
    if not (is_llm_loaded and llm_model and llm_tokenizer):
        raise HTTPException(status_code=503, detail="LLM şu anda kullanılamıyor. Lütfen daha sonra tekrar deneyin.")
    return llm_model, llm_tokenizer

def get_vector_db_components_dependency():
    if not (is_db_loaded and faiss_index is not None and documents is not None and ids is not None):
        raise HTTPException(status_code=503, detail="Vektör DB şu anda kullanılamıyor.")
    return faiss_index, documents, ids

def get_db_path_dependency():
    return DB_PATH

# --- Pydantic Modelleri ---
class ChatMessageInput(BaseModel):
    role: str
    content: str

class ChatQuery(BaseModel):
    query: str
    history: Optional[List[ChatMessageInput]] = Field(default_factory=list)
    current_chat_id: Optional[str] = None
    top_k: int = Field(default=3, ge=1, le=10)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0) # Temperature 0.0 da geçerli (greedy)
    max_new_tokens: int = Field(default=768, ge=10, le=N_CTX_HF // 2) # N_CTX_HF'in yarısını geçmesin
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(default=1.1, ge=1.0)

class ChatResponseAPI(BaseModel):
    model_answer: str
    retrieved_context: str
    sources: List[str]
    chat_id: Optional[str] = None

class HealthStatus(BaseModel):
    status: str
    llm_loaded: bool
    db_loaded: bool
    model_path: Optional[str] = None
    model_dtype: Optional[str] = None
    db_path: Optional[str] = None

class UploadResponse(BaseModel):
    message: str
    processed_files: int
    added_chunks: int
    errors: List[str]

class UserRegister(BaseModel):
    username: str = Field(..., min_length=3)
    email: str # Pydantic otomatik olarak e-posta formatını doğrular
    password: str = Field(..., min_length=6)

class Token(BaseModel):
    access_token: str
    token_type: str

class UserInfo(BaseModel):
    username: str
    email: Optional[str] = None


# --- Yardımcı Fonksiyonlar (JWT, Model Cevabı Üretme) ---
def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    credentials_exception = HTTPException(
        status_code=401,
        detail="Geçersiz kimlik bilgileri",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if username is None:
            raise credentials_exception
        # Kullanıcının e-postasını da döndürebiliriz (eğer token'a ekliyorsak veya DB'den çekiyorsak)
        # Şimdilik sadece username yeterli.
        return {"username": username}
    except JWTError:
        raise credentials_exception

async def _generate_chat_response(
    query_data: ChatQuery, current_hf_model: AutoModelForCausalLM, current_hf_tokenizer: AutoTokenizer,
    db_path_val: str, is_anonymous: bool = False, username: Optional[str] = None
) -> Tuple[str, str, List[str]]:
    log_prefix = "[Anonim] " if is_anonymous else f"[{username if username else 'Kullanıcı'}] "
    print(f"{log_prefix}Sorgu alındı: '{query_data.query[:50]}...'")

    ret_result = retrieve_relevant_context(query=query_data.query, db_path=db_path_val, top_k=query_data.top_k, return_sources=True)
    ctx_text = ret_result.get("text", "").strip()
    sources = ret_result.get("sources", [])
    
    base_persona = ("Sen SAÜChat, Sakarya Üniversitesi'nin resmi Yapay Zeka Destek Asistanısın. "
                    "Amacın, öğrencilere ve personele üniversite yönetmelikleri, akademik süreçler, "
                    "kampüs yaşamı ve sıkça sorulan sorular hakkında doğru ve güncel bilgi sağlamaktır. "
                    "Cevapların net, anlaşılır, profesyonel ve her zaman saygılı olmalıdır.")
    general_instructions = [
        "Bilgi Kaynakları: Cevapların HER ZAMAN öncelikle sana <CONTEXT> etiketi içinde sağlanan bilgilere dayanmalıdır. Eğer <CONTEXT> boşsa veya soruyla ilgisizse, bunu belirt. Asla <CONTEXT> dışından bilgi uydurma veya spekülasyon yapma.",
        "Cevap Formatı: Cevaplarını kısa paragraflar veya madde işaretleri halinde yapılandır. Teknik terimleri açıklayın. Sadece sorulan soruya odaklan, gereksiz uzunluğa kaçma.",
        "Kapsam Dışı Sorular: Eğer soru Sakarya Üniversitesi veya genel akademik/yönetsel konularla ilgili değilse (örneğin kişisel tavsiye, güncel olaylar, felsefi tartışmalar), bu konunun uzmanlık alanının dışında olduğunu nazikçe belirt.",
        "Bilgi Eksikliği: Eğer bir soruya cevap verecek yeterli bilgiye sahip değilsen (ne <CONTEXT> içinde ne de genel bilginde), bunu açıkça ifade et. Yanlış veya eksik bilgi vermektense 'Bu konuda bilgim yok.' demek daha iyidir.",
        "Kişiselleştirme: Eğer kullanıcı giriş yapmışsa (username verildiyse) ve bu ilk mesajıysa, ona 'Merhaba [Kullanıcı Adı]!' gibi bir selamlama ile başla. Anonim kullanıcılara genel bir selamlama kullanabilirsin veya doğrudan cevaba geçebilirsin.",
        "Yönlendirme: Gerekirse, kullanıcıyı daha fazla bilgi için Sakarya Üniversitesi'nin resmi web sitesindeki ilgili sayfalara veya departmanlara yönlendirebilirsin."
    ]
    context_instructions = {
        "with_context": ("Aşağıda <CONTEXT> etiketi içinde verilen bilgilere dayanarak, kullanıcının sorusuna net, doğru ve öz bir cevap ver. Cevabında sadece sorulan konuyla ilgili bilgileri kullan, gereksiz detay verme. Eğer bağlamda cevap yoksa, bunu belirt ve spekülasyon yapma."),
        "without_context_specific_inquiry": ("Kullanıcının sorusu spesifik bir konuyla ilgili görünüyor ancak elimde bu konuda yardımcı olacak spesifik bir bağlam bilgisi (<CONTEXT>) yok. Nazikçe, bu konuda spesifik bilgiye sahip olmadığınızı veya yardımcı olamayacağınızı belirtin. Alternatif bilgi kaynakları önermeyin."),
        "without_context_general_inquiry": ("Kullanıcının sorusu genel bir konuyla ilgili ve elimde spesifik bir bağlam yok. Soruyu anladığınızı gösteren, genel ve yardımcı bir cevap vermeye çalışın. Sakarya Üniversitesi ile ilgiliyse, genel bilgi verebilirsiniz. Bilmiyorsanız, bilmediğinizi belirtin."),
        "clarification_needed": ("Kullanıcının sorusu belirsiz veya çok geniş. Cevap vermek için daha fazla detaya veya netleştirmeye ihtiyacınız olduğunu nazikçe belirtin. Kullanıcıdan sorusunu daha spesifik hale getirmesini isteyin."),
        "out_of_scope": ("Kullanıcının sorusu Sakarya Üniversitesi yönetmelikleri veya genel üniversite işleyişi dışında bir konuyla ilgili. Bu konuda yardımcı olamayacağınızı nazikçe belirtin ve ana uzmanlık alanınızın üniversite yönetmelikleri olduğunu hatırlatın.")
    }
    
    system_message_list = [base_persona]
    system_message_list.extend(general_instructions)
    msgs = []
    ret_ctx_resp = "İlgili yönetmelik bilgisi bulunamadı."

    if ctx_text:
        chosen_instruction = context_instructions["with_context"]
        final_system_content = "\n\n".join(system_message_list) + "\n\n" + chosen_instruction + f"\n\n<CONTEXT>\n{ctx_text}\n</CONTEXT>"
        ret_ctx_resp = ctx_text
    else: # Bağlam yoksa
        current_query_lower = query_data.query.lower()
        if len(query_data.query.split()) < 3 or "?" not in query_data.query: # Çok kısa veya soru değilse
             chosen_instruction = context_instructions["clarification_needed"]
        elif any(k in current_query_lower for k in ["nedir","nasıl","ne zaman","nerede","kimdir","kim","ne","kaç","hangi"]): # Soru kelimeleri içeriyorsa
            chosen_instruction = context_instructions["without_context_general_inquiry"]
        else: # Daha spesifik bir ifade gibi duruyorsa
            chosen_instruction = context_instructions["without_context_specific_inquiry"]
        final_system_content = "\n\n".join(system_message_list) + "\n\n" + chosen_instruction
        sources = [] # Bağlam yoksa kaynak da yok

    msgs.append({"role": "system", "content": final_system_content})
    
    is_first_turn = not query_data.history or len(query_data.history) == 0
    if query_data.history:
        for hist_msg in query_data.history:
            msgs.append({"role": hist_msg.role, "content": hist_msg.content})
    msgs.append({"role": "user", "content": query_data.query})

    try:
        prompt = current_hf_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception as e_template:
        print(f"{log_prefix}Chat template apply hatası: {e_template}")
        prompt = f"System: {final_system_content}\nUser: {query_data.query}\nAssistant:" # Basit bir fallback

    print(f"{log_prefix}DEBUG: Oluşturulan Prompt (ilk 300 karakter):\n{prompt[:300]}\n...")

    # Tokenizer'ın max_length'i, modelin max_new_tokens'ını da hesaba katmalı
    # Prompt zaten token içerdiği için, kalan yer max_new_tokens için olmalı.
    # N_CTX_HF modelin toplam bağlam penceresi olmalı.
    # inputs = current_hf_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=N_CTX_HF - query_data.max_new_tokens)
    # Daha güvenli: tokenizer'ın kendi max_length'ini kullanıp, üretilecek token sayısını da hesaba katalım.
    # Ancak Llama3 gibi modellerde genellikle N_CTX_HF = tokenizer.model_max_length olur.
    # Eğer prompt çok uzunsa ve max_new_tokens ile N_CTX_HF'i aşıyorsa, truncation=True bunu kırpacaktır.
    # Bu durumda, modelin üreteceği yer kalmayabilir. Bu yüzden max_length'i dikkatli ayarlamak lazım.
    # Şimdilik sizin kullandığınız mantıkla devam edelim ama bu bir potansiyel sorun noktası olabilir.
    effective_max_length = N_CTX_HF - query_data.max_new_tokens
    if effective_max_length <= 0: # Eğer üretilecek token sayısı bağlamı aşıyorsa
        print(f"{log_prefix}UYARI: max_new_tokens ({query_data.max_new_tokens}) N_CTX_HF ({N_CTX_HF}) için çok büyük. Kırpma sorun yaratabilir.")
        effective_max_length = N_CTX_HF // 2 # Güvenli bir değere çekelim
        
    inputs = current_hf_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=effective_max_length)


    try:
        target_device = next(current_hf_model.parameters()).device
    except StopIteration: # Modelin parametresi yoksa (çok olası değil ama)
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{log_prefix}DEBUG: Hedef cihaz: {target_device}")

    print(f"{log_prefix}DEBUG: --- Tokenizer Çıktısı Kontrolü ---")
    if "input_ids" not in inputs or inputs["input_ids"] is None or inputs["input_ids"].numel() == 0:
        print(f"{log_prefix}HATA: Tokenizer 'input_ids' üretmedi veya boş üretti!")
        return "Tokenizer hatası: Girdi işlenemedi.", ret_ctx_resp, sources
        
    print(f"{log_prefix}DEBUG: inputs['input_ids'] shape: {inputs['input_ids'].shape}")
    print(f"{log_prefix}DEBUG: inputs['input_ids'] dtype: {inputs['input_ids'].dtype}")
    print(f"{log_prefix}DEBUG: inputs['input_ids'] (CPU) min value: {inputs['input_ids'].cpu().min().item()}")
    print(f"{log_prefix}DEBUG: inputs['input_ids'] (CPU) max value: {inputs['input_ids'].cpu().max().item()}")

    if "attention_mask" not in inputs or inputs["attention_mask"] is None:
        print(f"{log_prefix}UYARI: Tokenizer 'attention_mask' üretmedi. Otomatik oluşturuluyor.")
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"]) # Basit bir attention mask

    print(f"{log_prefix}DEBUG: inputs['attention_mask'] shape: {inputs['attention_mask'].shape}")
    print(f"{log_prefix}DEBUG: inputs['attention_mask'] dtype: {inputs['attention_mask'].dtype}")
    print(f"{log_prefix}DEBUG: inputs['attention_mask'] (CPU) unique values: {torch.unique(inputs['attention_mask'].cpu())}")
    print(f"{log_prefix}DEBUG: --- Kontrol Sonu ---")

    input_ids_dev = inputs["input_ids"].to(target_device)
    attn_mask_dev = inputs["attention_mask"].to(target_device)
    print(f"{log_prefix}DEBUG: input_ids_dev ve attn_mask_dev GPU'ya taşındı.")

    gen_kwargs = {
        "max_new_tokens": query_data.max_new_tokens,
        "pad_token_id": current_hf_tokenizer.eos_token_id if current_hf_tokenizer.eos_token_id is not None else current_hf_tokenizer.pad_token_id,
        "eos_token_id": current_hf_tokenizer.eos_token_id,
        # Llama modelleri için genellikle bos_token_id de önemlidir, ancak generate'de doğrudan kullanılmaz.
    }
    # Temperature 0.0 ise (veya çok yakınsa), bu greedy decoding anlamına gelir.
    # Hugging Face'in generate fonksiyonu temperature=0.0 ve do_sample=True'yu desteklemez.
    # Eğer temperature çok düşükse, do_sample=False olmalı (greedy).
    if query_data.temperature <= 0.01: # Neredeyse sıfırsa greedy yap
        gen_kwargs["do_sample"] = False
        # gen_kwargs.pop("temperature", None) # Gerekirse temperature'ı kaldır
        # gen_kwargs.pop("top_p", None)       # Gerekirse top_p'yi kaldır
    else:
        gen_kwargs.update({
            "temperature": query_data.temperature,
            "do_sample": True,
            "top_p": query_data.top_p,
            # "top_k": query_data.top_k, # Eğer kullanılıyorsa
        })

    if query_data.repetition_penalty and query_data.repetition_penalty > 1.0: # 1.0 etkisizdir
        gen_kwargs["repetition_penalty"] = query_data.repetition_penalty

    print(f"{log_prefix}DEBUG: Generation kwargs: {gen_kwargs}")

    try:
        with torch.no_grad():
            print(f"{log_prefix}DEBUG: --- Logit ve Probs Kontrolü Başlıyor (generate öncesi) ---")
            # Modelden bir adım logit alalım (generate içindeki ilk adıma benzer)
            model_inputs_for_logit_check = {"input_ids": input_ids_dev, "attention_mask": attn_mask_dev}
            outputs_for_logit_check = current_hf_model(**model_inputs_for_logit_check)
            
            next_token_logits = outputs_for_logit_check.logits[0, -1, :]
            print(f"{log_prefix}DEBUG: next_token_logits.dtype: {next_token_logits.dtype}")
            print(f"{log_prefix}DEBUG: next_token_logits shape: {next_token_logits.shape}")

            if torch.isnan(next_token_logits).any():
                print(f"{log_prefix}DEBUG: ERROR - next_token_logits (ham) NaN içeriyor!")
            if torch.isinf(next_token_logits).any():
                print(f"{log_prefix}DEBUG: ERROR - next_token_logits (ham) Inf içeriyor!")
            
            # Sıcaklık ve diğer logit işlemcilerini (varsa) uygula
            # Bu kısım generate fonksiyonunun içini taklit etmeye çalışır, ama karmaşıktır.
            # Şimdilik sadece temperature'ı basitçe uygulayalım.
            current_temperature_for_debug = gen_kwargs.get("temperature", 1.0)
            logits_after_processing_for_debug = next_token_logits
            if gen_kwargs.get("do_sample", False) and current_temperature_for_debug > 0.01: # Sadece sampling ve geçerli temp varsa
                 logits_after_processing_for_debug = logits_after_processing_for_debug / current_temperature_for_debug
                 print(f"{log_prefix}DEBUG: Logits (debug) temperature ({current_temperature_for_debug}) ile bölündü.")
            else:
                 print(f"{log_prefix}DEBUG: Logits (debug) temperature uygulanmadı (greedy veya temp<=0.01).")


            probs_for_debug = torch.nn.functional.softmax(logits_after_processing_for_debug, dim=-1)
            print(f"{log_prefix}DEBUG: probs_for_debug shape: {probs_for_debug.shape}")
            print(f"{log_prefix}DEBUG: probs_for_debug dtype: {probs_for_debug.dtype}")


            if torch.isnan(probs_for_debug).any():
                print(f"{log_prefix}DEBUG: ERROR - probs_for_debug NaN içeriyor!")
            if torch.isinf(probs_for_debug).any():
                print(f"{log_prefix}DEBUG: ERROR - probs_for_debug Inf içeriyor!")
            if (probs_for_debug < 0).any():
                print(f"{log_prefix}DEBUG: ERROR - probs_for_debug negatif değer içeriyor!")
            
            if probs_for_debug.numel() > 0 and not (torch.isnan(probs_for_debug).any() or torch.isinf(probs_for_debug).any()):
                try:
                    target_sum = torch.tensor(1.0, device=probs_for_debug.device, dtype=probs_for_debug.dtype)
                    if not torch.allclose(probs_for_debug.sum(dim=-1), target_sum, atol=1e-3):
                        print(f"{log_prefix}DEBUG: UYARI - probs_for_debug toplamı ({probs_for_debug.sum(dim=-1).item()}) 1.0 değil! (dtype: {probs_for_debug.dtype})")
                except Exception as e_sum_check:
                    print(f"{log_prefix}DEBUG: probs_for_debug.sum() kontrolünde hata: {e_sum_check}")
            
            print(f"{log_prefix}DEBUG: --- Logit ve Probs Kontrolü Bitti ---")

            print(f"{log_prefix}DEBUG: current_hf_model.generate çağrılıyor...")
            out_seqs = current_hf_model.generate(
                input_ids=input_ids_dev,
                attention_mask=attn_mask_dev,
                **gen_kwargs
            )
            print(f"{log_prefix}DEBUG: current_hf_model.generate tamamlandı.")
    
    except RuntimeError as e_cuda: # Özellikle CUDA hatalarını yakala
        print(f"{log_prefix}CUDA Runtime Hatası (generate veya öncesi logit kontrolü sırasında):")
        traceback.print_exc() # CUDA hataları için tam traceback önemli
        # CUDA_LAUNCH_BLOCKING ayarlandıysa, traceback daha anlamlı olabilir.
        # Streamlit'e döndürülecek hata mesajını daha kullanıcı dostu yapabiliriz.
        # raise HTTPException(status_code=500, detail=f"Model yanıt üretirken GPU hatası oluştu: {str(e_cuda)}")
        raise e_cuda # Şimdilik orijinal hatayı fırlat, loglarda görünsün
    except Exception as e_gen:
        print(f"{log_prefix}Beklenmedik Genel Hata (generate veya öncesi logit kontrolü sırasında): {e_gen}")
        traceback.print_exc()
        # raise HTTPException(status_code=500, detail=f"Model yanıt üretirken beklenmedik bir hata oluştu: {str(e_gen)}")
        raise e_gen

    # Çıktıyı decode et
    # input_ids'nin batch boyutunu al (genellikle 1)
    batch_size = inputs["input_ids"].shape[0]
    output_sequences = []
    for i in range(batch_size):
        # Her bir sekans için, input kısmını atlayarak sadece üretilen tokenları al
        generated_tokens = out_seqs[i][inputs["input_ids"].shape[1]:]
        decoded_text = current_hf_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        output_sequences.append(decoded_text)
    
    answer = output_sequences[0] if output_sequences else "" # Sadece ilk batchedeki sonucu alıyoruz
    
    final_answer = answer
    # Selamlama mantığı (isteğe bağlı, basitleştirilebilir)
    if is_first_turn and username:
        greeting_base = "Merhaba"
        greeting = f"{greeting_base} {username.capitalize()}"
        answer_lower = answer.lower()
        greeting_base_lower = greeting_base.lower()
        if not (answer_lower.startswith(greeting_base_lower) or \
                answer_lower.startswith("selam") or \
                answer_lower.startswith("iyi günler")):
            final_answer = f"{greeting}! {answer}"

    print(f"{log_prefix}Üretilen Cevap (ilk 100kr): {final_answer[:100]}...")
    return final_answer, ret_ctx_resp, sources

# --- API Endpointleri ---
@app.get("/health", response_model=HealthStatus, summary="API ve Model Durumunu Kontrol Et")
async def health_check():
    status="ok"
    model_p = MODEL_PATH if os.path.isdir(MODEL_PATH) else f"Bulunamadı: {MODEL_PATH}"
    db_p = DB_PATH if os.path.exists(DB_PATH) else f"Bulunamadı: {DB_PATH}"
    model_dt = str(llm_model.dtype) if llm_model and hasattr(llm_model, 'dtype') else "Yüklenmedi"

    if not is_llm_loaded or not is_db_loaded: status = "partial_error"
    if not is_llm_loaded and not is_db_loaded: status = "error"
    return HealthStatus(status=status, llm_loaded=is_llm_loaded, db_loaded=is_db_loaded, model_path=model_p, model_dtype=model_dt, db_path=db_p)

@app.post("/chat", response_model=ChatResponseAPI, summary="Giriş Yapmış Kullanıcı için Sohbet")
async def chat_endpoint(
    query_data: ChatQuery,
    model_tokenizer_tuple: tuple = Depends(get_hf_model_and_tokenizer_dependency),
    db_path_val: str = Depends(get_db_path_dependency),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    current_llm_model, current_llm_tokenizer = model_tokenizer_tuple
    username = current_user.get("username")
    try:
        model_answer, retrieved_context, sources = await _generate_chat_response(
            query_data, current_llm_model, current_llm_tokenizer, db_path_val,
            is_anonymous=False, username=username
        )
        returned_chat_id = save_chat_message( # Bu fonksiyon string chat_id döndürmeli
            username=username,
            user_message=query_data.query,
            bot_response=model_answer,
            retrieved_docs=[s for s in sources if s], # Boş stringleri filtrele
            chat_id=query_data.current_chat_id
        )
        return ChatResponseAPI(
            model_answer=model_answer, retrieved_context=retrieved_context,
            sources=sources, chat_id=returned_chat_id
        )
    except HTTPException as http_exc:
        raise http_exc # Zaten HTTP hatasıysa tekrar fırlat
    except Exception as e_chat:
        print(f"Chat endpoint hatası (Kullanıcı: {username}): {e_chat}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Sunucuda bir hata oluştu: {str(e_chat)}")

@app.post("/anon-chat", response_model=ChatResponseAPI, summary="Anonim Kullanıcı için Sohbet")
async def anon_chat_endpoint(
    query_data: ChatQuery,
    model_tokenizer_tuple: tuple = Depends(get_hf_model_and_tokenizer_dependency),
    db_path_val: str = Depends(get_db_path_dependency)
):
    current_llm_model, current_llm_tokenizer = model_tokenizer_tuple
    try:
        model_answer, retrieved_context, sources = await _generate_chat_response(
            query_data, current_llm_model, current_llm_tokenizer, db_path_val,
            is_anonymous=True, username=None
        )
        return ChatResponseAPI(
            model_answer=model_answer, retrieved_context=retrieved_context,
            sources=sources, chat_id=None # Anonim için chat_id yok
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e_anon:
        print(f"[Anonim] Chat endpoint hatası: {e_anon}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Sunucuda bir hata oluştu: {str(e_anon)}")


@app.post("/upload-pdf", response_model=UploadResponse, summary="PDF Dosyalarını Yükle ve İndeksle")
async def upload_pdf_endpoint(files: List[UploadFile] = File(...), db_path_val: str = Depends(get_db_path_dependency)):
    global faiss_index, documents, ids, is_db_loaded
    processed_files_count = 0
    total_added_chunks = 0
    all_chunk_dicts: List[Dict[str, Any]] = []
    errors: List[str] = []

    with tempfile.TemporaryDirectory() as temp_dir:
        json_temp_path = os.path.join(temp_dir, "chunks_to_add.json")
        for uploaded_file in files:
            if not uploaded_file.filename or not uploaded_file.filename.lower().endswith(".pdf"):
                errors.append(f"'{uploaded_file.filename or 'Bilinmeyen dosya'}': Geçersiz dosya tipi, sadece PDF kabul edilir.")
                continue
            
            temp_pdf_path = os.path.join(temp_dir, uploaded_file.filename)
            try:
                with open(temp_pdf_path, "wb") as buffer:
                    shutil.copyfileobj(uploaded_file.file, buffer)
                
                extracted_text = extract_text_from_pdf(temp_pdf_path)
                if not extracted_text or not extracted_text.strip():
                    errors.append(f"'{uploaded_file.filename}': Dosyadan metin çıkarılamadı veya boş metin.")
                    continue
                
                cleaned_text = clean_text(extracted_text)
                text_chunks = simple_chunk_text(cleaned_text, chunk_size=1000, chunk_overlap=100) # Ayarlanabilir
                if not text_chunks:
                    errors.append(f"'{uploaded_file.filename}': Metin parçalanamadı.")
                    continue
                
                chunk_dicts_for_file = convert_chunks_to_dict(text_chunks, uploaded_file.filename)
                all_chunk_dicts.extend(chunk_dicts_for_file)
                processed_files_count += 1
            except Exception as e:
                errors.append(f"'{uploaded_file.filename}' dosyası işlenirken hata oluştu: {str(e)}")
                traceback.print_exc()
            finally:
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)
                uploaded_file.file.close()

        if not all_chunk_dicts:
            if not errors: errors.append("Yüklenecek geçerli içerik bulunamadı.")
            # raise HTTPException(status_code=400, detail=f"Dosya işleme başarısız: {'; '.join(errors)}")
            # Hata fırlatmak yerine, mesajla dönelim
            return UploadResponse(message=f"Dosya işleme başarısız: {'; '.join(errors)}", processed_files=0, added_chunks=0, errors=errors)

        try:
            with open(json_temp_path, 'w', encoding='utf-8') as f:
                json.dump(all_chunk_dicts, f, ensure_ascii=False, indent=2)
            
            # add_to_faiss_index True/False dönmeli
            if not add_to_faiss_index(json_temp_path, db_path_val, model_name=FAISS_MODEL):
                raise Exception("FAISS indeksine ekleme başarısız oldu.") # Daha spesifik hata
            
            total_added_chunks = len(all_chunk_dicts)
            
            # DB'yi yeniden yükle
            faiss_index, documents, ids = load_vector_db(db_path_val)
            is_db_loaded = bool(faiss_index is not None and documents is not None and ids is not None)
            if not is_db_loaded:
                errors.append("Veritabanı, yeni eklenenlerden sonra yeniden yüklenemedi.")

            message = f"{processed_files_count} PDF dosyası işlendi, toplam {total_added_chunks} metin parçası eklendi."
            if errors: message += " Bazı sorunlar oluştu."
            return UploadResponse(message=message, processed_files=processed_files_count, added_chunks=total_added_chunks, errors=errors)
        
        except Exception as e:
            error_message = f"Veritabanına ekleme veya yükleme sırasında genel hata: {str(e)}"
            print(error_message)
            traceback.print_exc()
            errors.append(error_message)
            # raise HTTPException(status_code=500, detail=error_message)
            return UploadResponse(message=error_message, processed_files=processed_files_count, added_chunks=0, errors=errors)


@app.post("/register", response_model=dict, summary="Yeni Kullanıcı Kaydı")
async def register_new_user_endpoint(user_data: UserRegister):
    try:
        success, message = create_user(username=user_data.username, email=user_data.email, password=user_data.password)
        if not success:
            raise HTTPException(status_code=400, detail=message or "Kullanıcı adı veya e-posta zaten kullanımda.")
        return {"success": True, "message": message or "Kullanıcı başarıyla oluşturuldu."}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Kullanıcı kayıt sırasında beklenmedik hata: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Sunucu tarafında bir kayıt hatası oluştu.")

@app.post("/token", response_model=Token, summary="Giriş Yap ve Access Token Al")
async def login_for_access_token_endpoint(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password) # Bu dict veya None döndürmeli
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Hatalı kullanıcı adı veya şifre.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user["username"]}) # Sadece username'i sub'a koyuyoruz
    return Token(access_token=access_token, token_type="bearer")

@app.get("/users/me", response_model=UserInfo, summary="Mevcut Kullanıcı Bilgilerini Al")
async def read_current_user_info_endpoint(current_user: Dict[str, Any] = Depends(get_current_user)):
    # get_current_user'dan gelen dict'te email olup olmadığını kontrol etmemiz gerekebilir
    # Şimdilik sadece username'i alıyoruz, mongo_db_helper.py'deki authenticate_user
    # ve get_current_user fonksiyonları email bilgisini de içeriyorsa burası güncellenebilir.
    return UserInfo(username=current_user["username"], email=current_user.get("email")) # email opsiyonel

@app.get("/chat-history", response_model=List[MDB_ChatHistory], summary="Kullanıcının Sohbet Oturumlarını Listele")
async def list_user_chat_history_endpoint(current_user: Dict[str, Any] = Depends(get_current_user)):
    try:
        username = current_user["username"]
        chat_sessions: List[MDB_ChatHistory] = get_user_chats(username) # Bu Pydantic listesi döndürmeli
        return chat_sessions
    except Exception as e:
        print(f"Sohbet geçmişi alınırken hata (kullanıcı: {current_user.get('username')}): {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Sohbet geçmişi alınırken bir sunucu hatası oluştu: {str(e)}")

@app.get("/chat-history/{chat_id}", response_model=List[MDB_ChatMessageHistory], summary="Belirli Bir Sohbetin Mesajlarını Getir")
async def get_chat_message_history_endpoint(chat_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    try:
        username = current_user["username"]
        # Kullanıcının bu sohbete erişim yetkisi var mı kontrolü (opsiyonel ama iyi bir pratik)
        user_chats = get_user_chats(username)
        if not any(chat.chat_id == chat_id for chat in user_chats):
            raise HTTPException(status_code=403, detail="Bu sohbete erişim yetkiniz yok veya sohbet bulunamadı.")
        
        messages: List[MDB_ChatMessageHistory] = get_chat_messages(chat_id) # Bu Pydantic listesi döndürmeli
        return messages
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Sohbet mesajları alınırken hata (chat_id: {chat_id}, kullanıcı: {current_user.get('username')}): {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Sohbet mesajları alınırken bir sunucu hatası oluştu: {str(e)}")

@app.delete("/chat-history/{chat_id}", response_model=dict, summary="Belirli Bir Sohbeti Sil")
async def delete_chat_history_endpoint(chat_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    try:
        username = current_user["username"]
        success = delete_chat(chat_id, username) # Bu fonksiyon, kullanıcının sadece kendi sohbetini silebildiğini doğrulamalı
        if not success:
            raise HTTPException(status_code=404, detail="Sohbet bulunamadı veya silme işlemi için yetkiniz yok.")
        return {"success": True, "message": "Sohbet başarıyla silindi."}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Sohbet silinirken hata (chat_id: {chat_id}, kullanıcı: {current_user.get('username')}): {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Sohbet silinirken bir sunucu hatası oluştu: {str(e)}")

if __name__ == "__main__":
    host = os.environ.get("SAUCHAT_API_HOST", "0.0.0.0")
    port = int(os.environ.get("SAUCHAT_API_PORT", "8001")) # Portu 8001 olarak değiştirdim
    reload_enabled = os.environ.get("SAUCHAT_API_RELOAD", "true").lower() == "true"

    print(f"SAÜChat API sunucusu başlatılıyor: http://{host}:{port}")
    print(f"Model Yolu: {MODEL_PATH}")
    print(f"Veritabanı Yolu: {DB_PATH}")
    print(f"Yeniden Yükleme (Reload): {'Aktif' if reload_enabled else 'Devre Dışı'}")
    
    # Gunicorn ile çalıştırmak için bir yapılandırma örneği (production için daha iyi olabilir)
    # workers = int(os.environ.get("WEB_CONCURRENCY", 2)) # Örnek worker sayısı
    # uvicorn.run("api_server:app", host=host, port=port, reload=reload_enabled, workers=workers, log_level="info")
    
    uvicorn.run("api_server:app", host=host, port=port, reload=reload_enabled, log_level="info")