import streamlit as st
import requests
import json
import os
from typing import Dict, Any, List
from datetime import datetime

# --- Sayfa Yapılandırması ---
st.set_page_config(
    page_title="SAÜChat - Yönetmelik Asistanı",
    page_icon="🎓",
    layout="wide"
)

# --- API URL'leri ---
# Bu URL'leri gerektiğinde ortam değişkenlerinden veya bir config dosyasından almak daha iyidir.
BASE_API_URL = os.environ.get("API_URL", "http://localhost:8000")
CHAT_API_URL = f"{BASE_API_URL}/chat"
UPLOAD_API_URL = f"{BASE_API_URL}/upload-pdf"
HEALTH_API_URL = f"{BASE_API_URL}/health"
REGISTER_API_URL = f"{BASE_API_URL}/register"
TOKEN_API_URL = f"{BASE_API_URL}/token"
CHAT_HISTORY_API_URL = f"{BASE_API_URL}/chat-history"

# --- Ortam Değişkenleri ---
# GPU öncelikle tercih edilsin
os.environ["N_GPU_LAYERS"] = "-1"  # -1 değeri tüm katmanları GPU'ya yükler
os.environ["USE_MLOCK"] = "1"  # GPU belleğini optimum kullanım için kilit

# --- API İstemci Fonksiyonları ---

def check_api_health() -> Dict[str, Any]:
    """API sağlık durumunu kontrol eder."""
    try:
        response = requests.get(HEALTH_API_URL, timeout=5) # Timeout ekle
        if response.status_code == 200:
            return {"status": "success", "data": response.json()}
        else:
            return {"status": "error", "code": response.status_code, "detail": response.text}
    except requests.RequestException as e:
        return {"status": "error", "detail": f"Sunucu bağlantı hatası: {e}"}
    except Exception as e:
        return {"status": "error", "detail": f"Beklenmeyen hata: {e}"}

def send_query_to_api(query: str, top_k: int, temperature: float, max_tokens: int) -> Dict[str, Any]:
    """Kullanıcı sorgusunu API'ye gönderir ve sonucu alır."""
    payload = {
        "query": query,
        "top_k": top_k,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    headers = {}
    if st.session_state.get("access_token"):
        headers["Authorization"] = f"Bearer {st.session_state.access_token}"
    
    try:
        response = requests.post(CHAT_API_URL, json=payload, headers=headers, timeout=300) # Daha uzun timeout (300 saniye)
        response.raise_for_status() # HTTP 2xx olmayan durumlar için hata fırlat
        return {"status": "success", "data": response.json()}
    except requests.exceptions.Timeout:
         return {"status": "error", "detail": "API isteği zaman aşımına uğradı. LLaMA modeli yanıt üretirken zaman aşımına uğradı. Lütfen tekrar deneyin veya daha kısa bir soru sorun."}
    except requests.exceptions.RequestException as e:
        error_detail = f"API bağlantı hatası: {e}."
        if e.response is not None:
            try:
                api_error = e.response.json().get("detail", e.response.text)
                error_detail += f" API Yanıtı: {api_error}"
            except json.JSONDecodeError:
                error_detail += f" API Yanıtı (JSON değil): {e.response.text}"
        return {"status": "error", "detail": error_detail}
    except Exception as e:
        return {"status": "error", "detail": f"Beklenmeyen hata: {e}"}

def upload_pdf_to_api(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> Dict[str, Any]:
    """Yüklenen PDF dosyalarını API'ye gönderir."""
    if not uploaded_files:
        return {"status": "error", "detail": "Yüklenecek dosya seçilmedi."}

    files_payload = []
    for uploaded_file in uploaded_files:
        # Dosyayı başa sar (önemli olabilir)
        uploaded_file.seek(0)
        files_payload.append(('files', (uploaded_file.name, uploaded_file, uploaded_file.type)))

    try:
        response = requests.post(UPLOAD_API_URL, files=files_payload, timeout=300) # Yükleme için daha uzun timeout
        response.raise_for_status()
        return {"status": "success", "data": response.json()}
    except requests.exceptions.Timeout:
         return {"status": "error", "detail": "Dosya yükleme isteği zaman aşımına uğradı."}
    except requests.exceptions.RequestException as e:
        error_detail = f"Dosya yükleme hatası: {e}."
        if e.response is not None:
            try:
                api_error = e.response.json().get("detail", e.response.text)
                error_detail += f" API Yanıtı: {api_error}"
            except json.JSONDecodeError:
                error_detail += f" API Yanıtı (JSON değil): {e.response.text}"
        return {"status": "error", "detail": error_detail}
    except Exception as e:
        return {"status": "error", "detail": f"Dosya yüklenirken beklenmeyen hata: {e}"}

def register_user(username: str, email: str, password: str) -> Dict[str, Any]:
    """Yeni kullanıcı kaydı yapar."""
    payload = {
        "username": username,
        "email": email,
        "password": password
    }
    try:
        response = requests.post(REGISTER_API_URL, json=payload, timeout=10)
        response.raise_for_status()
        return {"status": "success", "data": response.json()}
    except requests.exceptions.RequestException as e:
        error_detail = f"Kayıt hatası: {e}."
        if e.response is not None:
            try:
                api_error = e.response.json().get("detail", e.response.text)
                error_detail += f" API Yanıtı: {api_error}"
            except json.JSONDecodeError:
                error_detail += f" API Yanıtı (JSON değil): {e.response.text}"
        return {"status": "error", "detail": error_detail}
    except Exception as e:
        return {"status": "error", "detail": f"Beklenmeyen hata: {e}"}

def login_user(username: str, password: str) -> Dict[str, Any]:
    """Kullanıcı girişi yapar ve token alır."""
    data = {
        "username": username,
        "password": password
    }
    try:
        response = requests.post(
            TOKEN_API_URL, 
            data=data,  # form-data olarak gönder
            timeout=10
        )
        response.raise_for_status()
        return {"status": "success", "data": response.json()}
    except requests.exceptions.RequestException as e:
        error_detail = f"Giriş hatası: {e}."
        if e.response is not None:
            try:
                api_error = e.response.json().get("detail", e.response.text)
                error_detail += f" API Yanıtı: {api_error}"
            except json.JSONDecodeError:
                error_detail += f" API Yanıtı (JSON değil): {e.response.text}"
        return {"status": "error", "detail": error_detail}
    except Exception as e:
        return {"status": "error", "detail": f"Beklenmeyen hata: {e}"}

def get_chat_history() -> Dict[str, Any]:
    """Kullanıcının sohbet geçmişini getirir."""
    if not st.session_state.get("access_token"):
        return {"status": "error", "detail": "Oturum açık değil"}
    
    headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
    
    try:
        response = requests.get(CHAT_HISTORY_API_URL, headers=headers, timeout=10)
        response.raise_for_status()
        return {"status": "success", "data": response.json()}
    except requests.exceptions.RequestException as e:
        error_detail = f"Sohbet geçmişi alınırken hata: {e}."
        if e.response is not None:
            try:
                api_error = e.response.json().get("detail", e.response.text)
                error_detail += f" API Yanıtı: {api_error}"
            except json.JSONDecodeError:
                error_detail += f" API Yanıtı (JSON değil): {e.response.text}"
        return {"status": "error", "detail": error_detail}
    except Exception as e:
        return {"status": "error", "detail": f"Beklenmeyen hata: {e}"}

def get_chat_messages(chat_id: str) -> Dict[str, Any]:
    """Belirli bir sohbetin mesajlarını getirir."""
    if not st.session_state.get("access_token"):
        return {"status": "error", "detail": "Oturum açık değil"}
    
    headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
    
    try:
        response = requests.get(f"{CHAT_HISTORY_API_URL}/{chat_id}", headers=headers, timeout=10)
        response.raise_for_status()
        return {"status": "success", "data": response.json()}
    except requests.exceptions.RequestException as e:
        error_detail = f"Sohbet mesajları alınırken hata: {e}."
        if e.response is not None:
            try:
                api_error = e.response.json().get("detail", e.response.text)
                error_detail += f" API Yanıtı: {api_error}"
            except json.JSONDecodeError:
                error_detail += f" API Yanıtı (JSON değil): {e.response.text}"
        return {"status": "error", "detail": error_detail}
    except Exception as e:
        return {"status": "error", "detail": f"Beklenmeyen hata: {e}"}

def delete_chat(chat_id: str) -> Dict[str, Any]:
    """Belirli bir sohbeti siler."""
    if not st.session_state.get("access_token"):
        return {"status": "error", "detail": "Oturum açık değil"}
    
    headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
    
    try:
        response = requests.delete(f"{CHAT_HISTORY_API_URL}/{chat_id}", headers=headers, timeout=10)
        response.raise_for_status()
        return {"status": "success", "data": response.json()}
    except requests.exceptions.RequestException as e:
        error_detail = f"Sohbet silinirken hata: {e}."
        if e.response is not None:
            try:
                api_error = e.response.json().get("detail", e.response.text)
                error_detail += f" API Yanıtı: {api_error}"
            except json.JSONDecodeError:
                error_detail += f" API Yanıtı (JSON değil): {e.response.text}"
        return {"status": "error", "detail": error_detail}
    except Exception as e:
        return {"status": "error", "detail": f"Beklenmeyen hata: {e}"}

# --- Yardımcı Fonksiyonlar ---

def format_timestamp(timestamp_str):
    """Zaman damgasını formatlar"""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime("%d.%m.%Y %H:%M")
    except:
        return timestamp_str

# --- Kullanıcı Girişi ve Oturum Yönetimi ---

# Oturum durum değişkenlerini başlat
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "username" not in st.session_state:
    st.session_state.username = None
if "is_logged_in" not in st.session_state:
    st.session_state.is_logged_in = False

# Giriş yapma/çıkış yapma fonksiyonları
def do_login(username, password):
    login_result = login_user(username, password)
    if login_result["status"] == "success":
        st.session_state.access_token = login_result["data"]["access_token"]
        st.session_state.username = username
        st.session_state.is_logged_in = True
        st.success(f"Hoş geldiniz, {username}!")
        
        # Giriş yapıldığında sohbet geçmişini yükle
        history_result = get_chat_history()
        if history_result["status"] == "success":
            st.session_state.chat_history = history_result["data"]
        else:
            st.session_state.chat_history = []
            
        st.rerun()
    else:
        st.error(f"Giriş başarısız: {login_result['detail']}")

def do_logout():
    st.session_state.access_token = None
    st.session_state.username = None
    st.session_state.is_logged_in = False
    st.session_state.messages = []
    st.session_state.current_chat_id = None
    if 'chat_history' in st.session_state:
        del st.session_state.chat_history
    st.rerun()

# --- Başlık ---
st.title("🎓 SAÜChat: Yönetmelik Asistanı")
st.markdown("Sakarya Üniversitesi yönetmelikleri hakkında sorularınızı sorun veya yeni PDF'ler ekleyin.")

# --- Sidebar ---
with st.sidebar:
    st.image("https://www.sakarya.edu.tr/img/logo_tr.png", width=200)
    
    # Giriş/Kayıt Bölümü
    if not st.session_state.is_logged_in:
        st.subheader("Giriş Yap veya Üye Ol")
        tab1, tab2 = st.tabs(["Giriş", "Üye Ol"])
        
        with tab1:
            with st.form("login_form"):
                login_username = st.text_input("Kullanıcı Adı")
                login_password = st.text_input("Şifre", type="password")
                login_button = st.form_submit_button("Giriş Yap")
                
                if login_button:
                    do_login(login_username, login_password)
        
        with tab2:
            with st.form("register_form"):
                reg_username = st.text_input("Kullanıcı Adı", key="reg_username")
                reg_email = st.text_input("E-posta", key="reg_email")
                reg_password = st.text_input("Şifre", type="password", key="reg_password")
                reg_password_confirm = st.text_input("Şifre (Tekrar)", type="password")
                register_button = st.form_submit_button("Üye Ol")
                
                if register_button:
                    if reg_password != reg_password_confirm:
                        st.error("Şifreler eşleşmiyor!")
                    elif len(reg_password) < 6:
                        st.error("Şifre en az 6 karakter olmalıdır.")
                    else:
                        register_result = register_user(reg_username, reg_email, reg_password)
                        if register_result["status"] == "success":
                            st.success("Kayıt başarılı! Şimdi giriş yapabilirsiniz.")
                        else:
                            st.error(f"Kayıt başarısız: {register_result['detail']}")
    
    # Kullanıcı girişi yapıldıysa
    else:
        st.success(f"Hoş geldiniz, {st.session_state.username}!")
        if st.button("Çıkış Yap"):
            do_logout()
        
        # Sohbet Geçmişi
        st.subheader("Sohbet Geçmişi")
        
        # Sohbet geçmişini session_state'e ekle (ilk kez veya yenileme)
        if st.button("�� Geçmişi Yenile"):
            history_result = get_chat_history()
            if history_result["status"] == "success":
                st.session_state.chat_history = history_result["data"]
                st.success("Sohbet geçmişi yenilendi!")
            else:
                st.error(f"Geçmiş yüklenemedi: {history_result['detail']}")
                st.session_state.chat_history = []
        
        # İlk kez veya yeniden yükleme için geçmişi kontrol et
        if "chat_history" not in st.session_state:
            history_result = get_chat_history()
            if history_result["status"] == "success":
                st.session_state.chat_history = history_result["data"]
            else:
                st.session_state.chat_history = []
        
        # Geçmiş sohbetleri göster
        if hasattr(st.session_state, 'chat_history') and st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                # datetime objesi olabileceği için string'e çevir
                timestamp_str = str(chat["timestamp"])
                formatted_time = format_timestamp(timestamp_str)
                
                chat_title = chat["first_message"]
                col1, col2 = st.columns([4, 1])
                with col1:
                    # Sohbeti yükle butonu (başlık olarak göster)
                    if st.button(f"💬 {chat_title}", key=f"chat_{chat['chat_id']}", use_container_width=True):
                        with st.spinner("Sohbet yükleniyor..."):
                            messages_result = get_chat_messages(chat["chat_id"])
                            if messages_result["status"] == "success":
                                # Mesajları session_state'e yükle
                                st.session_state.messages = []
                                for msg in messages_result["data"]:
                                    st.session_state.messages.append({
                                        "role": "user",
                                        "content": msg["user_message"]
                                    })
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": msg["bot_response"]
                                    })
                                st.session_state.current_chat_id = chat["chat_id"]
                                st.rerun()  # Değişiklikleri göstermek için sayfayı yeniden yükle
                            else:
                                st.error(f"Sohbet yüklenemedi: {messages_result['detail']}")
                
                with col2:
                    st.caption(f"{formatted_time}")
                    if st.button("🗑️", key=f"del_{chat['chat_id']}", help="Sohbeti sil"):
                        delete_result = delete_chat(chat["chat_id"])
                        if delete_result["status"] == "success":
                            st.success("Sohbet silindi!")
                            # Geçmişi güncelle
                            history_result = get_chat_history()
                            if history_result["status"] == "success":
                                st.session_state.chat_history = history_result["data"]
                            st.rerun()
                        else:
                            st.error(f"Sohbet silinemedi: {delete_result['detail']}")
                
                # Her sohbetten sonra ince bir çizgi ekle
                st.markdown("---")
        else:
            st.info("Henüz sohbet geçmişiniz bulunmuyor.")
        
        st.divider()
    
    # Sohbet Ayarları
    st.subheader("Sohbet Ayarları")
    top_k = st.slider("Kaynak Belge Sayısı", 1, 10, 3, 1, help="Yanıt için kaç adet ilgili belge kullanılacak?")
    temperature = st.slider("Yaratıcılık", 0.0, 1.0, 0.1, 0.05, help="Düşük değerler daha kesin, yüksek değerler daha yaratıcı yanıtlar üretir.")
    max_tokens = st.slider("Maks. Yanıt Uzunluğu", 100, 2000, 512, 50, help="Modelin üreteceği maksimum kelime/token sayısı.")

    st.divider()

    # PDF Yükleme (yönetici yetkisiyle)
    if st.session_state.is_logged_in and st.session_state.username in ["admin", "mehmet", "ozgur", "beyza"]:  # Sadece belirli kullanıcılar
        st.subheader("Yeni Yönetmelik Ekle")
        uploaded_files = st.file_uploader(
            "PDF Dosyalarını Seçin",
            type="pdf",
            accept_multiple_files=True,
            help="İndekslenmesini istediğiniz bir veya daha fazla PDF dosyası yükleyin."
        )

        if uploaded_files:
            st.write(f"{len(uploaded_files)} dosya seçildi:")
            for f in uploaded_files:
                st.caption(f"- {f.name}") # Daha küçük yazı tipi

            if st.button("Seçili PDF'leri İndeksle", key="upload_button"):
                with st.spinner(f"{len(uploaded_files)} dosya işleniyor ve veritabanına ekleniyor... Bu işlem biraz sürebilir."):
                    upload_result = upload_pdf_to_api(uploaded_files)

                if upload_result["status"] == "success":
                    st.success(upload_result["data"].get("message", "Dosyalar başarıyla işlendi."))
                    st.info(f"İşlenen dosya sayısı: {upload_result['data'].get('processed_files', 'N/A')}")
                    st.info(f"Eklenen toplam parça: {upload_result['data'].get('added_chunks', 'N/A')}")
                    if upload_result["data"].get("errors"):
                        st.warning("Bazı dosyalarda hatalar oluştu:")
                        for err in upload_result["data"]["errors"]:
                            st.error(f"- {err}")
                else:
                    st.error(f"Yükleme Hatası: {upload_result['detail']}")

        st.divider()

    # API Sağlık Kontrolü
    st.subheader("API Durumu")
    if st.button("API Durumunu Kontrol Et", key="health_check_button"):
        with st.spinner("API durumu kontrol ediliyor..."):
            health_result = check_api_health()
        if health_result["status"] == "success":
            data = health_result["data"]
            st.success("✅ API sunucusu aktif!")
            llm_status = "✅ LLM Yüklü" if data.get("llm_loaded") else "❌ LLM Yüklü Değil"
            db_status = "✅ DB Yüklü" if data.get("db_loaded") else "❌ DB Yüklü Değil"
            st.markdown(f"{llm_status}\n{db_status}")
            st.caption(f"Model: {data.get('model_path', 'N/A')}")
            st.caption(f"Veritabanı: {data.get('db_path', 'N/A')}")
        else:
            st.error(f"❌ API Bağlantı Hatası: {health_result['detail']}")

# --- Ana İçerik Alanı ---
if not st.session_state.is_logged_in:
    st.info("Sohbet geçmişinizin kaydedilmesi ve tüm özelliklere erişim için lütfen giriş yapın veya üye olun.")

# --- Sohbet Arayüzü ---

# Oturum durumunu başlat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Geçmiş mesajları göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Asistan yanıtıysa ve ek bilgiler varsa göster
        if message["role"] == "assistant" and "details" in message:
            with st.expander("Detaylar ve Kaynaklar"):
                st.markdown("**İlgili Bilgiler:**")
                st.markdown(message["details"]["retrieved_context"])
                st.markdown("**Kaynaklar:**")
                if message["details"]["sources"]:
                    for source in message["details"]["sources"]:
                        st.caption(os.path.basename(source) if source else "Bilinmeyen")
                else:
                    st.caption("Kaynak bulunamadı.")

# Yeni sohbet başlatma butonu
if st.session_state.is_logged_in:
    if st.button("🆕 Yeni Sohbet Başlat", key="new_chat_button", type="primary"):
        st.session_state.messages = []
        st.session_state.current_chat_id = None
        # Sohbet geçmişini güncelle
        history_result = get_chat_history()
        if history_result["status"] == "success":
            st.session_state.chat_history = history_result["data"]
        st.rerun()

# Kullanıcı girdisi al
user_query = st.chat_input("Sorunuzu buraya yazın...")

if user_query:
    # Kullanıcı mesajını ekle ve göster
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # API'ye sorguyu gönder ve yanıtı bekle
    with st.chat_message("assistant"):
        # Eğer giriş yapılmamışsa ve fazla mesaj gönderilmişse uyar
        if not st.session_state.is_logged_in and len(st.session_state.messages) > 10:
            st.warning("Sınırsız sohbet için lütfen giriş yapın. Giriş yapmadığınız sürece sohbet geçmişiniz kaybolabilir.")
            
        with st.spinner("Yanıtınız hazırlanıyor..."):
            # Kullanıcı giriş yapmış mı kontrol et
            if st.session_state.is_logged_in:
                api_response = send_query_to_api(user_query, top_k, temperature, max_tokens)
            else:
                # Giriş yapılmadıysa, anonim kullanıcı olarak istek gönder
                payload = {
                    "query": user_query,
                    "top_k": top_k,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                try:
                    response = requests.post(f"{BASE_API_URL}/anon-chat", json=payload, timeout=300)
                    response.raise_for_status()
                    api_response = {"status": "success", "data": response.json()}
                except requests.exceptions.RequestException as e:
                    error_detail = f"API bağlantı hatası: {e}."
                    if e.response is not None:
                        try:
                            api_error = e.response.json().get("detail", e.response.text)
                            error_detail += f" API Yanıtı: {api_error}"
                        except json.JSONDecodeError:
                            error_detail += f" API Yanıtı (JSON değil): {e.response.text}"
                    api_response = {"status": "error", "detail": error_detail}
                except Exception as e:
                    api_response = {"status": "error", "detail": f"Beklenmeyen hata: {e}"}

        if api_response["status"] == "success":
            response_data = api_response["data"]
            assistant_response = response_data.get("model_answer", "Üzgünüm, bir yanıt alamadım.")
            retrieved_context = response_data.get("retrieved_context", "Bağlam bilgisi alınamadı.")
            sources = response_data.get("sources", [])

            # Yanıtı göster
            st.markdown(assistant_response)

            # Yanıtı ve detayları oturum durumuna ekle
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_response,
                "details": {
                    "retrieved_context": retrieved_context,
                    "sources": sources
                }
            })

            # 3. Sohbet geçmişine kaydet
            try:
                username = st.session_state.username
                chat_id = st.session_state.get("current_chat_id", None)  # Eğer varsa mevcut sohbet ID'sini al
                save_chat_message(
                    username=username,
                    user_message=user_query,
                    bot_response=assistant_response,
                    retrieved_docs=[s for s in sources if s],  # Boş kaynak olmadığından emin ol
                    chat_id=chat_id  # Sohbet ID'sini ekle, None ise yeni sohbet oluşturulur
                )
                
                # Eğer bu ilk mesajsa ve yeni bir sohbet oluşturulduysa, ID'yi al
                if not chat_id:
                    # Sohbet geçmişini güncelle
                    history_result = get_chat_history()
                    if history_result["status"] == "success":
                        # En son eklenen sohbetin ID'sini al
                        if history_result["data"]:
                            st.session_state.current_chat_id = history_result["data"][0]["chat_id"]
                             
                # Her mesaj sonrası sohbet geçmişi listesini güncelle
                history_result = get_chat_history()
                if history_result["status"] == "success":
                    st.session_state.chat_history = history_result["data"]
            except Exception as e:
                print(f"Sohbet geçmişi kaydedilirken hata: {e}")
                # Geçmiş kaydedilemese bile, yanıt dönmeye devam et

        else:
            # Hata mesajını göster
            error_message = f"Hata: {api_response['detail']}"
            st.error(error_message)
            # Hata mesajını oturum durumuna ekle (içerik olarak)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

# Footer
st.markdown("---")
st.caption("SAÜChat © 2025 - Sakarya Üniversitesi")