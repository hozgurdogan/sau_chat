import streamlit as st
import requests
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

# --- Sayfa Yapılandırması ---
# Logo dosyasının yolu (Streamlit uygulamasının çalıştığı dizine göre)
# Bu dosyanın Colab'da /content/sau_chat/logo.png gibi bir yerde olması lazım
# veya Drive'daysa /content/drive/MyDrive/..../logo.png gibi.
# Şimdilik, Streamlit uygulamasının olduğu dizinde "logo.png" olduğunu varsayalım.
# Eğer /content/sau_chat/logo.png ise:
page_icon_path = "/logo/logo.png" # VEYA "/content/sau_chat/logo.png"
                            # VEYA Drive'daki tam yolu, örn: "/content/drive/MyDrive/HasanProje/sau_chat/logo.png"

try:
    st.set_page_config(
        page_title="SAÜChat - Yönetmelik Asistanı",
        page_icon=page_icon_path, # Logo dosyasının yolu
        layout="wide",
        initial_sidebar_state="expanded" # Kenar çubuğu başlangıçta açık olsun
    )
except Exception as e:
    st.error(f"Sayfa ikonu yüklenirken hata: {e}. 'logo.png' dosyasının doğru yolda olduğundan emin olun.")
    st.set_page_config(
        page_title="SAÜChat - Yönetmelik Asistanı",
        page_icon="🎓", # Hata durumunda varsayılan emoji
        layout="wide",
        initial_sidebar_state="expanded"
    )


# --- API URL'leri ---
BASE_API_URL = os.environ.get("API_URL", "http://localhost:8001") # FastAPI sunucunuzun çalıştığı port
CHAT_API_URL = f"{BASE_API_URL}/chat" # Giriş yapmış kullanıcılar için
ANON_CHAT_API_URL = f"{BASE_API_URL}/anon-chat" # Anonim kullanıcılar için
UPLOAD_API_URL = f"{BASE_API_URL}/upload-pdf"
HEALTH_API_URL = f"{BASE_API_URL}/health"
REGISTER_API_URL = f"{BASE_API_URL}/register"
TOKEN_API_URL = f"{BASE_API_URL}/token"
CHAT_HISTORY_API_URL = f"{BASE_API_URL}/chat-history" # /chat-history/{chat_id} ve /chat-history/{chat_id} (DELETE) için de bu temel URL

# --- Ortam Değişkenleri (API sunucusu için değil, Streamlit için gerekiyorsa) ---
# os.environ["N_GPU_LAYERS"] = "-1" # Bu Streamlit tarafında değil, API sunucusu tarafında ayarlanmalı
# os.environ["USE_MLOCK"] = "1"   # Bu da API sunucusu tarafında

# --- API İstemci Fonksiyonları (Bir önceki mesajınızdaki gibi) ---
def check_api_health() -> Dict[str, Any]:
    try:
        response = requests.get(HEALTH_API_URL, timeout=5)
        if response.status_code == 200: return {"status": "success", "data": response.json()}
        else: return {"status": "error", "code": response.status_code, "detail": response.text}
    except requests.RequestException as e: return {"status": "error", "detail": f"Sunucu bağlantı hatası: {e}"}
    except Exception as e: return {"status": "error", "detail": f"Beklenmeyen hata: {e}"}

def send_chat_query_to_api(
    query: str,
    history: List[Dict[str, str]],
    current_chat_id: Optional[str],
    top_k: int,
    temperature: float,
    max_new_tokens: int,
    top_p: Optional[float],
    repetition_penalty: Optional[float]
) -> Dict[str, Any]:
    payload = {
        "query": query,
        "history": history,
        "current_chat_id": current_chat_id,
        "top_k": top_k,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty
    }
    headers = {}
    target_url = ANON_CHAT_API_URL # Varsayılan olarak anonim
    if st.session_state.get("access_token"):
        headers["Authorization"] = f"Bearer {st.session_state.access_token}"
        target_url = CHAT_API_URL # Giriş yapıldıysa /chat endpoint'i

    try:
        response = requests.post(target_url, json=payload, headers=headers, timeout=300)
        response.raise_for_status()
        return {"status": "success", "data": response.json()}
    except requests.exceptions.Timeout:
        return {"status": "error", "detail": "API isteği zaman aşımına uğradı. Lütfen tekrar deneyin."}
    except requests.exceptions.RequestException as e:
        error_detail = f"API bağlantı hatası: {e}."
        if e.response is not None:
            try: api_error = e.response.json().get("detail", e.response.text); error_detail += f" API Yanıtı: {api_error}"
            except json.JSONDecodeError: error_detail += f" API Yanıtı (JSON değil): {e.response.text}"
        return {"status": "error", "detail": error_detail}
    except Exception as e: return {"status": "error", "detail": f"Beklenmeyen hata: {e}"}

def upload_pdf_to_api(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> Dict[str, Any]:
    if not uploaded_files: return {"status": "error", "detail": "Yüklenecek dosya seçilmedi."}
    files_payload = [('files', (f.name, f, f.type)) for f in uploaded_files]
    headers = {}
    if st.session_state.get("access_token"): # PDF yükleme de yetkilendirme gerektirebilir
        headers["Authorization"] = f"Bearer {st.session_state.access_token}"
    try:
        response = requests.post(UPLOAD_API_URL, files=files_payload, headers=headers, timeout=300)
        response.raise_for_status()
        return {"status": "success", "data": response.json()}
    except requests.exceptions.Timeout: return {"status": "error", "detail": "Dosya yükleme isteği zaman aşımına uğradı."}
    except requests.exceptions.RequestException as e:
        error_detail = f"Dosya yükleme hatası: {e}."
        if e.response is not None:
            try: api_error = e.response.json().get("detail", e.response.text); error_detail += f" API Yanıtı: {api_error}"
            except json.JSONDecodeError: error_detail += f" API Yanıtı (JSON değil): {e.response.text}"
        return {"status": "error", "detail": error_detail}
    except Exception as e: return {"status": "error", "detail": f"Dosya yüklenirken beklenmeyen hata: {e}"}

def register_user(username: str, email: str, password: str) -> Dict[str, Any]:
    payload = {"username": username, "email": email, "password": password}
    try:
        response = requests.post(REGISTER_API_URL, json=payload, timeout=10)
        response.raise_for_status(); return {"status": "success", "data": response.json()}
    except requests.exceptions.RequestException as e:
        error_detail = f"Kayıt hatası: {e}."
        if e.response is not None:
            try: api_error = e.response.json().get("detail", e.response.text); error_detail += f" API Yanıtı: {api_error}"
            except json.JSONDecodeError: error_detail += f" API Yanıtı (JSON değil): {e.response.text}"
        return {"status": "error", "detail": error_detail}
    except Exception as e: return {"status": "error", "detail": f"Beklenmeyen hata: {e}"}

def login_user(username: str, password: str) -> Dict[str, Any]:
    data = {"username": username, "password": password}
    try:
        response = requests.post(TOKEN_API_URL, data=data, timeout=10)
        response.raise_for_status(); return {"status": "success", "data": response.json()}
    except requests.exceptions.RequestException as e:
        error_detail = f"Giriş hatası: {e}."
        if e.response is not None:
            try: api_error = e.response.json().get("detail", e.response.text); error_detail += f" API Yanıtı: {api_error}"
            except json.JSONDecodeError: error_detail += f" API Yanıtı (JSON değil): {e.response.text}"
        return {"status": "error", "detail": error_detail}
    except Exception as e: return {"status": "error", "detail": f"Beklenmeyen hata: {e}"}

def get_chat_history_from_api() -> Dict[str, Any]:
    if not st.session_state.get("access_token"): return {"status": "error", "detail": "Oturum açık değil"}
    headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
    try:
        response = requests.get(CHAT_HISTORY_API_URL, headers=headers, timeout=10)
        response.raise_for_status(); return {"status": "success", "data": response.json()}
    except requests.exceptions.RequestException as e:
        error_detail = f"Sohbet geçmişi alınırken hata: {e}."
        if e.response is not None:
            try: api_error = e.response.json().get("detail", e.response.text); error_detail += f" API Yanıtı: {api_error}"
            except json.JSONDecodeError: error_detail += f" API Yanıtı (JSON değil): {e.response.text}"
        return {"status": "error", "detail": error_detail}
    except Exception as e: return {"status": "error", "detail": f"Beklenmeyen hata: {e}"}

def get_chat_messages_from_api(chat_id: str) -> Dict[str, Any]:
    if not st.session_state.get("access_token"): return {"status": "error", "detail": "Oturum açık değil"}
    headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
    try:
        response = requests.get(f"{CHAT_HISTORY_API_URL}/{chat_id}", headers=headers, timeout=10)
        response.raise_for_status(); return {"status": "success", "data": response.json()}
    except requests.exceptions.RequestException as e:
        error_detail = f"Sohbet mesajları alınırken hata: {e}."
        if e.response is not None:
            try: api_error = e.response.json().get("detail", e.response.text); error_detail += f" API Yanıtı: {api_error}"
            except json.JSONDecodeError: error_detail += f" API Yanıtı (JSON değil): {e.response.text}"
        return {"status": "error", "detail": error_detail}
    except Exception as e: return {"status": "error", "detail": f"Beklenmeyen hata: {e}"}

def delete_chat_from_api(chat_id: str) -> Dict[str, Any]:
    if not st.session_state.get("access_token"): return {"status": "error", "detail": "Oturum açık değil"}
    headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
    try:
        response = requests.delete(f"{CHAT_HISTORY_API_URL}/{chat_id}", headers=headers, timeout=10)
        response.raise_for_status(); return {"status": "success", "data": response.json()}
    except requests.exceptions.RequestException as e:
        error_detail = f"Sohbet silinirken hata: {e}."
        if e.response is not None:
            try: api_error = e.response.json().get("detail", e.response.text); error_detail += f" API Yanıtı: {api_error}"
            except json.JSONDecodeError: error_detail += f" API Yanıtı (JSON değil): {e.response.text}"
        return {"status": "error", "detail": error_detail}
    except Exception as e: return {"status": "error", "detail": f"Beklenmeyen hata: {e}"}

# --- Yardımcı Fonksiyonlar ---
def format_timestamp(timestamp_str: Optional[str]) -> str:
    if not timestamp_str: return "Bilinmeyen Zaman"
    try:
        # MongoDB'den gelen datetime objesi olabilir veya ISO string
        if isinstance(timestamp_str, datetime):
            dt = timestamp_str
        else: # String ise parse et
            dt = datetime.fromisoformat(str(timestamp_str).replace('Z', '+00:00'))
        return dt.strftime("%d.%m.%y %H:%M") # Yılı 2 haneli yaptım
    except Exception:
        return str(timestamp_str) # Hata durumunda orijinali döndür

# --- Kullanıcı Girişi ve Oturum Yönetimi ---
if "access_token" not in st.session_state: st.session_state.access_token = None
if "username" not in st.session_state: st.session_state.username = None
if "is_logged_in" not in st.session_state: st.session_state.is_logged_in = False
if "messages" not in st.session_state: st.session_state.messages = [] # Aktif sohbet mesajları
if "chat_history_list" not in st.session_state: st.session_state.chat_history_list = [] # Sidebar için geçmiş listesi
if "current_chat_id" not in st.session_state: st.session_state.current_chat_id = None
if "show_chat_history" not in st.session_state: st.session_state.show_chat_history = True # Başlangıçta geçmiş açık

def do_login(username, password):
    login_result = login_user(username, password)
    if login_result["status"] == "success":
        st.session_state.access_token = login_result["data"]["access_token"]
        st.session_state.username = username # API'den dönen username'i kullanmak daha iyi olabilir
        st.session_state.is_logged_in = True
        st.session_state.messages = [] # Yeni giriş, mesajları temizle
        st.session_state.current_chat_id = None
        st.success(f"Hoş geldiniz, {st.session_state.username}!")
        load_user_chat_history() # Giriş yapınca geçmişi yükle
        st.rerun()
    else: st.error(f"Giriş başarısız: {login_result['detail']}")

def do_logout():
    st.session_state.access_token = None
    st.session_state.username = None
    st.session_state.is_logged_in = False
    st.session_state.messages = []
    st.session_state.chat_history_list = []
    st.session_state.current_chat_id = None
    st.info("Başarıyla çıkış yapıldı.")
    st.rerun()

def load_user_chat_history():
    if st.session_state.is_logged_in:
        history_result = get_chat_history_from_api()
        if history_result["status"] == "success":
            st.session_state.chat_history_list = history_result["data"]
        else:
            st.session_state.chat_history_list = []
            # st.warning(f"Sohbet geçmişi yüklenemedi: {history_result['detail']}") # Opsiyonel uyarı
    else:
        st.session_state.chat_history_list = []

# --- Ana Başlık ---
# st.title("🎓 SAÜChat: Yönetmelik Asistanı") # Sidebar'a taşıdık veya set_page_config ile ayarlandı
# st.markdown("Sakarya Üniversitesi yönetmelikleri hakkında sorularınızı sorun veya yeni PDF'ler ekleyin.")

# --- Sidebar ---
with st.sidebar:
    # Sidebar'ın en üstüne logo
    # Logo dosyasının yolu (streamlit_app.py ile aynı dizinde veya belirtilen yolda)
    # Örneğin, Colab'da /content/logo.png veya /content/sau_chat/logo.png ise:
    sidebar_logo_path = "logo.png" # VEYA "/content/sau_chat/logo.png"
    try:
        st.image(sidebar_logo_path, use_column_width=True) # use_column_width sidebar genişliğine sığdırır
    except Exception as e:
        st.error(f"Sidebar logo yüklenemedi: {e}. '{sidebar_logo_path}' yolunu kontrol edin.")

    st.title("SAÜChat") # Logodan sonra başlık

    if not st.session_state.is_logged_in:
        st.subheader("Oturum")
        login_tab, register_tab = st.tabs(["Giriş Yap", "Üye Ol"])
        with login_tab:
            with st.form("login_form_sidebar"):
                login_username = st.text_input("Kullanıcı Adı", key="sb_login_user")
                login_password = st.text_input("Şifre", type="password", key="sb_login_pass")
                if st.form_submit_button("Giriş"): do_login(login_username, login_password)
        with register_tab:
            with st.form("register_form_sidebar"):
                reg_username = st.text_input("Kullanıcı Adı", key="sb_reg_user")
                reg_email = st.text_input("E-posta", key="sb_reg_email")
                reg_password = st.text_input("Şifre", type="password", key="sb_reg_pass")
                reg_pass_confirm = st.text_input("Şifre (Tekrar)", type="password", key="sb_reg_pass_confirm")
                if st.form_submit_button("Üye Ol"):
                    if reg_password != reg_pass_confirm: st.error("Şifreler eşleşmiyor!")
                    elif len(reg_password) < 6: st.error("Şifre en az 6 karakter olmalıdır.")
                    else:
                        res = register_user(reg_username, reg_email, reg_password)
                        if res["status"] == "success": st.success("Kayıt başarılı! Giriş yapabilirsiniz.")
                        else: st.error(f"Kayıt başarısız: {res['detail']}")
    else: # Kullanıcı giriş yapmışsa
        st.success(f"Hoş geldiniz, {st.session_state.username}!")
        if st.button("Çıkış Yap", key="logout_sb_btn", type="primary"): do_logout()

        st.divider()
        # "Sohbet Geçmişi" başlığı ve açılır/kapanır buton
        history_col1, history_col2 = st.columns([3,1])
        with history_col1:
            st.subheader("Sohbet Geçmişi")
        with history_col2:
            if st.button("🔄", key="refresh_history_btn_icon", help="Geçmişi Yenile"):
                load_user_chat_history()
                st.rerun() # Yenileme sonrası arayüzü güncellemek için

        # Geçmişi göster/gizle durumu için checkbox (veya buton)
        # st.session_state.show_chat_history = st.toggle("Geçmişi Göster/Gizle", value=st.session_state.show_chat_history, key="toggle_history_sb")
        # Daha çok "çekmece" gibi olması için st.expander kullanılabilir ama sidebar'da tam istenen gibi olmayabilir.
        # Butonla açıp kapama daha iyi olabilir.

        if st.session_state.is_logged_in and st.session_state.show_chat_history:
            if not st.session_state.chat_history_list: # Eğer liste boşsa yüklemeyi dene
                load_user_chat_history()

            if st.session_state.chat_history_list:
                for chat_item in st.session_state.chat_history_list:
                    chat_id = chat_item.get("chat_id", "") # Pydantic modeli değilse get kullan
                    first_message = chat_item.get("first_message", "Başlıksız Sohbet")
                    timestamp = chat_item.get("timestamp", "")

                    display_title = f"{first_message[:30]}..." if len(first_message) > 30 else first_message
                    
                    item_col1, item_col2 = st.columns([0.8, 0.2]) # Genişlik oranları ayarlandı
                    with item_col1:
                        if st.button(f"💬 {display_title}", key=f"load_chat_{chat_id}", help=f"{format_timestamp(timestamp)}", use_container_width=True):
                            with st.spinner("Sohbet yükleniyor..."):
                                messages_res = get_chat_messages_from_api(chat_id)
                                if messages_res["status"] == "success":
                                    st.session_state.messages = [] # Önceki mesajları temizle
                                    for msg_data in messages_res["data"]:
                                        st.session_state.messages.append({"role": "user", "content": msg_data.get("user_message")})
                                        st.session_state.messages.append({"role": "assistant", "content": msg_data.get("bot_response")})
                                    st.session_state.current_chat_id = chat_id
                                    st.rerun()
                                else: st.error(f"Sohbet mesajları yüklenemedi: {messages_res['detail']}")
                    with item_col2:
                        if st.button("🗑️", key=f"delete_chat_{chat_id}", help="Sohbeti Sil"):
                            delete_res = delete_chat_from_api(chat_id)
                            if delete_res["status"] == "success":
                                st.success("Sohbet silindi.")
                                load_user_chat_history() # Listeyi yenile
                                if st.session_state.current_chat_id == chat_id: # Eğer aktif sohbet silindiyse
                                    st.session_state.messages = []
                                    st.session_state.current_chat_id = None
                                st.rerun()
                            else: st.error(f"Sohbet silinemedi: {delete_res['detail']}")
                st.markdown("---")
            else:
                st.caption("Sohbet geçmişi bulunmuyor.")
        
        # Yeni sohbet butonu (kenar çubuğunda da olabilir)
        if st.button("Yeni Sohbet Başlat", key="new_chat_sb_btn"):
            st.session_state.messages = []
            st.session_state.current_chat_id = None
            st.success("Yeni sohbet başlatıldı. Sorunuzu yazabilirsiniz.")
            st.rerun()


    st.divider()
    st.subheader("Sohbet Ayarları")
    # Session state'de yoksa varsayılan değerleri ata
    if "top_k" not in st.session_state: st.session_state.top_k = 3
    if "temperature" not in st.session_state: st.session_state.temperature = 0.1
    if "max_new_tokens" not in st.session_state: st.session_state.max_new_tokens = 768 # API'deki default ile aynı olmalı
    if "top_p" not in st.session_state: st.session_state.top_p = 0.9
    if "repetition_penalty" not in st.session_state: st.session_state.repetition_penalty = 1.1


    st.session_state.top_k = st.slider("Kaynak Belge Sayısı", 1, 10, st.session_state.top_k, 1, help="Yanıt için kaç adet ilgili belge kullanılacak?")
    st.session_state.temperature = st.slider("Yaratıcılık (Temperature)", 0.0, 1.0, st.session_state.temperature, 0.01, help="Düşük değerler daha kesin, yüksek değerler daha çeşitli yanıtlar üretir. 0.0 greedy anlamına gelir.")
    st.session_state.max_new_tokens = st.slider("Maks. Yanıt Uzunluğu (Token)", 50, N_CTX_HF // 2, st.session_state.max_new_tokens, 50, help="Modelin üreteceği maksimum token sayısı.")
    st.session_state.top_p = st.slider("Top P (Nucleus Sampling)", 0.0, 1.0, st.session_state.top_p, 0.01, help="Daha düşük değerler daha odaklı, yüksekler daha çeşitli yanıtlar üretir. Temperature > 0.0 iken etkilidir.")
    st.session_state.repetition_penalty = st.slider("Tekrar Cezası", 1.0, 2.0, st.session_state.repetition_penalty, 0.05, help="1.0 ceza yok. Daha yüksek değerler tekrarları azaltır.")


    # PDF Yükleme (Yönetici Yetkisi)
    if st.session_state.is_logged_in and st.session_state.username in ["admin", "hasan"]: # Kullanıcı adlarını güncelle
        st.divider()
        st.subheader("Yönetim Paneli")
        with st.expander("Yeni Yönetmelik Ekle (PDF)"):
            uploaded_pdf_files = st.file_uploader(
                "PDF Dosyalarını Seçin", type="pdf", accept_multiple_files=True, key="pdf_uploader_sb"
            )
            if uploaded_pdf_files:
                if st.button("Seçili PDF'leri İndeksle", key="upload_sb_btn"):
                    with st.spinner(f"{len(uploaded_pdf_files)} dosya işleniyor..."):
                        upload_res = upload_pdf_to_api(uploaded_pdf_files)
                    if upload_res["status"] == "success": st.success(upload_res["data"].get("message", "Başarılı."))
                    else: st.error(f"Yükleme Hatası: {upload_res['detail']}")

    st.divider()
    st.subheader("API Durumu")
    if st.button("Kontrol Et", key="health_check_sb_btn"):
        with st.spinner("API durumu kontrol ediliyor..."): health_res = check_api_health()
        if health_res["status"] == "success":
            data = health_res["data"]
            st.success("✅ API Aktif")
            st.caption(f"LLM: {'✅' if data.get('llm_loaded') else '❌'} | DB: {'✅' if data.get('db_loaded') else '❌'}")
            st.caption(f"Model: {data.get('model_path','N/A')}")
        else: st.error(f"❌ API Hatası: {health_res['detail']}")


# --- Ana İçerik Alanı ---
if not st.session_state.is_logged_in and not st.session_state.messages: # Sadece giriş yapılmamışsa ve ilk açılışsa
    st.info("Sohbet geçmişinizin kaydedilmesi, PDF yükleme ve tüm özelliklere erişim için lütfen kenar çubuğundan giriş yapın veya üye olun.")

# Geçmiş mesajları göster
for msg_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "details" in message and message["details"]:
            with st.expander("Modelin Kullandığı Bilgiler ve Kaynaklar", expanded=False): # Başlangıçta kapalı
                st.markdown("**İlgili Bilgiler (Context):**")
                st.markdown(f"> {message['details']['retrieved_context']}")
                st.markdown("**Kaynak Dosyalar:**")
                if message["details"]["sources"]:
                    for src_idx, source in enumerate(message["details"]["sources"]):
                        st.caption(f"- {os.path.basename(source)}" if source else "Bilinmeyen Kaynak")
                else:
                    st.caption("Bu yanıt için spesifik kaynak dosya kullanılmadı.")

# Kullanıcı girdisi
user_prompt = st.chat_input("Sorunuzu buraya yazın...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"): st.markdown(user_prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🧠 Düşünüyorum...")
        
        history_for_api = []
        if st.session_state.is_logged_in and st.session_state.current_chat_id:
            # Sadece mevcut sohbete ait mesajları geçmiş olarak gönder
            # Bu kısım, API'nin tüm mesajları alıp işlemesi yerine,
            # sadece mevcut oturumdaki mesajları istemesi durumunda daha anlamlı olur.
            # Şimdilik tüm st.session_state.messages'ı göndermiyoruz, API bunu kendi yönetmeli.
            # Eğer API, history parametresini önceki konuşmaları almak için kullanıyorsa,
            # st.session_state.messages'dan sondan bir öncekileri (mevcut user_prompt hariç)
            # uygun formatta göndermek gerekir.
            # Şimdilik ChatQuery'deki history: Optional[List[ChatMessageInput]] = Field(default_factory=list)
            # varsayılan boş liste olarak kalacak. API tarafı bunu DB'den çekiyor.
            pass


        api_res = send_chat_query_to_api(
            query=user_prompt,
            history=st.session_state.messages[:-1], # Son kullanıcı mesajı hariç önceki mesajlar
            current_chat_id=st.session_state.current_chat_id,
            top_k=st.session_state.top_k,
            temperature=st.session_state.temperature,
            max_new_tokens=st.session_state.max_new_tokens,
            top_p=st.session_state.top_p,
            repetition_penalty=st.session_state.repetition_penalty
        )

        if api_res["status"] == "success":
            res_data = api_res["data"]
            assistant_text = res_data.get("model_answer", "Yanıt alınamadı.")
            message_placeholder.markdown(assistant_text)
            
            # Yanıtı ve detayları session_state'e ekle
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_text,
                "details": { # Bu detayları API'den aldığımızı varsayıyoruz
                    "retrieved_context": res_data.get("retrieved_context", ""),
                    "sources": res_data.get("sources", [])
                }
            })
            # Eğer yeni bir sohbetse ve API bir chat_id döndürdüyse, onu kaydet
            if st.session_state.is_logged_in and not st.session_state.current_chat_id and res_data.get("chat_id"):
                st.session_state.current_chat_id = res_data["chat_id"]
                load_user_chat_history() # Yeni sohbet eklendi, geçmişi yenile
                st.rerun() # Sidebar'daki listeyi güncellemek için
            elif st.session_state.is_logged_in and st.session_state.current_chat_id:
                 # Var olan bir sohbetse ve mesaj eklendiyse de geçmişi yenileyebiliriz
                 # (eğer API tarafı her mesajı ayrı kaydetmiyorsa)
                 # Şimdilik bu kısmı atlayalım, yeni sohbet ID'si alma durumu daha önemli.
                 pass

        else:
            error_txt = f"Hata: {api_res['detail']}"
            message_placeholder.error(error_txt)
            st.session_state.messages.append({"role": "assistant", "content": error_txt})

# Footer
st.markdown("---")
st.caption(f"SAÜChat © {datetime.now().year} - Sakarya Üniversitesi")