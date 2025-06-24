import streamlit as st
import requests
import json
import os
from typing import Dict, Any, List, Optional # Optional ekledim, çünkü send_query_to_api'de kullanılıyor olabilir
from datetime import datetime

# --- Logo Dosyasının Yolu ---
# Bu yolu kendi logo dosyanızın konumuna göre güncelleyin.
# Eğer streamlit_app.py ile aynı dizindeyse "logo.png" yeterlidir.
# Colab'da çalıştırıyorsanız ve dosya /content/sau_chat/logo.png ise:
# LOGO_FILE_PATH = "/content/sau_chat/logo.png"
LOGO_FILE_PATH = "/content/sau_chat/logo/logo.png"

# --- Sayfa Yapılandırması ---
try:
    st.set_page_config(
        page_title="SAÜChat - Yönetmelik Asistanı",
        page_icon=LOGO_FILE_PATH,  # DEĞİŞİKLİK BURADA
        layout="wide"
    )
except Exception as e:
    st.error(f"Sayfa ikonu '{LOGO_FILE_PATH}' yüklenirken hata: {e}. Varsayılan ikon kullanılacak.")
    st.set_page_config(
        page_title="SAÜChat - Yönetmelik Asistanı",
        page_icon="🎓", # Hata durumunda varsayılan
        layout="wide"
    )

# --- API URL'leri ---
# Bu URL'leri gerektiğinde ortam değişkenlerinden veya bir config dosyasından almak daha iyidir.
BASE_API_URL = os.environ.get("API_URL", "http://localhost:8001")
CHAT_API_URL = f"{BASE_API_URL}/chat" # Bu, giriş yapmış kullanıcılar için olmalı
ANON_CHAT_API_URL = f"{BASE_API_URL}/anon-chat" # Anonim kullanıcılar için
UPLOAD_API_URL = f"{BASE_API_URL}/upload-pdf"
HEALTH_API_URL = f"{BASE_API_URL}/health"
REGISTER_API_URL = f"{BASE_API_URL}/register"
TOKEN_API_URL = f"{BASE_API_URL}/token"
CHAT_HISTORY_API_URL = f"{BASE_API_URL}/chat-history"

# --- Ortam Değişkenleri (Streamlit tarafında değil, API sunucusunda ayarlanmalı) ---
# os.environ["N_GPU_LAYERS"] = "-1"
# os.environ["USE_MLOCK"] = "1"

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

# send_query_to_api fonksiyonunu API'nizdeki ChatQuery modeline uygun hale getirelim
def send_query_to_api(
    query: str,
    history: Optional[List[Dict[str, str]]], # API'nizdeki ChatQuery.history
    current_chat_id: Optional[str],        # API'nizdeki ChatQuery.current_chat_id
    top_k: int,
    temperature: float,
    max_new_tokens: int,                   # API'nizdeki ChatQuery.max_new_tokens
    top_p: Optional[float],                # API'nizdeki ChatQuery.top_p
    repetition_penalty: Optional[float]    # API'nizdeki ChatQuery.repetition_penalty
) -> Dict[str, Any]:
    """Kullanıcı sorgusunu API'ye gönderir ve sonucu alır."""
    payload = {
        "query": query,
        "history": history if history is not None else [], # API boş liste bekliyorsa
        "current_chat_id": current_chat_id,
        "top_k": top_k,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens, # API'nizdeki parametre adıyla eşleşmeli
        "top_p": top_p,
        "repetition_penalty": repetition_penalty
    }
    
    headers = {}
    target_url = ANON_CHAT_API_URL # Varsayılan olarak anonim endpoint
    if st.session_state.get("access_token"):
        headers["Authorization"] = f"Bearer {st.session_state.access_token}"
        target_url = CHAT_API_URL # Giriş yapılmışsa /chat endpoint'i
    
    try:
        response = requests.post(target_url, json=payload, headers=headers, timeout=300) 
        response.raise_for_status() 
        return {"status": "success", "data": response.json()}
    except requests.exceptions.Timeout:
         return {"status": "error", "detail": "API isteği zaman aşımına uğradı. Model yanıt üretirken sorun yaşıyor olabilir. Lütfen tekrar deneyin veya daha kısa bir soru sorun."}
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
    if not uploaded_files: return {"status": "error", "detail": "Yüklenecek dosya seçilmedi."}
    files_payload = []
    for uploaded_file in uploaded_files:
        uploaded_file.seek(0)
        files_payload.append(('files', (uploaded_file.name, uploaded_file, uploaded_file.type)))
    headers = {} # PDF yükleme de yetkilendirme gerektirebilir (API'nize bağlı)
    if st.session_state.get("access_token"):
        headers["Authorization"] = f"Bearer {st.session_state.access_token}"
    try:
        response = requests.post(UPLOAD_API_URL, files=files_payload, headers=headers, timeout=300)
        response.raise_for_status(); return {"status": "success", "data": response.json()}
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

def get_chat_history() -> Dict[str, Any]:
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

def get_chat_messages(chat_id: str) -> Dict[str, Any]:
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

def delete_chat(chat_id: str) -> Dict[str, Any]:
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
def format_timestamp(timestamp_str: Optional[str]): # Optional ekledim
    if not timestamp_str: return "Bilinmiyor"
    try:
        # API'den gelen timestamp string ise ve Z içeriyorsa:
        dt = datetime.fromisoformat(str(timestamp_str).replace('Z', '+00:00'))
        return dt.strftime("%d.%m.%Y %H:%M")
    except ValueError: # Farklı format veya parse edilemiyorsa
        return str(timestamp_str)
    except Exception: # Diğer beklenmedik hatalar için
        return "Zaman Hatalı"

# --- Kullanıcı Girişi ve Oturum Yönetimi ---
if "access_token" not in st.session_state: st.session_state.access_token = None
if "username" not in st.session_state: st.session_state.username = None
if "is_logged_in" not in st.session_state: st.session_state.is_logged_in = False
if "messages" not in st.session_state: st.session_state.messages = []
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "current_chat_id" not in st.session_state: st.session_state.current_chat_id = None


def do_login(username, password):
    login_result = login_user(username, password)
    if login_result["status"] == "success":
        st.session_state.access_token = login_result["data"]["access_token"]
        st.session_state.username = username
        st.session_state.is_logged_in = True
        st.session_state.messages = [] # Yeni giriş, aktif sohbeti temizle
        st.session_state.current_chat_id = None
        st.success(f"Hoş geldiniz, {username}!")
        history_result = get_chat_history()
        if history_result["status"] == "success": st.session_state.chat_history = history_result["data"]
        else: st.session_state.chat_history = []
        st.rerun()
    else: st.error(f"Giriş başarısız: {login_result['detail']}")

def do_logout():
    st.session_state.access_token = None
    st.session_state.username = None
    st.session_state.is_logged_in = False
    st.session_state.messages = []
    st.session_state.current_chat_id = None
    if 'chat_history' in st.session_state: del st.session_state.chat_history
    st.info("Başarıyla çıkış yapıldı.") # Çıkış yapıldığında mesaj
    st.rerun()

# --- Başlık ---
# st.title("🎓 SAÜChat: Yönetmelik Asistanı") # st.set_page_config'de page_title olarak ayarlandı
# st.markdown("Sakarya Üniversitesi yönetmelikleri hakkında sorularınızı sorun veya yeni PDF'ler ekleyin.")

# --- Sidebar ---
with st.sidebar:
    # Sidebar'ın en üstüne logo
    SIDEBAR_LOGO_PATH = LOGO_FILE_PATH # Sayfa ikonu ile aynı logoyu kullanalım
    try:
        st.image(SIDEBAR_LOGO_PATH, use_container_width=True) # DEĞİŞİKLİK BURADA: use_column_width -> use_container_width
    except Exception as e:
        st.error(f"Sidebar logo '{SIDEBAR_LOGO_PATH}' yüklenemedi: {e}")

    st.header("SAÜChat") # Daha büyük bir başlık için st.header veya st.title

    # Giriş/Kayıt Bölümü
    if not st.session_state.is_logged_in:
        st.subheader("Oturum") # "Giriş Yap veya Üye Ol" yerine daha genel
        tab1, tab2 = st.tabs(["Giriş", "Üye Ol"])
        with tab1:
            with st.form("login_form_sidebar"): # Key'leri değiştirdim
                login_username = st.text_input("Kullanıcı Adı", key="sidebar_login_username")
                login_password = st.text_input("Şifre", type="password", key="sidebar_login_password")
                if st.form_submit_button("Giriş Yap"): do_login(login_username, login_password)
        with tab2:
            with st.form("register_form_sidebar"): # Key'leri değiştirdim
                reg_username = st.text_input("Kullanıcı Adı", key="sidebar_reg_username")
                reg_email = st.text_input("E-posta", key="sidebar_reg_email")
                reg_password = st.text_input("Şifre", type="password", key="sidebar_reg_password")
                reg_password_confirm = st.text_input("Şifre (Tekrar)", type="password", key="sidebar_reg_password_confirm")
                if st.form_submit_button("Üye Ol"):
                    if reg_password != reg_password_confirm: st.error("Şifreler eşleşmiyor!")
                    elif len(reg_password) < 6: st.error("Şifre en az 6 karakter olmalıdır.")
                    else:
                        register_result = register_user(reg_username, reg_email, reg_password)
                        if register_result["status"] == "success": st.success("Kayıt başarılı! Şimdi giriş yapabilirsiniz.")
                        else: st.error(f"Kayıt başarısız: {register_result['detail']}")
    else: # Kullanıcı giriş yapmışsa
        st.success(f"Hoş geldiniz, {st.session_state.username}!")
        if st.button("Çıkış Yap", type="primary", key="sidebar_logout_button"): do_logout() # type="primary" ekledim

        st.divider() # Ayırıcı
        st.subheader("Sohbet Geçmişi")
        if st.button("🔄 Geçmişi Yenile", key="sidebar_refresh_history"): # İkon ve metin bir arada
            history_result = get_chat_history()
            if history_result["status"] == "success":
                st.session_state.chat_history = history_result["data"]
                st.toast("Sohbet geçmişi yenilendi!") # Daha kibar bildirim
            else:
                st.error(f"Geçmiş yüklenemedi: {history_result['detail']}")
                st.session_state.chat_history = []

        if "chat_history" not in st.session_state or not st.session_state.chat_history:
            # Eğer session_state'de yoksa veya boşsa, API'den çekmeyi dene (giriş yapılmışsa)
            if st.session_state.is_logged_in:
                history_result = get_chat_history()
                if history_result["status"] == "success":
                    st.session_state.chat_history = history_result["data"]
                else:
                    st.session_state.chat_history = []

        if st.session_state.chat_history: # Kontrolü düzelt
            for chat in st.session_state.chat_history:
                chat_id = chat.get("chat_id", f"no_id_{chat.get('timestamp', 'default')}") # Pydantic değilse .get()
                first_message = chat.get("first_message", "Başlıksız Sohbet")
                timestamp_str = str(chat.get("timestamp")) # str() ile güvenceye al

                formatted_time = format_timestamp(timestamp_str)
                chat_title = first_message[:25] + "..." if len(first_message) > 25 else first_message # Başlığı kısalt

                col1, col2 = st.columns([0.8, 0.2]) # Oranlar ayarlandı
                with col1:
                    if st.button(f"💬 {chat_title}", key=f"chat_{chat_id}_sb", help=f"{formatted_time}", use_container_width=True): # DEĞİŞİKLİK: use_container_width
                        with st.spinner("Sohbet yükleniyor..."):
                            messages_result = get_chat_messages(chat_id)
                        if messages_result["status"] == "success":
                            st.session_state.messages = []
                            for msg_data in messages_result["data"]: # API'den dönen MDB_ChatMessageHistory
                                st.session_state.messages.append({"role": "user", "content": msg_data.get("user_message")})
                                st.session_state.messages.append({"role": "assistant", "content": msg_data.get("bot_response")})
                            st.session_state.current_chat_id = chat_id
                            st.rerun()
                        else: st.error(f"Sohbet yüklenemedi: {messages_result['detail']}")
                with col2:
                    if st.button("🗑️", key=f"del_{chat_id}_sb", help="Sohbeti sil"):
                        delete_result = delete_chat(chat_id)
                        if delete_result["status"] == "success":
                            st.toast("Sohbet silindi!") # toast
                            history_result_del = get_chat_history() # Yenile
                            if history_result_del["status"] == "success": st.session_state.chat_history = history_result_del["data"]
                            if st.session_state.current_chat_id == chat_id: # Aktif sohbet silindiyse
                                st.session_state.messages = []
                                st.session_state.current_chat_id = None
                            st.rerun()
                        else: st.error(f"Sohbet silinemedi: {delete_result['detail']}")
            st.markdown("---") # Her sohbetten sonra değil, listenin sonuna
        else:
            st.caption("Henüz sohbet geçmişiniz bulunmuyor.") # info yerine caption
        st.divider()

    st.subheader("Sohbet Ayarları")
    # Sohbet ayarlarını session_state'den al, yoksa varsayılan ata
    top_k_default = st.session_state.get("top_k_slider", 3)
    temp_default = st.session_state.get("temperature_slider", 0.1)
    max_tokens_default = st.session_state.get("max_tokens_slider", 768) # API'deki ChatQuery default
    top_p_default = st.session_state.get("top_p_slider", 0.9)
    rep_penalty_default = st.session_state.get("repetition_penalty_slider", 1.1)

    # Slider'lar için yeni key'ler ve session_state'e kaydetme
    st.session_state.top_k_slider = st.slider("Kaynak Belge Sayısı", 1, 10, top_k_default, 1, help="Yanıt için kaç adet ilgili belge kullanılacak?", key="top_k_slider_widget")
    st.session_state.temperature_slider = st.slider("Yaratıcılık", 0.0, 1.0, temp_default, 0.01, help="Düşük değerler daha kesin, yüksek değerler daha yaratıcı yanıtlar üretir. 0.0 greedy olur.", key="temp_slider_widget") # 0.01 adımlı
    st.session_state.max_tokens_slider = st.slider("Maks. Yanıt Uzunluğu", 100, 2000, max_tokens_default, 50, help="Modelin üreteceği maksimum kelime/token sayısı.", key="max_tokens_slider_widget")
    st.session_state.top_p_slider = st.slider("Top P (Nucleus Sampling)", 0.0, 1.0, top_p_default, 0.01, help="Düşük değerler daha odaklı, yüksekler daha çeşitli yanıtlar üretir. Sıcaklık > 0 iken etkilidir.", key="top_p_slider_widget")
    st.session_state.repetition_penalty_slider = st.slider("Tekrar Cezası", 1.0, 2.0, rep_penalty_default, 0.05, help="1.0 ceza yok. Daha yüksek değerler tekrarları azaltır.", key="rep_penalty_slider_widget")


    st.divider()
    if st.session_state.is_logged_in and st.session_state.username in ["admin", "mehmet", "ozgur", "beyza", "hasan"]: # "hasan" eklendi
        st.subheader("Yeni Yönetmelik Ekle")
        with st.expander("PDF Yükle", expanded=False): # Başlangıçta kapalı
            uploaded_files = st.file_uploader("PDF Dosyalarını Seçin", type="pdf", accept_multiple_files=True, key="pdf_upload_sidebar")
            if uploaded_files:
                st.write(f"{len(uploaded_files)} dosya seçildi:")
                for f_up in uploaded_files: st.caption(f"- {f_up.name}")
                if st.button("Seçili PDF'leri İndeksle", key="upload_button_sidebar"):
                    with st.spinner(f"{len(uploaded_files)} dosya işleniyor..."):
                        upload_result = upload_pdf_to_api(uploaded_files)
                    if upload_result["status"] == "success":
                        st.success(upload_result["data"].get("message", "Dosyalar başarıyla işlendi."))
                        st.info(f"İşlenen: {upload_result['data'].get('processed_files', 'N/A')}, Eklenen: {upload_result['data'].get('added_chunks', 'N/A')}")
                        if upload_result["data"].get("errors"):
                            st.warning("Hatalar:")
                            for err_item in upload_result["data"]["errors"]: st.error(f"- {err_item}")
                    else: st.error(f"Yükleme Hatası: {upload_result['detail']}")
        st.divider()

    st.subheader("API Durumu")
    if st.button("API Durumunu Kontrol Et", key="health_check_sidebar_btn"): # Key değiştirildi
        with st.spinner("API durumu kontrol ediliyor..."): health_result = check_api_health()
        if health_result["status"] == "success":
            data = health_result["data"]
            st.success("✅ API sunucusu aktif!")
            st.markdown(f"{'✅ LLM Yüklü' if data.get('llm_loaded') else '❌ LLM Yüklü Değil'}")
            st.markdown(f"{'✅ DB Yüklü' if data.get('db_loaded') else '❌ DB Yüklü Değil'}")
            st.caption(f"Model: {data.get('model_path', 'N/A')}")
            st.caption(f"Veritabanı: {data.get('db_path', 'N/A')}")
        else: st.error(f"❌ API Bağlantı Hatası: {health_result['detail']}")

# --- Ana İçerik Alanı ---
# Ana başlık ve açıklama
col_main_title, _ = st.columns([3,1]) # Başlık için daha fazla yer
with col_main_title:
    st.header("🎓 SAÜChat: Yönetmelik Asistanı")
    st.caption("Sakarya Üniversitesi yönetmelikleri hakkında sorularınızı sorun veya yeni PDF'ler ekleyin.")
st.markdown("---")


if not st.session_state.is_logged_in and not st.session_state.messages:
    st.info("Sohbet geçmişinizin kaydedilmesi ve tüm özelliklere erişim için lütfen kenar çubuğundan giriş yapın veya üye olun.")

if st.session_state.is_logged_in:
    if st.button("💬 Yeni Sohbet Başlat", key="new_chat_mainarea_button", type="primary"):
        st.session_state.messages = []
        st.session_state.current_chat_id = None
        st.toast("Yeni sohbet başlatıldı.")
        # API tarafı yeni sohbeti ilk mesajla oluşturuyorsa, geçmişi burada yenilemeye gerek yok
        # Ama sidebar'ın güncellenmesi için rerun gerekebilir.
        st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "details" in message and message["details"]: # details var mı diye kontrol
            with st.expander("Modelin Kullandığı Bilgiler ve Kaynaklar", expanded=False):
                st.markdown("**İlgili Bilgiler (Context):**")
                st.markdown(f"> {message['details'].get('retrieved_context', 'Bağlam yok.')}")
                st.markdown("**Kaynak Dosyalar:**")
                sources_list = message['details'].get('sources', [])
                if sources_list:
                    for source_item in sources_list:
                        st.caption(f"- {os.path.basename(source_item)}" if source_item else "Bilinmeyen Kaynak")
                else:
                    st.caption("Bu yanıt için spesifik kaynak dosya kullanılmadı.")

user_query = st.chat_input("Sorunuzu buraya yazın...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"): st.markdown(user_query)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🧠 Düşünüyorum...")
        
        if not st.session_state.is_logged_in and len(st.session_state.messages) > 10: # Anonim için mesaj sınırı
            st.warning("Sınırsız sohbet için lütfen giriş yapın. Giriş yapmadığınız sürece sohbet geçmişiniz kaybolabilir.")
            # Anonim kullanıcı için mesaj göndermeyi burada durdurabiliriz veya devam ettirebiliriz.
            # Şimdilik devam ediyor.
            
        # API'ye gönderilecek geçmiş mesajlar (mevcut kullanıcı sorgusu hariç)
        # API'niz tüm geçmişi ChatQuery.history içinde bekliyorsa:
        history_for_api = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages[:-1] # Son kullanıcı mesajı hariç
            if msg["role"] in ["user", "assistant"] # Sadece user ve assistant rollerini al
        ]

        api_response = send_query_to_api(
            query=user_query,
            history=history_for_api,
            current_chat_id=st.session_state.current_chat_id,
            top_k=st.session_state.top_k_slider,
            temperature=st.session_state.temperature_slider,
            max_new_tokens=st.session_state.max_tokens_slider,
            top_p=st.session_state.top_p_slider,
            repetition_penalty=st.session_state.repetition_penalty_slider
        )

        if api_response["status"] == "success":
            response_data = api_response["data"]
            assistant_response = response_data.get("model_answer", "Üzgünüm, bir yanıt alamadım.")
            message_placeholder.markdown(assistant_response)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_response,
                "details": { # Bu detayları API'den aldığımızı varsayıyoruz
                    "retrieved_context": response_data.get("retrieved_context", ""),
                    "sources": response_data.get("sources", [])
                }
            })
            
            # Eğer API yeni bir chat_id döndürdüyse (giriş yapmış kullanıcı için)
            if st.session_state.is_logged_in and response_data.get("chat_id"):
                new_chat_id_from_api = response_data["chat_id"]
                if st.session_state.current_chat_id != new_chat_id_from_api:
                    st.session_state.current_chat_id = new_chat_id_from_api
                    # Yeni sohbet oluştuysa veya farklı bir sohbete geçildiyse geçmiş listesini yenile
                    history_result_update = get_chat_history()
                    if history_result_update["status"] == "success":
                        st.session_state.chat_history = history_result_update["data"]
                    st.rerun() # Sidebar'ı güncellemek için
        else:
            error_txt = f"Hata: {api_response['detail']}"
            message_placeholder.error(error_txt)
            st.session_state.messages.append({"role": "assistant", "content": error_txt})

st.markdown("---")
st.caption(f"SAÜChat © {datetime.now().year} - Sakarya Üniversitesi")