import streamlit as st
import requests
import json
import os
from typing import Dict, Any, List

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
    try:
        response = requests.post(CHAT_API_URL, json=payload, timeout=120) # Daha uzun timeout
        response.raise_for_status() # HTTP 2xx olmayan durumlar için hata fırlat
        return {"status": "success", "data": response.json()}
    except requests.exceptions.Timeout:
         return {"status": "error", "detail": "API isteği zaman aşımına uğradı."}
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

# --- Streamlit Arayüzü ---

# Başlık
st.title("🎓 SAÜChat: Yönetmelik Asistanı")
st.markdown("Sakarya Üniversitesi yönetmelikleri hakkında sorularınızı sorun veya yeni PDF'ler ekleyin.")

# Sidebar
with st.sidebar:
    st.image("https://www.sakarya.edu.tr/img/logo_tr.png", width=200)
    st.title("Ayarlar ve İşlemler")
    st.info("Bu asistan, SAÜ yönetmelikleri hakkında bilgi vermek üzere tasarlanmıştır.")

    # Sohbet Ayarları
    st.subheader("Sohbet Ayarları")
    top_k = st.slider("Kaynak Belge Sayısı", 1, 10, 3, 1, help="Yanıt için kaç adet ilgili belge kullanılacak?")
    temperature = st.slider("Yaratıcılık", 0.0, 1.0, 0.1, 0.05, help="Düşük değerler daha kesin, yüksek değerler daha yaratıcı yanıtlar üretir.")
    max_tokens = st.slider("Maks. Yanıt Uzunluğu", 100, 2000, 512, 50, help="Modelin üreteceği maksimum kelime/token sayısı.")

    st.divider()

    # PDF Yükleme
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
            # Yükleme sonrası seçimi temizlemek için state yönetimi gerekir, şimdilik manuel.

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

# Kullanıcı girdisi al
user_query = st.chat_input("Sorunuzu buraya yazın...")

if user_query:
    # Kullanıcı mesajını ekle ve göster
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # API'ye sorguyu gönder ve yanıtı bekle
    with st.chat_message("assistant"):
        with st.spinner("Yanıtınız hazırlanıyor..."):
            api_response = send_query_to_api(user_query, top_k, temperature, max_tokens)

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

            # Detayları expander içinde göster (isteğe bağlı, yukarıdaki döngüde de gösteriliyor)
            # with st.expander("Detaylar ve Kaynaklar"):
            #     st.markdown("**İlgili Bilgiler:**")
            #     st.markdown(retrieved_context)
            #     st.markdown("**Kaynaklar:**")
            #     if sources:
            #         for source in sources:
            #             st.caption(os.path.basename(source) if source else "Bilinmeyen")
            #     else:
            #         st.caption("Kaynak bulunamadı.")

        else:
            # Hata mesajını göster
            error_message = f"Hata: {api_response['detail']}"
            st.error(error_message)
            # Hata mesajını oturum durumuna ekle (içerik olarak)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

# Footer
st.markdown("---")
st.caption("SAÜChat © 2025 - Sakarya Üniversitesi Bilgi İşlem Daire Başkanlığı (Konsept)")