# import streamlit as st
# import requests
# import json
# import os
# import sys
# from typing import Dict, Any

# # Sayfa yapılandırması
# st.set_page_config(
#     page_title="SAÜChat - Sakarya Üniversitesi Yönetmelik Asistanı",
#     page_icon="🎓",
#     layout="wide"
# )

# # API URL (yerel veya uzak sunucu için ayarlanabilir)
# API_URL = "http://localhost:8000/chat"

# def send_query_to_api(query: str, top_k: int = 3, temperature: float = 0.1, max_tokens: int = 512) -> Dict[str, Any]:
#     """
#     Kullanıcı sorgusunu API'ye gönderir ve sonucu alır
    
#     Args:
#         query: Kullanıcının sorgusu
#         top_k: Döndürülecek belge sayısı
#         temperature: Model yaratıcılık seviyesi
#         max_tokens: Maksimum yanıt uzunluğu
        
#     Returns:
#         API yanıtı (retrieved_context, model_answer ve sources içeren sözlük)
#     """
#     try:
#         payload = {
#             "query": query,
#             "top_k": top_k,
#             "temperature": temperature,
#             "max_tokens": max_tokens
#         }
#         response = requests.post(API_URL, json=payload)
        
#         # HTTP hatalarını kontrol et
#         if response.status_code != 200:
#             error_detail = response.json().get("detail", "Bilinmeyen hata")
#             return {
#                 "error": f"API Hatası ({response.status_code}): {error_detail}",
#                 "status": "error"
#             }
            
#         return response.json()
    
#     except requests.RequestException as e:
#         return {
#             "error": f"Sunucu bağlantı hatası: {str(e)}. API sunucusunun çalıştığından emin olun.",
#             "status": "error"
#         }
#     except Exception as e:
#         return {
#             "error": f"Beklenmeyen hata: {str(e)}",
#             "status": "error"
#         }

# # Uygulama başlığı ve açıklaması
# st.title("🎓 SAÜChat: Yönetmelik Asistanı")
# st.markdown("""
# Bu uygulama, Sakarya Üniversitesi'nin resmi yönerge ve yönetmelikleri hakkında bilgi almanızı sağlar.
# Sorularınızı Türkçe olarak sorun, cevapları alın!
# """)

# # Sidebar bilgileri ve ayarlar
# with st.sidebar:
#     st.image("https://www.sakarya.edu.tr/img/logo_tr.png", width=200)
#     st.title("SAÜChat")
#     st.info("""
#     Bu uygulama, Sakarya Üniversitesi'nin resmi yönerge ve yönetmeliklerinden bilgileri almanızı sağlar.
#     LLaMA 3 tabanlı yapay zeka destekli bir bilgi erişim sistemidir.
#     """)
    
#     # Ayarlar
#     st.subheader("Ayarlar")
#     top_k = st.slider("Kaynak belge sayısı", min_value=1, max_value=10, value=3, step=1)
#     temperature = st.slider("Yaratıcılık seviyesi", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
#     max_tokens = st.slider("Maksimum yanıt uzunluğu", min_value=100, max_value=2000, value=512, step=50)
    
#     # API bağlantı testi
#     if st.button("API Bağlantısını Test Et"):
#         try:
#             health_check = requests.get("http://localhost:8000/health")
#             if health_check.status_code == 200:
#                 data = health_check.json()
#                 llm_status = "✅ LLM modeli yüklü" if data.get("llm_loaded", False) else "❌ LLM modeli yüklü değil"
#                 st.success(f"✅ API sunucusu aktif!\n\n{llm_status}")
#             else:
#                 st.error("❌ API sunucusu yanıt veriyor ancak hata döndürüyor.")
#         except:
#             st.error("❌ API sunucusuna bağlanılamıyor. Sunucunun çalıştığından emin olun.")

# # Oturum durumunu başlat
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Geçmiş mesajları göster
# for message in st.session_state.messages:
#     # Kullanıcı mesajları
#     if message["role"] == "user":
#         with st.chat_message("user"):
#             st.markdown(message["content"])
#     # Asistan mesajları
#     elif message["role"] == "assistant":
#         with st.chat_message("assistant"):
#             if "error" in message:
#                 st.error(message["error"])
#             else:
#                 # Asıl yanıtı göster
#                 st.markdown(message["model_answer"])
#                 # İlgili bağlam bilgilerini göster
#                 with st.expander("İlgili yönetmelik bilgileri"):
#                     st.markdown(message["retrieved_context"])
                
#                 # Kaynak bilgileri varsa göster
#                 if message.get("sources") and len(message["sources"]) > 0:
#                     with st.expander("Bilgi kaynakları"):
#                         for source in message["sources"]:
#                             st.info(source)

# # Kullanıcı girdisi
# user_query = st.chat_input("Sorunuzu yazın...")

# if user_query:
#     # Kullanıcı mesajını göster
#     with st.chat_message("user"):
#         st.markdown(user_query)
    
#     # Kullanıcı mesajını kaydet
#     st.session_state.messages.append({"role": "user", "content": user_query})
    
#     # API'ye sorguyu gönder
#     with st.spinner("Bilgiler aranıyor ve yanıtınız hazırlanıyor..."):
#         response = send_query_to_api(
#             user_query, 
#             top_k=top_k,
#             temperature=temperature,
#             max_tokens=max_tokens
#         )
    
#     # Yanıtı göster
#     with st.chat_message("assistant"):
#         if "error" in response:
#             st.error(response["error"])
#             # Hata mesajını kaydet
#             st.session_state.messages.append({
#                 "role": "assistant",
#                 "error": response["error"]
#             })
#         else:
#             # Asıl yanıtı göster
#             st.markdown(response["model_answer"])
#             # İlgili bağlam bilgilerini göster
#             with st.expander("İlgili yönetmelik bilgileri"):
#                 st.markdown(response["retrieved_context"])
            
#             # Kaynak bilgileri varsa göster
#             if response.get("sources") and len(response["sources"]) > 0:
#                 with st.expander("Bilgi kaynakları"):
#                     for source in response["sources"]:
#                         st.info(source)
            
#             # Asistan yanıtını kaydet
#             st.session_state.messages.append({
#                 "role": "assistant",
#                 "model_answer": response["model_answer"],
#                 "retrieved_context": response["retrieved_context"],
#                 "sources": response.get("sources")
#             })

# # Footer
# st.markdown("---")
# st.markdown("**SAÜChat** - Sakarya Üniversitesi Yönerge ve Yönetmelikler Bilgi Sistemi © 2025")



import streamlit as st
import requests
import json
import os
import sys
from typing import Dict, Any

# Sayfa yapılandırması
st.set_page_config(
    page_title="SAÜChat - Sakarya Üniversitesi Yönetmelik Asistanı",
    page_icon="🎓",
    layout="wide"
)

# API URL'leri
CHAT_API_URL = "http://localhost:8000/chat"
UPLOAD_API_URL = "http://localhost:8000/upload-pdf"
HEALTH_API_URL = "http://localhost:8000/health"

def send_query_to_api(query: str, top_k: int = 3, temperature: float = 0.1, max_tokens: int = 512) -> Dict[str, Any]:
    """
    Kullanıcı sorgusunu API'ye gönderir ve sonucu alır
    """
    try:
        payload = {
            "query": query,
            "top_k": top_k,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        response = requests.post(CHAT_API_URL, json=payload)

        if response.status_code != 200:
            error_detail = response.json().get("detail", "Bilinmeyen hata")
            return {
                "error": f"API Hatası ({response.status_code}): {error_detail}",
                "status": "error"
            }

        return response.json()

    except requests.RequestException as e:
        return {
            "error": f"Sunucu bağlantı hatası: {str(e)}. API sunucusunun çalıştığından emin olun.",
            "status": "error"
        }
    except Exception as e:
        return {
            "error": f"Beklenmeyen hata: {str(e)}",
            "status": "error"
        }

def upload_pdf_to_api(uploaded_file):
    """
    Yüklenen PDF dosyasını API'ye gönderir.
    """
    if uploaded_file is None:
        return {"error": "Yüklenecek dosya seçilmedi.", "status": "error"}

    try:
        files = {'file': (uploaded_file.name, uploaded_file, uploaded_file.type)}
        response = requests.post(UPLOAD_API_URL, files=files)

        if response.status_code != 200:
            error_detail = response.json().get("detail", "Bilinmeyen hata")
            return {
                "error": f"API Hatası ({response.status_code}): {error_detail}",
                "status": "error"
            }

        return response.json()

    except requests.RequestException as e:
        return {
            "error": f"Sunucu bağlantı hatası: {str(e)}. API sunucusunun çalıştığından emin olun.",
            "status": "error"
        }
    except Exception as e:
        return {
            "error": f"Dosya yüklenirken beklenmeyen hata: {str(e)}",
            "status": "error"
        }


# Uygulama başlığı ve açıklaması
st.title("🎓 SAÜChat: Yönetmelik Asistanı")
st.markdown("""
Bu uygulama, Sakarya Üniversitesi'nin resmi yönerge ve yönetmelikleri hakkında bilgi almanızı sağlar.
Sorularınızı Türkçe olarak sorun, cevapları alın! Ayrıca yeni yönetmelik PDF'lerini sisteme ekleyebilirsiniz.
""")

# Sidebar bilgileri ve ayarlar
with st.sidebar:
    st.image("https://www.sakarya.edu.tr/img/logo_tr.png", width=200)
    st.title("SAÜChat")
    st.info("""
    Bu uygulama, Sakarya Üniversitesi'nin resmi yönerge ve yönetmeliklerinden bilgileri almanızı sağlar.
    LLaMA 3 tabanlı yapay zeka destekli bir bilgi erişim sistemidir.
    """)

    # Ayarlar
    st.subheader("Sohbet Ayarları")
    top_k = st.slider("Kaynak belge sayısı", min_value=1, max_value=10, value=3, step=1)
    temperature = st.slider("Yaratıcılık seviyesi", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
    max_tokens = st.slider("Maksimum yanıt uzunluğu", min_value=100, max_value=2000, value=512, step=50)

    st.divider() # Ayırıcı

    # PDF Yükleme Bölümü
    st.subheader("Yeni Yönetmelik Ekle")
    uploaded_file = st.file_uploader("İndekslenecek PDF dosyasını seçin", type="pdf")

    if uploaded_file is not None:
        st.write(f"Yüklenen dosya: **{uploaded_file.name}**")
        if st.button("PDF'i İndeksle"):
            with st.spinner(f"'{uploaded_file.name}' işleniyor ve veritabanına ekleniyor..."):
                upload_response = upload_pdf_to_api(uploaded_file)

            if "error" in upload_response:
                st.error(f"Hata: {upload_response['error']}")
            else:
                st.success(upload_response.get("message", "Dosya başarıyla işlendi."))
                st.info(f"Eklenen parça sayısı: {upload_response.get('added_chunks', 'N/A')}")
                # Başarılı yükleme sonrası dosya seçiciyi temizle (isteğe bağlı)
                # uploaded_file = None # Streamlit'te bu doğrudan yapılamaz, state yönetimi gerekir

    st.divider() # Ayırıcı

    # API bağlantı testi
    if st.button("API Bağlantısını Test Et"):
        try:
            health_check = requests.get(HEALTH_API_URL)
            if health_check.status_code == 200:
                data = health_check.json()
                llm_status = "✅ LLM modeli yüklü" if data.get("llm_loaded", False) else "❌ LLM modeli yüklü değil"
                st.success(f"✅ API sunucusu aktif!\n\n{llm_status}")
            else:
                st.error("❌ API sunucusu yanıt veriyor ancak hata döndürüyor.")
        except requests.RequestException:
            st.error("❌ API sunucusuna bağlanılamıyor. Sunucunun çalıştığından emin olun.")
        except Exception as e:
            st.error(f"❌ Test sırasında hata: {e}")


# Oturum durumunu başlat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Geçmiş mesajları göster
for message in st.session_state.messages:
    # Kullanıcı mesajları
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    # Asistan mesajları
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            if "error" in message:
                st.error(message["error"])
            else:
                # Asıl yanıtı göster
                st.markdown(message["model_answer"])
                # İlgili bağlam bilgilerini göster
                with st.expander("İlgili yönetmelik bilgileri"):
                    st.markdown(message["retrieved_context"])

                # Kaynak bilgileri varsa göster
                if message.get("sources") and len(message["sources"]) > 0:
                    with st.expander("Bilgi kaynakları"):
                        for source in message["sources"]:
                            # Kaynak adını daha okunabilir hale getir
                            display_source = os.path.basename(source) if source else "Bilinmeyen Kaynak"
                            st.info(display_source)

# Kullanıcı girdisi
user_query = st.chat_input("Sorunuzu yazın...")

if user_query:
    # Kullanıcı mesajını göster
    with st.chat_message("user"):
        st.markdown(user_query)

    # Kullanıcı mesajını kaydet
    st.session_state.messages.append({"role": "user", "content": user_query})

    # API'ye sorguyu gönder
    with st.spinner("Bilgiler aranıyor ve yanıtınız hazırlanıyor..."):
        response = send_query_to_api(
            user_query,
            top_k=top_k,
            temperature=temperature,
            max_tokens=max_tokens
        )

    # Yanıtı göster
    with st.chat_message("assistant"):
        if "error" in response:
            st.error(response["error"])
            # Hata mesajını kaydet
            st.session_state.messages.append({
                "role": "assistant",
                "error": response["error"]
            })
        else:
            # Asıl yanıtı göster
            st.markdown(response["model_answer"])
            # İlgili bağlam bilgilerini göster
            with st.expander("İlgili yönetmelik bilgileri"):
                st.markdown(response["retrieved_context"])

            # Kaynak bilgileri varsa göster
            sources = response.get("sources")
            if sources and len(sources) > 0:
                with st.expander("Bilgi kaynakları"):
                    for source in sources:
                         # Kaynak adını daha okunabilir hale getir
                        display_source = os.path.basename(source) if source else "Bilinmeyen Kaynak"
                        st.info(display_source)

            # Asistan yanıtını kaydet
            st.session_state.messages.append({
                "role": "assistant",
                "model_answer": response["model_answer"],
                "retrieved_context": response["retrieved_context"],
                "sources": sources
            })

# Footer
st.markdown("---")
st.markdown("**SAÜChat** - Sakarya Üniversitesi Yönerge ve Yönetmelikler Bilgi Sistemi © 2025")