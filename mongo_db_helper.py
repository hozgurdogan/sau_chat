# mongo_db_helper.py
import pymongo
from pymongo import MongoClient
import datetime as dt # Bu import kullanılmıyor gibi, datetime zaten import edilmiş
from datetime import datetime, timedelta
import hashlib
import os
# import uuid # ObjectId kullanıldığı için uuid'ye gerek yok
from typing import List, Dict, Optional, Any
from bson.objectid import ObjectId

# --- Pydantic Modelleri (api_server.py'deki ile aynı olmalı) ---
# İdeal olarak bu modeller ayrı bir 'models.py' dosyasında olmalı ve her iki yerden de import edilmeli.
# Şimdilik buraya kopyalıyorum.
from pydantic import BaseModel, Field # pydantic import etmeniz gerekebilir: pip install pydantic

class MDB_ChatHistory(BaseModel): # MongoDB'den dönerken kullanılacak Pydantic modeli
    chat_id: str
    first_message: str
    timestamp: datetime
    message_count: int

class MDB_ChatMessageHistory(BaseModel): # MongoDB'den dönerken kullanılacak Pydantic modeli
    user_message: str
    bot_response: str
    timestamp: datetime
    # retrieved_docs: Optional[List[str]] = [] # İsteğe bağlı olarak eklenebilir

# --- MongoDB Bağlantısı ---
MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://denemecursor1bedava:oF27WsS8MqA1nYPk@bitirme.ne3ofr5.mongodb.net/sau_chat_db?retryWrites=true&w=majority&appName=bitirme")
DB_NAME = "sau_chat_db"

_db_instance = None

def get_database():
    global _db_instance
    if _db_instance is None:
        client = MongoClient(MONGO_URI)
        _db_instance = client[DB_NAME]
    return _db_instance

def hash_password(password: str) -> str:
    salt = os.environ.get("PASSWORD_SALT", "sau_chat_salt_value") # Varsayılan salt değeri eklendi
    return hashlib.sha256((password + salt).encode()).hexdigest()

def create_user(username: str, email: str, password: str) -> tuple[bool, str]: # Mesaj da döndürsün
    db = get_database()
    if db.users.find_one({"$or": [{"username": username}, {"email": email}]}):
        return False, "Kullanıcı adı veya e-posta zaten mevcut."
    user_doc = {
        "username": username, "email": email, "password": hash_password(password),
        "created_at": datetime.now(), "last_login": None,
        "chat_sessions_summary": [] # Kullanıcının sohbet özetlerini burada tutabiliriz
    }
    result = db.users.insert_one(user_doc)
    return result.acknowledged, "Kullanıcı başarıyla oluşturuldu."


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    db = get_database()
    user = db.users.find_one({"username": username})
    if user and user["password"] == hash_password(password):
        db.users.update_one({"_id": user["_id"]}, {"$set": {"last_login": datetime.now()}})
        return {"username": user["username"], "email": user["email"]} # created_at gereksiz olabilir
    return None

# --- Sohbet İşlemleri (Mevcut Yapıyı Koruyarak Pydantic Eklemeleriyle) ---
def save_chat_message(username: str, user_message: str, bot_response: str, retrieved_docs: List[str] = None, chat_id: Optional[str] = None) -> str:
    db = get_database()
    timestamp = datetime.utcnow()
    retrieved_docs = retrieved_docs or []

    active_chat_id = chat_id
    
    # Kullanıcının sohbet oturumları özetini tuttuğumuz users koleksiyonundaki alanı kontrol et
    user_doc = db.users.find_one({"username": username})
    chat_sessions_summary = user_doc.get("chat_sessions_summary", []) if user_doc else []

    existing_session_summary = None
    if active_chat_id:
        for session in chat_sessions_summary:
            if session.get("chat_id") == active_chat_id:
                existing_session_summary = session
                break
    
    if not active_chat_id or not existing_session_summary:
        # Yeni sohbet oturumu oluştur
        active_chat_id = str(ObjectId()) # Yeni chat_id
        print(f"Yeni sohbet oturumu oluşturuluyor (mongo_db_helper): {active_chat_id} kullanıcı: {username}")
        new_session_summary = {
            "chat_id": active_chat_id,
            "first_message": user_message[:75] + ("..." if len(user_message) > 75 else ""),
            "timestamp": timestamp, # Oluşturulma zamanı
            "last_updated": timestamp,
            "message_count": 0 # Aşağıda artırılacak
        }
        db.users.update_one(
            {"username": username},
            {"$push": {"chat_sessions_summary": new_session_summary}},
            upsert=True # Eğer kullanıcı dokümanı yoksa oluştur (normalde olmamalı)
        )
    else:
        print(f"Var olan sohbet oturumuna ekleniyor (mongo_db_helper): {active_chat_id} kullanıcı: {username}")

    # Her zaman message_count'u ve last_updated'ı güncelle
    db.users.update_one(
        {"username": username, "chat_sessions_summary.chat_id": active_chat_id},
        {
            "$set": {"chat_sessions_summary.$.last_updated": timestamp},
            "$inc": {"chat_sessions_summary.$.message_count": 1}
        }
    )

    # Mesajı chat_messages koleksiyonuna kaydet
    message_doc = {
        "chat_id": active_chat_id,
        "username": username,
        "user_message": user_message,
        "bot_response": bot_response,
        "retrieved_docs": retrieved_docs,
        "timestamp": timestamp
    }
    db.chat_messages.insert_one(message_doc)

    return active_chat_id # Her zaman kullanılan veya yeni oluşturulan chat_id'yi döndür
    
def get_user_chats(username: str) -> List[MDB_ChatHistory]:
    """Kullanıcının tüm sohbet oturumlarının özetlerini Pydantic modeli listesi olarak getirir."""
    db = get_database()
    user = db.users.find_one({"username": username}, {"chat_sessions_summary": 1}) # Sadece ilgili alanı çek
    
    if not user or "chat_sessions_summary" not in user:
        return []

    chat_sessions_data = user.get("chat_sessions_summary", [])
    
    # Pydantic modellerine dönüştür
    pydantic_chats = []
    for chat_data in chat_sessions_data:
        try:
            # timestamp MongoDB'den datetime objesi olarak gelmeli, değilse parse et
            if isinstance(chat_data.get("timestamp"), str):
                chat_data["timestamp"] = datetime.fromisoformat(chat_data["timestamp"])
            
            pydantic_chats.append(MDB_ChatHistory(**chat_data))
        except Exception as e:
            print(f"get_user_chats - Pydantic dönüşüm hatası: {e}, Veri: {chat_data}")
            continue # Hatalı veriyi atla

    # En son güncellenen sohbet en üstte olacak şekilde sırala (last_updated'a göre)
    return sorted(pydantic_chats, key=lambda x: x.timestamp, reverse=True) # timestamp (oluşturulma) veya last_updated


def get_chat_messages(chat_id: str) -> List[MDB_ChatMessageHistory]:
    """Belirli bir sohbetin tüm mesajlarını Pydantic modeli listesi olarak getirir."""
    db = get_database()
    # chat_messages koleksiyonundan ilgili sohbetin mesajlarını çek
    messages_cursor = db.chat_messages.find(
        {"chat_id": chat_id},
        sort=[("timestamp", pymongo.ASCENDING)] # Mesajları zamanına göre sıralı al
    )
    
    pydantic_messages = []
    for msg_data in messages_cursor:
        try:
            # timestamp MongoDB'den datetime objesi olarak gelmeli
            if isinstance(msg_data.get("timestamp"), str):
                msg_data["timestamp"] = datetime.fromisoformat(msg_data["timestamp"])

            pydantic_messages.append(MDB_ChatMessageHistory(
                user_message=msg_data.get("user_message", ""),
                bot_response=msg_data.get("bot_response", ""),
                timestamp=msg_data.get("timestamp")
                # retrieved_docs da eklenebilir eğer ChatMessageHistory modelinde varsa
            ))
        except Exception as e:
            print(f"get_chat_messages - Pydantic dönüşüm hatası: {e}, Veri: {msg_data}")
            continue
            
    return pydantic_messages


def delete_chat(chat_id: str, username: str) -> bool:
    """Kullanıcıya ait bir sohbeti ve tüm mesajlarını siler."""
    db = get_database()
    
    # 1. Kullanıcının sohbet oturumu özetinden sil
    user_update_result = db.users.update_one(
        {"username": username},
        {"$pull": {"chat_sessions_summary": {"chat_id": chat_id}}}
    )
    
    # 2. chat_messages koleksiyonundan o sohbete ait tüm mesajları sil
    messages_delete_result = db.chat_messages.delete_many({
        "chat_id": chat_id,
        "username": username # Ekstra güvenlik katmanı
    })
    
    # Eğer en azından özet silindiyse veya mesajlar silindiyse başarılı say
    return user_update_result.modified_count > 0 or messages_delete_result.deleted_count > 0