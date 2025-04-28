import pymongo
from pymongo import MongoClient
import datetime
import hashlib
import os
import uuid
from typing import List, Dict, Optional, Any

# MongoDB bağlantı bilgileri - güvenlik için çevre değişkenlerinden alınmalı
MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://<username>:<password>@<cluster>.mongodb.net/")
DB_NAME = "sau_chat_db"

def get_database():
    """MongoDB veritabanı bağlantısını döndürür"""
    client = MongoClient(MONGO_URI)
    return client[DB_NAME]

def hash_password(password: str) -> str:
    """Şifreyi güvenli bir şekilde hashler"""
    salt = os.environ.get("PASSWORD_SALT", "sau_chat_salt_value")
    return hashlib.sha256((password + salt).encode()).hexdigest()

# Kullanıcı işlemleri
def create_user(username: str, email: str, password: str) -> bool:
    """Yeni kullanıcı oluşturur"""
    db = get_database()
    
    # Kullanıcı adı veya e-posta zaten kullanılıyor mu kontrol et
    if db.users.find_one({"$or": [{"username": username}, {"email": email}]}):
        return False
    
    user = {
        "username": username,
        "email": email,
        "password": hash_password(password),
        "created_at": datetime.datetime.utcnow(),
        "last_login": None
    }
    
    result = db.users.insert_one(user)
    return result.acknowledged

def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Kullanıcı adı ve şifre ile kullanıcıyı doğrular"""
    db = get_database()
    user = db.users.find_one({"username": username})
    
    if user and user["password"] == hash_password(password):
        # Son giriş zamanını güncelle
        db.users.update_one(
            {"_id": user["_id"]},
            {"$set": {"last_login": datetime.datetime.utcnow()}}
        )
        return {
            "username": user["username"],
            "email": user["email"],
            "created_at": user["created_at"]
        }
    return None

# Sohbet geçmişi işlemleri
def save_chat_message(username: str, user_message: str, bot_response: str, 
                      retrieved_docs: List[str] = None) -> str:
    """Kullanıcı mesajını ve bot yanıtını kaydeder"""
    db = get_database()
    
    # Eğer yeni bir konuşma ise chat_id oluştur, değilse son konuşmanın ID'sini kullan
    last_chat = db.chat_history.find_one(
        {"username": username},
        sort=[("timestamp", pymongo.DESCENDING)]
    )
    
    # Son mesajdan 30 dakika geçtiyse yeni konuşma başlat
    if (not last_chat or 
        (datetime.datetime.utcnow() - last_chat["timestamp"]).total_seconds() > 1800):
        chat_id = str(uuid.uuid4())
    else:
        chat_id = last_chat["chat_id"]
    
    message = {
        "chat_id": chat_id,
        "username": username,
        "user_message": user_message,
        "bot_response": bot_response,
        "retrieved_docs": retrieved_docs or [],
        "timestamp": datetime.datetime.utcnow()
    }
    
    result = db.chat_history.insert_one(message)
    return chat_id if result.acknowledged else None

def get_user_chats(username: str) -> List[Dict[str, Any]]:
    """Kullanıcının tüm sohbet oturumlarını getirir"""
    db = get_database()
    
    # Benzersiz chat_id'leri bul ve her biri için ilk mesajı al
    pipeline = [
        {"$match": {"username": username}},
        {"$sort": {"timestamp": 1}},
        {"$group": {
            "_id": "$chat_id",
            "first_message": {"$first": "$user_message"},
            "timestamp": {"$first": "$timestamp"},
            "message_count": {"$sum": 1}
        }},
        {"$sort": {"timestamp": -1}}
    ]
    
    chat_sessions = list(db.chat_history.aggregate(pipeline))
    return [{
        "chat_id": chat["_id"],
        "first_message": chat["first_message"][:50] + "..." if len(chat["first_message"]) > 50 else chat["first_message"],
        "timestamp": chat["timestamp"],
        "message_count": chat["message_count"]
    } for chat in chat_sessions]

def get_chat_messages(chat_id: str) -> List[Dict[str, Any]]:
    """Belirli bir sohbetin tüm mesajlarını getirir"""
    db = get_database()
    messages = list(db.chat_history.find(
        {"chat_id": chat_id},
        sort=[("timestamp", 1)]
    ))
    
    return [{
        "user_message": msg["user_message"],
        "bot_response": msg["bot_response"],
        "timestamp": msg["timestamp"]
    } for msg in messages]

def delete_chat(chat_id: str, username: str) -> bool:
    """Kullanıcıya ait bir sohbeti siler"""
    db = get_database()
    result = db.chat_history.delete_many({
        "chat_id": chat_id,
        "username": username
    })
    return result.deleted_count > 0 