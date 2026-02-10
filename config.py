# config.py — Конфигурация проекта
# Загрузка переменных окружения и проверка наличия ключей

import os
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv()

# Токен Telegram бота (получить через @BotFather)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# API ключ Groq (получить на console.groq.com)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Проверка наличия обязательных переменных
if not TELEGRAM_BOT_TOKEN:
    raise ValueError(
        "❌ Не задан TELEGRAM_BOT_TOKEN!\n"
        "Получите токен у @BotFather в Telegram и добавьте в файл .env:\n"
        "TELEGRAM_BOT_TOKEN=ваш_токен"
    )

if not GROQ_API_KEY:
    raise ValueError(
        "❌ Не задан GROQ_API_KEY!\n"
        "Получите ключ на https://console.groq.com и добавьте в файл .env:\n"
        "GROQ_API_KEY=ваш_ключ"
    )

# Пути к директориям
KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# Параметры RAG
CHUNK_SIZE = 1500         # Размер чанка в символах (крупные чанки сохраняют контекст)
CHUNK_OVERLAP = 300       # Перекрытие между чанками
TOP_K_RESULTS = 8         # Количество релевантных чанков для контекста
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # Мультиязычная модель

# Параметры Groq
GROQ_MODEL = "llama-3.1-8b-instant"
