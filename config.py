# config.py — Конфигурация проекта
# Загрузка переменных окружения и проверка наличия ключей

import os
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv()

# Токен Telegram бота (получить через @BotFather)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# API ключ Google Gemini (получить на aistudio.google.com)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Проверка наличия обязательных переменных
if not TELEGRAM_BOT_TOKEN:
    raise ValueError(
        "❌ Не задан TELEGRAM_BOT_TOKEN!\n"
        "Получите токен у @BotFather в Telegram и добавьте в файл .env:\n"
        "TELEGRAM_BOT_TOKEN=ваш_токен"
    )

if not GEMINI_API_KEY:
    raise ValueError(
        "❌ Не задан GEMINI_API_KEY!\n"
        "Получите ключ на https://aistudio.google.com и добавьте в файл .env:\n"
        "GEMINI_API_KEY=ваш_ключ"
    )

# Пути к директориям
KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# Параметры RAG
CHUNK_SIZE = 500          # Размер чанка в символах
CHUNK_OVERLAP = 100       # Перекрытие между чанками
TOP_K_RESULTS = 3         # Количество релевантных чанков для контекста
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Модель для эмбеддингов

# Параметры Gemini
GEMINI_MODEL = "gemini-2.0-flash"
