# knowledge_loader.py — Загрузка документов базы знаний в ChromaDB
# Запустите этот скрипт перед первым запуском бота: python knowledge_loader.py

import logging
import shutil
import os

from config import CHROMA_DB_DIR

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    """Главная функция загрузки базы знаний."""
    print("=" * 60)
    print("  Загрузчик базы знаний Satbayev University Mentor Bot")
    print("=" * 60)
    print()

    # Удаляем старую базу ChromaDB для чистой перезагрузки
    if os.path.exists(CHROMA_DB_DIR):
        logger.info(f"Удаление старой базы данных: {CHROMA_DB_DIR}")
        shutil.rmtree(CHROMA_DB_DIR)
        logger.info("Старая база удалена.")

    # Создаём экземпляр RAG движка и загружаем базу знаний
    from rag_engine import RAGEngine
    rag = RAGEngine()
    stats = rag.load_knowledge_base()

    # Выводим статистику
    print()
    print("=" * 60)
    print(f"  📁 Файлов обработано: {stats['files']}")
    print(f"  📄 Чанков создано:    {stats['chunks']}")
    print(f"  💾 База сохранена в:  {CHROMA_DB_DIR}")
    print("=" * 60)
    print()

    if stats["chunks"] > 0:
        print("✅ База знаний успешно загружена!")
        print("Теперь можно запускать бота: python bot.py")
    else:
        print("⚠️  Внимание: база знаний пуста.")
        print("Убедитесь, что в папке knowledge_base/ есть .md файлы.")


if __name__ == "__main__":
    main()
