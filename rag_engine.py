# rag_engine.py — RAG движок для поиска по базе знаний
# Гибридный поиск: векторный (ChromaDB) + ключевые слова

import os
import re
import logging
from typing import List

import chromadb
from chromadb.utils import embedding_functions
from config import (
    KNOWLEDGE_BASE_DIR,
    CHROMA_DB_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
    EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)

# Маппинг ключевых слов на файлы базы знаний
# Если запрос содержит ключевое слово — принудительно включаем чанки из этого файла
KEYWORD_FILE_MAP = {
    # Руководство
    "ректор": "02_leadership.md",
    "проректор": "02_leadership.md",
    "руководство": "02_leadership.md",
    "бегентаев": "02_leadership.md",
    "ермекбаев": "02_leadership.md",
    "кульдеев": "02_leadership.md",
    "ускенбаева": "02_leadership.md",
    "шалабаев": "02_leadership.md",
    # Институты
    "институт": "03_institutes.md",
    "кафедр": "03_institutes.md",
    "иаиит": "03_institutes.md",
    "факультет": "03_institutes.md",
    "структура": "03_institutes.md",
    # Общежитие
    "общежити": "08_dormitory.md",
    "заселен": "08_dormitory.md",
    "дормитор": "08_dormitory.md",
    # Учебный процесс
    "кредит": "04_study_process.md",
    "аттестац": "04_study_process.md",
    "экзамен": "04_study_process.md",
    "сессия": "04_study_process.md",
    "силлабус": "04_study_process.md",
    "эдвайзер": "04_study_process.md",
    "пропуск": "04_study_process.md",
    # Оценки и GPA
    "оценк": "05_grades_gpa.md",
    "gpa": "05_grades_gpa.md",
    "балл": "05_grades_gpa.md",
    "ретейк": "05_grades_gpa.md",
    "retake": "05_grades_gpa.md",
    # Регистрация
    "регистрац": "06_registration.md",
    "дисциплин": "06_registration.md",
    # Службы
    "библиотек": "07_services.md",
    "медицинск": "07_services.md",
    "медпункт": "07_services.md",
    "психолог": "07_services.md",
    "карьер": "07_services.md",
    "оплат": "07_services.md",
    "воинск": "07_services.md",
    "справк": "07_services.md",
    "регистратор": "07_services.md",
    "международн": "07_services.md",
    "мобильност": "07_services.md",
    # Студенческая жизнь
    "стипенди": "09_student_life.md",
    "онай": "09_student_life.md",
    "оңай": "09_student_life.md",
    "организаци": "09_student_life.md",
    "клуб": "09_student_life.md",
    "банковск": "09_student_life.md",
    # Контакты
    "контакт": "10_contacts.md",
    "телефон": "10_contacts.md",
    "email": "10_contacts.md",
    "почт": "10_contacts.md",
    # FAQ
    "отчислен": "11_faq.md",
    "перевод": "11_faq.md",
    "академ": "11_faq.md",
    "грант": "11_faq.md",
    "первая неделя": "11_faq.md",
    # Общая информация
    "адрес": "01_general_info.md",
    "корпус": "01_general_info.md",
    "кампус": "01_general_info.md",
    "добраться": "01_general_info.md",
    "расположен": "01_general_info.md",
    "гук": "01_general_info.md",
    "гмк": "01_general_info.md",
    "нк": "01_general_info.md",
    "мук": "01_general_info.md",
    "ккц": "01_general_info.md",
}


class RAGEngine:
    """RAG движок — загрузка документов, создание эмбеддингов, гибридный поиск."""

    def __init__(self):
        """Инициализация RAG движка с ChromaDB и sentence-transformers."""
        logger.info("Инициализация RAG движка...")

        # Создаём функцию эмбеддингов на основе sentence-transformers
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )

        # Инициализируем ChromaDB с постоянным хранилищем
        self.client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

        # Получаем или создаём коллекцию
        self.collection = self.client.get_or_create_collection(
            name="satbayev_knowledge",
            embedding_function=self.embedding_fn,
            metadata={"description": "База знаний Satbayev University для первокурсников"},
        )

        # Кэш всех документов для поиска по ключевым словам
        self._all_docs_cache = None

        logger.info(
            f"RAG движок инициализирован. "
            f"Документов в базе: {self.collection.count()}"
        )

    def _split_into_chunks(self, text: str, source: str) -> List[dict]:
        """Разбивает текст на чанки с перекрытием."""
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + CHUNK_SIZE

            # Ищем конец абзаца или предложения для аккуратного разбиения
            if end < len(text):
                newline_pos = text.rfind("\n\n", start, end)
                if newline_pos > start + CHUNK_SIZE // 2:
                    end = newline_pos + 2
                else:
                    for sep in [". ", ".\n", "!\n", "?\n"]:
                        sep_pos = text.rfind(sep, start, end)
                        if sep_pos > start + CHUNK_SIZE // 2:
                            end = sep_pos + len(sep)
                            break

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "chunk_id": f"{source}_chunk_{chunk_id}",
                })
                chunk_id += 1

            start = end - CHUNK_OVERLAP if end < len(text) else end

        return chunks

    def load_knowledge_base(self) -> dict:
        """Загружает все .md файлы из папки knowledge_base в ChromaDB."""
        logger.info(f"Загрузка базы знаний из {KNOWLEDGE_BASE_DIR}...")

        # Очищаем существующую коллекцию
        existing_count = self.collection.count()
        if existing_count > 0:
            logger.info(f"Очистка существующей коллекции ({existing_count} записей)...")
            all_ids = self.collection.get()["ids"]
            if all_ids:
                self.collection.delete(ids=all_ids)

        # Ищем все .md файлы
        md_files = sorted([
            f for f in os.listdir(KNOWLEDGE_BASE_DIR)
            if f.endswith(".md")
        ])

        if not md_files:
            logger.warning(f"В папке {KNOWLEDGE_BASE_DIR} не найдено .md файлов!")
            return {"files": 0, "chunks": 0}

        all_chunks = []
        files_processed = 0

        for filename in md_files:
            filepath = os.path.join(KNOWLEDGE_BASE_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                chunks = self._split_into_chunks(content, filename)
                all_chunks.extend(chunks)
                files_processed += 1
                logger.info(f"  + {filename}: {len(chunks)} chunk(s)")

            except Exception as e:
                logger.error(f"  ! Error reading {filename}: {e}")

        # Загружаем чанки в ChromaDB
        if all_chunks:
            logger.info(f"Загрузка {len(all_chunks)} чанков в ChromaDB...")

            batch_size = 100
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                self.collection.add(
                    ids=[chunk["chunk_id"] for chunk in batch],
                    documents=[chunk["text"] for chunk in batch],
                    metadatas=[{"source": chunk["source"]} for chunk in batch],
                )

            logger.info(
                f"База знаний загружена: {files_processed} файлов, "
                f"{len(all_chunks)} чанков"
            )

        # Сбрасываем кэш
        self._all_docs_cache = None

        stats = {"files": files_processed, "chunks": len(all_chunks)}
        return stats

    def _get_all_docs(self) -> dict:
        """Получает все документы из ChromaDB (с кэшированием)."""
        if self._all_docs_cache is None:
            self._all_docs_cache = self.collection.get(
                include=["documents", "metadatas"]
            )
        return self._all_docs_cache

    def _keyword_search(self, query: str) -> List[dict]:
        """Поиск чанков по ключевым словам из запроса."""
        query_lower = query.lower()
        matched_files = set()

        # Определяем файлы по ключевым словам
        for keyword, filename in KEYWORD_FILE_MAP.items():
            if keyword in query_lower:
                matched_files.add(filename)

        if not matched_files:
            return []

        # Достаём все документы и фильтруем по файлам
        all_docs = self._get_all_docs()
        results = []

        for i, meta in enumerate(all_docs["metadatas"]):
            if meta["source"] in matched_files:
                results.append({
                    "text": all_docs["documents"][i],
                    "source": meta["source"],
                    "distance": 0.5,  # Средний приоритет для keyword-результатов
                })

        logger.info(
            f"Keyword search: '{query[:40]}...' -> "
            f"файлы {matched_files}, найдено {len(results)} чанков"
        )
        return results

    def _text_match_search(self, query: str) -> List[dict]:
        """Прямой поиск подстроки в тексте чанков (для имён, названий)."""
        query_lower = query.lower()
        all_docs = self._get_all_docs()
        results = []

        for i, doc in enumerate(all_docs["documents"]):
            if query_lower in doc.lower():
                results.append({
                    "text": doc,
                    "source": all_docs["metadatas"][i]["source"],
                    "distance": 0.1,  # Высокий приоритет — точное совпадение
                })

        if results:
            logger.info(
                f"Text match: '{query[:40]}...' -> найдено {len(results)} чанков"
            )
        return results

    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[dict]:
        """Гибридный поиск: вектор + ключевые слова + прямой текст."""
        if self.collection.count() == 0:
            logger.warning("База знаний пуста! Запустите knowledge_loader.py")
            return []

        # 1. Векторный поиск
        vector_results = []
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, self.collection.count()),
            )
            if results and results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    vector_results.append({
                        "text": doc,
                        "source": results["metadatas"][0][i]["source"],
                        "distance": results["distances"][0][i] if results["distances"] else 1.0,
                    })
        except Exception as e:
            logger.error(f"Ошибка векторного поиска: {e}")

        # 2. Поиск по ключевым словам
        keyword_results = self._keyword_search(query)

        # 3. Прямой поиск по тексту (для имён, названий)
        text_results = self._text_match_search(query)

        # Объединяем результаты, убирая дубликаты
        seen_texts = set()
        combined = []

        # Сначала точные совпадения текста (высший приоритет)
        for r in text_results:
            text_key = r["text"][:100]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                combined.append(r)

        # Затем векторные результаты
        for r in vector_results:
            text_key = r["text"][:100]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                combined.append(r)

        # Затем keyword-результаты
        for r in keyword_results:
            text_key = r["text"][:100]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                combined.append(r)

        # Сортируем: сначала точные совпадения, потом по расстоянию
        combined.sort(key=lambda x: x["distance"])

        # Ограничиваем количество (но берём больше чем top_k для keyword)
        max_results = max(top_k, 10)
        combined = combined[:max_results]

        logger.info(
            f"Гибридный поиск '{query[:50]}...' -> "
            f"вектор: {len(vector_results)}, keyword: {len(keyword_results)}, "
            f"text: {len(text_results)}, итого: {len(combined)}"
        )
        return combined

    def get_context(self, query: str) -> str:
        """Получает объединённый контекст из релевантных чанков."""
        results = self.search(query)

        if not results:
            return "Информация по данному запросу не найдена в базе знаний."

        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Источник {i}: {result['source']}]\n{result['text']}"
            )

        return "\n\n---\n\n".join(context_parts)
