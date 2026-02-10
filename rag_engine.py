# rag_engine.py — RAG движок для поиска по базе знаний
# Использует ChromaDB для векторного хранилища и sentence-transformers для эмбеддингов

import os
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


class RAGEngine:
    """RAG движок — загрузка документов, создание эмбеддингов, поиск по базе знаний."""

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

        logger.info(
            f"RAG движок инициализирован. "
            f"Документов в базе: {self.collection.count()}"
        )

    def _split_into_chunks(self, text: str, source: str) -> List[dict]:
        """Разбивает текст на чанки с перекрытием.

        Args:
            text: Исходный текст документа.
            source: Имя файла-источника.

        Returns:
            Список словарей с полями 'text', 'source', 'chunk_id'.
        """
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + CHUNK_SIZE

            # Ищем конец предложения или абзаца для более аккуратного разбиения
            if end < len(text):
                # Пробуем найти конец абзаца
                newline_pos = text.rfind("\n\n", start, end)
                if newline_pos > start + CHUNK_SIZE // 2:
                    end = newline_pos + 2
                else:
                    # Пробуем найти конец предложения
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

            # Сдвигаем начало с учётом перекрытия
            start = end - CHUNK_OVERLAP if end < len(text) else end

        return chunks

    def load_knowledge_base(self) -> dict:
        """Загружает все .md файлы из папки knowledge_base в ChromaDB.

        Returns:
            Словарь со статистикой: количество файлов и чанков.
        """
        logger.info(f"Загрузка базы знаний из {KNOWLEDGE_BASE_DIR}...")

        # Очищаем существующую коллекцию для перезагрузки
        existing_count = self.collection.count()
        if existing_count > 0:
            logger.info(f"Очистка существующей коллекции ({existing_count} записей)...")
            # Получаем все ID и удаляем
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
                logger.info(f"  ✓ {filename}: {len(chunks)} чанков")

            except Exception as e:
                logger.error(f"  ✗ Ошибка при чтении {filename}: {e}")

        # Загружаем чанки в ChromaDB
        if all_chunks:
            logger.info(f"Загрузка {len(all_chunks)} чанков в ChromaDB...")

            # ChromaDB принимает данные пакетами
            batch_size = 100
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                self.collection.add(
                    ids=[chunk["chunk_id"] for chunk in batch],
                    documents=[chunk["text"] for chunk in batch],
                    metadatas=[{"source": chunk["source"]} for chunk in batch],
                )

            logger.info(
                f"✅ База знаний загружена: {files_processed} файлов, "
                f"{len(all_chunks)} чанков"
            )

        stats = {"files": files_processed, "chunks": len(all_chunks)}
        return stats

    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[dict]:
        """Поиск релевантных чанков по запросу.

        Args:
            query: Текст запроса пользователя.
            top_k: Количество результатов для возврата.

        Returns:
            Список словарей с полями 'text', 'source', 'distance'.
        """
        if self.collection.count() == 0:
            logger.warning("База знаний пуста! Сначала запустите knowledge_loader.py")
            return []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, self.collection.count()),
            )

            search_results = []
            if results and results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    search_results.append({
                        "text": doc,
                        "source": results["metadatas"][0][i]["source"],
                        "distance": results["distances"][0][i] if results["distances"] else None,
                    })

            logger.info(
                f"Поиск по запросу '{query[:50]}...' — "
                f"найдено {len(search_results)} результатов"
            )
            return search_results

        except Exception as e:
            logger.error(f"Ошибка при поиске: {e}")
            return []

    def get_context(self, query: str) -> str:
        """Получает объединённый контекст из релевантных чанков.

        Args:
            query: Текст запроса пользователя.

        Returns:
            Строка с объединённым контекстом для передачи в LLM.
        """
        results = self.search(query)

        if not results:
            return "Информация по данному запросу не найдена в базе знаний."

        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Источник {i}: {result['source']}]\n{result['text']}"
            )

        return "\n\n---\n\n".join(context_parts)
