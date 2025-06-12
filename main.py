import logging
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from functools import lru_cache
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from elasticsearch import Elasticsearch, exceptions

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('search_logs.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Семантический поиск продукции API",
    version="1.0.0",
    description="REST API для семантического поиска строительных материалов с поддержкой синонимов и жаргона"
)


# Модели для API
class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10
    threshold: Optional[float] = 0.6


class SearchResult(BaseModel):
    category: str
    subcategory: Optional[str] = None
    score: float
    method: str  # 'exact', 'synonym', 'semantic'


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total: int
    processing_time: float


# Кэширование модели
@lru_cache(maxsize=1)
def load_model():
    """Загружаем легкую русскоязычную модель для M2"""
    logger.info("Загружаем модель sentence-transformers...")
    # Используем легкую модель для экономии памяти на M2
    model = SentenceTransformer('cointegrated/rubert-tiny2')
    logger.info("Модель загружена успешно")
    return model


class ProductSearchEngine:
    def __init__(self):
        # Словарь категорий с подкатегориями и синонимами
        self.categories_data = {
            "отделочные_материалы": {
                "name": "Отделочные материалы",
                "subcategories": {
                    "шпатлевка": {
                        "name": "Шпатлевка",
                        "synonyms": ["шпаклевка", "шпатлёвка", "замазка", "выравнивающая смесь", "финишка", "стартовая"]
                    },
                    "обои": {
                        "name": "Обои",
                        "synonyms": ["шпалеры", "стеновые покрытия", "обойка", "бумажные обои", "виниловые обои"]
                    },
                    "краска": {
                        "name": "Краска",
                        "synonyms": ["эмаль", "покрытие", "лкм", "лакокрасочные материалы", "водоэмульсионка", "акрил"]
                    }
                }
            },
            "напольные_покрытия": {
                "name": "Напольные покрытия",
                "subcategories": {
                    "ламинат": {
                        "name": "Ламинат",
                        "synonyms": ["ламинированный пол", "деревянные панели", "напольные панели"]
                    },
                    "плитка": {
                        "name": "Плитка",
                        "synonyms": ["кафель", "керамика", "керамогранит", "плиточка", "кафельная плитка"]
                    },
                    "линолеум": {
                        "name": "Линолеум",
                        "synonyms": ["линолиум", "рулонное покрытие", "пвх покрытие"]
                    }
                }
            },
            "инструменты": {
                "name": "Инструменты",
                "subcategories": {
                    "шпатели": {
                        "name": "Шпатели",
                        "synonyms": ["шпатель", "лопатка", "скребок", "инструмент для шпаклевки"]
                    },
                    "кисти": {
                        "name": "Кисти",
                        "synonyms": ["кисточка", "малярная кисть", "инструмент для покраски"]
                    }
                }
            }
        }

        # Создаем плоский список категорий для эмбеддингов
        self.flat_categories = []
        self.category_mapping = {}

        for cat_id, cat_data in self.categories_data.items():
            for subcat_id, subcat_data in cat_data["subcategories"].items():
                full_name = f"{cat_data['name']} -> {subcat_data['name']}"
                self.flat_categories.append(full_name)
                self.category_mapping[full_name] = {
                    "category": cat_id,
                    "subcategory": subcat_id,
                    "category_name": cat_data['name'],
                    "subcategory_name": subcat_data['name'],
                    "synonyms": subcat_data['synonyms']
                }

        # Инициализация Elasticsearch
        self.es = None
        self.init_elasticsearch()

        # Загружаем модель с кэшированием
        self.model = load_model()

        # Создаем эмбеддинги для категорий
        logger.info("Создаем эмбеддинги для категорий...")
        self.category_embeddings = self.model.encode(self.flat_categories)
        logger.info(f"Созданы эмбеддинги для {len(self.flat_categories)} категорий")

    def init_elasticsearch(self):
        """Инициализация Elasticsearch (опционально)"""
        try:
            self.es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
            if self.es.ping():
                logger.info("Elasticsearch подключен")
                self.setup_elasticsearch_index()
            else:
                logger.warning("Elasticsearch недоступен, используем только векторный поиск")
                self.es = None
        except Exception as e:
            logger.warning(f"Ошибка подключения к Elasticsearch: {e}")
            self.es = None

    def setup_elasticsearch_index(self):
        """Настройка индекса Elasticsearch с русскоязычным анализатором"""
        if not self.es:
            return

        index_name = "products"

        # Настройки индекса с русскоязычным анализатором
        index_settings = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "russian_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "russian_stop", "russian_stemmer"]
                        }
                    },
                    "filter": {
                        "russian_stop": {
                            "type": "stop",
                            "stopwords": "_russian_"
                        },
                        "russian_stemmer": {
                            "type": "stemmer",
                            "language": "russian"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "category": {"type": "text", "analyzer": "russian_analyzer"},
                    "subcategory": {"type": "text", "analyzer": "russian_analyzer"},
                    "synonyms": {"type": "text", "analyzer": "russian_analyzer"},
                    "full_name": {"type": "text", "analyzer": "russian_analyzer"}
                }
            }
        }

        try:
            # Удаляем индекс если существует
            if self.es.indices.exists(index=index_name):
                self.es.indices.delete(index=index_name)

            # Создаем новый индекс
            self.es.indices.create(index=index_name, body=index_settings)

            # Индексируем данные
            for full_name, mapping in self.category_mapping.items():
                doc = {
                    "category": mapping["category_name"],
                    "subcategory": mapping["subcategory_name"],
                    "synonyms": " ".join(mapping["synonyms"]),
                    "full_name": full_name
                }
                self.es.index(index=index_name, body=doc)

            self.es.indices.refresh(index=index_name)
            logger.info("Elasticsearch индекс создан и заполнен")

        except Exception as e:
            logger.error(f"Ошибка настройки Elasticsearch: {e}")

    def search_exact_and_synonyms(self, query: str) -> List[Tuple[str, float, str]]:
        """Поиск по точным совпадениям и синонимам"""
        results = []
        query_lower = query.lower().strip()

        for full_name, mapping in self.category_mapping.items():
            # Проверяем точное совпадение с названием категории
            if query_lower in mapping["subcategory_name"].lower():
                results.append((full_name, 1.0, "exact"))
                continue

            # Проверяем синонимы
            for synonym in mapping["synonyms"]:
                if query_lower in synonym.lower() or synonym.lower() in query_lower:
                    results.append((full_name, 0.9, "synonym"))
                    break

        return results

    def search_semantic(self, query: str, threshold: float = 0.6) -> List[Tuple[str, float, str]]:
        """Семантический поиск через эмбеддинги"""
        try:
            query_embedding = self.model.encode([query])
            similarities = cosine_similarity(query_embedding, self.category_embeddings)[0]

            results = []
            for i, score in enumerate(similarities):
                if score >= threshold:
                    results.append((self.flat_categories[i], float(score), "semantic"))

            return sorted(results, key=lambda x: x[1], reverse=True)
        except Exception as e:
            logger.error(f"Ошибка семантического поиска: {e}")
            return []

    def search_elasticsearch(self, query: str) -> List[Tuple[str, float, str]]:
        """Поиск через Elasticsearch"""
        if not self.es:
            return []

        try:
            search_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["category^2", "subcategory^3", "synonyms^1.5", "full_name^2"],
                        "type": "best_fields",
                        "fuzziness": "AUTO"
                    }
                },
                "size": 20
            }

            response = self.es.search(index="products", body=search_body)
            results = []

            for hit in response['hits']['hits']:
                full_name = hit['_source']['full_name']
                score = float(hit['_score']) / 10.0  # Нормализуем скор
                results.append((full_name, min(score, 1.0), "elasticsearch"))

            return results

        except Exception as e:
            logger.error(f"Ошибка поиска в Elasticsearch: {e}")
            return []

    async def search(self, query: str, threshold: float = 0.6, limit: int = 10) -> List[SearchResult]:
        """Основной метод поиска, комбинирующий все подходы"""
        start_time = datetime.now()

        # Логируем запрос
        logger.info(f"Поиск: '{query}' (threshold={threshold}, limit={limit})")

        all_results = []

        # 1. Точный поиск и синонимы (высший приоритет)
        exact_results = self.search_exact_and_synonyms(query)
        all_results.extend(exact_results)

        # 2. Elasticsearch поиск
        es_results = self.search_elasticsearch(query)
        all_results.extend(es_results)

        # 3. Семантический поиск
        semantic_results = self.search_semantic(query, threshold)
        all_results.extend(semantic_results)

        # Удаляем дубликаты и сортируем по релевантности
        seen = set()
        unique_results = []

        for full_name, score, method in all_results:
            if full_name not in seen:
                seen.add(full_name)
                mapping = self.category_mapping[full_name]

                result = SearchResult(
                    category=mapping["category_name"],
                    subcategory=mapping["subcategory_name"],
                    score=score,
                    method=method
                )
                unique_results.append(result)

        # Сортируем по релевантности
        unique_results.sort(key=lambda x: (-x.score, x.method != "exact"))

        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Поиск завершен за {processing_time:.3f}с, найдено {len(unique_results)} результатов")

        return unique_results[:limit]


# Глобальный экземпляр поискового движка
search_engine = ProductSearchEngine()


@app.get("/")
async def root():
    """Корневой эндпоинт с информацией об API"""
    return {
        "name": "Семантический поиск продукции API",
        "version": "1.0.0",
        "description": "API для семантического поиска строительных материалов",
        "endpoints": {
            "search": "POST /search - Поиск продукции",
            "categories": "GET /categories - Получить все категории",
            "health": "GET /health - Проверка работоспособности",
            "docs": "GET /docs - Swagger документация"
        },
        "examples": {
            "search_query": {
                "query": "шпаклевка",
                "threshold": 0.6,
                "limit": 10
            }
        }
    }


@app.post("/search", response_model=SearchResponse)
async def search_products(search_request: SearchRequest):
    """API эндпоинт для поиска"""
    start_time = datetime.now()

    try:
        results = await search_engine.search(
            query=search_request.query,
            threshold=search_request.threshold,
            limit=search_request.limit
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        return SearchResponse(
            query=search_request.query,
            results=results,
            total=len(results),
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Ошибка поиска: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/categories")
async def get_categories():
    """Получить все категории и подкатегории"""
    return {
        "categories": search_engine.categories_data,
        "total_categories": len(search_engine.categories_data),
        "total_subcategories": sum(len(cat["subcategories"]) for cat in search_engine.categories_data.values())
    }


@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса"""
    try:
        # Проверяем модель
        model_status = "ok" if search_engine.model else "error"

        # Проверяем Elasticsearch
        es_status = "ok" if search_engine.es and search_engine.es.ping() else "unavailable"

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "model": model_status,
                "elasticsearch": es_status,
                "categories": len(search_engine.flat_categories)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Получить статистику поиска"""
    try:
        # Читаем логи для подсчета статистики
        stats = {
            "total_categories": len(search_engine.flat_categories),
            "model_name": "cointegrated/rubert-tiny2",
            "elasticsearch_available": search_engine.es is not None,
            "supported_methods": ["exact", "synonym", "semantic", "elasticsearch"]
        }

        # Можно добавить статистику из логов
        try:
            with open('search_logs.log', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                search_queries = [line for line in lines if 'Поиск:' in line]
                stats["total_searches"] = len(search_queries)
        except FileNotFoundError:
            stats["total_searches"] = 0

        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)