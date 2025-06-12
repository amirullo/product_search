from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import uvicorn
import threading

app = FastAPI(title="Semantic Category Search")

# === Категории с подкатегориями ===
categories = {
    "Отделочные материалы": {
        "Шпатлевка": ["Финишная шпатлевка", "Гипсовая шпатлевка"],
        "Грунтовка": ["Глубокого проникновения", "Универсальная"],
        "Плиточный клей": [],
        "Обои флизелиновые": []
    },
    "Строительные материалы": {
        "Гипсокартон": [],
        "Профиль металлический": [],
        "Пена монтажная": [],
        "Утеплитель": []
    },
    "Напольные покрытия": {
        "Ламинат": [],
        "Паркет": []
    }
}

# === Плоский список всех категорий и подкатегорий ===
def flatten_categories(tree):
    flat_list = []
    for parent, subcats in tree.items():
        for cat, subs in subcats.items():
            flat_list.append(cat)
            flat_list.extend(subs)
    return flat_list

flat_categories = flatten_categories(categories)

# === Синонимы и жаргон ===
synonym_dict = {
    "шпаклевка": "Шпатлевка",
    "шпаклёвка": "Шпатлевка",
    "шпакля": "Шпатлевка",
    "мазилка": "Шпатлевка",
    "обойка": "Обои флизелиновые",
    "пенка": "Пена монтажная"
}

# === Кеш модели и индекса ===
class SearchEngine:
    def __init__(self):
        self.model = SentenceTransformer('cointegrated/rubert-tiny2')
        self.flat_categories = flat_categories
        self.embeddings = self.model.encode(self.flat_categories, convert_to_numpy=True)
        faiss.normalize_L2(self.embeddings)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def preprocess_query(self, q):
        q = q.strip().lower()
        return synonym_dict.get(q, q)

    def search(self, query: str, top_k: int = 3):
        q = self.preprocess_query(query)
        q_emb = self.model.encode([q], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        return [(self.flat_categories[i], float(D[0][rank])) for rank, i in enumerate(I[0])]

# === Инициализация при запуске ===
engine = SearchEngine()

# === Модель запроса ===
class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

# === Эндпоинт поиска ===
@app.post("/search")
def search_category(request: SearchRequest):
    results = engine.search(request.query, request.top_k)
    return {"results": [{"category": cat, "score": round(score, 3)} for cat, score in results]}

# === Точка входа ===
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
