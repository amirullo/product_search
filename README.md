# 🔍 Семантический поиск продукции - REST API

Минимальный рабочий пример семантического поиска с поддержкой синонимов, жаргона и подкатегорий, оптимизированный для Apple M2 16GB.

## ✨ Возможности

- **Семантический поиск** через легкую русскоязычную модель `rubert-tiny2`
- **Поддержка синонимов и жаргона** (шпакля → шпатлевка, кафель → плитка)
- **Многоуровневые категории** с подкатегориями
- **Elasticsearch** с русскоязычными анализаторами
- **REST API** на FastAPI с автоматической документацией
- **Кеширование модели** для экономии памяти
- **Логирование запросов** пользователей
- **Гибридный поиск** (точный + синонимы + семантический + Elasticsearch)

## 📋 Предустановленные категории

### Отделочные материалы
- **Шпатлевка**: шпаклевка, замазка, выравнивающая смесь, финишка
- **Обои**: шпалеры, стеновые покрытия, обойка
- **Краска**: эмаль, покрытие, лкм, водоэмульсионка, акрил

### Напольные покрытия  
- **Ламинат**: ламинированный пол, деревянные панели
- **Плитка**: кафель, керамика, керамогранит
- **Линолеум**: рулонное покрытие, пвх покрытие

### Инструменты
- **Шпатели**: лопатка, скребок
- **Кисти**: кисточка, малярная кисть

## 🚀 Быстрый старт

### 1. Клонирование и подготовка
```bash
# Создайте директорию проекта
mkdir semantic-search && cd semantic-search

# Скопируйте файлы из артефактов:
# ├── main.py
# ├── requirements.txt  
# ├── docker-compose.yml
# ├── run.sh
# └── README.md
```

### 2. Автоматический запуск (рекомендуется)
```bash
chmod +x run.sh
./run.sh
```

### 3. Ручная установка
```bash
# Создание виртуального окружения
python3 -m venv venv
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt

# Запуск Elasticsearch и Redis (опционально)
docker-compose up -d

# Запуск приложения
python3 main.py
```

## 🌐 Использование API

### Основные эндпоинты

**Swagger документация:** http://localhost:8000/docs

#### 1. Поиск продукции
```bash
POST /search
Content-Type: application/json

{
  "query": "шпаклевка",
  "threshold": 0.6,
  "limit": 10
}
```

**Пример ответа:**
```json
{
  "query": "шпаклевка",
  "results": [
    {
      "category": "Отделочные материалы",
      "subcategory": "Шпатлевка", 
      "score": 0.9,
      "method": "synonym"
    }
  ],
  "total": 1,
  "processing_time": 0.123
}
```

#### 2. Получить все категории
```bash
GET /categories
```

#### 3. Проверка работоспособности  
```bash
GET /health
```

#### 4. Статистика поиска
```bash
GET /stats
```

### Примеры cURL запросов

```bash
# Поиск по синониму
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "шпаклевка", "threshold": 0.6}'

# Семантический поиск
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "напольные панели", "threshold": 0.5}'

# Получить категории
curl "http://localhost:8000/categories"

# Проверить статус
curl "http://localhost:8000/health"
```

### Python клиент

```python
import requests

# Поиск
response = requests.post("http://localhost:8000/search", json={
    "query": "кафель",
    "threshold": 0.6,
    "limit": 5
})

results = response.json()
for result in results["results"]:
    print(f"{result['category']} -> {result['subcategory']} ({result['score']:.2f})")
```

## 🔧 Настройки для Apple M2

### Оптимизация памяти
- Используется легкая модель `rubert-tiny2` (~40MB вместо ~500MB)
- Кеширование модели через `@lru_cache` 
- Ограничение памяти Elasticsearch: 512MB
- Отключена CUDA для совместимости с M2

### Производительность
- Векторизация на CPU оптимизирована для ARM64
- Batch-обработка для множественных запросов
- Асинхронные операции в FastAPI

## 📊 Алгоритм поиска

1. **Точное совпадение** (score = 1.0)
2. **Поиск по синонимам** (score = 0.9) 
3. **Elasticsearch** с нечетким поиском (score нормализован)
4. **Семантический поиск** через эмбеддинги (configurable threshold)

Результаты дедуплицируются и сортируются по релевантности.

## 📝 Логирование

Все запросы логируются в:
- **Консоль** для разработки
- **Файл** `search_logs.log` для анализа

Формат лога:
```
2024-06-09 12:34:56 - __main__ - INFO - Поиск: 'шпаклевка' (threshold=0.6, limit=10)
2024-06-09 12:34:56 - __main__ - INFO - Поиск завершен за 0.123с, найдено 3 результатов
```

## 🗃️ Добавление новых категорий

Отредактируйте `categories_data` в `main.py`:

```python
"новая_категория": {
    "name": "Новая категория",
    "subcategories": {
        "подкатегория": {
            "name": "Подкатегория",
            "synonyms": ["синоним1", "синоним2", "жаргон"]
        }
    }
}
```

## 🐳 Docker

```bash
# Только инфраструктура
docker-compose up -d

# Остановка
docker-compose down

# Очистка данных
docker-compose down -v
```

## 🔍 Мониторинг

- **Elasticsearch**: http://localhost:9200/_cluster/health
- **API docs**: http://localhost:8000/docs  
- **Логи**: `tail -f search_logs.log`

## ⚡ Устранение неполадок

### Elasticsearch не запускается
```bash
# Проверка статуса
curl http://localhost:9200

# Увеличение памяти Docker (в Docker Desktop)
# Settings → Resources → Memory → 4GB+
```

### Модель не загружается  
```bash
# Очистка кеша
rm -rf ~/.cache/huggingface/

# Ручная загрузка
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('cointegrated/rubert-tiny2')"
```

### Ошибки памяти на M2
```bash
# Уменьшение batch size
export TOKENIZERS_PARALLELISM=false
ulimit -v 4194304  # 4GB limit
```

## 📈 Расширение функционала

### Добавление новых моделей
```python
# В main.py замените модель:
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
```

### Интеграция с базой данных
```python
# Замените categories_data на запросы к БД
async def load_categories_from_db():
    # Ваш код подключения к БД
    pass
```

### API аутентификация
```python
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")
```


## 📄 Лицензия

MIT License - свободное использование и изменение.

## 🆘 Поддержка

При возникновении проблем:
1. Проверьте логи в `search_logs.log`
2. Убедитесь, что все зависимости установлены
3. Проверьте доступность портов 8000, 9200, 6379
4. Попробуйте перезапустить Docker контейнеры
