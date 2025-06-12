#!/bin/bash

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Запуск семантического поиска продукции${NC}"

# Проверяем наличие Python 3.8+
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 не найден. Установите Python 3.8+${NC}"
    exit 1
fi

# Проверяем версию Python
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "${BLUE}🐍 Используется Python ${PYTHON_VERSION}${NC}"

# Создаем директории
mkdir -p logs data

echo -e "${YELLOW}📦 Установка зависимостей...${NC}"

# Создаем виртуальное окружение если его нет
if [ ! -d "venv" ]; then
    echo -e "${BLUE}🔧 Создание виртуального окружения...${NC}"
    python3 -m venv venv
fi

# Активируем виртуальное окружение
source venv/bin/activate

# Обновляем pip
pip install --upgrade pip

# Устанавливаем зависимости
echo -e "${BLUE}📚 Установка библиотек...${NC}"
pip install -r requirements.txt

# Запускаем Docker контейнеры (если Docker доступен)
if command -v docker-compose &> /dev/null; then
    echo -e "${BLUE}🐳 Запуск Elasticsearch и Redis...${NC}"
    docker-compose up -d
    
    # Ждем запуска Elasticsearch
    echo -e "${YELLOW}⏳ Ожидание запуска Elasticsearch...${NC}"
    while ! curl -s http://localhost:9200 > /dev/null; do
        sleep 2
        echo -n "."
    done
    echo -e "${GREEN}✅ Elasticsearch запущен${NC}"
else
    echo -e "${YELLOW}⚠️  Docker не найден. Elasticsearch будет недоступен${NC}"
fi

# Скачиваем модель (если еще не скачана)
echo -e "${BLUE}🤖 Подготовка модели машинного обучения...${NC}"
python3 -c "
from sentence_transformers import SentenceTransformer
import os
print('📥 Загрузка модели...')
model = SentenceTransformer('cointegrated/rubert-tiny2')
print('✅ Модель готова')
"

echo -e "${GREEN}🎉 Все готово! Запуск API сервера...${NC}"

# Запускаем FastAPI сервер
echo -e "${BLUE}🌐 API доступен по адресу: http://localhost:8000${NC}"
echo -e "${BLUE}📖 API документация: http://localhost:8000/docs${NC}"
echo -e "${BLUE}🔍 Тестовый поиск: curl -X POST http://localhost:8000/search -H 'Content-Type: application/json' -d '{\"query\":\"шпаклевка\"}'${NC}"
echo -e "${YELLOW}⏹️  Для остановки нажмите Ctrl+C${NC}"

python3 main.py