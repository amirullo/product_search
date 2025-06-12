#!/usr/bin/env python3
"""
Скрипт для тестирования API семантического поиска
"""

import requests
import json
import time
from typing import Dict, List

API_BASE_URL = "http://localhost:8000"

def test_health():
    """Тест работоспособности API"""
    print("🔍 Проверка работоспособности API...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API работает: {data['status']}")
            print(f"   - Модель: {data['components']['model']}")
            print(f"   - Elasticsearch: {data['components']['elasticsearch']}")
            print(f"   - Категорий: {data['components']['categories']}")
            return True
        else:
            print(f"❌ API недоступен: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Не удается подключиться к API. Убедитесь, что сервер запущен.")
        return False

def test_root():
    """Тест корневого эндпоинта"""
    print("\n📋 Проверка информации об API...")
    
    response = requests.get(f"{API_BASE_URL}/")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ {data['name']} v{data['version']}")
        print(f"   Доступные эндпоинты: {len(data['endpoints'])}")
        return True
    else:
        print(f"❌ Ошибка получения информации: {response.status_code}")
        return False

def test_categories():
    """Тест получения категорий"""
    print("\n📂 Получение категорий...")
    
    response = requests.get(f"{API_BASE_URL}/categories")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Загружено категорий: {data['total_categories']}")
        print(f"   Подкатегорий: {data['total_subcategories']}")
        
        # Выводим первые несколько категорий
        for cat_id, cat_data in list(data['categories'].items())[:2]:
            print(f"   📁 {cat_data['name']}:")
            for subcat_id, subcat_data in list(cat_data['subcategories'].items())[:2]:
                synonyms = ', '.join(subcat_data['synonyms'][:3])
                print(f"      - {subcat_data['name']} (синонимы: {synonyms}...)")
        
        return True
    else:
        print(f"❌ Ошибка получения категорий: {response.status_code}")
        return False

def test_search(query: str, expected_category: str = None, threshold: float = 0.6):
    """Тест поиска"""
    print(f"\n🔍 Поиск: '{query}'")
    
    payload = {
        "query": query,
        "threshold": threshold,
        "limit": 5
    }
    
    start_time = time.time()
    response = requests.post(f"{API_BASE_URL}/search", json=payload)
    response_time = time.time() - start_time
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Найдено результатов: {data['total']}")
        print(f"   Время обработки: {data['processing_time']:.3f}с (запрос: {response_time:.3f}с)")
        
        if data['results']:
            print("   Результаты:")
            for i, result in enumerate(data['results'][:3], 1):
                score_percent = int(result['score'] * 100)
                method_emoji = {
                    'exact': '🎯',
                    'synonym': '🔄', 
                    'semantic': '🧠',
                    'elasticsearch': '🔍'
                }
                emoji = method_emoji.get(result['method'], '❓')
                print(f"   {i}. {emoji} {result['category']} → {result['subcategory']} ({score_percent}%, {result['method']})")
            
            # Проверяем ожидаемую категорию
            if expected_category:
                found_categories = [r['subcategory'].lower() for r in data['results']]
                if expected_category.lower() in found_categories:
                    print(f"   ✅ Ожидаемая категория '{expected_category}' найдена")
                else:
                    print(f"   ⚠️  Ожидаемая категория '{expected_category}' не найдена")
        else:
            print("   🤷‍♂️ Результатов не найдено")
        
        return True
    else:
        print(f"❌ Ошибка поиска: {response.status_code}")
        try:
            error_data = response.json()
            print(f"   Детали: {error_data.get('detail', 'Неизвестная ошибка')}")
        except:
            print(f"   Ответ сервера: {response.text}")
        return False

def test_stats():
    """Тест статистики"""
    print("\n📊 Получение статистики...")
    
    response = requests.get(f"{API_BASE_URL}/stats")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Статистика получена:")
        print(f"   - Всего категорий: {data['total_categories']}")
        print(f"   - Модель: {data['model_name']}")
        print(f"   - Elasticsearch: {'✅' if data['elasticsearch_available'] else '❌'}")
        print(f"   - Методы поиска: {', '.join(data['supported_methods'])}")
        print(f"   - Всего поисков: {data.get('total_searches', 'N/A')}")
        return True
    else:
        print(f"❌ Ошибка получения статистики: {response.status_code}")
        return False

def run_performance_test():
    """Тест производительности"""
    print("\n⚡ Тест производительности...")
    
    test_queries = [
        "шпаклевка", "кафель", "эмаль", "ламинат", "обои",
        "линолеум", "керамогранит", "водоэмульсионка"
    ]
    
    times = []
    successful = 0
    
    for query in test_queries:
        payload = {"query": query, "threshold": 0.6, "limit": 3}
        
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/search", json=payload)
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            times.append(data['processing_time'])
            successful += 1
            print(f"   {query}: {data['processing_time']:.3f}с ({len(data['results'])} результатов)")
        else:
            print(f"   {query}: ❌ ошибка")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n📈 Результаты производительности:")
        print(f"   - Успешных запросов: {successful}/{len(test_queries)}")
        print(f"   - Среднее время: {avg_time:.3f}с")
        print(f"   - Мин/Макс время: {min_time:.3f}с / {max_time:.3f}с")
        print(f"   - Запросов в секунду: ~{1/avg_time:.1f}")

def main():
    """Основная функция тестирования"""
    print("🧪 Тестирование API семантического поиска\n")
    
    # Базовые тесты
    if not test_health():
        return
    
    test_root()
    test_categories()
    test_stats()
    
    print("\n" + "="*50)
    print("🔍 ТЕСТИРОВАНИЕ ПОИСКА")
    print("="*50)
    
    # Тесты поиска
    test_cases = [
        ("шпаклевка", "Шпатлевка", 0.6),    # Синоним
        ("кафель", "Плитка", 0.6),          # Синоним
        ("эмаль", "Краска", 0.6),           # Синоним  
        ("водоэмульсионка", "Краска", 0.6), # Жаргон
        ("напольные панели", "Ламинат", 0.5), # Семантика
        ("керамогранит", "Плитка", 0.6),    # Синоним
        ("малярная кисть", "Кисти", 0.5),   # Семантика
        ("линолиум", "Линолеум", 0.6),      # Опечатка + точное
        ("несуществующий товар", None, 0.6) # Негативный тест
    ]
    
    successful_searches = 0
    for query, expected, threshold in test_cases:
        if test_search(query, expected, threshold):
            successful_searches += 1
    
    # Тест производительности
    run_performance_test()
    
    print(f"\n🎯 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print(f"   Успешных поисков: {successful_searches}/{len(test_cases)}")
    print(f"   API готов к использованию: {'✅' if successful_searches >= len(test_cases) - 1 else '⚠️'}")

if __name__ == "__main__":
    main()