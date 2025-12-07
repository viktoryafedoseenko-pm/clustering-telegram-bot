"""
Модуль для классификации текстов с использованием LLM (YandexGPT).
Позволяет распределить тексты по заданным пользователем категориям.
"""

import logging
import os
import json
import time
from typing import List, Dict, Tuple
import requests
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LLMClassifier:
    """Классификатор текстов с использованием YandexGPT."""
    
    def __init__(self):
        """Инициализация классификатора."""
        self.api_key = os.getenv("YANDEX_API_KEY")
        self.folder_id = os.getenv("YANDEX_FOLDER_ID")
        self.api_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        
        if not self.api_key or not self.folder_id:
            raise ValueError(
                "Для классификации необходимы YANDEX_API_KEY и YANDEX_FOLDER_ID в .env"
            )
    
    def _create_classification_prompt(
        self, text: str, categories: List[str], descriptions: Dict[str, str] = None
    ) -> str:
        """
        Создает промпт для классификации текста.
        
        Args:
            text: Текст для классификации
            categories: Список категорий
            descriptions: Опциональные описания категорий
            
        Returns:
            Промпт для LLM
        """
        categories_text = "\n".join([
            f"{i+1}. {cat}" + (f" - {descriptions.get(cat, '')}" if descriptions and cat in descriptions else "")
            for i, cat in enumerate(categories)
        ])
        
        prompt = f"""Ты - эксперт по классификации текстов. Твоя задача - определить, к какой категории относится текст.

Доступные категории:
{categories_text}

Текст для классификации:
"{text}"

Проанализируй текст и выбери ОДНУ наиболее подходящую категорию из списка выше.

Ответ верни СТРОГО в формате JSON:
{{
    "category": "название_категории_БЕЗ_НОМЕРА",
    "confidence": 0.95,
    "reasoning": "краткое объяснение выбора"
}}

Важно:
- Используй точное название категории из списка (учитывай регистр)
- Если категория написана с ошибкой – не исправляй её
- НЕ добавляй номера типа "1.", "2." и т.д.
- confidence должен быть числом от 0 до 1
- reasoning - краткое объяснение (1-2 предложения)
"""
        return prompt
    
    def _call_yandex_gpt(self, prompt: str, temperature: float = 0.3) -> str:
        """
        Вызов YandexGPT API.
        
        Args:
            prompt: Промпт для модели
            temperature: Температура генерации (0.0-1.0)
            
        Returns:
            Ответ модели
        """
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "modelUri": f"gpt://{self.folder_id}/yandexgpt-lite",
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": 500
            },
            "messages": [
                {
                    "role": "user",
                    "text": prompt
                }
            ]
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result["result"]["alternatives"][0]["message"]["text"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при вызове YandexGPT: {e}")
            raise
    
    def _parse_classification_result(self, response: str) -> Tuple[str, float, str]:
        """Парсит ответ модели."""
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                
                category = result.get("category", "")
                confidence = float(result.get("confidence", 0.0))
                reasoning = result.get("reasoning", "")
                
                # Если категория начинается с "1. ", "2. " и т.д. - убираем номер
                import re
                category = re.sub(r'^\d+\.\s*', '', category)
                
                return (category, confidence, reasoning)
            else:
                logger.warning(f"Не удалось найти JSON в ответе: {response}")
                return ("", 0.0, "Ошибка парсинга")
                
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON: {e}\nОтвет: {response}")
            return ("", 0.0, "Ошибка парсинга JSON")

    
    def classify_text(
        self,
        text: str,
        categories: List[str],
        descriptions: Dict[str, str] = None
    ) -> Dict[str, any]:
        """
        Классифицирует один текст.
        
        Args:
            text: Текст для классификации
            categories: Список категорий
            descriptions: Опциональные описания категорий
            
        Returns:
            Словарь с результатами классификации
        """
        prompt = self._create_classification_prompt(text, categories, descriptions)
        response = self._call_yandex_gpt(prompt)
        category, confidence, reasoning = self._parse_classification_result(response)
        
        # Проверяем, что категория из списка
        if category not in categories:
            logger.warning(
                f"Модель вернула категорию '{category}', которой нет в списке. "
                f"Используем первую категорию из списка."
            )
            category = categories[0]
            confidence = 0.5
            reasoning = "Категория выбрана автоматически из-за ошибки модели"
        
        return {
            "category": category,
            "confidence": confidence,
            "reasoning": reasoning
        }
    
    def classify_batch(
        self,
        texts: List[str],
        categories: List[str],
        descriptions: Dict[str, str] = None,
        batch_delay: float = 1.0,
        progress_callback=None
    ) -> pd.DataFrame:
        """
        Классифицирует батч текстов.
        
        Args:
            texts: Список текстов
            categories: Список категорий
            descriptions: Опциональные описания категорий
            batch_delay: Задержка между запросами (сек)
            progress_callback: Функция для обновления прогресса
            
        Returns:
            DataFrame с результатами классификации
        """
        results = []
        
        logger.info(f"Начинаем классификацию {len(texts)} текстов по {len(categories)} категориям")
        
        for i, text in enumerate(tqdm(texts, desc="Классификация")):
            try:
                result = self.classify_text(text, categories, descriptions)
                results.append({
                    "text": text,
                    "category": result["category"],
                    "confidence": result["confidence"],
                    "reasoning": result["reasoning"]
                })
                
                if progress_callback:
                    progress = (i + 1) / len(texts) * 100
                    progress_callback(progress, i + 1, len(texts))
                
                # Задержка между запросами для соблюдения rate limits
                if i < len(texts) - 1:
                    time.sleep(batch_delay)
                    
            except Exception as e:
                logger.error(f"Ошибка при классификации текста {i}: {e}")
                results.append({
                    "text": text,
                    "category": categories[0],  # Fallback на первую категорию
                    "confidence": 0.0,
                    "reasoning": f"Ошибка: {str(e)}"
                })
        
        df = pd.DataFrame(results)
        
        # Добавляем статистику
        logger.info(f"Классификация завершена. Распределение по категориям:")
        for category in categories:
            count = len(df[df['category'] == category])
            percentage = count / len(df) * 100
            logger.info(f"  {category}: {count} ({percentage:.1f}%)")
        
        return df
    
    def get_classification_stats(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Получает статистику по результатам классификации.
        
        Args:
            df: DataFrame с результатами классификации
            
        Returns:
            Словарь со статистикой
        """
        stats = {
            "total_texts": len(df),
            "categories": {},
            "avg_confidence": float(df['confidence'].mean()),
            "min_confidence": float(df['confidence'].min()),
            "max_confidence": float(df['confidence'].max())
        }
        
        # Статистика по категориям
        for category in df['category'].unique():
            category_df = df[df['category'] == category]
            stats["categories"][category] = {
                "count": len(category_df),
                "percentage": len(category_df) / len(df) * 100,
                "avg_confidence": float(category_df['confidence'].mean())
            }
        
        return stats


def validate_categories(categories: List[str]) -> Tuple[bool, str]:
    """
    Валидирует список категорий.
    
    Args:
        categories: Список категорий
        
    Returns:
        Кортеж (валидно, сообщение об ошибке)
    """
    if not categories:
        return False, "Список категорий пуст"
    
    if len(categories) < 2:
        return False, "Необходимо минимум 2 категории"
    
    if len(categories) > 20:
        return False, "Максимум 20 категорий"
    
    # Проверка на дубликаты
    if len(categories) != len(set(categories)):
        return False, "Есть дублирующиеся категории"
    
    # Проверка длины названий
    for cat in categories:
        if not cat or not cat.strip():
            return False, "Есть пустые категории"
        if len(cat) > 100:
            return False, f"Категория '{cat}' слишком длинная (макс. 100 символов)"
    
    return True, ""


def parse_categories_from_text(text: str) -> List[str]:
    """
    Парсит категории из текста пользователя.
    Поддерживает разделители: новая строка, запятая, точка с запятой.
    
    Args:
        text: Текст с категориями
        
    Returns:
        Список категорий
    """
    # Пробуем разные разделители
    if '\n' in text:
        categories = text.split('\n')
    elif ';' in text:
        categories = text.split(';')
    elif ',' in text:
        categories = text.split(',')
    else:
        categories = [text]
    
    # Очистка и фильтрация
    categories = [
        cat.strip().strip('0123456789.-) ')  # Убираем номера списков
        for cat in categories
        if cat.strip()
    ]
    
    return categories
