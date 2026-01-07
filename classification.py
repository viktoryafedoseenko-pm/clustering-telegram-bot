"""
Модуль для классификации текстов с использованием LLM (YandexGPT).
Позволяет распределить тексты по заданным пользователем категориям.
"""

import logging
import os
import json
import time
import re
from typing import List, Dict, Tuple, Optional
import requests
from requests.exceptions import RequestException, Timeout, HTTPError
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Константы для retry логики
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # секунды
RATE_LIMIT_DELAY = 5  # секунды при 429


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
- Если текст НЕ подходит ни под одну категорию - используй null 
- НЕ добавляй номера типа "1.", "2." и т.д.
- confidence должен быть числом от 0 до 1
- reasoning - краткое объяснение (1-2 предложения)
"""
        return prompt
    
    def _call_yandex_gpt(self, prompt: str, temperature: float = 0.3, max_retries: int = MAX_RETRIES) -> str:
        """
        Вызов YandexGPT API с retry логикой.
        
        Args:
            prompt: Промпт для модели
            temperature: Температура генерации (0.0-1.0)
            max_retries: Максимальное количество попыток
            
        Returns:
            Ответ модели
            
        Raises:
            RequestException: При ошибках API после всех попыток
            ValueError: При невалидном ответе API
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
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                # Обработка rate limit (429)
                if response.status_code == 429:
                    wait_time = RATE_LIMIT_DELAY * (attempt + 1)
                    logger.warning(f"Rate limit (429), ждём {wait_time} сек (попытка {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                
                # Обработка других HTTP ошибок
                try:
                    response.raise_for_status()
                except HTTPError as e:
                    error_detail = ""
                    try:
                        error_json = response.json()
                        error_detail = f" | Детали: {error_json}"
                    except:
                        error_detail = f" | Ответ: {response.text[:200]}"
                    
                    # Для 5xx ошибок делаем retry
                    if 500 <= response.status_code < 600:
                        wait_time = RETRY_DELAY_BASE * (attempt + 1)
                        logger.warning(f"Серверная ошибка {response.status_code}, ждём {wait_time} сек (попытка {attempt + 1}/{max_retries}){error_detail}")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Для 4xx ошибок не делаем retry (кроме 429)
                        logger.error(f"Ошибка API {response.status_code}: {e}{error_detail}")
                        raise ValueError(f"Ошибка API {response.status_code}: {response.text[:200]}")
                
                # Парсинг успешного ответа
                try:
                    result = response.json()
                except json.JSONDecodeError as e:
                    logger.error(f"Не удалось распарсить JSON ответа: {e} | Ответ: {response.text[:200]}")
                    raise ValueError(f"Невалидный JSON в ответе API: {response.text[:200]}")
                
                # Проверка структуры ответа
                if "result" not in result:
                    logger.error(f"Неожиданная структура ответа: {result.keys()}")
                    raise ValueError("Ответ API не содержит поле 'result'")
                
                if "alternatives" not in result["result"] or len(result["result"]["alternatives"]) == 0:
                    logger.error(f"Нет альтернатив в ответе: {result}")
                    raise ValueError("Ответ API не содержит альтернатив")
                
                message_text = result["result"]["alternatives"][0]["message"]["text"]
                if not message_text or not message_text.strip():
                    logger.warning("Пустой ответ от модели")
                    raise ValueError("Модель вернула пустой ответ")
                
                return message_text
                
            except Timeout as e:
                wait_time = RETRY_DELAY_BASE * (attempt + 1)
                logger.warning(f"Timeout при запросе к API, ждём {wait_time} сек (попытка {attempt + 1}/{max_retries})")
                last_exception = e
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                continue
                
            except RequestException as e:
                # Для сетевых ошибок делаем retry
                wait_time = RETRY_DELAY_BASE * (attempt + 1)
                logger.warning(f"Сетевая ошибка при запросе к API: {e}, ждём {wait_time} сек (попытка {attempt + 1}/{max_retries})")
                last_exception = e
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                continue
                
            except ValueError as e:
                # Для ошибок валидации не делаем retry
                logger.error(f"Ошибка валидации ответа API: {e}")
                raise
        
        # Если все попытки исчерпаны
        error_msg = f"Не удалось получить ответ от API после {max_retries} попыток"
        if last_exception:
            error_msg += f": {last_exception}"
        logger.error(error_msg)
        raise RequestException(error_msg) from last_exception
    
    def _parse_classification_result(self, response: str) -> Tuple[Optional[str], float, str]:
        """
        Парсит ответ модели с улучшенной обработкой ошибок.
        
        Args:
            response: Ответ от модели
            
        Returns:
            Кортеж (category, confidence, reasoning)
        """
        if not response or not response.strip():
            logger.warning("Получен пустой ответ от модели")
            return (None, 0.0, "Пустой ответ от модели")
        
        try:
            # Ищем JSON в ответе (может быть обёрнут в текст)
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start == -1 or json_end <= json_start:
                # Пробуем найти JSON в markdown code blocks
                code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
                if code_block_match:
                    json_str = code_block_match.group(1)
                else:
                    logger.warning(f"Не удалось найти JSON в ответе. Ответ: {response[:300]}")
                    return (None, 0.0, f"Не найден JSON в ответе: {response[:100]}...")
            else:
                json_str = response[json_start:json_end]
            
            # Парсим JSON
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"Ошибка парсинга JSON: {e}\nJSON строка: {json_str[:200]}")
                # Пробуем исправить частые проблемы
                json_str_fixed = json_str.replace("'", '"')  # Заменяем одинарные кавычки
                try:
                    result = json.loads(json_str_fixed)
                    logger.info("Удалось исправить JSON заменой кавычек")
                except:
                    return (None, 0.0, f"Ошибка парсинга JSON: {str(e)[:100]}")
            
            # Извлекаем поля
            category = result.get("category")
            
            # Обработка null/None категории
            if category is None or (isinstance(category, str) and category.lower() in ("null", "none", "")):
                category = None
            else:
                # Очистка категории от номеров и лишних символов
                category = str(category).strip()
                category = re.sub(r'^\d+\.\s*', '', category)  # Убираем номера типа "1. "
                category = category.strip('"\'')  # Убираем кавычки если есть
            
            # Валидация confidence
            confidence_raw = result.get("confidence")
            try:
                confidence = float(confidence_raw) if confidence_raw is not None else 0.0
                # Ограничиваем диапазон
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                logger.warning(f"Невалидное значение confidence: {confidence_raw}, используем 0.0")
                confidence = 0.0
            
            # Извлекаем reasoning
            reasoning = result.get("reasoning", "")
            if not isinstance(reasoning, str):
                reasoning = str(reasoning) if reasoning else ""
            reasoning = reasoning.strip()
            
            return (category, confidence, reasoning)
                
        except Exception as e:
            logger.error(f"Неожиданная ошибка при парсинге ответа: {e}\nОтвет: {response[:300]}", exc_info=True)
            return (None, 0.0, f"Ошибка парсинга: {str(e)[:100]}")


    
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
            
        Raises:
            ValueError: При невалидных входных данных
        """
        # Валидация входных данных
        if not text or not isinstance(text, str) or not text.strip():
            logger.warning("Получен пустой текст для классификации")
            return {
                "category": "⚠️ Не удалось определить",
                "confidence": 0.0,
                "reasoning": "Пустой текст"
            }
        
        if not categories or len(categories) == 0:
            raise ValueError("Список категорий не может быть пустым")
        
        # Валидация категорий
        is_valid, error_msg = validate_categories(categories)
        if not is_valid:
            raise ValueError(f"Невалидные категории: {error_msg}")
        
        try:
            prompt = self._create_classification_prompt(text, categories, descriptions)
            response = self._call_yandex_gpt(prompt)
            category, confidence, reasoning = self._parse_classification_result(response)
            
            # Если модель не смогла определить категорию
            if category is None or category == "" or str(category).lower() in ("none", "null"):
                logger.info(f"Модель не смогла определить категорию для: {text[:50]}...")
                category = "⚠️ Не удалось определить"
                confidence = 1.0  # Высокая уверенность что не подходит
                reasoning = reasoning or "Текст не соответствует ни одной из заданных категорий"
            
            # Проверка что категория из списка (но пропускаем специальную категорию)
            elif category not in categories:
                # Пробуем найти похожую категорию (case-insensitive)
                category_lower = category.lower()
                matched_category = None
                for cat in categories:
                    if cat.lower() == category_lower:
                        matched_category = cat
                        break
                
                if matched_category:
                    logger.info(f"Исправлена категория '{category}' -> '{matched_category}' (регистр)")
                    category = matched_category
                else:
                    logger.warning(f"Модель вернула неизвестную категорию: '{category}' (доступные: {categories[:3]}...)")
                    # Оставляем как есть - покажет что модель вернула
            
            return {
                "category": category,
                "confidence": confidence,
                "reasoning": reasoning
            }
            
        except (RequestException, ValueError) as e:
            # Для ошибок API и валидации логируем и возвращаем fallback
            logger.error(f"Ошибка при классификации текста '{text[:50]}...': {e}")
            return {
                "category": "⚠️ Не удалось определить",
                "confidence": 0.0,
                "reasoning": f"Ошибка API: {str(e)[:100]}"
            }
        except Exception as e:
            # Для неожиданных ошибок логируем с traceback
            logger.error(f"Неожиданная ошибка при классификации текста '{text[:50]}...': {e}", exc_info=True)
            return {
                "category": "⚠️ Не удалось определить",
                "confidence": 0.0,
                "reasoning": f"Неожиданная ошибка: {str(e)[:100]}"
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
            
        Raises:
            ValueError: При невалидных входных данных
        """
        # Валидация входных данных
        if not texts or len(texts) == 0:
            raise ValueError("Список текстов не может быть пустым")
        
        if not categories or len(categories) == 0:
            raise ValueError("Список категорий не может быть пустым")
        
        is_valid, error_msg = validate_categories(categories)
        if not is_valid:
            raise ValueError(f"Невалидные категории: {error_msg}")
        
        results = []
        error_count = 0
        undefined_count = 0
        
        logger.info(f"Начинаем классификацию {len(texts)} текстов по {len(categories)} категориям")
        
        for i, text in enumerate(tqdm(texts, desc="Классификация")):
            try:
                result = self.classify_text(text, categories, descriptions)
                
                # Подсчитываем ошибки и неопределённые
                if result["category"] == "⚠️ Не удалось определить":
                    undefined_count += 1
                if result["confidence"] == 0.0 and "Ошибка" in result.get("reasoning", ""):
                    error_count += 1
                
                results.append({
                    "text": text,
                    "category": result["category"],
                    "confidence": result["confidence"],
                    "reasoning": result["reasoning"]
                })
                
                # Обновление прогресса
                if progress_callback:
                    try:
                        progress = (i + 1) / len(texts) * 100
                        progress_callback(progress, i + 1, len(texts))
                    except Exception as e:
                        logger.warning(f"Ошибка в progress_callback: {e}")
                
                # Задержка между запросами для соблюдения rate limits
                if i < len(texts) - 1:
                    time.sleep(batch_delay)
                    
            except ValueError as e:
                # Ошибки валидации не должны происходить здесь, но на всякий случай
                logger.error(f"Ошибка валидации при классификации текста {i}: {e}")
                error_count += 1
                results.append({
                    "text": text,
                    "category": "⚠️ Не удалось определить",
                    "confidence": 0.0,
                    "reasoning": f"Ошибка валидации: {str(e)[:100]}"
                })
            except Exception as e:
                # Неожиданные ошибки
                logger.error(f"Неожиданная ошибка при классификации текста {i}: {e}", exc_info=True)
                error_count += 1
                results.append({
                    "text": text,
                    "category": "⚠️ Не удалось определить",
                    "confidence": 0.0,
                    "reasoning": f"Ошибка: {str(e)[:100]}"
                })
        
        df = pd.DataFrame(results)
        
        # Логируем статистику
        logger.info(f"Классификация завершена:")
        logger.info(f"  Всего текстов: {len(df)}")
        logger.info(f"  Ошибок: {error_count} ({error_count/len(df)*100:.1f}%)")
        logger.info(f"  Неопределённых: {undefined_count} ({undefined_count/len(df)*100:.1f}%)")
        logger.info(f"  Распределение по категориям:")
        
        # Статистика по категориям
        category_counts = df['category'].value_counts()
        for category in categories:
            count = category_counts.get(category, 0)
            percentage = count / len(df) * 100 if len(df) > 0 else 0
            logger.info(f"    {category}: {count} ({percentage:.1f}%)")
        
        # Показываем неизвестные категории если есть
        unknown_categories = [cat for cat in category_counts.index if cat not in categories and cat != "⚠️ Не удалось определить"]
        if unknown_categories:
            logger.warning(f"  Неизвестные категории от модели: {unknown_categories}")
        
        return df
    
    def get_classification_stats(self, df: pd.DataFrame) -> Dict[str, any]:
        """Получает статистику по результатам классификации."""
        
        # Считаем случаи когда модель не определила
        undefined_count = len(df[df['category'] == "⚠️ Не удалось определить"])
        
        stats = {
            "total_texts": len(df),
            "undefined_count": undefined_count,  # НОВОЕ ПОЛЕ
            "undefined_percentage": (undefined_count / len(df) * 100) if len(df) > 0 else 0,
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
