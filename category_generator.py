# category_generator.py
import logging
import random
import json
import html
import re
from typing import List, Optional, Tuple
from dataclasses import dataclass
import requests

logger = logging.getLogger(__name__)

@dataclass
class CategorySuggestion:
    """Сгенерированная категория"""
    name: str
    description: str
    examples: List[str]

class CategoryGenerator:
    """Генерация категорий через YandexGPT"""
    
    DEFAULT_PROMPT = """Проанализируй следующие тексты и предложи 5-8 категорий для их классификации.

Тексты (выборка):
{texts}

Требования к категориям:
1. Взаимоисключающие (каждый текст — в одну категорию)
2. Покрывают основные темы
3. Понятные короткие названия
4. С кратким описанием

Формат ответа JSON:
{{
  "categories": [
    {{
      "name": "Название категории",
      "description": "Краткое описание",
      "examples": ["пример1", "пример2"]
    }}
  ]
}}

Только JSON, без дополнительного текста."""

    def __init__(self, api_key: str, folder_id: str):
        self.api_key = api_key
        self.folder_id = folder_id
        self.url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    
    def get_sample(self, texts: List[str], max_size: int = 1000) -> List[str]:
        """Получить репрезентативную выборку"""
        n = len(texts)
        
        if n <= 1000:
            sample_size = n
        elif n <= 5000:
            sample_size = 500
        else:
            sample_size = 1000
        
        sample_size = min(sample_size, max_size)
        
        # Случайная выборка
        if n > sample_size:
            sample = random.sample(texts, sample_size)
        else:
            sample = texts
        
        logger.info(f"📊 Sample created: {len(sample)} texts from {n} total")
        return sample
    
    def generate_categories(
        self, 
        texts_sample: List[str],
        custom_prompt: Optional[str] = None
    ) -> Tuple[bool, Optional[List[CategorySuggestion]], Optional[str]]:
        """
        Генерация категорий
        
        Returns:
            (success, categories, error_message)
        """
        try:
            # Формируем промт
            prompt_template = custom_prompt or self.DEFAULT_PROMPT
            
            # Берём до 100 текстов для промта (чтобы не превысить лимит токенов)
            sample_for_prompt = texts_sample[:100]
            texts_str = "\n".join([f"{i+1}. {t[:200]}" for i, t in enumerate(sample_for_prompt)])
            
            prompt = prompt_template.format(texts=texts_str)
            
            # Запрос к API
            headers = {
                "Authorization": f"Api-Key {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "modelUri": f"gpt://{self.folder_id}/yandexgpt-lite",
                "completionOptions": {
                    "temperature": 0.3,
                    "maxTokens": 2000
                },
                "messages": [
                    {
                        "role": "system",
                        "text": "Ты эксперт по анализу текстов. Отвечай только в формате JSON."
                    },
                    {
                        "role": "user",
                        "text": prompt
                    }
                ]
            }
            
            logger.info("🤖 Sending request to YandexGPT for category generation")
            
            response = requests.post(
                self.url,
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code != 200:
                logger.error(f"YandexGPT API error: {response.status_code} - {response.text}")
                return False, None, f"Ошибка API: {response.status_code}"
            
            result = response.json()
            text_response = result['result']['alternatives'][0]['message']['text']
            
            # Парсим JSON
            try:
                # Убираем markdown если есть
                if "```json" in text_response:
                    text_response = text_response.split("```json")[1].split("```")[0]
                elif "```" in text_response:
                    text_response = text_response.split("```")[1].split("```")[0]
                
                data = json.loads(text_response.strip())
                categories_data = data.get('categories', [])
                
                if not categories_data:
                    return False, None, "API не вернул категории"
                
                # Функция очистки HTML-тегов
                def clean_html(text: str) -> str:
                    """Удаляет HTML-теги, оставляет текст"""
                    if not text:
                        return ""
                    # <br/> → перенос строки
                    text = re.sub(r'<br\s*/?>', '\n', text)
                    # Удаляем остальные HTML-теги
                    text = re.sub(r'<[^>]+>', '', text)
                    return text.strip()
                
                # Преобразуем в CategorySuggestion
                categories = []
                for cat in categories_data:
                    categories.append(CategorySuggestion(
                        name=clean_html(cat.get('name', 'Без названия')),
                        description=clean_html(cat.get('description', '')),
                        examples=[clean_html(ex) for ex in cat.get('examples', [])[:3]]
                    ))
                
                logger.info(f"✅ Generated {len(categories)} categories")
                return True, categories, None
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}\nResponse: {text_response}")
                return False, None, "Не удалось распознать ответ от AI"
        
        except requests.Timeout:
            logger.error("YandexGPT request timeout")
            return False, None, "Превышено время ожидания"
        
        except Exception as e:
            logger.error(f"Category generation error: {e}", exc_info=True)
            return False, None, f"Ошибка: {str(e)}"
    
    def format_categories_for_display(self, categories: List[CategorySuggestion]) -> str:
        """Форматирование для показа пользователю"""
        msg = f"🏷️ <b>Предложенные категории ({len(categories)}):</b>\n\n"
        
        for i, cat in enumerate(categories, 1):
            emoji = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"][i-1] if i <= 10 else "▪️"
            
            # Экранируем спецсимволы HTML
            safe_name = html.escape(cat.name)
            
            msg += f"{emoji} <b>{safe_name}</b>\n"
            
            if cat.description:
                safe_desc = html.escape(cat.description)
                # Обрезаем длинные описания
                if len(safe_desc) > 150:
                    safe_desc = safe_desc[:150] + "..."
                msg += f"   <i>{safe_desc}</i>\n"
            
            if cat.examples:
                safe_examples = [html.escape(ex[:50]) for ex in cat.examples[:2]]
                examples_str = "; ".join(safe_examples)
                msg += f"   💬 Примеры: {examples_str}\n"
            
            msg += "\n"
        
        return msg
