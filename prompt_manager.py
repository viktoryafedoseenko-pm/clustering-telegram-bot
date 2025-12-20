# prompt_manager.py
import logging
from typing import Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class UserPrompts:
    """Промты пользователя"""
    generation_prompt: Optional[str] = None
    classification_prompt: Optional[str] = None

class PromptManager:
    """Управление промтами"""
    
    DEFAULT_GENERATION_PROMPT = """Проанализируй тексты и предложи 5-8 категорий.

Тексты:
{texts}

Требования:
- Взаимоисключающие категории
- Понятные названия
- С описанием

Ответ в JSON."""
    
    DEFAULT_CLASSIFICATION_PROMPT = """Определи категорию текста.

Текст: {text}

Категории:
{categories}

Ответ в JSON:
{{"category": "название", "confidence": 0.95, "reasoning": "объяснение"}}"""
    
    def __init__(self):
        self.user_prompts: Dict[int, UserPrompts] = {}
    
    def get_generation_prompt(self, user_id: int) -> str:
        """Получить промт для генерации"""
        if user_id in self.user_prompts and self.user_prompts[user_id].generation_prompt:
            return self.user_prompts[user_id].generation_prompt
        return self.DEFAULT_GENERATION_PROMPT
    
    def get_classification_prompt(self, user_id: int) -> str:
        """Получить промт для классификации"""
        if user_id in self.user_prompts and self.user_prompts[user_id].classification_prompt:
            return self.user_prompts[user_id].classification_prompt
        return self.DEFAULT_CLASSIFICATION_PROMPT
    
    def set_generation_prompt(self, user_id: int, prompt: str):
        """Сохранить промт генерации"""
        if user_id not in self.user_prompts:
            self.user_prompts[user_id] = UserPrompts()
        self.user_prompts[user_id].generation_prompt = prompt
        logger.info(f"💾 Generation prompt saved for user {user_id}")
    
    def set_classification_prompt(self, user_id: int, prompt: str):
        """Сохранить промт классификации"""
        if user_id not in self.user_prompts:
            self.user_prompts[user_id] = UserPrompts()
        self.user_prompts[user_id].classification_prompt = prompt
        logger.info(f"💾 Classification prompt saved for user {user_id}")
    
    def reset_prompts(self, user_id: int):
        """Сбросить промты"""
        if user_id in self.user_prompts:
            del self.user_prompts[user_id]
            logger.info(f"🔄 Prompts reset for user {user_id}")
