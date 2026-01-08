"""
Состояния бота — Классификатор текстов
Соответствует спецификации процессов v1.1
"""

from enum import Enum, auto


class BotState(Enum):
    """Состояния бота согласно спецификации"""
    
    # 1. Приветствие
    START = auto()
    
    # 2. Загрузка файла
    WAITING_FOR_FILE = auto()
    FILE_RECEIVED = auto()
    
    # 3. Настройка параметров
    SETTINGS_MENU = auto()
    WAITING_FOR_CATEGORIES = auto()
    CATEGORIES_CONFIRMED = auto()
    WAITING_FOR_PROMPT = auto()
    GENERATING_CATEGORIES = auto()
    SHOWING_GENERATED = auto()
    EDITING_CATEGORIES = auto()
    
    # 4. Классификация
    CLASSIFYING = auto()
    SHOWING_RESULT = auto()
    COLLECTING_FEEDBACK = auto()
    WAITING_FOR_FEEDBACK_TEXT = auto()
    SESSION_END = auto()
    
    # 5. Демо
    DEMO_MENU = auto()
    
    # Служебные
    ERROR = auto()


def get_expected_input(state: BotState) -> str:
    """Возвращает описание ожидаемого ввода для состояния"""
    expectations = {
        BotState.START: "кнопку или CSV-файл",
        BotState.WAITING_FOR_FILE: "CSV-файл",
        BotState.FILE_RECEIVED: "выбор действия (кнопку)",
        BotState.SETTINGS_MENU: "выбор настройки (кнопку)",
        BotState.WAITING_FOR_CATEGORIES: "список категорий (текст)",
        BotState.WAITING_FOR_PROMPT: "текст промпта или кнопку",
        BotState.EDITING_CATEGORIES: "отредактированные категории (текст)",
        BotState.SHOWING_GENERATED: "подтверждение (кнопку)",
        BotState.WAITING_FOR_FEEDBACK_TEXT: "описание проблемы (текст)",
        BotState.SESSION_END: "кнопку или CSV-файл",
        BotState.DEMO_MENU: "выбор демо-датасета (кнопку) или CSV-файл",
    }
    return expectations.get(state, "действие")
