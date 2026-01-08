"""
Состояния бота — Классификатор текстов
Соответствует спецификации процессов v1.1
"""

from enum import Enum, auto


class BotState(Enum):
    """Состояния бота согласно спецификации"""
    
    # 1. Приветствие
    START = auto()                      # 1 — Начальное состояние
    
    # 2. Загрузка файла
    WAITING_FOR_FILE = auto()           # 2.0 — Ожидание файла
    FILE_RECEIVED = auto()              # 2.3 — Файл получен, выбор действия
    
    # 3. Настройка параметров
    SETTINGS_MENU = auto()              # 3.1 — Меню настроек
    WAITING_FOR_CATEGORIES = auto()     # 3.2.1 — Ожидание ввода категорий
    CATEGORIES_CONFIRMED = auto()       # 3.2.3 — Категории подтверждены
    WAITING_FOR_PROMPT = auto()         # 3.3.1 — Ожидание кастомного промпта
    GENERATING_CATEGORIES = auto()      # 3.5 — Процесс генерации
    SHOWING_GENERATED = auto()          # 3.6 — Показ сгенерированных категорий
    EDITING_CATEGORIES = auto()         # 3.6.2 — Редактирование категорий
    
    # 4. Классификация
    CLASSIFYING = auto()                # 4.1-4.2 — Процесс классификации
    SHOWING_RESULT = auto()             # 4.3 — Показ результата
    COLLECTING_FEEDBACK = auto()        # 4.3.3-4.3.4 — Сбор обратной связи
    WAITING_FOR_FEEDBACK_TEXT = auto()  # 4.3.4.1 — Ожидание текста фидбека
    SESSION_END = auto()                # 4.4 — Завершение сессии
    
    # 5. Демо
    DEMO_MENU = auto()                  # 5.1 — Меню демо-режима
    
    # Ошибки
    ERROR = auto()                      # Состояние ошибки


class StateTransition:
    """Допустимые переходы между состояниями"""
    
    TRANSITIONS = {
        BotState.START: [
            BotState.WAITING_FOR_FILE,
            BotState.DEMO_MENU,
            BotState.FILE_RECEIVED,  # G1: CSV на любом шаге
        ],
        
        BotState.WAITING_FOR_FILE: [
            BotState.FILE_RECEIVED,
            BotState.START,
            BotState.ERROR,
        ],
        
        BotState.FILE_RECEIVED: [
            BotState.CLASSIFYING,        # Запуск со стандартными настройками
            BotState.SETTINGS_MENU,      # Настроить параметры
            BotState.START,
        ],
        
        BotState.SETTINGS_MENU: [
            BotState.WAITING_FOR_CATEGORIES,
            BotState.GENERATING_CATEGORIES,
            BotState.WAITING_FOR_PROMPT,
            BotState.FILE_RECEIVED,
        ],
        
        BotState.WAITING_FOR_CATEGORIES: [
            BotState.CATEGORIES_CONFIRMED,
            BotState.SETTINGS_MENU,
            BotState.ERROR,
        ],
        
        BotState.CATEGORIES_CONFIRMED: [
            BotState.CLASSIFYING,
            BotState.WAITING_FOR_CATEGORIES,
        ],
        
        BotState.WAITING_FOR_PROMPT: [
            BotState.GENERATING_CATEGORIES,
            BotState.SETTINGS_MENU,
        ],
        
        BotState.GENERATING_CATEGORIES: [
            BotState.SHOWING_GENERATED,
            BotState.ERROR,
        ],
        
        BotState.SHOWING_GENERATED: [
            BotState.CLASSIFYING,        # Подтвердить
            BotState.EDITING_CATEGORIES, # Редактировать
            BotState.WAITING_FOR_PROMPT, # Изменить промпт
        ],
        
        BotState.EDITING_CATEGORIES: [
            BotState.CATEGORIES_CONFIRMED,
            BotState.SHOWING_GENERATED,
            BotState.ERROR,
        ],
        
        BotState.CLASSIFYING: [
            BotState.SHOWING_RESULT,
            BotState.ERROR,
        ],
        
        BotState.SHOWING_RESULT: [
            BotState.SESSION_END,        # Положительная оценка
            BotState.COLLECTING_FEEDBACK,# Отрицательная оценка
            BotState.SETTINGS_MENU,      # Перенастроить
        ],
        
        BotState.COLLECTING_FEEDBACK: [
            BotState.WAITING_FOR_FEEDBACK_TEXT,
            BotState.SETTINGS_MENU,
            BotState.SESSION_END,
        ],
        
        BotState.WAITING_FOR_FEEDBACK_TEXT: [
            BotState.SETTINGS_MENU,
            BotState.SESSION_END,
        ],
        
        BotState.SESSION_END: [
            BotState.WAITING_FOR_FILE,   # Да, ещё файл
            BotState.START,              # Завершение
            BotState.FILE_RECEIVED,      # G1: CSV на любом шаге
        ],
        
        BotState.DEMO_MENU: [
            BotState.CLASSIFYING,        # Автозапуск
            BotState.SETTINGS_MENU,      # Настроить демо
            BotState.FILE_RECEIVED,      # Свой файл
            BotState.START,
        ],
        
        BotState.ERROR: [
            BotState.START,
            BotState.WAITING_FOR_FILE,
            BotState.SETTINGS_MENU,
        ],
    }
    
    @classmethod
    def is_valid(cls, from_state: BotState, to_state: BotState) -> bool:
        """Проверяет допустимость перехода"""
        if from_state not in cls.TRANSITIONS:
            return False
        return to_state in cls.TRANSITIONS[from_state]


def get_expected_input(state: BotState) -> str:
    """Возвращает описание ожидаемого ввода для состояния"""
    expectations = {
        BotState.START: "кнопку или CSV-файл",
        BotState.WAITING_FOR_FILE: "CSV-файл",
        BotState.FILE_RECEIVED: "выбор действия (кнопку)",
        BotState.SETTINGS_MENU: "выбор настройки (кнопку)",
        BotState.WAITING_FOR_CATEGORIES: "список категорий",
        BotState.WAITING_FOR_PROMPT: "текст промпта",
        BotState.EDITING_CATEGORIES: "отредактированные категории",
        BotState.WAITING_FOR_FEEDBACK_TEXT: "описание проблемы",
    }
    return expectations.get(state, "действие")

