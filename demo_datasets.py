# demo_datasets.py
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Описание демо-датасетов
DEMO_DATASETS = {
    'reviews_app': {
        'name': '📱 Отзывы о мобильном приложении',
        'file': 'demo_data/reviews_app.csv',
        'rows': 15,
        'description': 'Реальные отзывы пользователей мобильного приложения',
        'emoji': '📱'
    },
    'support_ecommerce': {
        'name': '🛒 Обращения в поддержку e-commerce',
        'file': 'demo_data/support_ecommerce.csv',
        'rows': 15,
        'description': 'Тикеты службы поддержки интернет-магазина',
        'emoji': '🛒'
    },
    'course_feedback': {
        'name': '🎓 Фидбек студентов онлайн-курса',
        'file': 'demo_data/course_feedback.csv',
        'rows': 15,
        'description': 'Отзывы студентов после завершения онлайн-курса',
        'emoji': '🎓'
    }
}


def get_demo_file_path(key: str) -> Optional[str]:
    """
    Получить путь к демо-файлу
    
    Args:
        key: Ключ датасета (reviews_app, support_ecommerce, course_feedback)
    
    Returns:
        Полный путь к файлу или None
    """
    if key not in DEMO_DATASETS:
        logger.error(f"Unknown demo dataset key: {key}")
        return None
    
    file_path = Path(DEMO_DATASETS[key]['file'])
    
    if not file_path.exists():
        logger.error(f"Demo file not found: {file_path}")
        return None
    
    return str(file_path)


def get_demo_description(key: str) -> str:
    """Получить описание датасета"""
    if key not in DEMO_DATASETS:
        return "Неизвестный датасет"
    
    dataset = DEMO_DATASETS[key]
    return f"{dataset['emoji']} {dataset['name']} ({dataset['rows']} примеров)"


def format_demo_list() -> str:
    """Форматировать список датасетов для показа пользователю"""
    lines = []
    for key, dataset in DEMO_DATASETS.items():
        lines.append(
            f"{dataset['emoji']} <b>{dataset['name']}</b>\n"
            f"   {dataset['description']} ({dataset['rows']} строк)"
        )
    
    return "\n\n".join(lines)
