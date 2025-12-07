"""
Модуль для оценки качества классификации.
Рассчитывает метрики: accuracy, precision, recall, F1-score.
"""

import logging
from typing import Dict, List, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_metrics(
    y_true: List[str], 
    y_pred: List[str], 
    categories: List[str]
) -> Dict:
    """
    Рассчитывает метрики качества классификации.
    
    Args:
        y_true: Правильные категории
        y_pred: Предсказанные категории
        categories: Список всех категорий
        
    Returns:
        Словарь с метриками
    """
    n_total = len(y_true)
    n_correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = n_correct / n_total if n_total > 0 else 0
    
    metrics = {
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_total": n_total,
        "per_category": {}
    }
    
    # Метрики по каждой категории
    for category in categories:
        # True Positives: правильно предсказали эту категорию
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == category and p == category)
        
        # False Positives: неправильно предсказали эту категорию
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != category and p == category)
        
        # False Negatives: пропустили эту категорию
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == category and p != category)
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics["per_category"][category] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": tp + fn  # Сколько реально было в ground truth
        }
    
    logger.info(f"Metrics calculated: accuracy={accuracy:.3f}, categories={len(categories)}")
    return metrics


def get_error_examples(
    df: pd.DataFrame, 
    n: int = 3
) -> List[Dict]:
    """
    Возвращает примеры ошибок классификации.
    
    Args:
        df: DataFrame с колонками: text, true_category, predicted_category
        n: Количество примеров
        
    Returns:
        Список словарей с примерами ошибок
    """
    errors = df[df['true_category'] != df['category']]
    
    if len(errors) == 0:
        return []
    
    # Берём первые N ошибок
    examples = []
    for _, row in errors.head(n).iterrows():
        examples.append({
            "text": row['text'],
            "true_category": row['true_category'],
            "predicted_category": row['category']
        })
    
    logger.info(f"Found {len(errors)} errors, returning {len(examples)} examples")
    return examples


def format_evaluation_report(
    metrics: Dict, 
    examples: List[Dict],
    categories: List[str]
) -> str:
    """
    Форматирует отчёт об оценке качества.
    
    Args:
        metrics: Словарь с метриками
        examples: Список примеров ошибок
        categories: Список категорий
        
    Returns:
        Форматированная строка отчёта
    """
    report = f"""📊 <b>Оценка качества ({metrics['n_total']} текстов)</b>

✅ <b>Accuracy:</b> {metrics['accuracy']*100:.1f}% ({metrics['n_correct']} из {metrics['n_total']})

📋 <b>По категориям:</b>
"""
    
    # Сортируем категории по F1 (от лучших к худшим)
    sorted_categories = sorted(
        categories,
        key=lambda c: metrics['per_category'][c]['f1'],
        reverse=True
    )
    
    for cat in sorted_categories:
        m = metrics['per_category'][cat]
        safe_cat = html.escape(cat)
        report += (
            f"\n<b>{safe_cat}</b> (примеров: {m['support']})\n"
            f"  • F1: {m['f1']:.2f} | "
            f"Precision: {m['precision']:.2f} | "
            f"Recall: {m['recall']:.2f}\n"
        )
    
    # Примеры ошибок
    if examples:
        report += f"\n\n❌ <b>Примеры ошибок ({len(examples)} шт.):</b>\n"
        for i, ex in enumerate(examples, 1):
            text_preview = ex['text'][:80] + "..." if len(ex['text']) > 80 else ex['text']
            safe_text = html.escape(text_preview)
            safe_true = html.escape(ex['true_category'])
            safe_pred = html.escape(ex['predicted_category'])
            
            report += (
                f"\n{i}. <i>\"{safe_text}\"</i>\n"
                f"   Правильно: <b>{safe_true}</b>\n"
                f"   Модель: <b>{safe_pred}</b>\n"
            )
    else:
        report += "\n\n✅ <b>Ошибок нет! Все тексты классифицированы верно.</b>"

    return report


def validate_ground_truth(
    df: pd.DataFrame,
    expected_categories: List[str]
) -> Tuple[bool, str]:
    """Валидирует файл с ground truth."""
    
    # Проверка наличия второй колонки
    if len(df.columns) < 2:
        return False, "В файле должно быть минимум 2 колонки: текст и правильная_категория"
    
    # Проверка на пустые значения
    if df.iloc[:, 0].isna().any():
        return False, "Первая колонка (текст) содержит пустые значения"
    
    if df.iloc[:, 1].isna().any():
        empty_count = df.iloc[:, 1].isna().sum()
        return False, f"Вторая колонка (категория) содержит {empty_count} пустых значений"
    
    # Проверка категорий
    true_categories = set(df.iloc[:, 1].astype(str).str.strip().unique())
    expected_set = set(expected_categories)
    
    unknown_categories = true_categories - expected_set
    if unknown_categories:
        # html.escape для безопасности
        import html
        cats_str = ", ".join([html.escape(cat) for cat in list(unknown_categories)[:5]])
        expected_str = "\n".join([f"• {html.escape(cat)}" for cat in expected_categories])
        
        return False, (
            f"В файле найдены неизвестные категории: {cats_str}\n\n"
            f"Ожидаемые категории:\n{expected_str}"
        )
    
    return True, ""

