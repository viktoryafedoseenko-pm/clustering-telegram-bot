# analytics.py
"""
Модуль аналитики для Telegram бота кластеризации текстов
Предоставляет высокоуровневый интерфейс для генерации отчётов
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
from typing import Optional, Tuple
from cache_manager import cache
from pdf_generator import PDFReportGenerator
from config import TEMP_DIR

# Thread pool для блокирующих операций
executor = ThreadPoolExecutor(max_workers=2)

async def generate_detailed_report(
    cache_key: str,
    user_id: int
) -> Optional[Tuple[str, str]]:
    """
    Генерирует детальный PDF отчёт и расширенный CSV
    
    Args:
        cache_key: Ключ кэша с результатами кластеризации
        user_id: Telegram user ID
    
    Returns:
        Tuple[pdf_path, csv_path] или None при ошибке
    """
    # Загружаем из кэша
    data = cache.load(cache_key)
    if not data:
        return None
    
    df = data['df']
    stats = data['stats']
    cluster_names = data['cluster_names']
    
    # Пути для результатов
    pdf_path = TEMP_DIR / f"report_{user_id}_{cache_key[:8]}.pdf"
    csv_path = TEMP_DIR / f"extended_stats_{user_id}_{cache_key[:8]}.csv"
    
    # Генерация в отдельном потоке (блокирующие операции)
    loop = asyncio.get_event_loop()
    
    try:
        # PDF
        generator = PDFReportGenerator(df, stats, cluster_names)
        success = await loop.run_in_executor(
            executor,
            generator.generate,
            str(pdf_path)
        )
        
        if not success:
            return None
        
        # Extended CSV
        await loop.run_in_executor(
            executor,
            _generate_extended_csv,
            df, cluster_names, str(csv_path)
        )
        
        return str(pdf_path), str(csv_path)
        
    except Exception as e:
        print(f"⚠️ Error generating report: {e}")
        return None

def _generate_extended_csv(df: pd.DataFrame, cluster_names: dict, output_path: str):
    """Генерирует расширенную статистику в CSV"""
    cluster_stats = df.groupby('cluster_id').agg({
        'cluster_name': 'first'
    }).reset_index()
    
    # Размер кластера
    cluster_stats['size'] = df['cluster_id'].value_counts()
    
    # Процент
    cluster_stats['percent'] = (cluster_stats['size'] / len(df) * 100).round(2)
    
    # Сортировка по размеру
    cluster_stats = cluster_stats.sort_values('size', ascending=False)
    
    # Сохранение
    cluster_stats.to_csv(output_path, index=False, encoding='utf-8')
