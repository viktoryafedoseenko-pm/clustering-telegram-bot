# analytics.py
"""
Модуль аналитики для Telegram бота кластеризации текстов
Предоставляет высокоуровневый интерфейс для генерации отчётов
"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
from typing import Optional, Tuple
from cache_manager import cache
from pdf_generator import PDFReportGenerator
from config import TEMP_DIR

# Thread pool для блокирующих операций
executor = ThreadPoolExecutor(max_workers=2)

logger = logging.getLogger(__name__)

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
        logger.error(f"❌ Cache not found for key: {cache_key}")
        return None
    
    logger.info(f"✅ Cache loaded: {len(data['df'])} rows, {data['stats']['n_clusters']} clusters")

    df = data['df']
    stats = data['stats']
    cluster_names = data['cluster_names']
    
    # ДОБАВЛЯЕМ: Получаем мастер-категории из кеша
    master_hierarchy = data.get('hierarchy', {})
    master_names = data.get('master_names', {})
    
    logger.info(f"🏷️ Master categories: {len(master_hierarchy)} hierarchies, {len(master_names)} names")
    
    # Пути для результатов
    pdf_path = TEMP_DIR / f"report_{user_id}_{cache_key[:8]}.pdf"
    csv_path = TEMP_DIR / f"extended_stats_{user_id}_{cache_key[:8]}.csv"
    
    # Генерация в отдельном потоке (блокирующие операции)
    loop = asyncio.get_event_loop()
    
    try:
        # PDF с мастер-категориями
        generator = PDFReportGenerator(
            df=df,
            stats=stats, 
            cluster_names=cluster_names,
            master_hierarchy=master_hierarchy,    # ← ДОБАВЛЯЕМ
            master_names=master_names             # ← ДОБАВЛЯЕМ
        )
        success = await loop.run_in_executor(
            executor,
            generator.generate,
            str(pdf_path)
        )
        
        if not success:
            return None
        
        # Extended CSV с мастер-категориями
        await loop.run_in_executor(
            executor,
            _generate_extended_csv,
            df, cluster_names, str(csv_path), master_hierarchy, master_names  # ← ДОБАВЛЯЕМ
        )
        
        return str(pdf_path), str(csv_path)
        
    except Exception as e:
        print(f"⚠️ Error generating report: {e}")
        return None

def _generate_extended_csv(
    df: pd.DataFrame, 
    cluster_names: dict, 
    output_path: str,
    master_hierarchy: dict = None,
    master_names: dict = None
):
    """Генерирует расширенную статистику в CSV с мастер-категориями"""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Проверка данных
        if df.empty:
            logger.error("❌ DataFrame is empty")
            raise ValueError("Empty DataFrame")
        
        if 'cluster_id' not in df.columns:
            logger.error(f"❌ 'cluster_id' column not found. Available: {df.columns.tolist()}")
            raise ValueError("cluster_id column missing")
        
        # Подсчёт
        logger.info(f"📊 Calculating stats for {len(df)} rows")
        cluster_counts = df['cluster_id'].value_counts().sort_values(ascending=False)
        
        logger.info(f"📊 Found {len(cluster_counts)} clusters")
        logger.info(f"🏷️ Master categories: {len(master_hierarchy or {})} hierarchies")
        
        # Создаём таблицу
        stats_data = []
        
        for cluster_id, size in cluster_counts.items():
            # Название кластера
            name = cluster_names.get(cluster_id, f"Кластер {cluster_id}")
            
            # Процент
            percent = round((size / len(df)) * 100, 2)
            
            # Определяем мастер-категорию
            master_category = ""
            master_category_id = ""
            master_category_size = 0
            
            if master_hierarchy:
                for master_id, sub_clusters in master_hierarchy.items():
                    if cluster_id in sub_clusters:
                        master_category = master_names.get(master_id, f"Категория {master_id}")
                        master_category_id = master_id
                        # Считаем размер мастер-категории для сортировки
                        master_category_size = sum(
                            len(df[df['cluster_id'] == cid]) 
                            for cid in sub_clusters
                        )
                        break
            
            stats_data.append({
                'cluster_id': cluster_id,
                'cluster_name': name,
                'master_category_id': master_category_id,
                'master_category_name': master_category,
                'master_category_size': master_category_size,  # Для сортировки
                'size': int(size),
                'percent': percent
            })
        
        # Сохранение с улучшенной сортировкой
        cluster_stats = pd.DataFrame(stats_data)
        
        if master_hierarchy:
            # Сортируем сначала по размеру мастер-категории (убывание), потом по размеру кластера (убывание)
            cluster_stats = cluster_stats.sort_values(
                ['master_category_size', 'size'], 
                ascending=[False, False]
            )
            cluster_stats = cluster_stats.drop('master_category_size', axis=1)
        else:
            cluster_stats = cluster_stats.sort_values('size', ascending=False)
        
        cluster_stats.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"✅ Extended CSV with sorted master categories saved: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Error in _generate_extended_csv: {e}", exc_info=True)
        raise