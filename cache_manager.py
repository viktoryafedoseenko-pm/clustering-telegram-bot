# cache_manager.py
import pickle
import hashlib
import time
from pathlib import Path
from typing import Optional, Dict, Any
from config import CACHE_DIR, MAX_CACHE_AGE_SECONDS, MAX_CACHE_ITEMS

class ClusteringCache:
    """Кэш результатов кластеризации для генерации PDF"""
    
    def __init__(self):
        self.cache_dir = CACHE_DIR
        self._cleanup_old_cache()
    
    def _get_cache_key(self, user_id: int, file_name: str) -> str:
        """Генерирует ключ кэша: user_id + timestamp"""
        timestamp = int(time.time())
        raw = f"{user_id}_{file_name}_{timestamp}"
        return hashlib.md5(raw.encode()).hexdigest()
    
    def save(self, user_id: int, file_name: str, data: Dict[str, Any]) -> str:
        """
        Сохраняет результаты кластеризации
        
        Args:
            user_id: Telegram user ID
            file_name: Имя исходного файла
            data: {
                'df': pd.DataFrame,           # Кластеризованный датафрейм
                'stats': dict,                # Статистика из calculate_metrics
                'cluster_names': dict,        # {cluster_id: name}
                'file_name': str,             # Исходное имя файла
                'timestamp': float            # Время создания
            }
        
        Returns:
            str: Ключ кэша для последующего извлечения
        """
        cache_key = self._get_cache_key(user_id, file_name)
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        data['timestamp'] = time.time()
        data['user_id'] = user_id
        
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        
        self._cleanup_old_cache()
        return cache_key
    
    def load(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Загружает данные из кэша"""
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_path.exists():
            return None
        
        # Проверка возраста
        age = time.time() - cache_path.stat().st_mtime
        if age > MAX_CACHE_AGE_SECONDS:
            cache_path.unlink()
            return None
        
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    def _cleanup_old_cache(self):
        """Удаляет старые файлы кэша"""
        cache_files = sorted(
            self.cache_dir.glob("*.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        # Удаляем старые файлы (больше лимита)
        for old_file in cache_files[MAX_CACHE_ITEMS:]:
            old_file.unlink()
        
        # Удаляем устаревшие
        now = time.time()
        for cache_file in cache_files:
            if now - cache_file.stat().st_mtime > MAX_CACHE_AGE_SECONDS:
                cache_file.unlink()

# Глобальный экземпляр
cache = ClusteringCache()
