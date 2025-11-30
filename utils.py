# utils.py
"""
Утилиты для бота: cleanup, проверки, санитизация
"""

import logging
import time
import shutil
from pathlib import Path
from typing import Tuple
from config import TEMP_DIR

logger = logging.getLogger(__name__)

# Константа для возраста файлов (24 часа)
TEMP_FILE_MAX_AGE_HOURS = 24


def cleanup_old_temp_files():
    """Удаляет временные файлы старше N часов"""
    try:
        if not TEMP_DIR.exists():
            logger.warning(f"⚠️ Temp directory does not exist: {TEMP_DIR}")
            return
        
        now = time.time()
        max_age = TEMP_FILE_MAX_AGE_HOURS * 3600
        removed_count = 0
        freed_bytes = 0
        
        for file_path in TEMP_DIR.glob("*"):
            if file_path.is_file():
                try:
                    age = now - file_path.stat().st_mtime
                    if age > max_age:
                        size = file_path.stat().st_size
                        file_path.unlink()
                        removed_count += 1
                        freed_bytes += size
                except Exception as e:
                    logger.warning(f"⚠️ Failed to delete {file_path}: {e}")
        
        if removed_count > 0:
            freed_mb = freed_bytes / (1024 * 1024)
            logger.info(f"🗑️ Cleaned {removed_count} old temp files ({freed_mb:.2f} MB freed)")
        else:
            logger.debug("✅ No old temp files to clean")
    
    except Exception as e:
        logger.error(f"❌ Error cleaning temp files: {e}", exc_info=True)


def cleanup_file_safe(file_path) -> bool:
    """Безопасное удаление файла"""
    try:
        if file_path:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                logger.debug(f"🗑️ Deleted: {file_path}")
                return True
    except Exception as e:
        logger.warning(f"⚠️ Failed to delete {file_path}: {e}")
    
    return False


def check_disk_space(path: str = "/", min_free_gb: float = 1.0) -> Tuple[bool, float]:
    """Проверяет свободное место на диске"""
    try:
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024**3)
        ok = free_gb >= min_free_gb
        
        if not ok:
            logger.warning(f"⚠️ LOW DISK SPACE | Free: {free_gb:.2f} GB | Min: {min_free_gb} GB")
        
        return ok, free_gb
    
    except Exception as e:
        logger.error(f"❌ Error checking disk space: {e}")
        return True, 0.0


def format_time_remaining(seconds: int) -> str:
    """Форматирует время ожидания"""
    if seconds < 60:
        return f"{seconds} секунд"
    
    minutes = seconds // 60
    
    if minutes < 60:
        return f"{minutes} минут"
    
    hours = minutes // 60
    remaining_minutes = minutes % 60
    
    if remaining_minutes == 0:
        return f"{hours} час{'а' if hours < 5 else 'ов'}"
    
    return f"{hours} час{'а' if hours < 5 else 'ов'} {remaining_minutes} минут"


def get_user_display_name(user) -> str:
    """Получает красивое имя пользователя для логов"""
    if not user:
        return "Unknown"
    
    parts = []
    
    if user.first_name:
        parts.append(user.first_name)
    if user.last_name:
        parts.append(user.last_name)
    
    name = " ".join(parts) if parts else str(user.id)
    
    if user.username:
        name += f" (@{user.username})"
    
    return name
