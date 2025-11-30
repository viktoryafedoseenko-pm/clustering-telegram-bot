# rate_limiter.py
"""
Rate Limiter для защиты от флуда
Ограничивает количество запросов на пользователя
"""

import time
import logging
from collections import defaultdict
from typing import Dict, Tuple, List

logger = logging.getLogger(__name__)


class RateLimiter:
    """Простой rate limiter на основе скользящего окна"""
    
    def __init__(self, max_requests: int = 5, window_seconds: int = 3600):
        """
        Args:
            max_requests: Максимум запросов в окне (по умолчанию 5)
            window_seconds: Размер окна в секундах (по умолчанию 1 час)
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        
        # {user_id: [timestamp1, timestamp2, ...]}
        self.requests: Dict[int, List[float]] = defaultdict(list)
        
        logger.info(f"🚦 Rate Limiter initialized: {max_requests} requests per {window_seconds}s")
    
    def is_allowed(self, user_id: int) -> Tuple[bool, int, int]:
        """
        Проверяет, можно ли пользователю сделать запрос
        
        Args:
            user_id: Telegram user ID
        
        Returns:
            (allowed: bool, remaining: int, wait_seconds: int)
        """
        now = time.time()
        
        # Очищаем старые запросы
        self.requests[user_id] = [
            ts for ts in self.requests[user_id]
            if now - ts < self.window_seconds
        ]
        
        current_count = len(self.requests[user_id])
        
        # Проверяем лимит
        if current_count >= self.max_requests:
            oldest_request = min(self.requests[user_id])
            wait_time = int(self.window_seconds - (now - oldest_request)) + 1
            
            logger.warning(
                f"⚠️ RATE LIMIT EXCEEDED | User: {user_id} | "
                f"Requests: {current_count}/{self.max_requests} | "
                f"Wait: {wait_time}s"
            )
            
            return False, 0, wait_time
        
        # Разрешаем запрос
        self.requests[user_id].append(now)
        remaining = self.max_requests - current_count - 1
        
        logger.info(
            f"✅ RATE LIMIT OK | User: {user_id} | "
            f"Requests: {current_count + 1}/{self.max_requests} | "
            f"Remaining: {remaining}"
        )
        
        return True, remaining, 0
    
    def reset(self, user_id: int):
        """Сброс лимита для пользователя (для админов)"""
        if user_id in self.requests:
            del self.requests[user_id]
            logger.info(f"🔄 Rate limit reset for user {user_id}")
    
    def cleanup_old_users(self, max_age_hours: int = 24):
        """Очистка данных неактивных пользователей"""
        now = time.time()
        cutoff = now - (max_age_hours * 3600)
        
        users_to_remove = []
        
        for user_id, timestamps in self.requests.items():
            if timestamps and max(timestamps) < cutoff:
                users_to_remove.append(user_id)
        
        for user_id in users_to_remove:
            del self.requests[user_id]
        
        if users_to_remove:
            logger.info(f"🗑️ Cleaned up {len(users_to_remove)} inactive users from rate limiter")


# Глобальный инстанс
rate_limiter = RateLimiter(
    max_requests=5,
    window_seconds=3600
)
