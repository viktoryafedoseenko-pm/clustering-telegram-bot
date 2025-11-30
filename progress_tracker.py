# progress_tracker.py
"""
Трекер прогресса с throttling для Telegram
"""
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ProgressTracker:
    """
    Отслеживает прогресс обработки и обновляет сообщение в Telegram
    с учётом rate limits API
    """
    
    def __init__(self, message, min_interval: float = 3.0):
        """
        Args:
            message: Telegram Message object для редактирования
            min_interval: Минимальный интервал между обновлениями (секунды)
        """
        self.message = message
        self.min_interval = min_interval
        self.last_update = 0
        self.current_stage = ""
        self.current_percent = 0
    
    async def update(self, stage: str, percent: int, details: str = "", force: bool = False):
        """
        Обновляет прогресс
        
        Args:
            stage: Название этапа (например, "Предобработка")
            percent: Процент выполнения (0-100)
            details: Дополнительные детали (опционально)
            force: Принудительное обновление (игнорирует throttling)
        """
        now = time.time()
        self.current_stage = stage
        self.current_percent = percent
        
        # Обновляем только если прошло достаточно времени или force=True
        should_update = force or (now - self.last_update) >= self.min_interval
        
        if should_update:
            try:
                message_text = self._format_message(stage, percent, details)
                await self.message.edit_text(message_text, parse_mode='HTML')
                self.last_update = now
                logger.info(f"Progress updated: {stage} - {percent}%")
            except Exception as e:
                logger.warning(f"Failed to update progress: {e}")
    
    def _format_message(self, stage: str, percent: int, details: str) -> str:
        """Форматирует сообщение с прогресс-баром"""
        # Создаём прогресс-бар
        filled = int(percent / 10)
        bar = "█" * filled + "░" * (10 - filled)
        
        # Эмодзи в зависимости от процента
        if percent < 30:
            emoji = "🔄"
        elif percent < 70:
            emoji = "⚙️"
        elif percent < 100:
            emoji = "🔧"
        else:
            emoji = "✅"
        
        message = (
            f"{emoji} <b>Обработка файла</b>\n\n"
            f"{bar} <b>{percent}%</b>\n\n"
            f"<i>{stage}</i>"
        )
        
        if details:
            message += f"\n\n💡 {details}"
        
        # Подсказка для долгих операций
        if 40 <= percent < 90:
            message += "\n\n<i>Можете закрыть чат — отправим уведомление когда готово</i>"
        
        return message
    
    async def complete(self, message: str = "Готово!"):
        """Завершает прогресс"""
        await self.update(message, 100, force=True)

