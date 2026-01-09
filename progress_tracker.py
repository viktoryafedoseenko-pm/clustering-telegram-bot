# progress_tracker.py
"""
Трекер прогресса с throttling для Telegram
"""
import time
import logging

logger = logging.getLogger(__name__)

class ProgressTracker:  # ⬅️ ИМЯ КЛАССА ОСТАЕТСЯ ТЕМ ЖЕ!
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
        self.start_time = time.time()
    
    async def update(self, stage: str, percent: int, details: str = "", force: bool = False):
        """
        Обновляет прогресс
        
        Args:
            stage: Название этапа (например, "Предобработка")
            percent: Процент выполнения (0-100)
            details: Дополнительные детали (опционально)
            force: Принудительное обновление (игнорирует throttling)
        """
        # 1. Ограничиваем проценты
        if percent < 0:
            percent = 0
        elif percent > 100:
            percent = 100
        
        # 2. Не показываем 100% до завершения
        if percent == 100 and not force:
            percent = 99
        
        now = time.time()
        
        # 3. Проверяем throttling
        should_update = force or (now - self.last_update) >= self.min_interval
        
        if should_update:
            try:
                message_text = self._format_message(stage, percent, details)
                await self.message.edit_text(message_text, parse_mode='HTML')
                self.last_update = now
                self.current_percent = percent
                self.current_stage = stage
            except Exception as e:
                # Игнорируем ошибку "message not modified"
                if "message is not modified" not in str(e):
                    logger.debug(f"Не удалось обновить прогресс: {e}")
    
    def _format_message(self, stage: str, percent: int, details: str) -> str:
        """Форматирует сообщение БЕЗ ПРОГРЕСС-БАРА"""
        
        # Время с начала
        elapsed = int(time.time() - self.start_time)
        elapsed_str = f"{elapsed // 60:02d}:{elapsed % 60:02d}"
        
        # Эмодзи в зависимости от процента
        if percent < 30:
            emoji = "🔄"
        elif percent < 70:
            emoji = "⚙️"
        elif percent < 100:
            emoji = "🔧"
        else:
            emoji = "✅"
        
        # Простое текстовое сообщение
        message = (
            f"{emoji} <b>Классификация текстов</b>\n\n"
            f"📊 <i>{stage}</i>\n"
            f"✅ Прогресс: {percent}%\n"
            f"⏱ Прошло: {elapsed_str}\n"
            f"⏳ Подождите..."
        )
        
        if details:
            message += f"\n\n💡 {details}"
        
        return message
    
    async def complete(self, message: str = "Готово!"):
        """Завершает прогресс - показывает 100%"""
        await self.update(message, 100, force=True)