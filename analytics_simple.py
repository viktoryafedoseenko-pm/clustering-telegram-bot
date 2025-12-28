# analytics_simple.py
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class UserAnalytics:
    """
    Простая аналитика без БД
    Отправляет уведомления админу о ключевых событиях
    """
    
    def __init__(self, admin_chat_id: int):
        self.admin_chat_id = admin_chat_id
        logger.info(f"✅ UserAnalytics initialized | Admin ID: {admin_chat_id}")
    
    async def track_start(
        self, 
        bot, 
        user_id: int, 
        username: Optional[str], 
        source: str,
        first_name: Optional[str] = None
    ):
        """
        Отслеживание /start
        
        Args:
            bot: Telegram bot instance
            user_id: ID пользователя
            username: Username (без @)
            source: Источник (organic, from_site, ad_vk, ref_xxx)
            first_name: Имя пользователя
        """
        logger.info(f"👤 NEW START | User: {user_id} | Source: {source}")
        
        # Форматирование источника для читаемости
        source_label = self._format_source(source)
        
        # Формирование user display
        user_display = self._format_user(user_id, username, first_name)
        
        try:
            await bot.send_message(
                chat_id=self.admin_chat_id,
                text=(
                    f"🆕 <b>Новый пользователь</b>\n\n"
                    f"👤 {user_display}\n"
                    f"🔗 Источник: {source_label}\n"
                    f"⏰ {datetime.now().strftime('%H:%M:%S')}"
                ),
                parse_mode='HTML'
            )
        except Exception as e:
            logger.error(f"Failed to send start notification: {e}")
    
    async def track_file_processed(
        self,
        bot,
        user_id: int,
        username: Optional[str],
        files_count: int,
        mode: str,
        rows: int,
        filename: str,
        quiz_data: Optional[dict] = None,
        source: Optional[str] = None
    ):
        """
        Отслеживание обработки файла
        
        Args:
            bot: Telegram bot instance
            user_id: ID пользователя
            username: Username
            files_count: Общее количество файлов пользователя
            mode: Режим (clustering/classification)
            rows: Количество строк в файле
            filename: Имя файла
            quiz_data: Ответы из квиза (опционально)
            source: Источник пользователя
        """
        logger.info(
            f"📊 FILE PROCESSED | User: {user_id} | "
            f"Mode: {mode} | Files: {files_count} | Rows: {rows}"
        )
        
        user_display = self._format_user(user_id, username)
        mode_label = "🔍 Кластеризация" if mode == "clustering" else "🏷️ Классификация"
        
        # Формирование текста с квизом (если есть)
        quiz_text = ""
        if quiz_data:
            quiz_text = (
                f"\n📝 <b>Квиз:</b>\n"
                f"   • Данные: {quiz_data.get('q1', 'N/A')}\n"
                f"   • Задача: {quiz_data.get('q2', 'N/A')}"
            )
        
        # Источник
        source_text = ""
        if source:
            source_text = f"\n🔗 Источник: {self._format_source(source)}"
        
        try:
            await bot.send_message(
                chat_id=self.admin_chat_id,
                text=(
                    f"📊 <b>Файл обработан</b>\n\n"
                    f"👤 {user_display}\n"
                    f"📁 Файл: <code>{filename[:30]}...</code>\n"
                    f"📈 Режим: {mode_label}\n"
                    f"📋 Строк: {rows}\n"
                    f"🔢 Файлов всего: <b>{files_count}</b>"
                    f"{quiz_text}"
                    f"{source_text}\n"
                    f"⏰ {datetime.now().strftime('%H:%M:%S')}"
                ),
                parse_mode='HTML'
            )
        except Exception as e:
            logger.error(f"Failed to send file processed notification: {e}")
    
    async def track_feedback(
        self,
        bot,
        user_id: int,
        username: Optional[str],
        feedback_type: str,
        files_count: int,
        details: Optional[str] = None
    ):
        """
        Отслеживание фидбека от пользователя
        
        Args:
            bot: Telegram bot instance
            user_id: ID пользователя
            username: Username
            feedback_type: Тип (positive/negative/later)
            files_count: Количество файлов
            details: Дополнительные детали
        """
        logger.info(
            f"💬 FEEDBACK | User: {user_id} | "
            f"Type: {feedback_type} | Files: {files_count}"
        )
        
        user_display = self._format_user(user_id, username)
        
        # Emoji для типа фидбека
        feedback_emoji = {
            'positive': '✅',
            'negative': '🤔',
            'later': '⏸️'
        }
        emoji = feedback_emoji.get(feedback_type, '💬')
        
        feedback_label = {
            'positive': 'Да, увидел полезное',
            'negative': 'Можно улучшить',
            'later': 'Пока не смотрел'
        }
        label = feedback_label.get(feedback_type, feedback_type)
        
        details_text = f"\n\n💭 <i>{details}</i>" if details else ""
        
        try:
            await bot.send_message(
                chat_id=self.admin_chat_id,
                text=(
                    f"{emoji} <b>Фидбек от пользователя</b>\n\n"
                    f"👤 {user_display}\n"
                    f"📊 Файлов обработано: {files_count}\n"
                    f"💬 Реакция: <b>{label}</b>"
                    f"{details_text}\n\n"
                    f"⏰ {datetime.now().strftime('%H:%M:%S')}"
                ),
                parse_mode='HTML'
            )
        except Exception as e:
            logger.error(f"Failed to send feedback notification: {e}")
    
    async def track_cta_shown(
        self,
        bot,
        user_id: int,
        username: Optional[str],
        cta_type: str,
        files_count: int
    ):
        """
        Отслеживание показа CTA
        
        Args:
            bot: Telegram bot instance
            user_id: ID пользователя
            username: Username
            cta_type: Тип CTA (after_file_1, after_file_2, after_file_3)
            files_count: Количество файлов
        """
        logger.info(
            f"🎯 CTA SHOWN | User: {user_id} | "
            f"Type: {cta_type} | Files: {files_count}"
        )
        
        user_display = self._format_user(user_id, username)
        
        cta_labels = {
            'after_file_1': '📋 Опрос после 1 файла',
            'after_file_2': '🎁 CTA с Calendly после 2 файла',
            'after_file_3': '🔥 Супер-пользователь (3+ файла)'
        }
        label = cta_labels.get(cta_type, cta_type)
        
        try:
            await bot.send_message(
                chat_id=self.admin_chat_id,
                text=(
                    f"🎯 <b>CTA показан</b>\n\n"
                    f"👤 {user_display}\n"
                    f"📊 Файлов: {files_count}\n"
                    f"🎬 CTA: {label}\n"
                    f"⏰ {datetime.now().strftime('%H:%M:%S')}"
                ),
                parse_mode='HTML'
            )
        except Exception as e:
            logger.error(f"Failed to send CTA notification: {e}")
    
    async def track_super_user(
        self,
        bot,
        user_id: int,
        username: Optional[str],
        files_count: int,
        quiz_data: Optional[dict] = None,
        modes_used: Optional[list] = None
    ):
        """
        Специальное уведомление о супер-активном пользователе
        
        Args:
            bot: Telegram bot instance
            user_id: ID пользователя
            username: Username
            files_count: Количество файлов (3+)
            quiz_data: Ответы квиза
            modes_used: Список использованных режимов
        """
        logger.info(f"🔥 SUPER USER | User: {user_id} | Files: {files_count}")
        
        user_display = self._format_user(user_id, username)
        
        # Квиз
        quiz_text = ""
        if quiz_data:
            quiz_text = (
                f"\n📝 <b>Квиз:</b>\n"
                f"   • {quiz_data.get('q1', 'N/A')}\n"
                f"   • {quiz_data.get('q2', 'N/A')}"
            )
        
        # Режимы
        modes_text = ""
        if modes_used:
            modes_text = f"\n\n🎯 <b>Использовал:</b> {', '.join(modes_used)}"
        
        try:
            await bot.send_message(
                chat_id=self.admin_chat_id,
                text=(
                    f"🔥🔥🔥 <b>СУПЕР-ПОЛЬЗОВАТЕЛЬ!</b> 🔥🔥🔥\n\n"
                    f"👤 {user_display}\n"
                    f"📊 <b>Обработал {files_count} файлов!</b>"
                    f"{quiz_text}"
                    f"{modes_text}\n\n"
                    f"👉 <b>Напиши ему/ей:</b> @{username or 'username_not_set'}\n\n"
                    f"⏰ {datetime.now().strftime('%H:%M:%S')}"
                ),
                parse_mode='HTML'
            )
        except Exception as e:
            logger.error(f"Failed to send super user notification: {e}")
    
    # === Вспомогательные методы ===
    
    def _format_user(
        self, 
        user_id: int, 
        username: Optional[str], 
        first_name: Optional[str] = None
    ) -> str:
        """Форматирование информации о пользователе"""
        if username:
            display = f"@{username}"
        elif first_name:
            display = f"{first_name}"
        else:
            display = f"ID: {user_id}"
        
        return f"{display} (ID: {user_id})"
    
    def _format_source(self, source: str) -> str:
        """Форматирование источника для читаемости"""
        source_map = {
            'organic': '🌱 Органика',
            'from_site': '🌐 С сайта',
            'ad_vk': '📢 Реклама VK',
            'ad_telegram': '📢 Реклама Telegram',
            'ad_google': '📢 Реклама Google'
        }
        
        # Если источник начинается с ref_
        if source.startswith('ref_'):
            ref_name = source.replace('ref_', '')
            return f"👥 Реферал: {ref_name}"
        
        return source_map.get(source, f"🔗 {source}")
