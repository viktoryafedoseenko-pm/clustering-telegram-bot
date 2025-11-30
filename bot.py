# bot.py
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os
import asyncio
from dotenv import load_dotenv
import html
import pandas as pd
from metrics import ClusteringMetrics
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from clustering import clusterize_texts
from clustering import generate_insight_yandex
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from cache_manager import cache
from analytics import generate_detailed_report
from config import TEMP_DIR
from rate_limiter import rate_limiter
from utils import (
    cleanup_old_temp_files,
    cleanup_file_safe,
    check_disk_space,
    format_time_remaining,
    get_user_display_name
)
from config import ADMIN_TELEGRAM_ID
import datetime
from progress_tracker import ProgressTracker

PROCESSING_SEMAPHORE = asyncio.Semaphore(2)

# Настройки логирования
# Создаём директорию для логов
LOG_DIR = Path("/home/yc-user/logs")
LOG_DIR.mkdir(exist_ok=True)

# Форматирование логов
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Хендлер для файла (с автоматической ротацией)
file_handler = RotatingFileHandler(
    LOG_DIR / "bot.log",
    maxBytes=10*1024*1024,  # 10 МБ на файл
    backupCount=5,           # Храним 5 файлов (итого 50 МБ)
    encoding='utf-8'
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

# Хендлер для консоли (чтобы systemd тоже видел)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

# Настройка корневого логгера
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

# Загрузка токена
load_dotenv()
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_msg = """
⚠️ Бот в режиме бета-тестирования. Будем рады вашей обратной связи!

👋 <b>Привет! Я бот для кластеризации текстов.</b>

📝 <b>Что я умею:</b>
• Анализирую тексты и группирую их по темам
• Нахожу общие паттерны в обращениях
• Помогаю понять, о чём чаще всего пишут клиенты

📎 <b>Как использовать:</b>
1. Подготовьте CSV файл с текстами
2. Первая колонка должна содержать тексты для анализа
3. Отправьте файл мне

💡 <b>Совет:</b> Лучше всего работает на 50-5000 текстах

<b>Команды:</b>
/help - подробная инструкция
/about - технологии и методы
/feedback - обратная связь

Готовы? Отправьте мне CSV файл! 🚀
    """
    await update.message.reply_text(welcome_msg, parse_mode='HTML')


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_msg = """
❓ <b>Справка по использованию</b>

<b>Формат файла:</b>
• CSV файл с текстами
• Тексты для анализа должны содержаться в первой колонке
• Кодировка: UTF-8
• Лимиты: 5 файлов/час, макс. 20 МБ, 50k строк

<b>Пример CSV:</b>
<code>текст
Не пришел заказ вовремя
Качество товара плохое
Долго ждал доставку
Товар не соответствует описанию</code>

<b>Что получите:</b>
✅ Файл с кластерами
✅ Статистику по кластерам
✅ Подробный PDF-отчёт с примерами кластеров


<b>Команды:</b>
/start - начать работу
/help - эта справка
/about - о технологиях
/feedback - обратная связь

Есть вопросы? Просто отправьте файл! 📊
    """
    await update.message.reply_text(help_msg, parse_mode='HTML')


async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    about_msg = """
🤖 <b>О технологиях</b>

Этот бот использует современные методы машинного обучения для автоматической группировки текстов по смыслу.

<b>🔬 Технологический стек:</b>

<b>BERTopic</b> — алгоритм кластеризации текстов. Автоматически находит темы в текстах без предварительной разметки.

<b>Sentence Transformers</b> — нейросети для понимания смысла. Превращает тексты в числовые векторы, сохраняя их значение

<b>UMAP</b> — снижение размерности данных. Упрощает сложные данные, сохраняя важные связи между текстами

<b>HDBSCAN</b> — алгоритм кластеризации. Находит группы похожих текстов автоматически

<b>💪 Преимущества перед ручным анализом:</b>
• Не нужно заранее знать темы
• Работает на любом языке
• Быстро обрабатывает тысячи текстов
• Находит неожиданные паттерны

Вопросы? Просто попробуйте! 🚀
    """
    await update.message.reply_text(about_msg, parse_mode='HTML')


async def feedback_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    feedback_msg = """
💬 <b>Обратная связь</b>

Нашли баг, есть идеи по улучшению или просто хотите поделиться впечатлениями?

Пишите мне: @viktoryafedoseenko

Буду рада любым комментариям! 🙏
    """
    await update.message.reply_text(feedback_msg, parse_mode='HTML')


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Статистика для администратора"""
    # Проверка прав доступа
    if not ADMIN_TELEGRAM_ID or update.effective_user.id != int(ADMIN_TELEGRAM_ID):
        await update.message.reply_text(
            "❌ У вас нет доступа к этой команде.",
            parse_mode='HTML'
        )
        return
    
    try:
        # Проверка дискового пространства
        disk_ok, free_gb = check_disk_space(min_free_gb=0.1)
        
        # Анализ логов
        errors_count = 0
        files_processed = 0
        warnings_count = 0
        
        try:
            log_file = LOG_DIR / "bot.log"
            if log_file.exists():
                with open(log_file, "r", encoding='utf-8') as f:
                    lines = f.readlines()
                    # Подсчитываем ошибки и успешные обработки
                    errors_count = len([l for l in lines if "ERROR" in l or "CRITICAL ERROR" in l])
                    files_processed = len([l for l in lines if "CLUSTERING COMPLETE" in l])
                    warnings_count = len([l for l in lines if "WARNING" in l or "⚠️" in l])
        except Exception as log_error:
            logger.error(f"Error reading logs: {log_error}")
        
        # Статистика rate limiter
        active_users = len(rate_limiter.requests) if hasattr(rate_limiter, 'requests') else 0
        
        # Статистика кэша
        cache_items = 0
        try:
            cache_dir = Path("cache")
            if cache_dir.exists():
                cache_items = len(list(cache_dir.glob("*.pkl")))
        except:
            pass
        
        # Формируем сообщение
        msg = (
            f"📊 <b>Статистика бота</b>\n\n"
            f"💾 <b>Диск:</b> {free_gb:.1f} ГБ свободно\n"
            f"   Статус: {'✅ OK' if disk_ok else '⚠️ Мало места'}\n\n"
            f"📈 <b>Обработано файлов:</b> {files_processed}\n"
            f"❌ <b>Ошибок:</b> {errors_count}\n"
            f"⚠️ <b>Предупреждений:</b> {warnings_count}\n\n"
            f"👥 <b>Активных пользователей:</b> {active_users}\n"
            f"💾 <b>Элементов в кэше:</b> {cache_items}\n\n"
            f"⏰ <b>Время:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        await update.message.reply_text(msg, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Error in stats_command: {e}", exc_info=True)
        await update.message.reply_text(
            f"❌ Ошибка при получении статистики: {str(e)}",
            parse_mode='HTML'
        )


async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    progress_msg = None
    file_path = None
    result_path = None
    cache_key = None
    
    try:
        # Логирование: Начало обработки
        user_id = update.effective_user.id
        username = update.effective_user.username or "unknown"
        file_name = update.message.document.file_name
        
        logger.info(f"📥 NEW FILE | User: {user_id} (@{username}) | File: {file_name}")

        # Rate Limiting проверка
        allowed, remaining, wait_time = rate_limiter.is_allowed(user_id)
        
        if not allowed:
            await update.message.reply_text(
                f"⏱ <b>Превышен лимит запросов</b>\n\n"
                f"Вы можете обработать максимум 5 файлов в час.\n"
                f"Попробуйте снова через <b>{format_time_remaining(wait_time)}</b>.\n\n"
                f"💡 Это сделано для стабильности сервиса",
                parse_mode='HTML'
            )
            return
        
        logger.info(f"✅ Rate limit OK | User: {user_id} | Remaining: {remaining}")
        
        # Проверка дискового пространства
        disk_ok, free_gb = check_disk_space(min_free_gb=1.0)
        
        if not disk_ok:
            await update.message.reply_text(
                "⚠️ <b>Сервер временно перегружен</b>\n\n"
                "Попробуйте через несколько минут.",
                parse_mode='HTML'
            )
            logger.error(f"🚨 LOW DISK SPACE | Free: {free_gb:.2f} GB")
            return

        # Проверка размера файла
        MAX_FILE_SIZE_MB = 20
        file_size_mb = update.message.document.file_size / (1024 * 1024)

        logger.info(f"📊 FILE INFO | User: {user_id} | Size: {file_size_mb:.2f} MB")
        
        if file_size_mb > MAX_FILE_SIZE_MB:
            logger.warning(f"⚠️ FILE TOO LARGE | User: {user_id} | Size: {file_size_mb:.2f} MB > {MAX_FILE_SIZE_MB} MB")
            await update.message.reply_text(
                f"❌ <b>Файл слишком большой</b>\n\n"
                f"Размер: {file_size_mb:.1f} МБ\n"
                f"Максимум: {MAX_FILE_SIZE_MB} МБ\n\n"
                f"💡 Попробуйте разбить данные на части",
                parse_mode='HTML'
            )
            return
        
        # Проверка формата
        if not update.message.document.file_name.endswith('.csv'):
            await update.message.reply_text(
                "❌ <b>Неверный формат файла</b>\n\n"
                "Пожалуйста, отправьте CSV файл\n"
                "Файл должен иметь расширение .csv",
                parse_mode='HTML'
            )
            return
        
        # Шаг 1: Загрузка файла
        progress_msg = await update.message.reply_text(
            "⏳ <b>Начинаю обработку...</b>",
            parse_mode='HTML'
        )
        
        # Создаём tracker
        tracker = ProgressTracker(progress_msg, min_interval=3.0)
        
        # Этап 1: Загрузка файла
        await tracker.update(
            stage="📥 Загрузка файла",
            percent=5,
            force=True
        )
        
        file = await update.message.document.get_file()
        file_path = f"/tmp/{file.file_unique_id}.csv"
        await file.download_to_drive(file_path)
        
        # Шаг 2: Анализ файла
        await tracker.update(
            stage="📊 Анализ структуры файла",
            percent=10
        )
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8', dtype=str)
            n_rows = len(df)
            n_cols = len(df.columns)

            logger.info(f"📋 DATASET LOADED | User: {user_id} | Rows: {n_rows} | Cols: {n_cols}")
            
            # Проверка количества строк
            MAX_ROWS = 50000
            if n_rows > MAX_ROWS:
                logger.warning(f"⚠️ TOO MANY ROWS | User: {user_id} | Rows: {n_rows} > {MAX_ROWS}")
                await tracker.update(
                    f"❌ Слишком много строк ({n_rows} > {MAX_ROWS})",
                    0,
                    "Пожалуйста, разделите файл на части",
                    force=True
                )
                return
            
            if n_rows == 0:
                await tracker.update(
                    "❌ Файл пустой",
                    0,
                    "В файле нет данных для анализа",
                    force=True
                )
                return
            
        except Exception as e:
            await tracker.update(
                "❌ Ошибка чтения файла",
                0,
                "Проверьте кодировку UTF-8 и формат CSV",
                force=True
            )
            logger.error(f"CSV read error: {e}")
            return
        
        # Шаг 3: Предобработка
        await tracker.update(
            stage="🧹 Предобработка текстов",
            percent=20,
            details="Очистка HTML, удаление дубликатов"
        )
        
        # Шаг 4: Кластеризация (самый долгий)
        await tracker.update(
            stage="🎯 Кластеризация текстов",
            percent=40,
            details=f"Это займёт 2-15 минут для {n_rows} текстов"
        )
        
        # Callback для обновления из clustering.py
        async def clustering_progress_callback(msg: str):
            """Callback для обновления прогресса из процесса кластеризации"""
            # Парсим сообщение и определяем процент
            if "Предобработка" in msg or "предобработк" in msg.lower():
                await tracker.update("🧹 Предобработка", 25)
            elif "Загрузка модели" in msg or "модели" in msg.lower() or "🤖" in msg:
                await tracker.update("🤖 Загрузка AI модели", 35)
            elif "Кластеризация" in msg or "🎯" in msg:
                await tracker.update("🎯 Кластеризация", 50)
            elif "Объединение" in msg or "похожих" in msg or "🔗" in msg:
                await tracker.update("🔗 Объединение похожих кластеров", 65)
            elif "Генерация названий" in msg or "названий" in msg.lower() or "📝" in msg:
                await tracker.update("📝 Генерация названий (AI)", 75)
            elif "иерархии" in msg.lower() or "🗂️" in msg:
                await tracker.update("🗂️ Создание иерархии", 85)
            elif "Сохранение" in msg or "сохран" in msg.lower() or "💾" in msg:
                await tracker.update("💾 Сохранение результатов", 95)
        
        # Вызываем кластеризацию с callback
        result_path, stats, hierarchy, master_names = clusterize_texts(
            file_path, 
            progress_callback=clustering_progress_callback
        )
        
        # Этап 5: Формирование результата
        await tracker.update(
            stage="📋 Формирование отчёта",
            percent=98
        )
        
        # Логирование: Результаты кластеризации
        logger.info(
            f"✅ CLUSTERING COMPLETE | User: {user_id} | "
            f"Texts: {stats['total_texts']} | "
            f"Clusters: {stats['n_clusters']} | "
            f"Noise: {stats['noise_percent']:.1f}% | "
            f"Silhouette: {stats.get('quality_metrics', {}).get('silhouette_score', 0):.3f}"
        )
        
        # Шаг 4: Формирование статистики
        stats_message = format_statistics(stats)
        
        # Шаг 5: Формирование инсайта
        insight_text = generate_insight_yandex(stats)
        if insight_text:
            stats_message += f"\n\n💡 <b>Инсайт:</b>\n{html.escape(insight_text)}"

        stats_message += "\n\n✨ Готово! Хотите проанализировать другие тексты? Отправляйте новый файл — я готов!"

        # Сохраняем в кэш (перед отправкой файла)
        df_cached = pd.read_csv(result_path, encoding='utf-8')
        
        cache_data = {
            'df': df_cached,
            'stats': stats,
            'cluster_names': {  # Извлекаем из датафрейма
                row['cluster_id']: row['cluster_name']
                for _, row in df_cached[['cluster_id', 'cluster_name']].drop_duplicates().iterrows()
            },
            'file_name': update.message.document.file_name,
            'hierarchy': hierarchy,
            'master_names': master_names
        }
        
        cache_key = cache.save(
            user_id=update.effective_user.id,
            file_name=update.message.document.file_name,
            data=cache_data
        )

        # Завершение
        await tracker.complete("✅ Анализ завершён!")
        
        # Удаляем прогресс-сообщение перед отправкой результата
        await progress_msg.delete()
        
        # Показываем кнопки выбора
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("Детальный отчёт в PDF", callback_data=f"pdf_{cache_key}")],
            [InlineKeyboardButton("Поделиться", callback_data=f"share_{cache_key}")]
        ])

        MAX_CAPTION_LENGTH = 1000  # С запасом (лимит 1024)

        if len(stats_message) > MAX_CAPTION_LENGTH:
            # Короткий caption для файла
            short_caption = "✅ <b>Кластеризация завершена!</b>\n\n📎 Подробная статистика ниже"
            
            with open(result_path, 'rb') as result_file:
                await update.message.reply_document(
                    document=result_file,
                    filename=os.path.basename(result_path),
                    caption=short_caption,
                    parse_mode='HTML',
                    reply_markup=keyboard
                )
            
            # Статистика отдельно
            await update.message.reply_text(
                stats_message,
                parse_mode='HTML'
            )
        else:
            # Если короткая — всё в одном
            with open(result_path, 'rb') as result_file:
                await update.message.reply_document(
                    document=result_file,
                    filename=os.path.basename(result_path),
                    caption=stats_message,
                    parse_mode='HTML',
                    reply_markup=keyboard
                )
        
        # Отправка статистики
        if 'quality_metrics' in stats:
            quality_report = ClusteringMetrics.format_report(stats['quality_metrics'])
            await update.message.reply_text(
                quality_report,
                parse_mode='HTML'
            )

        # Удаление прогресс-сообщения
        await progress_msg.delete()

    except ValueError as e:
        # 🆕 ЛОГИРОВАНИЕ: Ошибка валидации
        logger.warning(f"⚠️ VALIDATION ERROR | User: {user_id} | Error: {str(e)[:200]}")
        error_msg = f"⚠️ <b>Проблема с данными</b>\n\n{html.escape(str(e))}\n\n💡 Проверьте формат файла"
        if progress_msg:
            await progress_msg.edit_text(error_msg, parse_mode='HTML')
        else:
            await update.message.reply_text(error_msg, parse_mode='HTML')
        logger.warning(f"ValueError: {e}")
        
    except Exception as e:
        # Логирование: Критическая ошибка
        logger.error(
            f"❌ CRITICAL ERROR | User: {user_id} | File: {file_name} | Error: {str(e)}",
            exc_info=True  # Добавляет полный traceback
        )
        
        # Уведомляем админа о критичной ошибке
        if ADMIN_TELEGRAM_ID:
            try:
                user_display = get_user_display_name(update.effective_user)
                await context.bot.send_message(
                    chat_id=int(ADMIN_TELEGRAM_ID),
                    text=(
                        f"🚨 <b>Критичная ошибка</b>\n\n"
                        f"👤 <b>Пользователь:</b> {user_display} (ID: {user_id})\n"
                        f"📄 <b>Файл:</b> {html.escape(file_name) if file_name else 'N/A'}\n"
                        f"❌ <b>Ошибка:</b> {html.escape(str(e)[:300])}\n\n"
                        f"⏰ <b>Время:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    ),
                    parse_mode='HTML'
                )
            except Exception as admin_error:
                logger.error(f"Failed to notify admin: {admin_error}")
        
        error_msg = (
            "❌ <b>Произошла ошибка</b>\n\n"
            "Не удалось обработать файл.\n\n"
            "🔍 <b>Возможные причины:</b>\n"
            "• Неверная структура CSV\n"
            "• Слишком мало данных\n"
            "• Некорректная кодировка\n\n"
            "💡 <b>Попробуйте:</b>\n"
            "• Проверить формат файла\n"
            "• Использовать UTF-8 кодировку\n"
            "• Убедиться, что есть текстовые данные"
        )
        if progress_msg:
            await progress_msg.edit_text(error_msg, parse_mode='HTML')
        else:
            await update.message.reply_text(error_msg, parse_mode='HTML')
        logger.error(f"Error processing file: {e}", exc_info=True)
        
    finally:
        # Очистка временных файлов
        cleanup_file_safe(file_path)
        if result_path and cache_key:
            cleanup_file_safe(result_path)


def format_statistics(stats):
    """Форматирование статистики в красивое сообщение (с экранированием HTML)"""
    msg = "✅ <b>Кластеризация завершена!</b>\n\n"
    msg += "📊 <b>Результаты:</b>\n"
    msg += f"• Обработано текстов: <b>{stats['total_texts']}</b>\n"
    msg += f"• Найдено кластеров: <b>{stats['n_clusters']}</b>\n"
    msg += f"• Средний размер: <b>{stats['avg_cluster_size']:.0f}</b> текстов\n"
    msg += f"• Шум: <b>{stats['noise_percent']:.1f}%</b>\n\n"
    
    # Топ-3 кластера (с экранированием названий)
    if 'top_clusters' in stats and stats['top_clusters']:
        msg += "<b>Топ-3 кластера:</b>\n"
        
        # Собираем уникальные названия для избежания дублей
        seen_names = set()
        unique_clusters = []
        
        for cluster in stats['top_clusters']:
            if cluster['name'] not in seen_names:
                seen_names.add(cluster['name'])
                unique_clusters.append(cluster)
            if len(unique_clusters) >= 3:
                break
        
        for i, cluster in enumerate(unique_clusters, 1):
            emoji = ["1️⃣", "2️⃣", "3️⃣"][i-1]
            # Экранируем название кластера
            safe_name = html.escape(cluster['name'])
            msg += f"{emoji} <i>{safe_name}</i> — {cluster['size']} текстов\n"
        msg += "\n"
    
    msg += "📎 Полные результаты в прикрепленном файле\n"
    
    return msg


def generate_critical_insight(stats, cluster_names):
    """Генерирует инсайт 'Что критично?'"""
    top_clusters = stats.get('top_clusters', [])[:3]
    
    message = "🔴 <b>Критичные проблемы (топ-3 по объёму):</b>\n\n"
    
    for i, cluster in enumerate(top_clusters, 1):
        percent = (cluster['size'] / stats['total_texts']) * 100
        
        message += f"{i}. <b>{html.escape(cluster['name'])}</b>\n"
        message += f"   📊 {cluster['size']} обращений ({percent:.1f}%)\n"
        
        # Добавляем рекомендацию в зависимости от процента
        if percent > 5:
            message += f"   ⚠️ <i>Критично! Требует немедленных действий</i>\n"
        elif percent > 3:
            message += f"   🟡 <i>Важно. Включить в ближайший спринт</i>\n"
        else:
            message += f"   🟢 <i>Средний приоритет</i>\n"
        
        message += "\n"
    
    message += (
        "💡 <b>Рекомендация:</b>\n"
        "Сосредоточьтесь на проблемах с долей >5% — "
        "это влияет на большинство пользователей.\n\n"
        "📊 Полный анализ доступен в PDF-отчёте"
    )
    
    return message


def generate_priority_insight(stats, cluster_names):
    """Генерирует инсайт 'Как приоритизировать?'"""
    top_clusters = stats.get('top_clusters', [])
    total = stats['total_texts']
    
    # Группируем по приоритетам
    critical = [c for c in top_clusters if (c['size'] / total) > 0.05]
    important = [c for c in top_clusters if 0.03 < (c['size'] / total) <= 0.05]
    medium = [c for c in top_clusters if (c['size'] / total) <= 0.03]
    
    message = "📋 <b>Матрица приоритизации:</b>\n\n"
    
    message += f"🔴 <b>КРИТИЧНО</b> (>5% обращений):\n"
    if critical:
        for c in critical[:3]:
            message += f"   • {html.escape(c['name'])} — {c['size']} текстов\n"
    else:
        message += "   Нет критичных проблем ✅\n"
    message += "\n"
    
    message += f"🟡 <b>ВАЖНО</b> (3-5% обращений):\n"
    if important:
        for c in important[:3]:
            message += f"   • {html.escape(c['name'])} — {c['size']} текстов\n"
    else:
        message += "   —\n"
    message += "\n"
    
    message += f"🟢 <b>СРЕДНИЙ ПРИОРИТЕТ</b> (<3%):\n"
    message += f"   {len(medium)} тем\n\n"
    
    message += (
        "💡 <b>Подход:</b>\n"
        "1. Решите критичные проблемы в первую очередь\n"
        "2. Важные — включите в roadmap на месяц\n"
        "3. Средние — фиксируйте как технический долг\n\n"
        "📊 Детали в PDF-отчёте"
    )
    
    return message


def generate_action_insight(stats, cluster_names):
    """Генерирует инсайт 'Что делать первым?'"""
    top_clusters = stats.get('top_clusters', [])
    if not top_clusters:
        return "⚠️ Недостаточно данных для генерации плана действий."
    
    top_cluster = top_clusters[0]
    total = stats['total_texts']
    percent = (top_cluster['size'] / total) * 100
    
    message = "💡 <b>План действий на ближайшую неделю:</b>\n\n"
    
    message += f"<b>Проблема #1: {html.escape(top_cluster['name'])}</b>\n"
    message += f"📊 Объём: {top_cluster['size']} обращений ({percent:.1f}%)\n\n"
    
    message += "🎯 <b>Что сделать:</b>\n\n"
    
    # Генерируем рекомендации в зависимости от типа проблемы
    name_lower = top_cluster['name'].lower()
    
    if any(word in name_lower for word in ['баг', 'ошибк', 'не работает', 'проблем']):
        message += (
            "1️⃣ <b>День 1-2:</b> Воспроизвести баг и оценить масштаб\n"
            "   → Создать задачу в Jira с приоритетом P0\n\n"
            "2️⃣ <b>День 3-4:</b> Hotfix + тестирование\n"
            "   → Привлечь QA для регрессионных тестов\n\n"
            "3️⃣ <b>День 5:</b> Деплой + мониторинг метрик\n"
            "   → Отследить снижение обращений в саппорт\n"
        )
    elif any(word in name_lower for word in ['оплат', 'платёж', 'деньг']):
        message += (
            "1️⃣ <b>День 1:</b> Проанализировать логи платёжной системы\n"
            "   → Найти паттерны неуспешных транзакций\n\n"
            "2️⃣ <b>День 2-3:</b> Связаться с платёжным провайдером\n"
            "   → Проверить лимиты и настройки\n\n"
            "3️⃣ <b>День 4-5:</b> Добавить альтернативный метод оплаты\n"
            "   → Например, СБП или криптовалюту\n"
        )
    elif any(word in name_lower for word in ['диплом', 'сертификат', 'документ']):
        message += (
            "1️⃣ <b>День 1:</b> Автоматизировать уведомления о статусе\n"
            "   → Email с трек-номером после выдачи\n\n"
            "2️⃣ <b>День 2-3:</b> Создать FAQ 'Где мой диплом?'\n"
            "   → Разместить на видном месте в ЛК\n\n"
            "3️⃣ <b>День 4-5:</b> Добавить опцию самовывоза\n"
            "   → Снизит нагрузку на доставку\n"
        )
    else:
        message += (
            "1️⃣ <b>День 1-2:</b> Глубже изучить проблему\n"
            "   → Прочитать 20-30 примеров из кластера\n\n"
            "2️⃣ <b>День 3-4:</b> Провести интервью с пользователями\n"
            "   → Понять root cause проблемы\n\n"
            "3️⃣ <b>День 5:</b> Создать план решения\n"
            "   → Оценить impact и effort\n"
        )
    
    message += (
        "\n📈 <b>Метрика успеха:</b>\n"
        f"Снижение обращений по теме '{html.escape(top_cluster['name'])}' "
        f"с {top_cluster['size']} до <{int(top_cluster['size'] * 0.5)} за месяц\n\n"
        "📊 Остальные проблемы см. в PDF-отчёте"
    )
    
    return message


async def handle_insight_request(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик запросов быстрых инсайтов"""
    query = update.callback_query
    await query.answer()
    
    callback_data = query.data
    user_id = update.effective_user.id
    
    logger.info(f"💡 INSIGHT REQUEST | User: {user_id} | Action: {callback_data}")
    
    # Парсим тип инсайта и cache_key
    # Формат: "insight_<type>_<cache_key>"
    parts = callback_data.split("_")
    if len(parts) < 3:
        await query.message.reply_text(
            "⚠️ Ошибка: неверный формат данных",
            parse_mode='HTML'
        )
        return
    
    insight_type = parts[1]  # critical, priority, action
    cache_key = "_".join(parts[2:])  # cache_key может содержать подчёркивания
    
    # Загружаем данные из кеша
    cached_data = cache.load(cache_key)
    if not cached_data:
        await query.message.reply_text(
            "⚠️ <b>Данные устарели</b>\n\n"
            "Результаты хранятся 1 час.\n"
            "Загрузите файл заново.",
            parse_mode='HTML'
        )
        return
    
    stats = cached_data['stats']
    cluster_names = cached_data.get('cluster_names', {})
    
    # Генерируем инсайт в зависимости от типа
    if insight_type == "critical":
        message = generate_critical_insight(stats, cluster_names)
    elif insight_type == "priority":
        message = generate_priority_insight(stats, cluster_names)
    elif insight_type == "action":
        message = generate_action_insight(stats, cluster_names)
    else:
        message = "⚠️ Неизвестный тип инсайта"
    
    await query.message.reply_text(message, parse_mode='HTML')


async def handle_share_request(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик кнопки 'Поделиться'"""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    logger.info(f"📤 SHARE REQUEST | User: {user_id}")
    
    # Получаем username бота
    bot_username = context.bot.username
    
    message = (
        "📤 <b>Как поделиться результатом:</b>\n\n"
        
        "<b>Переслать файлы</b>\n"
        "Просто перешлите CSV или PDF файл коллеге в Telegram.\n"
        "Он сможет открыть и изучить результаты.\n\n"
        
        "<b>Вариант 2: Отправить ссылку на бота</b>\n"
        f"Скопируйте и отправьте коллеге:\n"
        f"<code>https://t.me/{bot_username}</code>\n\n"
        
        "💬 <b>Сообщение для коллеги:</b>\n"
        "<i>Попробуй этот бот для анализа текстов! "
        "Я только что обработал файл за несколько минут. "
        "Результат — кластеры по темам + PDF с инсайтами. "
        "Бесплатно до 50,000 текстов.</i>\n\n"
        
    )
    
    await query.message.reply_text(message, parse_mode='HTML')


# Обработчик запроса детального PDF
async def handle_pdf_request(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик запроса детального PDF отчёта"""
    query = update.callback_query
    await query.answer()

    # Логирование
    user_id = update.effective_user.id
    callback_data = query.data
    
    logger.info(f"📊 PDF REQUEST | User: {user_id} | Action: {callback_data}")
    
    callback_data = query.data
    
    # Извлекаем cache_key
    if not callback_data.startswith("pdf_"):
        logger.warning(f"⚠️ INVALID CALLBACK | User: {user_id} | Data: {callback_data}")
        await query.message.reply_text("❌ Ошибка: неверный формат данных")
        return
    
    cache_key = callback_data[4:]  # Убираем "pdf_"
    logger.info(f"🔄 GENERATING PDF | User: {user_id} | Cache key: {cache_key[:8]}...")
    
    # Показываем прогресс
    progress_msg = await query.message.reply_text(
        "⏳ <b>Генерирую детальный отчёт...</b>\n\n"
        "📊 Создание графиков\n"
        "📄 Формирование PDF\n"
        "📈 Подготовка расширенной статистики\n\n"
        "Это займёт 10-30 секунд...",
        parse_mode='HTML'
    )
    
    try:
        # Генерация с таймаутом
        result = await asyncio.wait_for(
            generate_detailed_report(cache_key, update.effective_user.id),
            timeout=120  # 2 минуты макс
        )
        
        if not result:
            logger.warning(f"⚠️ PDF GENERATION FAILED | User: {user_id} | Cache key: {cache_key[:8]}")
            await progress_msg.edit_text(
                "❌ <b>Ошибка генерации отчёта</b>\n\n"
                "Возможные причины:\n"
                "• Данные устарели (прошло больше часа)\n"
                "• Превышен размер отчёта (макс. 10 МБ)\n\n"
                "💡 Попробуйте загрузить файл заново",
                parse_mode='HTML'
            )
            return
        
        pdf_path, csv_path = result
        logger.info(f"✅ PDF GENERATED | User: {user_id} | Files: {pdf_path}, {csv_path}")
        
        # Отправляем файлы
        logger.info(f"📤 PDF SENT | User: {user_id}")
        await progress_msg.edit_text(
            "✅ <b>Отчёт готов!</b>\n\n"
            "📤 Отправляю файлы...",
            parse_mode='HTML'
        )
        
        # PDF
        with open(pdf_path, 'rb') as pdf_file:
            await query.message.reply_document(
                document=pdf_file,
                filename=f"detailed_report_{cache_key[:8]}.pdf",
                caption=(
                    "📊 <b>Детальный отчёт PDF</b>\n\n"
                    "Содержит:\n"
                    "• Полную статистику\n"
                    "• Графики распределения\n"
                    "• Топ-10 кластеров с примерами\n"
                    "• Ключевые слова по каждой теме"
                ),
                parse_mode='HTML'
            )
        
        # Extended CSV
        with open(csv_path, 'rb') as csv_file:
            await query.message.reply_document(
                document=csv_file,
                filename=f"extended_stats_{cache_key[:8]}.csv",
                caption="📈 <b>Расширенная статистика</b>\n\nРаспределение по всем кластерам с процентами",
                parse_mode='HTML'
            )
        
        await progress_msg.delete()
        
        # Убираем кнопки из исходного сообщения
        await query.edit_message_reply_markup(reply_markup=None)
        
        # Финальное сообщение
        await query.message.reply_text(
            "✨ <b>Готово!</b>\n\n"
            "Хотите проанализировать другие тексты?\n"
            "Отправляйте новый файл — я готов! 🚀",
            parse_mode='HTML'
        )
        
        # Очистка временных файлов
        try:
            Path(pdf_path).unlink()
            Path(csv_path).unlink()
        except:
            pass
        
    except asyncio.TimeoutError:
        logger.error(f"⏱ PDF TIMEOUT | User: {user_id} | Cache key: {cache_key[:8]}")
        await progress_msg.edit_text(
            "⏱ <b>Превышено время ожидания</b>\n\n"
            "Генерация отчёта заняла слишком много времени.\n"
            "Попробуйте с меньшим файлом или повторите позже.",
            parse_mode='HTML'
        )
    
    except Exception as e:
        logger.error(f"❌ PDF ERROR | User: {user_id} | Error: {str(e)}", exc_info=True)
        await progress_msg.edit_text(
            "❌ <b>Ошибка генерации отчёта</b>\n\n"
            "Попробуйте повторить запрос через минуту",
            parse_mode='HTML'
        )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Глобальный обработчик ошибок"""
    # 🆕 ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ
    logger.error("=" * 60)
    logger.error("🚨 UNHANDLED EXCEPTION")
    
    if update and isinstance(update, Update):
        user_id = update.effective_user.id if update.effective_user else "unknown"
        logger.error(f"User: {user_id}")
        
        if update.message:
            logger.error(f"Message: {update.message.text[:100] if update.message.text else 'N/A'}")
    
    logger.error(f"Error: {context.error}")
    logger.error("Traceback:", exc_info=context.error)
    logger.error("=" * 60)


def main():
    logger.info("=" * 60)
    logger.info("🤖 BOT STARTING...")
    logger.info(f"📁 Log directory: {LOG_DIR}")
    logger.info(f"📁 Temp directory: {TEMP_DIR}")
    logger.info(f"🔑 Token configured: {'✅' if TOKEN else '❌'}")
    
    # Очистка старых файлов при старте
    logger.info("🗑️ Cleaning up old temp files...")
    cleanup_old_temp_files()
    
    logger.info("=" * 60)
    
    # Создаём application с job_queue
    from telegram.ext import JobQueue

    application = (
        Application.builder()
        .token(TOKEN)
        .build()
    )

    # Инициализируем job_queue если его нет
    if application.job_queue is None:
        logger.warning("⚠️ JobQueue not available, periodic tasks disabled")


    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("about", about_command))
    application.add_handler(CommandHandler("feedback", feedback_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_file))

    from telegram.ext import CallbackQueryHandler
    application.add_handler(CallbackQueryHandler(handle_pdf_request, pattern="^pdf_"))
    application.add_handler(CallbackQueryHandler(handle_insight_request, pattern="^insight_"))
    application.add_handler(CallbackQueryHandler(handle_share_request, pattern="^share_"))
    async def handle_csv_only(update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text(
            "✅ Отлично! CSV файл уже у вас.\n\n"
            "Хотите проанализировать другие тексты? Отправляйте новый файл!"
        )
    
    application.add_handler(CallbackQueryHandler(handle_csv_only, pattern="^csv_only$"))

    application.add_error_handler(error_handler)
    
    # Периодические задачи
    if application.job_queue:
        job_queue = application.job_queue
        
        # Очистка временных файлов каждые 6 часов
        job_queue.run_repeating(
            callback=lambda ctx: cleanup_old_temp_files(),
            interval=datetime.timedelta(hours=6),
            first=datetime.timedelta(seconds=10)
        )
        
        # Очистка неактивных пользователей из rate limiter раз в сутки
        job_queue.run_repeating(
            callback=lambda ctx: rate_limiter.cleanup_old_users(),
            interval=datetime.timedelta(hours=24),
            first=datetime.timedelta(hours=1)
        )
        
        logger.info("✅ Periodic tasks scheduled")
    else:
        logger.warning("⚠️ JobQueue not available - periodic cleanup disabled")

    logger.info("✅ All handlers registered")
    logger.info("🚀 Bot is running and ready to accept requests!")
    logger.info("=" * 60)
    
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()

