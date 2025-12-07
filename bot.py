"""
Telegram бот для кластеризации и классификации текстов.
Поддерживает два режима:
1. Автоматическая кластеризация (BERTopic + HDBSCAN)
2. Классификация по заданным категориям (YandexGPT)
"""

import logging
import os
import sys
from datetime import datetime
from typing import Dict, Optional
import pandas as pd

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ParseMode
from dotenv import load_dotenv

from clustering import process_clustering
from analytics import generate_detailed_report
from cache_manager import CacheManager
from rate_limiter import RateLimiter
from utils import clean_filename, format_file_size

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('/home/yc-user/logs/bot.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Константы для состояний диалога
CHOOSING_MODE, ENTERING_CATEGORIES, ENTERING_DESCRIPTIONS, PROCESSING_FILE = range(4)

# Глобальные объекты
cache_manager = CacheManager()
rate_limiter = RateLimiter()
classifier = None

# Проверка наличия YandexGPT API и модуля классификации
YANDEX_API_AVAILABLE = False
try:
    from classification import LLMClassifier, validate_categories, parse_categories_from_text
    if os.getenv("YANDEX_API_KEY") and os.getenv("YANDEX_FOLDER_ID"):
        classifier = LLMClassifier()
        YANDEX_API_AVAILABLE = True
        logger.info("YandexGPT API инициализирован для классификации")
except ImportError:
    logger.warning("Модуль classification.py не найден. Классификация отключена.")
except Exception as e:
    logger.warning(f"Не удалось инициализировать классификатор: {e}")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start."""
    user = update.effective_user
    logger.info(f"Пользователь {user.id} ({user.username}) запустил бота")
    
    # Очищаем старые данные
    context.user_data.clear()
    
    welcome_text = f"""👋 Привет, {user.first_name}!

Я помогу тебе проанализировать текстовые данные двумя способами:

🔍 **Автоматическая кластеризация**
Загружаешь CSV-файл → я нахожу темы и группирую похожие тексты

🏷️ **Классификация по категориям**
Задаешь свои категории → я распределяю тексты по ним с помощью AI

📋 Что умею:
• Обрабатывать до 50,000 текстов
• Находить скрытые темы и паттерны
• Создавать детальные отчеты
• Работать с русским и английским языками

🚀 Выбери режим работы:"""
    
    keyboard = []
    
    # Кнопка кластеризации всегда доступна
    keyboard.append([InlineKeyboardButton("🔍 Кластеризация", callback_data="mode_clustering")])
    
    # Кнопка классификации только если доступен API
    if YANDEX_API_AVAILABLE:
        keyboard.append([InlineKeyboardButton("🏷️ Классификация", callback_data="mode_classification")])
    
    keyboard.append([InlineKeyboardButton("❓ Помощь", callback_data="help")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Проверяем откуда пришел запрос
    if update.callback_query:
        await update.callback_query.edit_message_text(
            welcome_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN
        )
    else:
        await update.message.reply_text(
            welcome_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN
        )
    
    return CHOOSING_MODE


async def mode_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик выбора режима работы."""
    query = update.callback_query
    await query.answer()
    
    # Обработка кнопки "Помощь"
    if query.data == "help":
        await help_command(update, context)
        return CHOOSING_MODE
    
    # Обработка кнопки "Вернуться в меню"
    if query.data == "restart":
        return await start(update, context)
    
    mode = query.data.replace("mode_", "")
    context.user_data['mode'] = mode
    
    if mode == "clustering":
        text = """🔍 **Режим: Автоматическая кластеризация**

Я найду темы и сгруппирую похожие тексты автоматически.

📎 Отправь CSV-файл:
• Первая колонка — тексты для анализа
• Кодировка UTF-8
• Макс. размер: 20 МБ
• Макс. строк: 50,000

✨ Что получишь:
• CSV с кластерами и названиями тем
• Статистику по группам
• Подробный PDF-отчет (по запросу)"""
        
        await query.edit_message_text(
            text,
            parse_mode=ParseMode.MARKDOWN
        )
        return PROCESSING_FILE
        
    elif mode == "classification":
        if not YANDEX_API_AVAILABLE:
            await query.edit_message_text(
                "❌ Классификация недоступна: не настроен YandexGPT API\n\n"
                "Для использования этой функции добавьте в .env:\n"
                "• YANDEX_API_KEY\n"
                "• YANDEX_FOLDER_ID"
            )
            return ConversationHandler.END
        
        text = """🏷️ **Режим: Классификация по категориям**

Ты задаешь категории, я распределяю тексты по ним с помощью AI.

📝 Введи категории (каждая с новой строки):

**Например:**
Проблемы с оплатой
Вопросы по доставке
Качество товара
Технические проблемы
Общие вопросы
Или через запятую: Оплата, Доставка, Качество, Техподдержка

💡 **Требования:**
• Минимум 2 категории
• Максимум 20 категорий
• Четкие и понятные названия"""
        
        await query.edit_message_text(
            text,
            parse_mode=ParseMode.MARKDOWN
        )
        return ENTERING_CATEGORIES


async def receive_categories(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Получение категорий от пользователя."""
    text = update.message.text
    categories = parse_categories_from_text(text)
    
    # Валидация
    is_valid, error_msg = validate_categories(categories)
    if not is_valid:
        await update.message.reply_text(
            f"❌ {error_msg}\n\n"
            "Попробуй еще раз или /cancel для отмены."
        )
        return ENTERING_CATEGORIES
    
    context.user_data['categories'] = categories
    
    # Предложение добавить описания
    categories_list = "\n".join([f"{i+1}. {cat}" for i, cat in enumerate(categories)])
    
    keyboard = [
        [InlineKeyboardButton("✅ Продолжить без описаний", callback_data="skip_descriptions")],
        [InlineKeyboardButton("📝 Добавить описания", callback_data="add_descriptions")],
        [InlineKeyboardButton("✏️ Изменить категории", callback_data="edit_categories")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        f"✅ **Категории приняты** ({len(categories)} шт.):\n\n"
        f"{categories_list}\n\n"
        "Хочешь добавить описания для более точной классификации?",
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN
    )
    
    return ENTERING_DESCRIPTIONS


async def descriptions_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик выбора про описания категорий."""
    query = update.callback_query
    await query.answer()
    
    action = query.data
    
    if action == "skip_descriptions":
        context.user_data['descriptions'] = None
        
        text = """✅ **Категории готовы!**

📎 Теперь отправь CSV-файл с текстами:
• Первая колонка — тексты для классификации
• Кодировка UTF-8
• Макс. размер: 20 МБ
• Макс. строк: 10,000 (для классификации)

⏱️ Время обработки: ~1-2 секунды на текст"""
        
        await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN)
        return PROCESSING_FILE
        
    elif action == "add_descriptions":
        categories = context.user_data['categories']
        categories_list = "\n".join([f"{i+1}. {cat}" for i, cat in enumerate(categories)])
        
        text = f"""📝 **Добавь описания для категорий**

**Формат** (каждая с новой строки):
Название категории: краткое описание
**Например:**
Проблемы с оплатой: ошибки при оплате, не проходит платеж, возврат средств
Вопросы по доставке: сроки доставки, отслеживание, не пришел заказ
Качество товара: брак, несоответствие описанию, повреждения

**Твои категории:**
{categories_list}

Введи описания или /skip чтобы пропустить."""
        
        await query.edit_message_text(
            text,
            parse_mode=ParseMode.MARKDOWN
        )
        return ENTERING_DESCRIPTIONS
        
    elif action == "edit_categories":
        await query.edit_message_text(
            "📝 Введи категории заново (каждая с новой строки):"
        )
        return ENTERING_CATEGORIES


async def receive_descriptions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Получение описаний категорий от пользователя."""
    text = update.message.text
    categories = context.user_data['categories']
    
    # Парсинг описаний
    descriptions = {}
    lines = text.strip().split('\n')
    
    for line in lines:
        if ':' in line:
            parts = line.split(':', 1)
            cat_name = parts[0].strip()
            description = parts[1].strip()
            
            # Ищем категорию (нечеткое совпадение)
            for cat in categories:
                if cat.lower() == cat_name.lower() or cat_name.lower() in cat.lower():
                    descriptions[cat] = description
                    break
    
    context.user_data['descriptions'] = descriptions if descriptions else None
    
    # Показываем что получилось
    if descriptions:
        desc_text = "\n".join([
            f"• **{cat}**: {desc}" for cat, desc in descriptions.items()
        ])
        result_text = f"✅ **Описания добавлены:**\n\n{desc_text}"
    else:
        result_text = "⚠️ Не удалось распознать описания. Продолжаем без них."
    
    await update.message.reply_text(
        f"{result_text}\n\n"
        "📎 Теперь отправь CSV-файл с текстами для классификации.",
        parse_mode=ParseMode.MARKDOWN
    )
    
    return PROCESSING_FILE


async def process_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка загруженного файла."""
    user = update.effective_user
    mode = context.user_data.get('mode', 'clustering')
    
    # Проверка rate limit
    can_process, wait_time = rate_limiter.can_process(user.id)
    if not can_process:
        await update.message.reply_text(
            f"⏳ Превышен лимит запросов.\n"
            f"Попробуй снова через {wait_time} минут."
        )
        return PROCESSING_FILE
    
    # Получение файла
    document = update.message.document
    if not document:
        await update.message.reply_text(
            "❌ Пожалуйста, отправь CSV-файл."
        )
        return PROCESSING_FILE
    
    # Проверка расширения
    if not document.file_name.endswith('.csv'):
        await update.message.reply_text(
            "❌ Поддерживаются только CSV-файлы."
        )
        return PROCESSING_FILE
    
    # Проверка размера
    max_size = 20 * 1024 * 1024  # 20 МБ
    if document.file_size > max_size:
        await update.message.reply_text(
            f"❌ Файл слишком большой ({format_file_size(document.file_size)}).\n"
            f"Максимальный размер: 20 МБ"
        )
        return PROCESSING_FILE
    
    # Скачивание файла
    status_msg = await update.message.reply_text("📥 Скачиваю файл...")
    
    file_path = None
    try:
        file = await document.get_file()
        file_path = f"/tmp/{user.id}_{document.file_name}"
        await file.download_to_drive(file_path)
        
        logger.info(
            f"Файл скачан: {document.file_name} "
            f"({format_file_size(document.file_size)}) "
            f"от пользователя {user.id}, режим: {mode}"
        )
        
        # Чтение CSV
        await status_msg.edit_text("📊 Читаю данные...")
        df = pd.read_csv(file_path, encoding='utf-8')
        
        if df.empty:
            await status_msg.edit_text("❌ Файл пустой")
            return PROCESSING_FILE
        
        texts = df.iloc[:, 0].astype(str).tolist()
        
        # Проверка лимитов в зависимости от режима
        max_texts = 10000 if mode == "classification" else 50000
        if len(texts) > max_texts:
            await status_msg.edit_text(
                f"❌ Слишком много строк: {len(texts)}\n"
                f"Максимум для {mode}: {max_texts}"
            )
            return PROCESSING_FILE
        
        # Обработка в зависимости от режима
        if mode == "clustering":
            await process_clustering_mode(
                update, context, texts, document.file_name, status_msg
            )
        else:
            await process_classification_mode(
                update, context, texts, document.file_name, status_msg
            )
        
        # Обновление rate limit
        rate_limiter.add_request(user.id)
        
    except Exception as e:
        logger.error(f"Ошибка обработки файла: {e}", exc_info=True)
        await status_msg.edit_text(
            f"❌ Ошибка обработки файла:\n`{str(e)}`",
            parse_mode=ParseMode.MARKDOWN
        )
    
    finally:
        # Очистка временного файла
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    
    return ConversationHandler.END


async def process_clustering_mode(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    texts: list,
    filename: str,
    status_msg
):
    """Обработка в режиме кластеризации."""
    user = update.effective_user
    
    await status_msg.edit_text(
        f"🔄 Начинаю кластеризацию {len(texts)} текстов...\n"
        f"⏱️ Это может занять несколько минут"
    )
    
    # Функция обновления прогресса
    async def progress_callback(stage: str, progress: float):
        stages_emoji = {
            "preprocessing": "🧹",
            "embedding": "🧠",
            "clustering": "🔍",
            "naming": "🏷️",
            "hierarchy": "📊"
        }
        emoji = stages_emoji.get(stage, "⏳")
        await status_msg.edit_text(
            f"{emoji} {stage.capitalize()}: {progress:.0f}%"
        )
    
    # Запуск кластеризации
    result_df, stats, success = await process_clustering(
        texts,
        progress_callback=progress_callback
    )
    
    if not success:
        await status_msg.edit_text("❌ Ошибка кластеризации")
        return
    
    # Сохранение результатов
    output_filename = clean_filename(f"clustered_{filename}")
    output_path = f"/tmp/{user.id}_{output_filename}"
    result_df.to_csv(output_path, index=False, encoding='utf-8')
    
    # Кэширование для PDF
    cache_key = cache_manager.save_to_cache(
        user.id,
        result_df,
        stats,
        filename
    )
    
    # Отправка результатов
    caption = format_clustering_stats(stats)
    
    keyboard = [
        [InlineKeyboardButton("📄 Получить PDF-отчет", callback_data=f"pdf_{cache_key}")],
        [InlineKeyboardButton("🔄 Обработать новый файл", callback_data="restart")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_document(
        document=open(output_path, 'rb'),
        filename=output_filename,
        caption=caption,
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN
    )
    
    await status_msg.delete()
    os.remove(output_path)


async def process_classification_mode(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    texts: list,
    filename: str,
    status_msg
):
    """Обработка в режиме классификации."""
    user = update.effective_user
    categories = context.user_data['categories']
    descriptions = context.user_data.get('descriptions')
    
    await status_msg.edit_text(
        f"🏷️ Начинаю классификацию {len(texts)} текстов...\n"
        f"📋 Категории: {len(categories)}\n"
        f"⏱️ Примерное время: {len(texts) * 1.5 / 60:.0f} мин"
    )
    
    # Функция обновления прогресса
    async def progress_callback(progress: float, current: int, total: int):
        if current % 10 == 0:  # Обновляем каждые 10 текстов
            await status_msg.edit_text(
                f"🏷️ Классифицирую: {current}/{total}\n"
                f"📊 Прогресс: {progress:.0f}%"
            )
    
    # Запуск классификации
    try:
        result_df = classifier.classify_batch(
            texts,
            categories,
            descriptions,
            progress_callback=progress_callback
        )
        
        stats = classifier.get_classification_stats(result_df)
        
        # Сохранение результатов
        output_filename = clean_filename(f"classified_{filename}")
        output_path = f"/tmp/{user.id}_{output_filename}"
        result_df.to_csv(output_path, index=False, encoding='utf-8')
        
        # Отправка результатов
        caption = format_classification_stats(stats, categories)
        
        keyboard = [
            [InlineKeyboardButton("🔄 Обработать новый файл", callback_data="restart")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_document(
            document=open(output_path, 'rb'),
            filename=output_filename,
            caption=caption,
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN
        )
        
        await status_msg.delete()
        os.remove(output_path)
        
    except Exception as e:
        logger.error(f"Ошибка классификации: {e}", exc_info=True)
        await status_msg.edit_text(
            f"❌ Ошибка классификации:\n`{str(e)}`",
            parse_mode=ParseMode.MARKDOWN
        )


def format_clustering_stats(stats: Dict) -> str:
    """Форматирует статистику кластеризации."""
    top_clusters = sorted(
        stats['clusters'].items(),
        key=lambda x: x[1]['size'],
        reverse=True
    )[:3]
    
    top_text = "\n".join([
        f"{i+1}️⃣ {name} — {info['size']} текстов"
        for i, (name, info) in enumerate(top_clusters)
    ])
    
    return f"""✅ **Кластеризация завершена!**

📊 Результаты:
• Обработано текстов: {stats['total_texts']}
• Найдено кластеров: {stats['num_clusters']}
• Средний размер: {stats['avg_cluster_size']:.0f} текстов
• Шум: {stats['noise_percentage']:.1f}%

Топ-3 кластера:
{top_text}

🎯 Метрики качества:
• Silhouette Score: {stats.get('silhouette_score', 0):.3f}
• Davies-Bouldin: {stats.get('davies_bouldin', 0):.3f}"""


def format_classification_stats(stats: Dict, categories: list) -> str:
    """Форматирует статистику классификации."""
    sorted_cats = sorted(
        stats['categories'].items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )[:5]
    
    dist_text = "\n".join([
        f"• {cat}: {info['count']} ({info['percentage']:.1f}%) "
        f"[уверенность: {info['avg_confidence']:.2f}]"
        for cat, info in sorted_cats
    ])
    
    return f"""✅ **Классификация завершена!**

📊 Результаты:
• Обработано текстов: {stats['total_texts']}
• Категорий: {len(categories)}
• Средняя уверенность: {stats['avg_confidence']:.2f}

📋 Распределение (топ-5):
{dist_text}"""


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отмена текущей операции."""
    context.user_data.clear()
    await update.message.reply_text(
        "❌ Операция отменена.\n"
        "Используй /start чтобы начать заново."
    )
    return ConversationHandler.END


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /help."""
    help_text = """❓ **Справка**

**Режимы работы:**

🔍 **Кластеризация** (автоматическая)
• Бот сам находит темы и группирует тексты
• Не нужно задавать категории
• Подходит для исследовательского анализа
• Создает иерархию мастер-категорий

🏷️ **Классификация** (по категориям)
• Ты задаешь категории
• AI распределяет тексты по ним
• Подходит когда категории уже известны
• Показывает уверенность модели

**Команды:**
/start - начать работу
/help - эта справка
/cancel - отменить операцию

**Лимиты:**
• Размер файла: до 20 МБ
• Кластеризация: до 50,000 текстов
• Классификация: до 10,000 текстов
• Rate limit: 5 файлов в час"""
    
    # Если вызвана из inline кнопки
    if update.callback_query:
        await update.callback_query.edit_message_text(
            help_text,
            parse_mode=ParseMode.MARKDOWN
        )
    else:
        await update.message.reply_text(
            help_text,
            parse_mode=ParseMode.MARKDOWN
        )


def main():
    """Запуск бота."""
    token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN")
    
    if not token:
        logger.error("Не найден TELEGRAM_BOT_TOKEN в .env")
        sys.exit(1)
    
    # Создание приложения
    application = Application.builder().token(token).build()
    
    # Conversation handler
    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler("start", start),
            CallbackQueryHandler(start, pattern="^restart$")
        ],
        states={
            CHOOSING_MODE: [
                CallbackQueryHandler(mode_callback, pattern="^mode_"),
                CallbackQueryHandler(mode_callback, pattern="^help$"),
                CallbackQueryHandler(mode_callback, pattern="^restart$")
            ],
            ENTERING_CATEGORIES: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_categories)
            ],
            ENTERING_DESCRIPTIONS: [
                CallbackQueryHandler(descriptions_callback),
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_descriptions),
                CommandHandler("skip", descriptions_callback)
            ],
            PROCESSING_FILE: [
                MessageHandler(filters.Document.ALL, process_file)
            ]
        },
        fallbacks=[
            CommandHandler("cancel", cancel),
            CommandHandler("start", start)
        ],
        allow_reentry=True
    )
    
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("help", help_command))
    
    # Запуск
    logger.info("Бот запущен (с поддержкой классификации)")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
