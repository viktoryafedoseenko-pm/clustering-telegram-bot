# bot.py
"""
Бот-классификатор текстов
Спецификация v1.1
"""

import os
import time
import asyncio
import logging
import html
import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler

import pandas as pd
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)

from states import BotState, get_expected_input
from messages import (
    MSG_1, MSG_1_3_1, MSG_2_0, MSG_2_3, MSG_3_1, MSG_3_2_1, MSG_3_2_3,
    MSG_3_3_1, MSG_3_3_2, MSG_3_5_1, MSG_3_6, MSG_3_6_2_1,
    MSG_4_1, MSG_4_3, MSG_4_3_3_1, MSG_4_3_4_1, MSG_4_3_4_2,
    MSG_4_3_5, MSG_4_3_6, MSG_4_3_7, MSG_4_4, MSG_5_1, MSG_5_1_SELECTED,
    MSG_E1, MSG_E2, MSG_E4, MSG_E5, MSG_E6, MSG_E8, MSG_E9, MSG_E10,
    MSG_RATE_LIMIT,
    format_message, get_buttons
)
from config import TEMP_DIR
from rate_limiter import rate_limiter
from utils import cleanup_file_safe, format_time_remaining
from progress_tracker import ProgressTracker
from demo_datasets import DEMO_DATASETS, get_demo_file_path, get_demo_description

load_dotenv()

# =============================================================================
# ЛОГИРОВАНИЕ
# =============================================================================

LOG_DIR = Path(os.getenv("BOT_LOG_DIR", TEMP_DIR / "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

file_handler = RotatingFileHandler(
    LOG_DIR / "bot.log", maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# =============================================================================
# ИНИЦИАЛИЗАЦИЯ
# =============================================================================

TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

# Классификация
classifier = None
category_generator = None
CLASSIFICATION_AVAILABLE = False

try:
    from classification import LLMClassifier, validate_categories, parse_categories_from_text
    if os.getenv("YANDEX_API_KEY") and os.getenv("YANDEX_FOLDER_ID"):
        classifier = LLMClassifier()
        CLASSIFICATION_AVAILABLE = True
        logger.info("✅ Classification module loaded")
except ImportError:
    logger.warning("⚠️ classification.py not found")
except Exception as e:
    logger.warning(f"⚠️ Classification init failed: {e}")

if CLASSIFICATION_AVAILABLE:
    try:
        from category_generator import CategoryGenerator
        category_generator = CategoryGenerator(
            api_key=os.getenv("YANDEX_API_KEY"),
            folder_id=os.getenv("YANDEX_FOLDER_ID")
        )
        logger.info("✅ Category generator loaded")
    except Exception as e:
        logger.warning(f"⚠️ Category generator init failed: {e}")

# Аналитика
analytics = None
try:
    from analytics_simple import UserAnalytics
    admin_id = os.getenv('ADMIN_TELEGRAM_ID')
    if admin_id:
        analytics = UserAnalytics(admin_chat_id=int(admin_id))
        logger.info("✅ Analytics initialized")
except Exception as e:
    logger.warning(f"⚠️ Analytics init failed: {e}")


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def get_state(context: ContextTypes.DEFAULT_TYPE) -> BotState:
    return context.user_data.get('state', BotState.START)


def set_state(context: ContextTypes.DEFAULT_TYPE, state: BotState):
    old = get_state(context)
    context.user_data['state'] = state
    logger.info(f"STATE | {old.name} → {state.name}")


def build_keyboard(buttons: list) -> InlineKeyboardMarkup:
    if not buttons:
        return None
    keyboard = [[InlineKeyboardButton(b["text"], callback_data=b["callback"])] for b in buttons]
    return InlineKeyboardMarkup(keyboard)


async def send_msg(update: Update, msg, edit: bool = False, **kwargs):
    """Отправка сообщения"""
    text = format_message(msg, **kwargs)
    keyboard = build_keyboard(get_buttons(msg))
    
    if update.callback_query:
        if edit:
            await update.callback_query.edit_message_text(text, parse_mode='HTML', reply_markup=keyboard)
        else:
            await update.callback_query.message.reply_text(text, parse_mode='HTML', reply_markup=keyboard)
    else:
        await update.message.reply_text(text, parse_mode='HTML', reply_markup=keyboard)


def get_target(update: Update):
    """Получить message для ответа"""
    return update.callback_query.message if update.callback_query else update.message


# =============================================================================
# 1. ПРИВЕТСТВИЕ
# =============================================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """1. Стартовое сообщение"""
    user_id = update.effective_user.id
    logger.info(f"START | User: {user_id}")
    
    context.user_data.clear()
    context.user_data['files_processed'] = 0
    set_state(context, BotState.START)
    
    await send_msg(update, MSG_1)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """G3: /help без сброса состояния"""
    await send_msg(update, MSG_1_3_1)


async def cb_back_to_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Возврат к старту"""
    query = update.callback_query
    await query.answer()
    set_state(context, BotState.START)
    await send_msg(update, MSG_1, edit=True)


async def cb_help_file_format(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """1.3.1. Инструкция по файлу"""
    query = update.callback_query
    await query.answer()
    await send_msg(update, MSG_1_3_1, edit=True)


async def cb_ready_to_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Переход к загрузке файла"""
    query = update.callback_query
    await query.answer()
    set_state(context, BotState.WAITING_FOR_FILE)
    await send_msg(update, MSG_2_0, edit=True)


# =============================================================================
# 2. ЗАГРУЗКА ФАЙЛА
# =============================================================================

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """G1: Обработка CSV на любом шаге"""
    user_id = update.effective_user.id
    document = update.message.document
    
    logger.info(f"FILE | User: {user_id} | File: {document.file_name}")
    
    current_state = get_state(context)
    
    # Нельзя прерывать классификацию
    if current_state == BotState.CLASSIFYING:
        await update.message.reply_text(
            "⏳ Подождите, идёт обработка предыдущего файла.",
            parse_mode='HTML'
        )
        return
    
    # Rate limit
    allowed, remaining, wait_time = rate_limiter.is_allowed(user_id)
    if not allowed:
        await send_msg(update, MSG_RATE_LIMIT, wait_time=format_time_remaining(wait_time))
        return
    
    # Проверка формата
    if not document.file_name.endswith('.csv'):
        await send_msg(update, MSG_E1)
        return
    
    # Проверка размера
    MAX_SIZE_MB = 20
    file_size_mb = document.file_size / (1024 * 1024)
    if file_size_mb > MAX_SIZE_MB:
        await send_msg(update, MSG_E2, 
                      file_size=f"{file_size_mb:.1f} МБ",
                      max_size=f"{MAX_SIZE_MB} МБ",
                      max_rows="10 000")
        return
    
    # Загрузка
    progress_msg = await update.message.reply_text("⏳ Загружаю файл...", parse_mode='HTML')
    
    try:
        file = await document.get_file()
        file_path = f"/tmp/{user_id}_{int(time.time())}.csv"
        await file.download_to_drive(file_path)
        
        df = pd.read_csv(file_path, encoding='utf-8', dtype=str)
        n_rows = len(df)
        
        # Проверка на пустоту
        if n_rows == 0:
            await progress_msg.delete()
            await send_msg(update, MSG_E10)
            cleanup_file_safe(file_path)
            return
        
        # Проверка лимита строк
        MAX_ROWS = 10000
        if n_rows > MAX_ROWS:
            await progress_msg.delete()
            await send_msg(update, MSG_E2,
                          file_size=f"{n_rows} строк",
                          max_size=f"{MAX_SIZE_MB} МБ",
                          max_rows=str(MAX_ROWS))
            cleanup_file_safe(file_path)
            return
        
        # Сохраняем данные
        context.user_data['file_path'] = file_path
        context.user_data['file_name'] = document.file_name
        context.user_data['df'] = df
        context.user_data['records_count'] = n_rows
        
        # Примеры
        texts = df.iloc[:, 0].fillna("").astype(str).tolist()
        examples = "\n".join([
            f"• {html.escape(t[:60])}{'...' if len(t) > 60 else ''}"
            for t in texts[:3] if t.strip()
        ])
        
        await progress_msg.delete()
        
        # 2.3. Файл получен
        set_state(context, BotState.FILE_RECEIVED)
        await send_msg(update, MSG_2_3, records_count=n_rows, examples=examples or "—")
        
    except Exception as e:
        logger.error(f"FILE ERROR | {e}", exc_info=True)
        await progress_msg.delete()
        await send_msg(update, MSG_E8)


async def cb_back_to_file_received(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Возврат к экрану 2.3"""
    query = update.callback_query
    await query.answer()
    
    n_rows = context.user_data.get('records_count', 0)
    df = context.user_data.get('df')
    
    if df is None:
        set_state(context, BotState.START)
        await send_msg(update, MSG_1, edit=True)
        return
    
    texts = df.iloc[:, 0].fillna("").astype(str).tolist()
    examples = "\n".join([
        f"• {html.escape(t[:60])}{'...' if len(t) > 60 else ''}"
        for t in texts[:3] if t.strip()
    ])
    
    set_state(context, BotState.FILE_RECEIVED)
    await send_msg(update, MSG_2_3, edit=True, records_count=n_rows, examples=examples or "—")


# =============================================================================
# 3. НАСТРОЙКА ПАРАМЕТРОВ
# =============================================================================

async def cb_settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """3.1. Меню настроек"""
    query = update.callback_query
    await query.answer()
    set_state(context, BotState.SETTINGS_MENU)
    await send_msg(update, MSG_3_1, edit=True)


async def cb_categories_manual(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """3.2.1. Ручной ввод категорий"""
    query = update.callback_query
    await query.answer()
    set_state(context, BotState.WAITING_FOR_CATEGORIES)
    await send_msg(update, MSG_3_2_1, edit=True)


async def cb_prompt_custom(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """3.3.1. Настройка промпта"""
    query = update.callback_query
    await query.answer()
    
    default_prompt = "Проанализируй тексты и предложи 5-10 категорий для классификации..."
    if category_generator:
        default_prompt = getattr(category_generator, 'DEFAULT_PROMPT', default_prompt)[:300] + "..."
    
    set_state(context, BotState.WAITING_FOR_PROMPT)
    await send_msg(update, MSG_3_3_1, edit=True, default_prompt=default_prompt)


async def cb_prompt_default(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Использовать стандартный промпт → генерация"""
    query = update.callback_query
    await query.answer()
    context.user_data['custom_prompt'] = None
    await start_category_generation(update, context)


async def cb_run_default(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """2.4.1 / 3.1.3. Стандартные настройки → генерация категорий"""
    query = update.callback_query
    await query.answer()
    context.user_data['custom_prompt'] = None
    await start_category_generation(update, context)


# =============================================================================
# 3.5. ГЕНЕРАЦИЯ КАТЕГОРИЙ
# =============================================================================

async def start_category_generation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """3.5. Запуск генерации категорий"""
    user_id = update.effective_user.id
    
    df = context.user_data.get('df')
    if df is None:
        await send_msg(update, MSG_E8, edit=True)
        return
    
    if not category_generator:
        logger.error("Category generator not available")
        await send_msg(update, MSG_E6, edit=True)
        return
    
    set_state(context, BotState.GENERATING_CATEGORIES)
    
    texts = df.iloc[:, 0].fillna("").astype(str).tolist()
    sample = texts[:500] if len(texts) > 500 else texts
    
    # Показываем прогресс
    target = get_target(update)
    progress_msg = await target.reply_text(
        format_message(MSG_3_5_1, sample_size=len(sample)),
        parse_mode='HTML'
    )
    
    try:
        custom_prompt = context.user_data.get('custom_prompt')
        success, categories, error = category_generator.generate_categories(sample, custom_prompt)
        
        await progress_msg.delete()
        
        if not success:
            logger.error(f"Generation failed: {error}")
            set_state(context, BotState.SETTINGS_MENU)
            await send_msg(update, MSG_E6)
            return
        
        # Сохраняем
        context.user_data['generated_categories'] = categories
        category_names = [c.name for c in categories]
        context.user_data['categories'] = category_names
        
        # Форматируем для отображения
        categories_list = category_generator.format_categories_for_display(categories)
        
        set_state(context, BotState.SHOWING_GENERATED)
        await send_msg(update, MSG_3_6, categories_list=categories_list)
        
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        await progress_msg.delete()
        set_state(context, BotState.SETTINGS_MENU)
        await send_msg(update, MSG_E6)


async def cb_categories_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """3.6.1. Подтвердить категории → классификация"""
    query = update.callback_query
    await query.answer()
    await start_classification(update, context)


async def cb_categories_edit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """3.6.2. Редактировать категории"""
    query = update.callback_query
    await query.answer()
    
    categories = context.user_data.get('categories', [])
    categories_text = "\n".join(categories)
    
    set_state(context, BotState.EDITING_CATEGORIES)
    await send_msg(update, MSG_3_6_2_1, edit=True, categories_text=categories_text)


async def cb_categories_show_again(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать категории снова (отмена редактирования)"""
    query = update.callback_query
    await query.answer()
    
    categories = context.user_data.get('categories', [])
    categories_list = "\n".join([f"• {c}" for c in categories])
    
    set_state(context, BotState.SHOWING_GENERATED)
    await send_msg(update, MSG_3_6, edit=True, categories_list=categories_list)


# =============================================================================
# ОБРАБОТКА ТЕКСТОВОГО ВВОДА
# =============================================================================

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка текста в зависимости от состояния"""
    state = get_state(context)
    text = update.message.text.strip()
    user_id = update.effective_user.id
    
    logger.info(f"TEXT | User: {user_id} | State: {state.name} | Text: {text[:50]}...")
    
    if state == BotState.WAITING_FOR_CATEGORIES:
        await process_categories_input(update, context, text)
    
    elif state == BotState.EDITING_CATEGORIES:
        await process_categories_input(update, context, text)
    
    elif state == BotState.WAITING_FOR_PROMPT:
        await process_prompt_input(update, context, text)
    
    elif state == BotState.WAITING_FOR_FEEDBACK_TEXT:
        await process_feedback_text(update, context, text)
    
    else:
        expected = get_expected_input(state)
        await send_msg(update, MSG_E9, 
                      expected_input=expected,
                      available_actions="Используйте кнопки или отправьте CSV-файл.")


async def process_categories_input(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    """3.2.2. Валидация введённых категорий"""
    categories = parse_categories_from_text(text)
    is_valid, error_msg = validate_categories(categories)
    
    if not is_valid:
        if len(categories) < 2:
            await send_msg(update, MSG_E4)
        else:
            await send_msg(update, MSG_E5)
        return
    
    context.user_data['categories'] = categories
    categories_list = "\n".join([f"{i+1}. {c}" for i, c in enumerate(categories)])
    
    set_state(context, BotState.CATEGORIES_CONFIRMED)
    await send_msg(update, MSG_3_2_3, 
                  categories_count=len(categories),
                  categories_list=categories_list)


async def process_prompt_input(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    """3.3.2. Сохранение кастомного промпта"""
    context.user_data['custom_prompt'] = text
    
    await send_msg(update, MSG_3_3_2)
    await asyncio.sleep(1)
    await start_category_generation(update, context)


async def process_feedback_text(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    """4.3.4.2. Обработка текста фидбека"""
    user_id = update.effective_user.id
    logger.info(f"FEEDBACK | User: {user_id} | Text: {text[:100]}")
    
    # Можно сохранить/отправить админу
    
    await send_msg(update, MSG_4_3_4_2)
    await asyncio.sleep(1)
    await send_msg(update, MSG_4_3_6)
    set_state(context, BotState.SESSION_END)


# =============================================================================
# 4. КЛАССИФИКАЦИЯ
# =============================================================================

async def cb_run_classification(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Запуск классификации (из 3.2.3)"""
    query = update.callback_query
    await query.answer()
    await start_classification(update, context)


async def start_classification(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """4.1. Запуск классификации"""
    user_id = update.effective_user.id
    
    categories = context.user_data.get('categories', [])
    df = context.user_data.get('df')
    file_path = context.user_data.get('file_path')
    
    if not categories or df is None:
        await send_msg(update, MSG_E8)
        return
    
    if not classifier:
        logger.error("Classifier not available")
        await send_msg(update, MSG_E8)
        return
    
    set_state(context, BotState.CLASSIFYING)
    
    total_texts = len(df)
    
    # Оценка времени
    if total_texts < 100:
        time_estimate = "~1 минута"
    elif total_texts < 500:
        time_estimate = "1-3 минуты"
    elif total_texts < 2000:
        time_estimate = "3-10 минут"
    else:
        time_estimate = "10-30 минут"
    
    target = get_target(update)
    progress_msg = await target.reply_text(
        format_message(MSG_4_1,
                      total_texts=total_texts,
                      categories_count=len(categories),
                      time_estimate=time_estimate),
        parse_mode='HTML'
    )
    
    try:
        texts = df.iloc[:, 0].fillna("").astype(str).tolist()
        
        # Классификация
        tracker = ProgressTracker(progress_msg, min_interval=3.0)
        loop = asyncio.get_running_loop()
        
        def progress_callback(progress: float, current: int, total: int):
            if current % 10 == 0 or current == total:
                asyncio.run_coroutine_threadsafe(
                    tracker.update(
                        stage=f"🏷️ Классифицировано: {current}/{total}",
                        percent=int(progress * 100)
                    ),
                    loop
                )
        
        result_df = await loop.run_in_executor(
            None,
            lambda: classifier.classify_batch(texts, categories, progress_callback=progress_callback)
        )
        
        stats = classifier.get_classification_stats(result_df)
        
        # Сохраняем результат
        result_path = f"/tmp/{user_id}_classified_{int(time.time())}.csv"
        result_df.to_csv(result_path, index=False, encoding='utf-8')
        context.user_data['result_path'] = result_path
        
        await progress_msg.delete()
        
        # Формируем распределение
        sorted_cats = sorted(stats['categories'].items(), key=lambda x: x[1]['count'], reverse=True)[:5]
        distribution = "\n".join([
            f"• {cat}: {info['count']} ({info['percentage']:.1f}%)"
            for cat, info in sorted_cats
        ])
        
        # Увеличиваем счётчик
        context.user_data['files_processed'] = context.user_data.get('files_processed', 0) + 1
        
        set_state(context, BotState.SHOWING_RESULT)
        
        # Отправляем файл с результатами
        with open(result_path, 'rb') as f:
            await target.reply_document(
                document=f,
                filename=f"classified_{context.user_data.get('file_name', 'result.csv')}",
                caption=format_message(MSG_4_3,
                                      total_texts=total_texts,
                                      categories_count=len(categories),
                                      avg_confidence=f"{stats['avg_confidence']:.0%}",
                                      distribution=distribution),
                parse_mode='HTML',
                reply_markup=build_keyboard(get_buttons(MSG_4_3))
            )
        
        # Очистка
        cleanup_file_safe(file_path)
        
    except Exception as e:
        logger.error(f"Classification error: {e}", exc_info=True)
        await progress_msg.delete()
        set_state(context, BotState.FILE_RECEIVED)
        await send_msg(update, MSG_E8)


# =============================================================================
# ОБРАТНАЯ СВЯЗЬ
# =============================================================================

async def cb_feedback_positive(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """4.3.1-2. Положительная оценка"""
    query = update.callback_query
    await query.answer()
    
    await send_msg(update, MSG_4_3_5, edit=True)
    await asyncio.sleep(1)
    
    set_state(context, BotState.SESSION_END)
    await send_msg(update, MSG_4_3_7)


async def cb_feedback_bad(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """4.3.3. Плохая оценка"""
    query = update.callback_query
    await query.answer()
    
    set_state(context, BotState.COLLECTING_FEEDBACK)
    await send_msg(update, MSG_4_3_3_1, edit=True)


async def cb_feedback_terrible(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """4.3.4. Очень плохая оценка"""
    query = update.callback_query
    await query.answer()
    
    set_state(context, BotState.WAITING_FOR_FEEDBACK_TEXT)
    await send_msg(update, MSG_4_3_4_1, edit=True)


async def cb_problem_selected(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Выбрана проблема → предложить перенастроить"""
    query = update.callback_query
    await query.answer()
    
    await send_msg(update, MSG_4_3_6, edit=True)


async def cb_ask_continue(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Спросить о продолжении"""
    query = update.callback_query
    await query.answer()
    
    set_state(context, BotState.SESSION_END)
    await send_msg(update, MSG_4_3_7, edit=True)


async def cb_upload_new(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """4.3.7.1. Загрузить новый файл"""
    query = update.callback_query
    await query.answer()
    
    # Очищаем старые данные
    context.user_data.pop('df', None)
    context.user_data.pop('file_path', None)
    context.user_data.pop('categories', None)
    context.user_data.pop('result_path', None)
    
    set_state(context, BotState.WAITING_FOR_FILE)
    await send_msg(update, MSG_2_0, edit=True)


async def cb_finish_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """4.4. Завершение сессии"""
    query = update.callback_query
    await query.answer()
    
    set_state(context, BotState.SESSION_END)
    await send_msg(update, MSG_4_4, edit=True)


# =============================================================================
# 5. ДЕМО-РЕЖИМ
# =============================================================================

async def cb_demo_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """5.1. Меню демо"""
    query = update.callback_query
    await query.answer()
    
    set_state(context, BotState.DEMO_MENU)
    await send_msg(update, MSG_5_1, edit=True)


async def cb_demo_select(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Выбор демо-датасета"""
    query = update.callback_query
    await query.answer()
    
    demo_key = query.data.replace("demo_", "")
    
    if demo_key not in DEMO_DATASETS:
        await send_msg(update, MSG_E8, edit=True)
        return
    
    dataset = DEMO_DATASETS[demo_key]
    context.user_data['demo_key'] = demo_key
    
    await send_msg(update, MSG_5_1_SELECTED, edit=True,
                  dataset_name=dataset['name'],
                  dataset_description=get_demo_description(demo_key))


async def cb_demo_run(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """5.1.1. Запуск демо → генерация категорий"""
    query = update.callback_query
    await query.answer()
    
    demo_key = context.user_data.get('demo_key')
    if not demo_key:
        await send_msg(update, MSG_E8, edit=True)
        return
    
    file_path = get_demo_file_path(demo_key)
    if not file_path:
        await send_msg(update, MSG_E8, edit=True)
        return
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8', dtype=str)
        
        context.user_data['df'] = df
        context.user_data['file_path'] = file_path
        context.user_data['file_name'] = f"demo_{demo_key}.csv"
        context.user_data['records_count'] = len(df)
        context.user_data['is_demo'] = True
        context.user_data['custom_prompt'] = None
        
        await start_category_generation(update, context)
        
    except Exception as e:
        logger.error(f"Demo load error: {e}", exc_info=True)
        await send_msg(update, MSG_E8, edit=True)


async def cb_demo_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """5.1.2. Настроить демо"""
    query = update.callback_query
    await query.answer()
    
    demo_key = context.user_data.get('demo_key')
    if not demo_key:
        await send_msg(update, MSG_E8, edit=True)
        return
    
    file_path = get_demo_file_path(demo_key)
    if not file_path:
        await send_msg(update, MSG_E8, edit=True)
        return
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8', dtype=str)
        
        context.user_data['df'] = df
        context.user_data['file_path'] = file_path
        context.user_data['file_name'] = f"demo_{demo_key}.csv"
        context.user_data['records_count'] = len(df)
        context.user_data['is_demo'] = True
        
        set_state(context, BotState.SETTINGS_MENU)
        await send_msg(update, MSG_3_1, edit=True)
        
    except Exception as e:
        logger.error(f"Demo load error: {e}", exc_info=True)
        await send_msg(update, MSG_E8, edit=True)


# =============================================================================
# ОБРАБОТКА ОШИБОК
# =============================================================================

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Глобальный обработчик ошибок"""
    logger.error(f"ERROR | {context.error}", exc_info=context.error)


# =============================================================================
# РЕГИСТРАЦИЯ ОБРАБОТЧИКОВ
# =============================================================================

def register_handlers(app: Application):
    """Регистрация всех обработчиков"""
    
    # Команды
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    
    # 1. Приветствие
    app.add_handler(CallbackQueryHandler(cb_back_to_start, pattern="^back_to_start$"))
    app.add_handler(CallbackQueryHandler(cb_help_file_format, pattern="^help_file_format$"))
    app.add_handler(CallbackQueryHandler(cb_ready_to_upload, pattern="^ready_to_upload$"))
    
    # 2. Загрузка
    app.add_handler(CallbackQueryHandler(cb_back_to_file_received, pattern="^back_to_file_received$"))
    
    # 3. Настройки
    app.add_handler(CallbackQueryHandler(cb_settings_menu, pattern="^settings_menu$"))
    app.add_handler(CallbackQueryHandler(cb_categories_manual, pattern="^categories_manual$"))
    app.add_handler(CallbackQueryHandler(cb_prompt_custom, pattern="^prompt_custom$"))
    app.add_handler(CallbackQueryHandler(cb_prompt_default, pattern="^prompt_default$"))
    app.add_handler(CallbackQueryHandler(cb_run_default, pattern="^run_default$"))
    
    # 3.5-3.6. Генерация
    app.add_handler(CallbackQueryHandler(cb_categories_confirm, pattern="^categories_confirm$"))
    app.add_handler(CallbackQueryHandler(cb_categories_edit, pattern="^categories_edit$"))
    app.add_handler(CallbackQueryHandler(cb_categories_show_again, pattern="^categories_show_again$"))
    
    # 4. Классификация
    app.add_handler(CallbackQueryHandler(cb_run_classification, pattern="^run_classification$"))
    
    # Оценка
    app.add_handler(CallbackQueryHandler(cb_feedback_positive, pattern="^feedback_great$"))
    app.add_handler(CallbackQueryHandler(cb_feedback_positive, pattern="^feedback_ok$"))
    app.add_handler(CallbackQueryHandler(cb_feedback_bad, pattern="^feedback_bad$"))
    app.add_handler(CallbackQueryHandler(cb_feedback_terrible, pattern="^feedback_terrible$"))
    app.add_handler(CallbackQueryHandler(cb_problem_selected, pattern="^problem_"))
    app.add_handler(CallbackQueryHandler(cb_ask_continue, pattern="^ask_continue$"))
    app.add_handler(CallbackQueryHandler(cb_upload_new, pattern="^upload_new$"))
    app.add_handler(CallbackQueryHandler(cb_finish_session, pattern="^finish_session$"))
    
    # 5. Демо
    app.add_handler(CallbackQueryHandler(cb_demo_start, pattern="^demo_start$"))
    app.add_handler(CallbackQueryHandler(cb_demo_select, pattern="^demo_app_reviews$"))
    app.add_handler(CallbackQueryHandler(cb_demo_select, pattern="^demo_ecommerce$"))
    app.add_handler(CallbackQueryHandler(cb_demo_select, pattern="^demo_students$"))
    app.add_handler(CallbackQueryHandler(cb_demo_run, pattern="^demo_run$"))
    app.add_handler(CallbackQueryHandler(cb_demo_settings, pattern="^demo_settings$"))
    
    # Текст и файлы
    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    # Ошибки
    app.add_error_handler(error_handler)


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not set")
        return
    
    logger.info("=" * 50)
    logger.info("🚀 Bot starting...")
    logger.info(f"Classification: {'✅' if CLASSIFICATION_AVAILABLE else '❌'}")
    logger.info(f"Category generator: {'✅' if category_generator else '❌'}")
    logger.info("=" * 50)
    
    app = Application.builder().token(TOKEN).build()
    register_handlers(app)
    
    logger.info("✅ Handlers registered")
    logger.info("🤖 Bot is running!")
    
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
