# bot.py
import time
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
from analytics_simple import UserAnalytics
from config import ADMIN_TELEGRAM_ID
import datetime
from progress_tracker import ProgressTracker
from evaluation import (
    calculate_metrics, 
    get_error_examples, 
    format_evaluation_report,
    validate_ground_truth
)
from category_generator import CategoryGenerator, CategorySuggestion
from prompt_manager import PromptManager

# Создать глобальный экземпляр
prompt_manager = PromptManager()
category_generator = None

PROCESSING_SEMAPHORE = asyncio.Semaphore(2)

# Состояния для ConversationHandler
class BotStates:
    """Состояния бота"""
    CHOOSING_MODE = "choosing_mode"
    # Существующие
    WAITING_FOR_CATEGORIES = "waiting_for_categories"
    WAITING_FOR_FILE = "waiting_for_file"
    # Новые для автогенерации
    CHOOSING_CATEGORY_METHOD = "choosing_category_method"
    ASKING_GENERATION_PROMPT = "asking_generation_prompt"
    WAITING_FOR_GENERATION_PROMPT = "waiting_for_generation_prompt"
    WAITING_FOR_SAMPLE_FILE = "waiting_for_sample_file"
    GENERATING_CATEGORIES = "generating_categories"
    SHOWING_GENERATED_CATEGORIES = "showing_generated_categories"
    EDITING_CATEGORIES = "editing_categories"
    ASKING_CLASSIFICATION_PROMPT = "asking_classification_prompt"
    WAITING_FOR_CLASSIFICATION_PROMPT = "waiting_for_classification_prompt"

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
analytics = None

# Импорты для классификации (опциональные)
classifier = None
CLASSIFICATION_AVAILABLE = False
try:
    from classification import LLMClassifier, validate_categories, parse_categories_from_text
    if os.getenv("YANDEX_API_KEY") and os.getenv("YANDEX_FOLDER_ID"):
        classifier = LLMClassifier()
        CLASSIFICATION_AVAILABLE = True
        logger.info("✅ Classification module loaded")
except ImportError:
    logger.warning("⚠️ classification.py not found - classification disabled")
except Exception as e:
    logger.warning(f"⚠️ Classification init failed: {e}")

if classifier:
    try:
        category_generator = CategoryGenerator(
            api_key=os.getenv("YANDEX_API_KEY"),
            folder_id=os.getenv("YANDEX_FOLDER_ID")
        )
        logger.info("✅ Category generator loaded")
    except Exception as e:
        logger.warning(f"⚠️ Category generator init failed: {e}")


# Загрузка токена
load_dotenv()
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Стартовое сообщение с выбором режима"""
    user_id = update.effective_user.id
    username = update.effective_user.username or "unknown"
    first_name = update.effective_user.first_name
    logger.info(f"📥 START | User: {user_id} (@{username})")
    
    # Парсинг источника из deep link
    args = context.args
    source = args[0] if args else 'organic'
    
    logger.info(f"🔗 SOURCE | User: {user_id} | Source: {source}")
    
    # Очищаем старые данные
    context.user_data.clear()
    
    # Сохраняем источник и инициализируем счётчики
    context.user_data['source'] = source
    context.user_data['files_processed'] = 0
    context.user_data['modes_used'] = []  # Список использованных режимов
    
    # Отправка уведомления админу
    if analytics:
        try:
            await analytics.track_start(
                bot=context.bot,
                user_id=user_id,
                username=username,
                source=source,
                first_name=first_name
            )
        except Exception as e:
            logger.error(f"Analytics track_start failed: {e}")

    welcome_msg = """
👋 <b>Привет! Я помогу разобрать отзывы и обращения.</b>

<b>Что нужно сделать?</b>

📋 <b>Разложить по категориям</b>
Распределю тексты по темам для отчёта или анализа.
→ Категории известны или AI предложит
→ Точная классификация с помощью AI
→ До 5,000 текстов

🔍 <b>Изучить данные</b>
Автоматически найду все темы в больших объёмах.
→ Быстрый анализ (5-20 минут)
→ Бесплатно, до 50,000 текстов
→ Для первичного исследования

❓ <b>Не уверен, что выбрать?</b>
Пройди быстрый квиз (30 секунд)
    """
    
    # Создаем клавиатуру
    keyboard = [
        [InlineKeyboardButton("Разложить по категориям", callback_data="mode_classification")]
    ]
    
    keyboard.append([InlineKeyboardButton("Изучить данные", callback_data="mode_clustering")])
    keyboard.append([InlineKeyboardButton("Помочь выбрать (квиз)", callback_data="show_quiz")])
    keyboard.append([InlineKeyboardButton("Как это работает?", callback_data="show_help")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        welcome_msg,
        parse_mode='HTML',
        reply_markup=reply_markup
    )

async def show_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать квиз для выбора режима"""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    logger.info(f"❓ QUIZ START | User: {user_id}")
    
    # Инициализируем квиз
    context.user_data['quiz_answers'] = {}
    
    text = """
❓ <b>Квиз: Какой режим тебе подходит?</b>

Отвечу на 3 быстрых вопроса и порекомендую оптимальный вариант.

<b>Вопрос 1 из 3:</b>

Сколько у тебя текстов для анализа?
    """
    
    keyboard = [
        [InlineKeyboardButton("До 500 текстов", callback_data="quiz_q1_small")],
        [InlineKeyboardButton("500 - 5,000 текстов", callback_data="quiz_q1_medium")],
        [InlineKeyboardButton("Больше 5,000 текстов", callback_data="quiz_q1_large")],
        [InlineKeyboardButton("🔙 Назад в меню", callback_data="back_to_start")]
    ]
    
    await query.edit_message_text(
        text,
        parse_mode='HTML',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


async def handle_quiz_q1(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик ответа на вопрос 1"""
    query = update.callback_query
    await query.answer()
    
    # Сохраняем ответ
    answer = query.data.split('_')[2]  # small, medium, large
    context.user_data['quiz_answers']['q1_size'] = answer
    
    text = """
<b>Вопрос 2 из 3:</b>

Знаешь ли ты, какие категории нужны?
(Например: "Доставка", "Оплата", "Качество товара")
    """
    
    keyboard = [
        [InlineKeyboardButton("Да, знаю категории", callback_data="quiz_q2_yes")],
        [InlineKeyboardButton("Нет, не знаю", callback_data="quiz_q2_no")],
        [InlineKeyboardButton("Есть идеи, но не уверен", callback_data="quiz_q2_maybe")],
        [InlineKeyboardButton("🔙 Назад", callback_data="quiz_back_to_q1")]
    ]
    
    await query.edit_message_text(
        text,
        parse_mode='HTML',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


async def handle_quiz_q2(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик ответа на вопрос 2"""
    query = update.callback_query
    await query.answer()
    
    # Сохраняем ответ
    answer = query.data.split('_')[2]  # yes, no, maybe
    context.user_data['quiz_answers']['q2_categories'] = answer
    
    text = """
<b>Вопрос 3 из 3:</b>

Это разовая задача или регулярная работа?
    """
    
    keyboard = [
        [InlineKeyboardButton("Разовая (первый раз)", callback_data="quiz_q3_once")],
        [InlineKeyboardButton("Регулярная (каждую неделю/месяц)", callback_data="quiz_q3_regular")],
        [InlineKeyboardButton("Не знаю", callback_data="quiz_q3_dunno")],
        [InlineKeyboardButton("🔙 Назад", callback_data="quiz_back_to_q2")]
    ]
    
    await query.edit_message_text(
        text,
        parse_mode='HTML',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


async def handle_quiz_back(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик кнопки 'Назад' в квизе"""
    query = update.callback_query
    await query.answer()
    
    action = query.data
    
    if action == "quiz_back_to_q1":
        # Возвращаемся к вопросу 1 (повторяем логику из show_quiz)
        text = """
❓ <b>Квиз: Какой режим тебе подходит?</b>

<b>Вопрос 1 из 3:</b>

Сколько у тебя текстов для анализа?
        """
        
        keyboard = [
            [InlineKeyboardButton("До 500 текстов", callback_data="quiz_q1_small")],
            [InlineKeyboardButton("500 - 5,000 текстов", callback_data="quiz_q1_medium")],
            [InlineKeyboardButton("Больше 5,000 текстов", callback_data="quiz_q1_large")],
            [InlineKeyboardButton("🔙 Назад в меню", callback_data="back_to_start")]
        ]
        
        await query.edit_message_text(
            text,
            parse_mode='HTML',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    elif action == "quiz_back_to_q2":
        # Возвращаемся к вопросу 2
        text = """
<b>Вопрос 2 из 3:</b>

Знаешь ли ты, какие категории нужны?
(Например: "Доставка", "Оплата", "Качество товара")
        """
        
        keyboard = [
            [InlineKeyboardButton("Да, знаю категории", callback_data="quiz_q2_yes")],
            [InlineKeyboardButton("Нет, не знаю", callback_data="quiz_q2_no")],
            [InlineKeyboardButton("Есть идеи, но не уверен", callback_data="quiz_q2_maybe")],
            [InlineKeyboardButton("🔙 Назад", callback_data="quiz_back_to_q1")]
        ]
        
        await query.edit_message_text(
            text,
            parse_mode='HTML',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )


async def handle_quiz_result(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать результат квиза"""
    query = update.callback_query
    await query.answer()
    
    logger.info(f"❓ QUIZ Q3 ANSWERED | User: {update.effective_user.id} | Data: {query.data}")

    # Сохраняем последний ответ
    answer = query.data.split('_')[2]  # once, regular, dunno
    context.user_data['quiz_answers']['q3_frequency'] = answer
    
    # Логика рекомендации
    answers = context.user_data['quiz_answers']
    
    size = answers.get('q1_size')
    categories = answers.get('q2_categories')
    frequency = answers.get('q3_frequency')
    
    # Алгоритм рекомендации
    if size == 'large':  # > 5000
        if categories == 'no' or frequency == 'once':
            recommendation = 'clustering'
            reason = (
                "У тебя <b>много данных</b> и это <b>первый раз</b> — "
                "лучше начать с быстрого обзора всех тем."
            )
        else:
            recommendation = 'classification'
            reason = (
                "Даже с большим объёмом можно использовать классификацию, "
                "если категории известны. Но это займёт больше времени (1-2 часа)."
            )
    
    elif size == 'small':  # < 500
        if categories == 'yes':
            recommendation = 'classification'
            reason = "У тебя <b>готовые категории</b> — классификация идеально подойдёт."
        else:
            recommendation = 'classification_auto'
            reason = (
                "Для небольшого объёма (до 500 текстов) лучше использовать <b>автогенерацию категорий</b> — "
                "получишь понятные названия и точную раскладку."
            )
    
    else:  # medium (500-5000)
        if categories == 'yes':
            recommendation = 'classification'
            reason = "У тебя <b>готовые категории</b> и оптимальный объём — классификация подходит идеально."
        elif categories == 'no':
            recommendation = 'classification_auto'
            reason = (
                "Не знаешь категории? AI сгенерирует их автоматически, "
                "и ты сможешь отредактировать под свои задачи."
            )
        else:  # maybe
            recommendation = 'classification_auto'
            reason = (
                "Есть идеи о категориях? Отлично! AI предложит свои варианты, "
                "а ты дополнишь или скорректируешь."
            )
    
    # Формируем сообщение
    if recommendation == 'clustering':
        result_text = f"""
✅ <b>Рекомендация: Изучение данных</b>

{reason}

<b>Что получишь:</b>
• Все темы автоматически
• Быстро (5-20 минут)
• PDF-отчёт с графиками
• Бесплатно

<b>Дальше можешь:</b>
→ Использовать найденные темы как категории
→ Запустить классификацию на новых данных

<b>Начать изучение данных?</b>
        """
        
        keyboard = [
            [InlineKeyboardButton("Да, начать изучение", callback_data="mode_clustering")],
            [InlineKeyboardButton("Нет, лучше классификацию", callback_data="mode_classification")],
            [InlineKeyboardButton("🔙 В главное меню", callback_data="back_to_start")]
        ]
    
    elif recommendation == 'classification_auto':
        result_text = f"""
✅ <b>Рекомендация: Классификация с автогенерацией</b>

{reason}

<b>Как это работает:</b>
1. Загружаешь файл
2. AI анализирует тексты и предлагает категории
3. Ты редактируешь (опционально)
4. AI раскладывает все тексты

<b>Что получишь:</b>
• Понятные названия категорий
• Точную классификацию (85-95%)
• Возможность доработать категории

<b>Начать классификацию?</b>
        """
        
        keyboard = [
            [InlineKeyboardButton("Да, начать классификацию", callback_data="mode_classification")],
            [InlineKeyboardButton("Нет, лучше изучение", callback_data="mode_clustering")],
            [InlineKeyboardButton("🔙 В главное меню", callback_data="back_to_start")]
        ]
    
    else:  # classification
        result_text = f"""
✅ <b>Рекомендация: Классификация</b>

{reason}

<b>Как это работает:</b>
1. Вводишь свои категории
2. AI раскладывает тексты по ним
3. Получаешь результат с уверенностью модели

<b>Что получишь:</b>
• Точную классификацию (85-95%)
• Совместимость с твоей таксономией
• Метрики качества

<b>Начать классификацию?</b>
        """
        
        keyboard = [
            [InlineKeyboardButton("Да, начать классификацию", callback_data="mode_classification")],
            [InlineKeyboardButton("Нет, лучше изучение", callback_data="mode_clustering")],
            [InlineKeyboardButton("🔙 В главное меню", callback_data="back_to_start")]
        ]
    
    await query.edit_message_text(
        result_text,
        parse_mode='HTML',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    
    logger.info(
        f"❓ QUIZ COMPLETE | User: {update.effective_user.id} | "
        f"Size: {size} | Categories: {categories} | Frequency: {frequency} | "
        f"Recommendation: {recommendation}"
    )


async def handle_mode_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик выбора режима"""
    query = update.callback_query
    await query.answer()
    
    action = query.data
    user_id = update.effective_user.id
    
    logger.info(f"🎯 MODE SELECT | User: {user_id} | Mode: {action}")
    
    if action == "back_to_start":
        context.user_data.clear()
        await start(update, context)
        return

    if action == "show_help":
        await help_command(update, context)
        return
    
    if action == "mode_clustering":
        context.user_data['mode'] = 'clustering'
        
        text = """
🔍 <b>Режим: Автоматическая кластеризация</b>

Я автоматически найду темы и сгруппирую похожие тексты.

📎 <b>Отправь CSV-файл:</b>
• Первая колонка — тексты для анализа
• Кодировка UTF-8
• Макс. размер: 20 МБ
• Макс. строк: 50,000

✨ <b>Что получишь:</b>
• CSV с кластерами и названиями тем
• Статистику по группам
• Детальный PDF-отчет (по запросу)

⏱ <b>Время обработки:</b> 1-20 минут
        """
        
        await query.edit_message_text(text, parse_mode='HTML')
    
    elif action == "mode_classification":
        if not CLASSIFICATION_AVAILABLE:
            await query.edit_message_text(
                "❌ <b>Классификация недоступна</b>\n\n"
                "Для использования нужен YandexGPT API.\n"
                "Свяжитесь с администратором.",
                parse_mode='HTML'
            )
            return
        
        context.user_data['mode'] = 'classification'
        
        # НОВОЕ: Выбор способа задания категорий
        text = """
🏷️ <b>Режим: Классификация по категориям</b>

Выбери способ задания категорий:

🎯 <b>Ввести вручную</b>
• Ты знаешь нужные категории
• Быстрый старт

🤖 <b>Сгенерировать автоматически</b>
• AI проанализирует твои тексты
• Предложит категории
• Ты сможешь их отредактировать

💡 Автогенерация полезна, когда не знаешь, какие категории нужны.
        """
        
        keyboard = [
            [InlineKeyboardButton("Ввести вручную", callback_data="cat_method_manual")],
            [InlineKeyboardButton("Сгенерировать автоматически", callback_data="cat_method_auto")],
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_start")]
        ]
        
        await query.edit_message_text(
            text,
            parse_mode='HTML',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

#Обработчик для выбора метода категорий
async def handle_category_method_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик выбора метода задания категорий"""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    action = query.data
    
    logger.info(f"📝 CATEGORY METHOD | User: {user_id} | Method: {action}")
    
    if action == "cat_method_manual":
        # Ручной ввод (существующая логика)
        text = """
🏷️ <b>Ввод категорий вручную</b>

📝 <b>Введи категории</b> (каждая с новой строки):

<b>Пример:</b>
<code>Проблемы с оплатой
Вопросы по доставке
Качество товара
Технические проблемы
Общие вопросы</code>

Или через запятую:
<code>Оплата, Доставка, Качество, Техподдержка</code>

💡 <b>Требования:</b>
• Минимум 2 категории
• Максимум 20 категорий
• Чёткие названия
        """
        
        context.user_data['category_method'] = 'manual'
        await query.edit_message_text(text, parse_mode='HTML')
    
    elif action == "cat_method_auto":
        # Автогенерация
        if not category_generator:
            await query.edit_message_text(
                "❌ <b>Автогенерация недоступна</b>\n\n"
                "Требуется настройка YandexGPT API.",
                parse_mode='HTML'
            )
            return
        
        context.user_data['category_method'] = 'auto'
        
        text = """
🤖 <b>Автоматическая генерация категорий</b>

📂 <b>Отправь CSV-файл с текстами</b>

Я возьму выборку и сгенерирую категории через AI.

📊 <b>Размер выборки:</b>
• До 1000 строк: все тексты
• 1000-5000: 500 случайных
• 5000+: 1000 случайных

⚙️ <b>Далее ты сможешь:</b>
• Настроить промт (опционально)
• Отредактировать категории
• Перегенерировать при необходимости

📎 Отправь файл (макс. 20 МБ, UTF-8)
        """
        
        keyboard = [[InlineKeyboardButton("🔙 Назад", callback_data="mode_classification")]]
        
        await query.edit_message_text(
            text,
            parse_mode='HTML',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )


async def handle_prompt_customization_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик выбора: настроить промт или использовать дефолтный"""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    action = query.data
    
    logger.info(f"⚙️ PROMPT CHOICE | User: {user_id} | Action: {action}")
    
    if action == "use_default_gen_prompt":
        # Использовать дефолтный промт генерации
        context.user_data['custom_generation_prompt'] = None
        await start_category_generation(update, context, query.message)
    
    elif action == "customize_gen_prompt":
        # Показать дефолтный промт и попросить ввести свой
        default_prompt = category_generator.DEFAULT_PROMPT
        
        text = f"""
⚙️ <b>Настройка промта генерации</b>

Промт определяет, как AI будет анализировать тексты.

📝 <b>Стандартный промт:</b>
<code>{default_prompt[:500]}...</code>

<b>Отправь свой вариант промта</b> или нажми "Использовать стандартный".

💡 <b>Что можно указать:</b>
• Специфику домена (медицина, e-commerce и т.д.)
• Желаемое количество категорий
• Особые критерии (тональность, срочность)

<b>Пример кастомизации:</b>
<i>"Проанализируй отзывы на медицинские услуги. Предложи 6-8 категорий. Обязательно выдели отдельно жалобы на побочные эффекты."</i>
        """
        
        keyboard = [
            [InlineKeyboardButton("Использовать стандартный", callback_data="use_default_gen_prompt")],
            [InlineKeyboardButton("Отмена", callback_data="mode_classification")]
        ]
        
        context.user_data['awaiting_custom_prompt'] = 'generation'
        
        await query.edit_message_text(
            text,
            parse_mode='HTML',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    elif action == "use_default_class_prompt":
        # Дефолтный промт классификации
        context.user_data['custom_classification_prompt'] = None
        await proceed_to_classification_type(update, context, query.message)
    
    elif action == "customize_class_prompt":
        # Кастомный промт классификации
        default_prompt = prompt_manager.DEFAULT_CLASSIFICATION_PROMPT
        
        text = f"""
⚙️ <b>Настройка промта классификации</b>

Этот промт определяет, как AI будет распределять тексты по категориям.

📝 <b>Стандартный промт:</b>
<code>{default_prompt[:400]}...</code>

<b>Отправь свой вариант</b> или используй стандартный.

💡 <b>Что можно настроить:</b>
• Строгость классификации
• Правила для пограничных случаев
• Специфику контекста

<b>Пример:</b>
<i>"При классификации медицинских отзывов учитывай серьёзность проблемы. Если есть упоминание боли или осложнений — приоритет категории 'Побочные эффекты'."</i>
        """
        
        keyboard = [
            [InlineKeyboardButton("Использовать стандартный", callback_data="use_default_class_prompt")],
            [InlineKeyboardButton("Отмена", callback_data="mode_classification")]
        ]
        
        context.user_data['awaiting_custom_prompt'] = 'classification'
        
        await query.edit_message_text(
            text,
            parse_mode='HTML',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )


async def start_category_generation(update: Update, context: ContextTypes.DEFAULT_TYPE, message):
    """Запуск генерации категорий"""
    user_id = update.effective_user.id
    
    # Получаем сохранённую выборку
    sample_texts = context.user_data.get('sample_texts')
    if not sample_texts:
        await message.reply_text(
            "❌ Ошибка: файл не найден. Начните заново с /start",
            parse_mode='HTML'
        )
        return
    
    # Показываем прогресс
    progress_msg = await message.reply_text(
        "🔄 <b>Генерирую категории...</b>\n\n"
        f"📊 Анализирую выборку: {len(sample_texts)} текстов\n"
        "🤖 Отправляю запрос в YandexGPT...\n\n"
        "⏱ Это займёт 10-30 секунд",
        parse_mode='HTML'
    )
    
    try:
        custom_prompt = context.user_data.get('custom_generation_prompt')
        
        success, categories, error = category_generator.generate_categories(
            sample_texts,
            custom_prompt=custom_prompt
        )
        
        if not success:
            await progress_msg.edit_text(
                f"❌ <b>Ошибка генерации</b>\n\n{error}\n\n"
                "Попробуйте:\n"
                "• Проверить настройки API\n"
                "• Повторить через минуту\n"
                "• Ввести категории вручную",
                parse_mode='HTML'
            )
            return
        
        # Сохраняем сгенерированные категории
        context.user_data['generated_categories'] = categories
        
        # Форматируем для показа
        categories_text = category_generator.format_categories_for_display(categories)
        
        full_text = (
            f"✅ <b>Категории сгенерированы!</b>\n\n"
            f"Проанализировано текстов: {len(sample_texts)}\n\n"
            f"{categories_text}"
            f"<b>Что делать дальше?</b>"
        )
        
        keyboard = [
            [InlineKeyboardButton("Использовать эти категории", callback_data="approve_generated_cats")],
            [InlineKeyboardButton("Редактировать", callback_data="edit_generated_cats")],
            [InlineKeyboardButton("Перегенерировать", callback_data="regenerate_cats")],
            [InlineKeyboardButton("Отмена", callback_data="back_to_start")]
        ]
        
        await progress_msg.edit_text(
            full_text,
            parse_mode='HTML',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        
    except Exception as e:
        logger.error(f"Error in start_category_generation: {e}", exc_info=True)
        await progress_msg.edit_text(
            f"❌ Произошла ошибка при генерации категорий.\n\nПопробуйте еще раз или обратитесь к администратору.",
            parse_mode='HTML'
        )


async def handle_generated_categories_action(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик действий с сгенерированными категориями"""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    action = query.data
    
    logger.info(f"📋 GENERATED CATS ACTION | User: {user_id} | Action: {action}")
    
    if action == "approve_generated_cats":
        # Утверждаем категории
        categories = context.user_data.get('generated_categories', [])
        if not categories:
            await query.edit_message_text("❌ Ошибка: категории не найдены", parse_mode='HTML')
            return
        
        # Преобразуем в формат для классификации
        category_names = [cat.name for cat in categories]
        category_descriptions = {cat.name: cat.description for cat in categories if cat.description}
        
        context.user_data['categories'] = category_names
        context.user_data['descriptions'] = category_descriptions

        logger.info(f"✅ CATEGORIES APPROVED | User: {user_id} | Resetting category_method flag")
        
        # Переходим к настройке промта классификации
        text = """
✅ <b>Категории сохранены!</b>

⚙️ <b>Настроить промт для классификации?</b>

Промт определяет, как AI будет распределять тексты по этим категориям.

💡 Кастомизация нужна, если:
• Специфичная предметная область
• Важны особые критерии
• Нужна строгая/мягкая классификация

По умолчанию используется универсальный промт.
        """
        
        keyboard = [
            [InlineKeyboardButton("Использовать стандартный", callback_data="use_default_class_prompt")],
            [InlineKeyboardButton("Настроить промт", callback_data="customize_class_prompt")]
        ]
        
        await query.edit_message_text(
            text,
            parse_mode='HTML',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    elif action == "edit_generated_cats":
        # Редактирование категорий
        categories = context.user_data.get('generated_categories', [])
        
        # Форматируем для редактирования
        cats_text = "\n".join([f"{cat.name} | {cat.description}" for cat in categories])
        
        text = f"""
✏️ <b>Редактирование категорий</b>

Текущие категории:
<code>{cats_text}</code>

<b>Отправь отредактированный список:</b>

Формат 1 (с описаниями):
<code>Название 1 | Описание 1
Название 2 | Описание 2</code>

Формат 2 (без описаний):
<code>Название 1
Название 2</code>

💡 Можешь:
• Изменить названия
• Убрать категории
• Добавить новые
• Уточнить описания
        """
        
        context.user_data['awaiting_edited_categories'] = True
        
        keyboard = [[InlineKeyboardButton("❌ Отмена", callback_data="show_generated_cats_again")]]
        
        await query.edit_message_text(
            text,
            parse_mode='HTML',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    elif action == "regenerate_cats":
        # Перегенерировать
        text = """
🔄 <b>Перегенерация категорий</b>

Хочешь изменить промт перед повторной генерацией?

💡 Это полезно, если:
• Категории слишком общие/специфичные
• Не хватает/много категорий
• Нужен другой фокус анализа
        """
        
        keyboard = [
            [InlineKeyboardButton("Перегенерировать с тем же промтом", callback_data="use_default_gen_prompt")],
            [InlineKeyboardButton("Изменить промт", callback_data="customize_gen_prompt")],
            [InlineKeyboardButton("Отмена", callback_data="show_generated_cats_again")]
        ]
        
        await query.edit_message_text(
            text,
            parse_mode='HTML',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    elif action == "show_generated_cats_again":
        # Показать категории снова (после отмены редактирования)
        categories = context.user_data.get('generated_categories', [])
        categories_text = category_generator.format_categories_for_display(categories)
        
        text = f"🏷️ <b>Сгенерированные категории:</b>\n\n{categories_text}\n<b>Что делать дальше?</b>"
        
        keyboard = [
            [InlineKeyboardButton("Использовать эти категории", callback_data="approve_generated_cats")],
            [InlineKeyboardButton("Редактировать", callback_data="edit_generated_cats")],
            [InlineKeyboardButton("Перегенерировать", callback_data="regenerate_cats")]
        ]
        
        await query.edit_message_text(
            text,
            parse_mode='HTML',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )


async def proceed_to_classification_type(update: Update, context: ContextTypes.DEFAULT_TYPE, message):
    """Переход к выбору типа классификации"""
    categories = context.user_data.get('categories', [])
    
    categories_list = "\n".join([f"{i+1}. {cat}" for i, cat in enumerate(categories)])
    
    # Проверяем, есть ли уже загруженный файл (для автогенерации)
    has_file = bool(context.user_data.get('full_file_path'))
    
    if has_file:
        text = f"""
✅ <b>Готово к классификации!</b>

<b>Категории ({len(categories)}):</b>
{categories_list}

📎 <b>Файл уже загружен</b>

<b>Выбери режим:</b>

📋 <b>Обычная классификация</b>
AI распределит тексты по категориям

📊 <b>Оценка качества</b>
Проверка качества классификации (нужен файл с правильными ответами)
        """
    else:
        text = f"""
✅ <b>Категории сохранены!</b>

<b>Категории ({len(categories)}):</b>
{categories_list}

<b>Выбери режим:</b>

📋 <b>Обычная классификация</b>
Загрузишь файл → AI распределит тексты

📊 <b>Оценка качества</b>
Проверка качества на размеченных данных
        """
    
    keyboard = [
        [InlineKeyboardButton("📋 Обычная классификация", callback_data="class_normal")],
        [InlineKeyboardButton("📊 Оценка качества", callback_data="class_eval")]
    ]
    
    await message.reply_text(
        text,
        parse_mode='HTML',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )



async def handle_categories_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик ввода категорий"""
    text = update.message.text
    user_id = update.effective_user.id
    
    logger.info(f"📝 TEXT INPUT | User: {user_id} | Mode: {context.user_data.get('mode')}")
    
    # ПРИОРИТЕТ 1: Проверка на кастомный промт (НОВОЕ - было в предыдущем плане, но не сработало)
    if context.user_data.get('awaiting_custom_prompt'):
        prompt_type = context.user_data['awaiting_custom_prompt']
        
        logger.info(f"📝 CUSTOM PROMPT RECEIVED | User: {user_id} | Type: {prompt_type}")
        
        if prompt_type == 'generation':
            context.user_data['custom_generation_prompt'] = text
            del context.user_data['awaiting_custom_prompt']
            
            await update.message.reply_text(
                "✅ <b>Промт сохранён!</b>\n\n🔄 Начинаю генерацию категорий...",
                parse_mode='HTML'
            )
            
            await start_category_generation(update, context, update.message)
            return
        
        elif prompt_type == 'classification':
            prompt_manager.set_classification_prompt(user_id, text)
            context.user_data['custom_classification_prompt'] = text
            del context.user_data['awaiting_custom_prompt']
            
            await update.message.reply_text(
                "✅ <b>Промт классификации сохранён!</b>",
                parse_mode='HTML'
            )
            
            await proceed_to_classification_type(update, context, update.message)
            return
    
    # ПРИОРИТЕТ 2: Проверка на редактирование сгенерированных категорий
    if context.user_data.get('awaiting_edited_categories'):
        logger.info(f"📝 EDITED CATEGORIES | User: {user_id}")
        
        del context.user_data['awaiting_edited_categories']
        
        # Парсим отредактированные категории
        categories = parse_categories_from_text(text)
        is_valid, error_msg = validate_categories(categories)
        
        if not is_valid:
            await update.message.reply_text(
                f"❌ <b>Ошибка:</b> {error_msg}\n\n"
                "Попробуйте еще раз или /start для отмены.",
                parse_mode='HTML'
            )
            return
        
        context.user_data['categories'] = categories
        context.user_data['descriptions'] = None
        
        categories_list = "\n".join([f"{i+1}. {cat}" for i, cat in enumerate(categories)])
        
        await update.message.reply_text(
            f"✅ <b>Категории обновлены ({len(categories)}):</b>\n\n{categories_list}",
            parse_mode='HTML'
        )
        
        # Переход к настройке промта классификации
        text_msg = """
⚙️ <b>Настроить промт для классификации?</b>

Промт — это инструкция для AI, как распределять тексты.

По умолчанию используется универсальный промт.
Кастомизация нужна для специфичных задач.
        """
        
        keyboard = [
            [InlineKeyboardButton("Использовать стандартный", callback_data="use_default_class_prompt")],
            [InlineKeyboardButton("Настроить промт", callback_data="customize_class_prompt")]
        ]
        
        await update.message.reply_text(
            text_msg,
            parse_mode='HTML',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return
    
    # ПРИОРИТЕТ 3: Обычный ввод категорий (существующая логика)
    if context.user_data.get('mode') != 'classification':
        logger.info(f"⚠️ TEXT INPUT IGNORED | User: {user_id} | Not in classification mode")
        return
    
    # Проверка, что не ждём файл для автогенерации
    if context.user_data.get('category_method') == 'auto':
        await update.message.reply_text(
            "⚠️ <b>Ожидается файл, а не текст</b>\n\n"
            "Отправьте CSV-файл для генерации категорий.",
            parse_mode='HTML'
        )
        return
    
    logger.info(f"📝 MANUAL CATEGORIES INPUT | User: {user_id}")
    
    # Далее существующая логика парсинга категорий...
    categories = parse_categories_from_text(text)
    is_valid, error_msg = validate_categories(categories)
    
    if not is_valid:
        await update.message.reply_text(
            f"❌ <b>Ошибка:</b> {error_msg}\n\n"
            "Попробуй еще раз или /start для отмены.",
            parse_mode='HTML'
        )
        return
    
    context.user_data['categories'] = categories
    context.user_data['descriptions'] = None

    categories_list = "\n".join([f"{i+1}. {cat}" for i, cat in enumerate(categories)])

    keyboard = [
        [InlineKeyboardButton("📋 Обычная классификация", callback_data="class_normal")],
        [InlineKeyboardButton("📊 Оценка качества", callback_data="class_eval")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        f"✅ <b>Категории приняты ({len(categories)} шт.):</b>\n\n"
        f"{categories_list}\n\n"
        f"<b>Выбери режим:</b>",
        reply_markup=reply_markup,
        parse_mode='HTML'
    )

async def handle_classification_mode_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик выбора режима классификации (обычная/оценка)"""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    action = query.data
    
    logger.info(f"📊 CLASSIFICATION MODE | User: {user_id} | Mode: {action}")
    
    if action == "class_normal":
        context.user_data['eval_mode'] = False
        
        # Проверяем, есть ли уже файл
        if context.user_data.get('full_file_path'):
            logger.info(f"📋 CLASSIFICATION WITH EXISTING FILE | User: {user_id}")
            
            # Используем уже загруженный файл
            file_path = context.user_data['full_file_path']
            
            # Проверяем, что файл существует
            import os
            if not os.path.exists(file_path):
                logger.error(f"❌ FILE NOT FOUND | Path: {file_path}")
                await query.message.reply_text(
                    "❌ <b>Ошибка: файл не найден</b>\n\n"
                    "Пожалуйста, загрузите файл заново.",
                    parse_mode='HTML'
                )
                # Очищаем несуществующий путь
                context.user_data.pop('full_file_path', None)
                context.user_data.pop('sample_texts', None)
                context.user_data.pop('original_filename', None)
                return
            
            # Показываем прогресс
            progress_msg = await query.message.reply_text(
                "🔄 <b>Запускаю классификацию...</b>\n\n"
                "Использую уже загруженный файл.",
                parse_mode='HTML'
            )
            
            try:
                # Читаем файл
                df = pd.read_csv(file_path, encoding='utf-8', dtype=str)
                filename = context.user_data.get('original_filename', 'classified.csv')
                
                logger.info(f"📊 FILE LOADED | Rows: {len(df)} | Filename: {filename}")
                
                # Создаём tracker
                from progress_tracker import ProgressTracker
                tracker = ProgressTracker(progress_msg, min_interval=3.0)
                
                # Запускаем классификацию
                await process_classification_mode(
                    update, context, df, file_path, 
                    filename, tracker, progress_msg
                )
                
                # ⭐ ВАЖНО: Удаляем файл ПОСЛЕ успешной классификации
                cleanup_file_safe(file_path)
                logger.info(f"🗑️ TEMP FILE DELETED | Path: {file_path}")
                
                # Очищаем сохранённые данные
                context.user_data.pop('full_file_path', None)
                context.user_data.pop('sample_texts', None)
                context.user_data.pop('category_method', None)
                context.user_data.pop('original_filename', None)
                
                logger.info(f"✅ CLASSIFICATION COMPLETE | User: {user_id}")
                
            except Exception as e:
                logger.error(f"❌ Error in classification with existing file: {e}", exc_info=True)
                
                # Удаляем файл даже при ошибке
                cleanup_file_safe(file_path)
                
                await progress_msg.edit_text(
                    "❌ <b>Ошибка классификации</b>\n\n"
                    "Попробуйте загрузить файл заново или обратитесь к администратору.",
                    parse_mode='HTML'
                )
                
                # Очищаем данные
                context.user_data.pop('full_file_path', None)
                context.user_data.pop('sample_texts', None)
                context.user_data.pop('category_method', None)
                context.user_data.pop('original_filename', None)
            
            return
        
        # Если файла НЕТ — просим загрузить
        logger.info(f"📋 NO FILE FOUND | User: {user_id} | Requesting file upload")
        
        text = (
            "📋 <b>Обычная классификация</b>\n\n"
            "📎 <b>Отправь CSV-файл с текстами:</b>\n"
            "• Одна колонка с текстами\n"
            "• Кодировка UTF-8\n"
            "• Макс. 10,000 строк\n\n"
            "⏱ Время: 1-2 сек на текст"
        )
        
        await query.edit_message_text(text, parse_mode='HTML')
        return
    
    elif action == "class_eval":
        context.user_data['eval_mode'] = True
        
        categories = context.user_data.get('categories', [])
        categories_list = "\n".join([f"• {cat}" for cat in categories])
        
        text = (
            "📊 <b>Оценка качества классификации</b>\n\n"
            "📎 Отправь CSV-файл с <b>двумя колонками</b>:\n"
            "1. <b>текст</b> - текст для классификации\n"
            "2. <b>правильная_категория</b> - эталонная категория\n\n"
            "<b>Пример CSV:</b>\n"
            "<code>текст,правильная_категория\n"
            '"Не могу оплатить",Вопросы по оплате\n'
            '"Где диплом?",Вопросы по дипломам</code>\n\n'
            f"<b>Ожидаемые категории:</b>\n{categories_list}\n\n"
            "⚠️ Категории в файле должны точно совпадать с введёнными"
        )
        
        await query.edit_message_text(text, parse_mode='HTML')
        return

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /help"""
    # Если вызвана из callback
    if update.callback_query:
        query = update.callback_query
        await query.answer()
        
        help_msg = """
💡 <b>Справка по использованию</b>

<b>📋 КЛАССИФИКАЦИЯ (рекомендуется)</b>

Подходит если:
✅ Нужно разложить тексты по темам
✅ Важна точность (для отчётов, дашбордов)
✅ До 5,000 текстов
✅ Категории известны ИЛИ AI их предложит

Не подходит если:
❌ Больше 10,000 текстов (будет долго)

<b>Пример:</b> Разложить отзывы клиентов по категориям "Доставка", "Качество", "Оплата" для еженедельного отчёта.

━━━━━━━━━━━━━━━━━

<b>🔍 ИЗУЧЕНИЕ ДАННЫХ</b>

Подходит если:
✅ Первый раз смотришь на данные
✅ Больше 1,000 текстов
✅ Нужно быстро и бесплатно
✅ Хочешь найти неожиданные темы

Не подходит если:
❌ Нужны категории для отчёта начальству
❌ Меньше 100 текстов

<b>Пример:</b> Проанализировать 10,000 обращений в поддержку за год, чтобы понять основные проблемы.

━━━━━━━━━━━━━━━━━

<b>📊 ФОРМАТ ФАЙЛА:</b>

<b>Для классификации и изучения:</b>
• CSV с текстами в первой колонке
• Кодировка UTF-8
• Макс. 20 МБ

<b>Пример:</b>
<code>текст
Не пришел заказ
Качество плохое
Долго ждал</code>

<b>Для оценки качества:</b>
• Две колонки: текст, правильная_категория

<b>Пример:</b>
<code>текст,правильная_категория
Не пришел заказ,Доставка
Качество плохое,Качество товара</code>

━━━━━━━━━━━━━━━━━

<b>🎯 ТИПИЧНЫЙ СЦЕНАРИЙ:</b>

1️⃣ <b>Первый раз:</b> ИЗУЧЕНИЕ → смотрю топ-темы
2️⃣ Формирую категории на основе результата
3️⃣ <b>Дальше:</b> КЛАССИФИКАЦИЯ → регулярная работа

<b>Команды:</b>
/start - начать работу
/help - эта справка
/about - о технологиях
        """
        
        keyboard = [[InlineKeyboardButton("🔙 Назад в меню", callback_data="back_to_start")]]
        await query.edit_message_text(help_msg, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    else:
        # Вызвана как команда (не из меню)
        help_msg = """
💡 <b>Справка по использованию</b>

<b>🎯 ЧТО ВЫБРАТЬ?</b>

📋 <b>Классификация</b> — когда нужно разложить по категориям
🔍 <b>Изучение</b> — когда нужно понять, что вообще есть в данных

Не уверен? Используй /start → "Помочь выбрать (квиз)"

━━━━━━━━━━━━━━━━━

<b>📊 ФОРМАТ ФАЙЛА:</b>

CSV с текстами в первой колонке, UTF-8

<b>Пример:</b>
<code>текст
Не пришел заказ
Качество плохое</code>

Максимум: 20 МБ, 50k строк

━━━━━━━━━━━━━━━━━

<b>Команды:</b>
/start - начать работу
/help - эта справка
/about - о технологиях
/feedback - обратная связь

Есть вопросы? Просто отправь файл! 📊
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

        # ⭐ ДЕБАГ: Логируем состояние context.user_data
        logger.info(
            f"📊 CONTEXT STATE | User: {user_id} | "
            f"mode={context.user_data.get('mode')} | "
            f"category_method={context.user_data.get('category_method')} | "
            f"has_categories={'categories' in context.user_data} | "
            f"has_file={'full_file_path' in context.user_data} | "
            f"eval_mode={context.user_data.get('eval_mode')}"
        )

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
        
        # Проверка: это файл для автогенерации категорий?
        is_auto_generation = (
            context.user_data.get('category_method') == 'auto' 
            and context.user_data.get('mode') == 'classification'
            and 'categories' not in context.user_data  # ⭐ КЛЮЧЕВАЯ ПРОВЕРКА
        )
        
        if is_auto_generation:
            # Режим автогенерации категорий (ПЕРВАЯ загрузка)
            logger.info(f"📊 AUTO-GENERATION MODE | User: {user_id}")
            
            progress_msg = await update.message.reply_text(
                "⏳ <b>Загружаю файл для генерации категорий...</b>",
                parse_mode='HTML'
            )
            
            try:
                # Загружаем файл
                file = await update.message.document.get_file()
                
                # ⭐ ВРЕМЕННЫЙ файл для скачивания
                temp_download_path = f"/tmp/{file.file_unique_id}.csv"
                await file.download_to_drive(temp_download_path)
                
                logger.info(f"📥 FILE DOWNLOADED | Path: {temp_download_path}")
                
                # Читаем CSV
                df = pd.read_csv(temp_download_path, encoding='utf-8', dtype=str)
                texts = df.iloc[:, 0].astype(str).tolist()
                
                if len(texts) < 10:
                    await progress_msg.edit_text(
                        "❌ <b>Слишком мало текстов</b>\n\n"
                        "Для генерации категорий нужно минимум 10 текстов.",
                        parse_mode='HTML'
                    )
                    cleanup_file_safe(temp_download_path)
                    return
                
                # ⭐ КОПИРУЕМ в безопасное место (TEMP_DIR под нашим контролем)
                from config import TEMP_DIR
                import os
                import shutil
                
                # Создаём уникальное имя файла
                safe_filename = f"autogen_{user_id}_{int(time.time())}.csv"
                safe_file_path = os.path.join(TEMP_DIR, safe_filename)
                
                # Копируем файл
                shutil.copy2(temp_download_path, safe_file_path)
                
                # Удаляем временный файл из /tmp
                cleanup_file_safe(temp_download_path)
                
                logger.info(f"💾 FILE SAVED | Safe path: {safe_file_path}")
                
                # Получаем выборку
                sample = category_generator.get_sample(texts)
                context.user_data['sample_texts'] = sample
                context.user_data['full_file_path'] = safe_file_path  # ⭐ Сохраняем безопасный путь
                context.user_data['original_filename'] = update.message.document.file_name

                # Спрашиваем про промт
                text = f"""
✅ <b>Файл загружен!</b>

📊 Найдено текстов: {len(texts)}
📦 Выборка для анализа: {len(sample)}

⚙️ <b>Настроить промт для генерации категорий?</b>

Стандартный промт подходит для большинства задач.
Кастомизация нужна для специфичных доменов.
                """
                
                keyboard = [
                    [InlineKeyboardButton("Использовать стандартный", callback_data="use_default_gen_prompt")],
                    [InlineKeyboardButton("Настроить промт", callback_data="customize_gen_prompt")],
                    [InlineKeyboardButton("Отмена", callback_data="back_to_start")]
                ]
                
                await progress_msg.edit_text(
                    text,
                    parse_mode='HTML',
                    reply_markup=InlineKeyboardMarkup(keyboard)
                )
                
                return 
                
            except Exception as e:
                logger.error(f"❌ Error loading file for auto-generation: {e}", exc_info=True)
                await progress_msg.edit_text(
                    "❌ Ошибка чтения файла.\n\nПроверьте формат (CSV, UTF-8).",
                    parse_mode='HTML'
                )
                cleanup_file_safe(file_path)
                return
        
        # ⭐ Если категории УЖЕ есть, но файл загружается снова — это классификация
        if context.user_data.get('mode') == 'classification' and 'categories' in context.user_data:
            logger.info(f"📋 CLASSIFICATION FILE UPLOADED | User: {user_id}")
            # Дальше идёт обычная обработка классификации
            # НЕ прерываем, пусть идёт дальше в код



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
                await progress_msg.edit_text(
                    f"❌ <b>Слишком много строк</b>\n\n"
                    f"Найдено: {n_rows} строк\n"
                    f"Максимум: {MAX_ROWS} строк\n\n"
                    f"💡 Пожалуйста, разделите файл на части",
                    parse_mode='HTML'
                )
                return
            
            if n_rows == 0:
                await progress_msg.edit_text(
                    "❌ <b>Файл пустой</b>\n\n"
                    "В файле нет данных для анализа",
                    parse_mode='HTML'
                )
                return
            
            # Показываем информацию о файле (с экранированием HTML)
            first_texts = df.iloc[:3, 0].fillna("").astype(str).tolist()
            examples = "\n".join([f"  • {html.escape(t[:50])}{'...' if len(t) > 50 else ''}" 
                                for t in first_texts if t.strip()])
            
            file_info = (
                f"✅ <b>Файл загружен!</b>\n\n"
                f"📄 <b>Информация о файле:</b>\n"
                f"• Название: {html.escape(update.message.document.file_name)}\n"
                f"• Размер: {file_size_mb:.2f} МБ\n"
                f"• Строк: <b>{n_rows}</b>\n"
                f"• Колонок: {n_cols}\n\n"
            )
            
            if examples:
                file_info += f"📝 <b>Примеры текстов:</b>\n{examples}\n\n"
            
            # Оценка времени обработки
            if n_rows < 1000:
                time_estimate = "1-2 минуты"
            elif n_rows < 5000:
                time_estimate = "2-5 минут"
            elif n_rows < 20000:
                time_estimate = "5-15 минут"
            else:
                time_estimate = "15-20 минут"
            
            file_info += (
                f"⏱ <b>Примерное время обработки:</b> {time_estimate}\n\n"
                f"🔄 <b>Начинаю анализ...</b>\n"
                f"Можете закрыть чат — я пришлю сообщение, когда всё будет готово."
            )
            
            # Обновляем прогресс-сообщение с информацией о файле
            await progress_msg.edit_text(file_info, parse_mode='HTML')
            
            # Даём пользователю время прочитать (2 секунды)
            await asyncio.sleep(2)
            
        except Exception as e:
            await progress_msg.edit_text(
                f"❌ <b>Ошибка чтения файла</b>\n\n"
                f"Не удалось прочитать CSV файл.\n\n"
                f"💡 Проверьте:\n"
                f"• Кодировка UTF-8\n"
                f"• Корректный CSV формат\n"
                f"• Файл не поврежден",
                parse_mode='HTML'
            )
            logger.error(f"CSV read error: {e}")
            return

        # Проверяем режим работы
        mode = context.user_data.get('mode', 'clustering')
        logger.info(f"🎯 MODE | User: {user_id} | Mode: {mode}")
        
        if mode == 'classification':
            # Проверка наличия категорий
            if 'categories' not in context.user_data:
                await progress_msg.edit_text(
                    "❌ <b>Ошибка:</b> Категории не заданы.\n\n"
                    "Используй /start для начала.",
                    parse_mode='HTML'
                )
                return
            
            # Лимит для классификации меньше
            MAX_ROWS_CLASSIFICATION = 10000
            if n_rows > MAX_ROWS_CLASSIFICATION:
                await progress_msg.edit_text(
                    f"❌ <b>Слишком много строк для классификации</b>\n\n"
                    f"Найдено: {n_rows}\n"
                    f"Максимум: {MAX_ROWS_CLASSIFICATION}\n\n"
                    f"💡 Для больших файлов используй кластеризацию",
                    parse_mode='HTML'
                )
                return
            
            # Вызываем классификацию
            await process_classification_mode(
                update, context, df, file_path, 
                update.message.document.file_name, tracker, progress_msg
            )
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

        # Увеличиваем счётчик файлов
        context.user_data['files_processed'] = context.user_data.get('files_processed', 0) + 1
        
        # Добавляем режим в список использованных
        if 'clustering' not in context.user_data.get('modes_used', []):
            context.user_data.setdefault('modes_used', []).append('кластеризация')
        
        # Отправка уведомления админу
        if analytics:
            try:
                await analytics.track_file_processed(
                    bot=context.bot,
                    user_id=user_id,
                    username=username,
                    files_count=context.user_data['files_processed'],
                    mode='clustering',
                    rows=n_rows,
                    filename=file_name,
                    quiz_data=context.user_data.get('quiz_answers'),
                    source=context.user_data.get('source')
                )
            except Exception as e:
                logger.error(f"Analytics track_file_processed failed: {e}")
        
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
        try:
            await progress_msg.delete()
            progress_msg = None  # Помечаем, что сообщение удалено
        except Exception as e:
            logger.warning(f"Failed to delete progress message: {e}")
        
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
        try:
            await progress_msg.delete()
        except:
            pass
        
        await update.message.reply_text(error_msg, parse_mode='HTML')
        
    finally:
        # Очистка временных файлов
        cleanup_file_safe(file_path)
        if result_path and cache_key:
            cleanup_file_safe(result_path)

async def process_classification_mode(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    df: pd.DataFrame,
    file_path: str,
    filename: str,
    tracker: ProgressTracker,
    progress_msg
):
    """Обработка в режиме классификации"""
    user_id = update.effective_user.id
    categories = context.user_data['categories']
    descriptions = context.user_data.get('descriptions')
    eval_mode = context.user_data.get('eval_mode', False)
    
    # ⭐ ТОЛЬКО ЗДЕСЬ определяем message
    if update.callback_query:
        message = update.callback_query.message
    else:
        message = update.message
    
    logger.info(
        f"🏷️ CLASSIFICATION START | User: {user_id} | "
        f"Texts: {len(df)} | Categories: {len(categories)} | Eval: {eval_mode}"
    )
    
    # Фильтрация мусорных данных
    original_count = len(df)
    
    # Если режим оценки - валидируем файл
    if eval_mode:
        is_valid, error_msg = validate_ground_truth(df, categories)
        if not is_valid:
            try:
                await progress_msg.delete()
            except:
                pass
            
            await message.reply_text(
                f"❌ <b>Ошибка в файле:</b>\n\n{error_msg}",
                parse_mode='HTML'
            )
            return
        
        texts = df.iloc[:, 0].astype(str).tolist()
        ground_truth = df.iloc[:, 1].astype(str).tolist()
    else:
        # Обычная классификация - фильтруем
        df = df[df.iloc[:, 0].notna()]
        df = df[df.iloc[:, 0].astype(str).str.strip() != '']
        
        texts_series = df.iloc[:, 0].astype(str)
        
        mask = (
            ~texts_series.str.startswith('/') &
            ~texts_series.str.endswith(('.png', '.jpg', '.pdf', '.jpeg', '.gif')) &
            (texts_series.str.len() > 5)
        )
        
        df = df[mask]
        df = df.reset_index(drop=True)
        
        filtered_count = len(df)
        
        if filtered_count == 0:
            try:
                await progress_msg.delete()
            except:
                pass
            
            await message.reply_text(
                "❌ <b>Нет данных для классификации</b>\n\n"
                "После фильтрации не осталось корректных текстов.",
                parse_mode='HTML'
            )
            return
        
        if filtered_count < original_count:
            logger.info(
                f"🧹 FILTERED | Original: {original_count} | "
                f"After: {filtered_count} | Removed: {original_count - filtered_count}"
            )
        
        texts = df.iloc[:, 0].astype(str).tolist()
        ground_truth = None
    
    n_texts = len(texts)
    
    try:
        await tracker.update(
            stage="🏷️ Классификация с помощью AI",
            percent=30,
            details=f"Обработка {n_texts} текстов...",
            force=True
        )
        
        async def classification_progress(progress: float, current: int, total: int):
            if current % 5 == 0 or current == total:
                await tracker.update(
                    stage=f"🏷️ Классифицировано: {current}/{total}",
                    percent=30 + int(progress * 0.6),
                    details=f"Осталось ~{(total-current)*1.5//60} мин"
                )
        
        result_df = classifier.classify_batch(
            texts,
            categories,
            descriptions,
            progress_callback=classification_progress
        )
        
        stats = classifier.get_classification_stats(result_df)
        
        await tracker.update(stage="💾 Сохранение результатов", percent=95)
        
        result_path = f"/tmp/{user_id}_classified_{filename}"
        result_df.to_csv(result_path, index=False, encoding='utf-8')
        
        logger.info(f"✅ CLASSIFICATION COMPLETE | User: {user_id} | Texts: {n_texts}")
        
        # Увеличиваем счётчик файлов
        context.user_data['files_processed'] = context.user_data.get('files_processed', 0) + 1
        
        # Добавляем режим в список
        if 'classification' not in context.user_data.get('modes_used', []):
            context.user_data.setdefault('modes_used', []).append('классификация')
        
        # Отправка уведомления админу
        if analytics:
            try:
                await analytics.track_file_processed(
                    bot=context.bot,
                    user_id=user_id,
                    username=update.effective_user.username,
                    files_count=context.user_data['files_processed'],
                    mode='classification',
                    rows=n_texts,
                    filename=filename,
                    quiz_data=context.user_data.get('quiz_answers'),
                    source=context.user_data.get('source')
                )
            except Exception as e:
                logger.error(f"Analytics track_file_processed failed: {e}")

        
        if eval_mode:
            result_df['true_category'] = ground_truth
            result_df['correct'] = result_df['category'] == result_df['true_category']
            
            metrics = calculate_metrics(
                y_true=ground_truth,
                y_pred=result_df['category'].tolist(),
                categories=categories
            )
            
            examples = get_error_examples(result_df, n=3)
            stats_msg = format_evaluation_report(metrics, examples, categories)
            stats_msg += "\n\n✨ CSV-файл с результатами прикреплён ниже"
            
        else:
            sorted_cats = sorted(
                stats['categories'].items(),
                key=lambda x: x[1]['count'],
                reverse=True
            )[:5]
            
            dist_text = "\n".join([
                f"• {cat}: {info['count']} ({info['percentage']:.1f}%)"
                for cat, info in sorted_cats
            ])
            
            stats_msg = (
                f"✅ <b>Классификация завершена!</b>\n\n"
                f"📊 <b>Результаты:</b>\n"
                f"• Обработано текстов: {n_texts}\n"
                f"• Категорий: {len(categories)}\n"
                f"• Средняя уверенность: {stats['avg_confidence']:.2f}\n\n"
            )
            
            if stats.get('undefined_count', 0) > 0:
                stats_msg += f"⚠️ <b>Не удалось определить:</b> {stats['undefined_count']} ({stats['undefined_percentage']:.1f}%)\n\n"
            
            stats_msg += f"📋 <b>Распределение (топ-5):</b>\n{dist_text}\n\n"
            stats_msg += f"✨ Готово! Хотите классифицировать другие тексты? Отправляйте новый файл!"

        await tracker.complete("✅ Классификация завершена!")
        
        try:
            await progress_msg.delete()
        except:
            pass
        
        with open(result_path, 'rb') as result_file:
            await message.reply_document(
                document=result_file,
                filename=f"classified_{filename}",
                caption=stats_msg,
                parse_mode='HTML'
            )
        
        cleanup_file_safe(result_path)
        
    except Exception as e:
        logger.error(f"❌ CLASSIFICATION ERROR | User: {user_id} | Error: {str(e)}", exc_info=True)
        
        try:
            await progress_msg.delete()
        except:
            pass
        
        await message.reply_text(
            f"❌ <b>Ошибка классификации</b>\n\n"
            f"Попробуйте еще раз или обратитесь к администратору.",
            parse_mode='HTML'
        )


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
    
    # Инициализация аналитики
    admin_id = os.getenv('ADMIN_TELEGRAM_ID')
    if admin_id:
        try:
            global analytics
            analytics = UserAnalytics(admin_chat_id=int(admin_id))
            logger.info("✅ Analytics initialized")
        except Exception as e:
            logger.error(f"⚠️ Analytics init failed: {e}")
            analytics = None
    else:
        logger.warning("⚠️ ADMIN_TELEGRAM_ID not set - analytics disabled")
        analytics = None

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
    from telegram.ext import CallbackQueryHandler
    # Обработчики для автогенерации категорий
    application.add_handler(CallbackQueryHandler(
        handle_category_method_choice,
        pattern="^cat_method_"
    ))
    
    application.add_handler(CallbackQueryHandler(
        handle_prompt_customization_choice,
        pattern="^use_default_|^customize_"
    ))
    
    application.add_handler(CallbackQueryHandler(
        handle_generated_categories_action,
pattern="^approve_generated_cats$|^edit_generated_cats$|^regenerate_cats$|^show_generated_cats_again$"
    ))

    application.add_handler(CallbackQueryHandler(handle_mode_selection, pattern="^mode_|^show_help$|^back_to_start$"))
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
    application.add_handler(CallbackQueryHandler(handle_classification_mode_choice, pattern="^class_"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_categories_input))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    application.add_error_handler(error_handler)
    # Обработчики квиза
    application.add_handler(CallbackQueryHandler(show_quiz, pattern="^show_quiz$"))
    application.add_handler(CallbackQueryHandler(handle_quiz_q1, pattern="^quiz_q1_"))
    application.add_handler(CallbackQueryHandler(handle_quiz_q2, pattern="^quiz_q2_"))
    application.add_handler(CallbackQueryHandler(handle_quiz_result, pattern="^quiz_q3_"))
    application.add_handler(CallbackQueryHandler(handle_quiz_back, pattern="^quiz_back_"))


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

