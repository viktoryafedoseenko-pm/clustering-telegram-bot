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
• Максимум: 20 МБ, 30000 строк

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
            "⏳ <b>Обработка началась...</b>\n\n"
            "📥 Загружаю файл...",
            parse_mode='HTML'
        )
        
        file = await update.message.document.get_file()
        file_path = f"/tmp/{file.file_unique_id}.csv"
        await file.download_to_drive(file_path)
        
        # Шаг 2: Анализ файла
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
            
            file_info += "🔄 <b>Начинаю анализ. Файл в 5000 строк обрабатывается до 5 минут. На файл в 30000 строк может уйти до 20 минут. Можете закрыть чат – я пришлю сообщение, когда всё будет готово.</b>"
            
            await progress_msg.edit_text(file_info, parse_mode='HTML')
            
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
        
        # Шаг 3: Кластеризация
        async def progress_callback(msg):
            try:
                await progress_msg.edit_text(msg, parse_mode='HTML')
            except:
                pass
        
        result_path, stats, hierarchy, master_names = clusterize_texts(file_path, progress_callback)
        
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

        # Информируем, что почти готово
        await progress_msg.edit_text(
            "⏳ <b>Почти готово...</b>\n\n"
            "✅ Файл загружен\n"
            "✅ Тексты проанализированы\n"
            "📤 Отправляю результат...",
            parse_mode='HTML'
        )
        
        # Показываем кнопки выбора
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 Детальный отчёт PDF", callback_data=f"pdf_{cache_key}")],
            [InlineKeyboardButton("❌ Только CSV", callback_data="csv_only")]
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
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"🗑️ Deleted temp file: {file_path}")
            if result_path and os.path.exists(result_path) and cache_key:
                os.remove(result_path)
                logger.debug(f"🗑️ Deleted result file: {result_path}")
        except Exception as e:
            logger.warning(f"⚠️ CLEANUP FAILED | Files: {file_path}, {result_path} | Error: {e}")


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
    
    if callback_data == "csv_only":
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text(
            "✅ Отлично! CSV файл уже у вас.\n\n"
            "Хотите проанализировать другие тексты? Отправляйте новый файл!"
        )
        return
    
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
    # Логирование: Старт бота
    logger.info("=" * 60)
    logger.info("🤖 BOT STARTING...")
    logger.info(f"📁 Log directory: {LOG_DIR}")
    logger.info(f"📁 Temp directory: {TEMP_DIR}")
    logger.info(f"🔑 Token configured: {'✅' if TOKEN else '❌'}")
    logger.info("=" * 60)
    
    application = Application.builder().token(TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("about", about_command))
    application.add_handler(CommandHandler("feedback", feedback_command))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_file))

    from telegram.ext import CallbackQueryHandler
    application.add_handler(CallbackQueryHandler(handle_pdf_request))

    application.add_error_handler(error_handler)
    
    logger.info("✅ All handlers registered")
    logger.info("🚀 Bot is running and ready to accept requests!")
    logger.info("=" * 60)
    
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()

