# bot.py
import logging
import os
from dotenv import load_dotenv
import html
import pandas as pd
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from clustering import clusterize_texts

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

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
• Первая колонка = тексты для анализа
• Кодировка: UTF-8
• Максимум: 10 МБ, 10000 строк

<b>Пример CSV:</b>
<code>текст
Не пришел заказ вовремя
Качество товара плохое
Долго ждал доставку
Товар не соответствует описанию</code>

<b>Что получите:</b>
✅ Файл с кластерами
✅ Статистику по темам
✅ Названия кластеров

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

<b>BERTopic</b> — алгоритм кластеризации текстов
Автоматически находит темы в текстах без предварительной разметки

<b>Sentence Transformers</b> — нейросети для понимания смысла
Превращает тексты в числовые векторы, сохраняя их значение

<b>UMAP</b> — снижение размерности данных
Упрощает сложные данные, сохраняя важные связи между текстами

<b>HDBSCAN</b> — алгоритм кластеризации
Находит группы похожих текстов автоматически

<b>🎯 Как это работает:</b>
1. Текст → понимание смысла (нейросеть)
2. Поиск похожих текстов (математика)
3. Группировка по темам (алгоритмы)
4. Автоматическое название каждой группы

<b>💪 Преимущества:</b>
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

Нашли баг? Есть идеи по улучшению? 
Или просто хотите поделиться впечатлениями?

Пишите мне: @viktoryafedoseenko

Буду рада любым комментариям! 🙏
    """
    await update.message.reply_text(feedback_msg, parse_mode='HTML')


async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    progress_msg = None
    file_path = None
    result_path = None
    
    try:
        # Проверка размера файла
        MAX_FILE_SIZE_MB = 20
        file_size_mb = update.message.document.file_size / (1024 * 1024)
        
        if file_size_mb > MAX_FILE_SIZE_MB:
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
            
            # Проверка количества строк
            MAX_ROWS = 50000
            if n_rows > MAX_ROWS:
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
            
            file_info += "🔄 <b>Начинаю анализ...</b>"
            
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
        
        result_path, stats = clusterize_texts(file_path, progress_callback)
        
        # Шаг 4: Формирование статистики
        stats_message = format_statistics(stats)
        
        await progress_msg.edit_text(
            "⏳ <b>Почти готово...</b>\n\n"
            "✅ Файл загружен\n"
            "✅ Тексты проанализированы\n"
            "📤 Отправляю результат...",
            parse_mode='HTML'
        )
        
        # Отправка результата
        with open(result_path, 'rb') as result_file:
            await update.message.reply_document(
                document=result_file,
                filename=os.path.basename(result_path),
                caption=stats_message,
                parse_mode='HTML'
            )
        
        # Удаление сообщения о прогрессе
        await progress_msg.delete()
        
    except ValueError as e:
        error_msg = f"⚠️ <b>Проблема с данными</b>\n\n{html.escape(str(e))}\n\n💡 Проверьте формат файла"
        if progress_msg:
            await progress_msg.edit_text(error_msg, parse_mode='HTML')
        else:
            await update.message.reply_text(error_msg, parse_mode='HTML')
        logger.warning(f"ValueError: {e}")
        
    except Exception as e:
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
            if result_path and os.path.exists(result_path):
                os.remove(result_path)
        except:
            pass


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
        msg += "🏆 <b>Топ-3 кластера:</b>\n"
        for i, cluster in enumerate(stats['top_clusters'][:3], 1):
            emoji = ["1️⃣", "2️⃣", "3️⃣"][i-1]
            # Экранируем название кластера
            safe_name = html.escape(cluster['name'])
            msg += f"{emoji} <i>{safe_name}</i> — {cluster['size']} текстов\n"
        msg += "\n"
    
    msg += "📎 Полные результаты в прикрепленном файле\n\n"
    msg += "✨ Готово! Хотите проанализировать другие тексты? Отправляйте новый файл — я готов!"
    
    return msg


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Глобальный обработчик ошибок"""
    logger.error("Exception while handling an update:", exc_info=context.error)


def main():
    application = Application.builder().token(TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("about", about_command))
    application.add_handler(CommandHandler("feedback", feedback_command))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    application.add_error_handler(error_handler)
    
    logger.info("🤖 Бот запущен и готов к работе!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
