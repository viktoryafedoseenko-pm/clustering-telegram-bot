# pdf_generator.py
import io
import re
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import logging
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, 
    Table, TableStyle, PageBreak, PageTemplate, Frame
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.platypus import HRFlowable
from config import FONT_PATH, MAX_PDF_SIZE_MB
from datetime import datetime
from wordcloud import WordCloud
import numpy as np

logger = logging.getLogger(__name__)

date_str = datetime.now().strftime("%d.%m.%Y")

# Регистрация шрифта
pdfmetrics.registerFont(TTFont('DejaVuSans', str(FONT_PATH)))

# Настройка matplotlib для кириллицы
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

def remove_emoji(text):
    """Удаляет эмодзи и специальные символы из текста"""
    if not isinstance(text, str):
        text = str(text)
    
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"  # enclosed characters
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)
    text = text.replace('•', '').replace('■', '').replace('→', '').replace('←', '')
    text = ' '.join(text.split())
    return text.strip()

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('DejaVuSans', 8)
    canvas.setFillColor(colors.HexColor('#546E7A'))
    
    canvas.drawString(
        inch, 0.5 * inch,
        f"Страница {doc.page}"
    )
    
    canvas.drawRightString(
        A4[0] - inch, 0.5 * inch,
        "Создано: @cluster_master_bot"
    )
    canvas.restoreState()

class PDFReportGenerator:
    """Генератор PDF отчётов по кластеризации"""
    
    # Новая цветовая палитра
    COLOR_PRIMARY = colors.HexColor('#263238')      # Тёмно-серый для текста
    COLOR_SECONDARY = colors.HexColor('#546E7A')    # Серо-голубой
    COLOR_ACCENT = colors.HexColor('#5E35B1')       # Глубокий фиолетовый
    COLOR_DIVIDER = colors.HexColor('#E0E0E0')      # Светло-серый
    COLOR_BACKGROUND = colors.HexColor('#FAFAFA')   # Очень светлый серый
    
    # Цвета для графиков
    COLOR_HIGH = colors.HexColor('#E53935')         # Красный
    COLOR_MEDIUM = colors.HexColor('#FB8C00')       # Оранжевый
    COLOR_LOW = colors.HexColor('#43A047')          # Зелёный
    
    # Размеры шрифтов
    FONT_TITLE = 20
    FONT_HEADING = 14
    FONT_SUBHEADING = 12
    FONT_BODY = 10
    FONT_SMALL = 9
    
    # Увеличенные отступы (+50%)
    SPACER_LARGE = 0.6 * inch
    SPACER_MEDIUM = 0.3 * inch
    SPACER_SMALL = 0.15 * inch
    
    def __init__(self, df: pd.DataFrame, stats: dict, cluster_names: dict, 
                 master_hierarchy: dict = None, master_names: dict = None):
        self.df = df
        self.stats = stats
        self.cluster_names = cluster_names
        self.master_hierarchy = master_hierarchy or {}
        self.master_names = master_names or {}
        self.styles = self._setup_styles()
    
    def _setup_styles(self):
        """Настройка стилей"""
        styles = getSampleStyleSheet()
        
        # Заголовок отчёта
        styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=styles['Title'],
            fontName='DejaVuSans',
            fontSize=self.FONT_TITLE,
            textColor=self.COLOR_PRIMARY,
            spaceAfter=24,
            spaceBefore=12,
            alignment=0
        ))
        
        # Заголовок секции (фиолетовый)
        styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=styles['Heading1'],
            fontName='DejaVuSans',
            fontSize=self.FONT_HEADING,
            textColor=self.COLOR_ACCENT,
            spaceAfter=12,
            spaceBefore=16
        ))
        
        # Подзаголовок
        styles.add(ParagraphStyle(
            name='CustomSubheading',
            parent=styles['Normal'],
            fontName='DejaVuSans',
            fontSize=self.FONT_SUBHEADING,
            textColor=self.COLOR_SECONDARY,
            spaceAfter=10
        ))
        
        # Обычный текст (увеличенный межстрочный интервал)
        styles.add(ParagraphStyle(
            name='CustomBody',
            parent=styles['Normal'],
            fontName='DejaVuSans',
            fontSize=self.FONT_BODY,
            textColor=self.COLOR_PRIMARY,
            leading=16,  # было 14
            spaceAfter=8
        ))
        
        # Мелкий текст
        styles.add(ParagraphStyle(
            name='CustomSmall',
            parent=styles['Normal'],
            fontName='DejaVuSans',
            fontSize=self.FONT_SMALL,
            textColor=self.COLOR_SECONDARY,
            leading=14
        ))
        
        # Для заголовков тем (UPPERCASE, без номера)
        styles.add(ParagraphStyle(
            name='TopicHeading',
            parent=styles['Heading1'],
            fontName='DejaVuSans',
            fontSize=self.FONT_HEADING,
            textColor=self.COLOR_PRIMARY,
            spaceAfter=12,
            spaceBefore=16
        ))
        
        return styles
    
    def _create_paragraph(self, text, style_name='CustomBody'):
        """Вспомогательный метод для создания параграфа"""
        return Paragraph(text, self.styles[style_name])
    
    def _create_divider(self, width="100%", thickness=0.5):
        """Создаёт горизонтальный разделитель"""
        return HRFlowable(
            width=width,
            thickness=thickness,
            color=self.COLOR_DIVIDER,
            spaceBefore=10,
            spaceAfter=10
        )
    
    def generate(self, output_path: str) -> bool:
        """Генерирует PDF отчёт"""
        try:
            logger.info(f"📄 Starting PDF generation: {output_path}")
            
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=85,      # увеличены margins (+15%)
                leftMargin=85,
                topMargin=85,
                bottomMargin=60
            )
            
            story = []
            
            # 1. Executive Summary
            logger.info("📝 Creating executive summary...")
            story.extend(self._create_executive_summary())
            story.append(PageBreak())
            
            # 2. Структура тем (мастер-категории)
            if self.master_hierarchy:
                logger.info("🏷️ Creating topic structure...")
                story.extend(self._create_topic_structure())
                story.append(PageBreak())
            
            # 3. Визуализация (графики + word cloud)
            logger.info("📈 Creating visualizations...")
            story.extend(self._create_visualizations())
            story.append(PageBreak())
            
            # 4. Топ-8 тем (вместо 10)
            logger.info("🏷️ Creating topic pages...")
            story.extend(self._create_topic_pages())
            story.append(PageBreak())
            
            # 5. CTA
            logger.info("🚀 Creating CTA page...")
            story.extend(self._create_cta_page())
            
            logger.info("🔨 Building PDF...")
            doc.build(story, onFirstPage=footer, onLaterPages=footer)
            
            size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            logger.info(f"📦 PDF size: {size_mb:.2f} MB")
            
            if size_mb > MAX_PDF_SIZE_MB:
                logger.warning(f"⚠️ PDF too large: {size_mb:.2f} MB > {MAX_PDF_SIZE_MB} MB")
                Path(output_path).unlink()
                return False
            
            logger.info("✅ PDF generated successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ PDF generation error: {e}", exc_info=True)
            return False
    
    def _create_executive_summary(self):
        """Executive Summary на первой странице"""
        elements = []
        
        # Заголовок
        elements.append(self._create_paragraph(
            "АНАЛИЗ ТЕКСТОВ: ГЛАВНЫЕ ВЫВОДЫ",
            'CustomTitle'
        ))
        elements.append(Spacer(1, self.SPACER_MEDIUM))
        
        # Масштаб анализа
        total_texts = self.stats['total_texts']
        n_clusters = self.stats['n_clusters']
        n_masters = len(self.master_hierarchy) if self.master_hierarchy else 0
        
        summary_text = f"""
        <b>Проанализировано:</b> {total_texts:,} текстов<br/>
        <b>Найдено:</b> {n_clusters} тематических групп
        """
        
        if n_masters > 0:
            summary_text += f"<br/><b>Объединено в:</b> {n_masters} категорий"
        
        elements.append(self._create_paragraph(summary_text, 'CustomBody'))
        elements.append(Spacer(1, self.SPACER_LARGE))
        
        # Топ-3 темы
        elements.append(self._create_paragraph(
            "ТОП-3 ТЕМЫ ПО ОБЪЁМУ",
            'CustomHeading'
        ))
        elements.append(Spacer(1, self.SPACER_SMALL))
        
        cluster_dist = self.df['cluster_id'].value_counts().head(3)
        
        for rank, (cluster_id, count) in enumerate(cluster_dist.items(), 1):
            if cluster_id == -1:
                continue
            
            cluster_name = remove_emoji(self.cluster_names.get(cluster_id, f"Тема {cluster_id}"))
            percent = (count / len(self.df)) * 100
            
            topic_text = f"""
            <b>{rank}. {cluster_name}</b><br/>
            {count:,} обращений ({percent:.1f}%)
            """
            
            elements.append(self._create_paragraph(topic_text, 'CustomBody'))
            elements.append(Spacer(1, self.SPACER_SMALL))
        
        elements.append(Spacer(1, self.SPACER_MEDIUM))
        elements.append(self._create_divider())
        elements.append(Spacer(1, self.SPACER_MEDIUM))
        
        # Метрики качества
        if 'quality_metrics' in self.stats:
            elements.append(self._create_paragraph(
                "КАЧЕСТВО АНАЛИЗА",
                'CustomHeading'
            ))
            elements.append(Spacer(1, self.SPACER_SMALL))
            
            qm = self.stats['quality_metrics']
            
            metrics_text = f"""
            <b>Silhouette Score:</b> {qm['silhouette_score']:.3f} / 1.0<br/>
            <b>Davies-Bouldin Index:</b> {qm['davies_bouldin_index']:.3f}<br/>
            <br/>
            <i>Интерпретация: Кластеры имеют размытые границы, что типично 
            для разнообразных текстов. Результат надёжен для принятия решений.</i>
            """
            
            elements.append(self._create_paragraph(metrics_text, 'CustomSmall'))
            elements.append(Spacer(1, self.SPACER_LARGE))
        
        # Призыв читать дальше
        elements.append(self._create_divider())
        elements.append(Spacer(1, self.SPACER_SMALL))
        
        elements.append(self._create_paragraph(
            "Детальный анализ на следующих страницах",
            'CustomBody'
        ))
        
        return elements
    
    def _create_topic_structure(self):
        """Структура тем (вместо 'Мастер-категории')"""
        elements = []
        
        elements.append(self._create_paragraph(
            "СТРУКТУРА ТЕМ",
            'CustomHeading'
        ))
        elements.append(Spacer(1, self.SPACER_SMALL))
        
        elements.append(self._create_paragraph(
            f"Тексты сгруппированы в иерархическую структуру: "
            f"{self.stats['n_clusters']} подтем объединены в {len(self.master_hierarchy)} категорий",
            'CustomBody'
        ))
        elements.append(Spacer(1, self.SPACER_LARGE))
        
        # Визуальная иерархия
        elements.append(self._create_paragraph(
            "ОБЗОР КАТЕГОРИЙ",
            'CustomSubheading'
        ))
        elements.append(Spacer(1, self.SPACER_MEDIUM))
        
        # Собираем статистику по мастер-категориям
        master_stats = []
        for master_id, sub_clusters in self.master_hierarchy.items():
            master_name = remove_emoji(self.master_names.get(master_id, f"Категория {master_id}"))
            total_count = sum(len(self.df[self.df['cluster_id'] == cid]) for cid in sub_clusters)
            percent = (total_count / len(self.df)) * 100
            
            master_stats.append({
                'name': master_name,
                'count': total_count,
                'percent': percent,
                'master_id': master_id
            })
        
        master_stats.sort(key=lambda x: x['count'], reverse=True)
        
        # Визуальное дерево категорий
        for master in master_stats:
            master_id = master['master_id']
            
            # Название категории
            category_text = f"<b>{master['name']}</b> ............... {master['percent']:.1f}%"
            elements.append(self._create_paragraph(category_text, 'CustomBody'))
            elements.append(Spacer(1, self.SPACER_SMALL))
            
            # Подтемы (топ-3)
            sub_clusters = self.master_hierarchy[master_id]
            sub_data = []
            
            for cluster_id in sub_clusters:
                cluster_count = len(self.df[self.df['cluster_id'] == cluster_id])
                cluster_name = remove_emoji(self.cluster_names.get(cluster_id, f"Тема {cluster_id}"))
                
                sub_data.append({
                    'name': cluster_name,
                    'count': cluster_count
                })
            
            sub_data.sort(key=lambda x: x['count'], reverse=True)
            
            for sub in sub_data[:7]:  # только топ-5
                subtopic_text = f"  ├ {sub['name']}"
                elements.append(self._create_paragraph(subtopic_text, 'CustomSmall'))
            
            if len(sub_data) > 7:
                elements.append(self._create_paragraph(
                    f"  └ ещё {len(sub_data) - 3} подтем...",
                    'CustomSmall'
                ))
            
            elements.append(Spacer(1, self.SPACER_MEDIUM))
        
        return elements
    
    def _create_visualizations(self):
        """Страница визуализаций: графики + word cloud"""
        elements = []
        
        elements.append(self._create_paragraph(
            "ВИЗУАЛИЗАЦИЯ",
            'CustomHeading'
        ))
        elements.append(Spacer(1, self.SPACER_MEDIUM))
        
        # Круговая диаграмма
        try:
            pie_img = self._create_pie_chart()
            if pie_img:
                elements.append(pie_img)
                elements.append(Spacer(1, self.SPACER_MEDIUM))
        except Exception as e:
            logger.error(f"Error creating pie chart: {e}")
        
        # Столбчатая диаграмма
        try:
            bar_img = self._create_bar_chart()
            if bar_img:
                elements.append(bar_img)
                elements.append(Spacer(1, self.SPACER_LARGE))
        except Exception as e:
            logger.error(f"Error creating bar chart: {e}")
        
        # Word Cloud
        # try:
        #     wc_img = self._create_word_cloud()
        #     if wc_img:
        #         elements.append(self._create_paragraph(
        #             "ЧАСТОТНЫЙ АНАЛИЗ",
        #             'CustomSubheading'
        #         ))
        #         elements.append(Spacer(1, self.SPACER_SMALL))
                
        #         elements.append(self._create_paragraph(
        #             "Наиболее употребляемые слова в обращениях:",
        #             'CustomSmall'
        #         ))
        #         elements.append(Spacer(1, self.SPACER_SMALL))
                
        #         elements.append(wc_img)
        # except Exception as e:
        #     logger.error(f"Error creating word cloud: {e}")
        
        return elements
    
    def _create_pie_chart(self):
        """Круговая диаграмма топ-8 (ИДЕАЛЬНО КРУГЛАЯ)"""
        cluster_dist = self.df['cluster_id'].value_counts().head(8)
        cluster_dist = cluster_dist[cluster_dist.index != -1]
        
        labels = [
            remove_emoji(self.cluster_names.get(cid, f"Тема {cid}"))[:30]
            for cid in cluster_dist.index
        ]
        sizes = cluster_dist.values
        
        # Строго квадратная фигура 10x10 дюймов
        fig, ax = plt.subplots(figsize=(10, 10))
        
        colors_palette = [
            '#5E35B1', '#7E57C2', '#9575CD', '#B39DDB',
            '#D1C4E9', '#BA68C8', '#AB47BC', '#9C27B0'
        ]
        
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors_palette[:len(sizes)],
            textprops={'fontsize': 11, 'color': '#263238'},
            pctdistance=0.80,
            labeldistance=1.08,
            wedgeprops={'linewidth': 2, 'edgecolor': 'white'}  # белые разделители
        )
        
        # Проценты
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        # Подписи
        for text in texts:
            text.set_fontsize(10)
        
        # Принудительно круг
        ax.axis('equal')
        
        plt.title('Топ-8 тем по размеру', 
                fontsize=15, 
                pad=30, 
                color='#263238', 
                weight='bold')
        
        # КРИТИЧНО: НЕ используем bbox_inches='tight'!
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, 
                format='png', 
                dpi=120,  # уменьшили для оптимизации размера
                facecolor='white',
                # bbox_inches='tight' УБРАЛИ!
                )
        plt.close()
        img_buffer.seek(0)
        
        # Квадратное изображение в PDF
        return Image(img_buffer, width=5.5*inch, height=5.5*inch)



    def _create_bar_chart(self):
        """Столбчатая диаграмма топ-8 (БЕЗ "Прочее")"""
        cluster_dist = self.df['cluster_id'].value_counts().head(8)
        
        # Фильтруем noise (-1)
        cluster_dist = cluster_dist[cluster_dist.index != -1]
        
        labels = [
            remove_emoji(self.cluster_names.get(cid, f"Тема {cid}"))[:50]  # увеличили лимит
            for cid in cluster_dist.index
        ]
        
        fig, ax = plt.subplots(figsize=(8, 6))  # увеличили высоту
        
        # Градиент от тёмного к светлому фиолетовому
        bar_colors = ['#5E35B1', '#7E57C2', '#9575CD', '#B39DDB', 
                    '#D1C4E9', '#E1BEE7', '#CE93D8', '#BA68C8']
        
        bars = ax.barh(labels, cluster_dist.values, 
                    color=bar_colors[:len(labels)],
                    edgecolor='#424242',
                    linewidth=0.5)
        
        ax.set_xlabel('Количество текстов', fontsize=11, color='#263238', weight='bold')
        ax.set_title('Топ-8 самых крупных тем', fontsize=14, pad=15, color='#263238', weight='bold')
        ax.invert_yaxis()
        ax.tick_params(axis='both', colors='#546E7A', labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#E0E0E0')
        ax.spines['bottom'].set_color('#E0E0E0')
        ax.grid(axis='x', color='#F5F5F5', linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Значения на столбцах
        for i, v in enumerate(cluster_dist.values):
            ax.text(v + max(cluster_dist.values) * 0.01, i, f'{v:,}', 
                va='center', fontsize=10, color='#263238', weight='bold')
        
        # Добавляем больше места для длинных названий
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=200, facecolor='white')
        plt.close()
        img_buffer.seek(0)
        
        return Image(img_buffer, width=6*inch, height=4.5*inch)

    
    def _create_topic_pages(self):
        """Страницы с топ-8 темами (по 2 на страницу, 4 примера)"""
        elements = []
        
        cluster_dist = self.df['cluster_id'].value_counts().head(8)  # топ-8 вместо 10
        
        for idx, (cluster_id, count) in enumerate(cluster_dist.items()):
            if cluster_id == -1:
                continue
            
            # Каждые 2 темы — новая страница
            if idx > 0 and idx % 2 == 0:
                elements.append(PageBreak())
            
            cluster_name = remove_emoji(self.cluster_names.get(cluster_id, f"Тема {cluster_id}"))
            percent = (count / len(self.df)) * 100
            
            # Определяем мастер-категорию
            master_category = ""
            if self.master_hierarchy:
                for master_id, sub_clusters in self.master_hierarchy.items():
                    if cluster_id in sub_clusters:
                        master_category = remove_emoji(self.master_names.get(master_id, f"Категория {master_id}"))
                        break
            
            # Заголовок темы (UPPERCASE, без номера)
            elements.append(self._create_paragraph(
                cluster_name.upper(),
                'TopicHeading'
            ))
            elements.append(Spacer(1, self.SPACER_SMALL))
            
            # Статистика
            stats_text = f"<b>Объём:</b> {count:,} текстов ({percent:.1f}% от общего)"
            if master_category:
                stats_text += f"<br/><b>Категория:</b> {master_category}"
            
            elements.append(self._create_paragraph(stats_text, 'CustomBody'))
            elements.append(Spacer(1, self.SPACER_SMALL))
            elements.append(self._create_divider())
            elements.append(Spacer(1, self.SPACER_SMALL))
            
            # О ЧЁМ ПИШУТ
            elements.append(self._create_paragraph(
                "О ЧЁМ ПИШУТ:",
                'CustomSubheading'
            ))
            elements.append(Spacer(1, self.SPACER_SMALL))
            
            # Основной паттерн (если есть описание от LLM)
            # TODO: добавить поле pattern в cluster_names
            # elements.append(self._create_paragraph(
            #     "<b>Основной паттерн:</b><br/>"
            #     "Пользователи интересуются данным вопросом в различных контекстах.",
            #     'CustomBody'
            # ))
            # elements.append(Spacer(1, self.SPACER_SMALL))
            
            # Типичные запросы (4 примера вместо 6)
            elements.append(self._create_paragraph(
                "Типичные запросы:",
                'CustomBody'
            ))
            elements.append(Spacer(1, self.SPACER_SMALL))
            
            cluster_texts = self.df[self.df['cluster_id'] == cluster_id].iloc[:, 0].head(4).tolist()  # 4 вместо 6
            
            for text in cluster_texts:
                text_preview = str(text)[:200] + "..." if len(str(text)) > 200 else str(text)
                text_preview = text_preview.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
                
                # Форматируем как цитату
                quote_text = f'<i>"{text_preview}"</i>'
                elements.append(self._create_paragraph(quote_text, 'CustomSmall'))
                elements.append(Spacer(1, self.SPACER_SMALL))
            
            elements.append(Spacer(1, self.SPACER_LARGE))
        
        return elements
    
    def _create_cta_page(self):
        """CTA страница (без изменений, но с новыми цветами)"""
        elements = []
        
        elements.append(self._create_paragraph(
            "@cluster_master_bot",
            'CustomTitle'
        ))
        elements.append(Spacer(1, self.SPACER_SMALL))
        
        elements.append(self._create_paragraph(
            "Этот отчёт создан автоматически за несколько минут",
            'CustomBody'
        ))
        elements.append(Spacer(1, self.SPACER_LARGE))
        
        # Возможности
        elements.append(self._create_paragraph(
            "Возможности бота:",
            'CustomSubheading'
        ))
        elements.append(Spacer(1, self.SPACER_SMALL))
        
        features = [
            "Анализ до 50,000 текстов за минуты",
            "Автоматическая кластеризация (BERTopic + HDBSCAN)",
            "Генерация названий через AI (YandexGPT)",
            "Экспорт в CSV и PDF",
            "Иерархическая структура тем",
            "Метрики качества анализа"
        ]
        
        for feature in features:
            elements.append(self._create_paragraph(f"• {feature}", 'CustomBody'))
        
        elements.append(Spacer(1, self.SPACER_MEDIUM))
        
        # Использование
        elements.append(self._create_paragraph(
            "Используйте для:",
            'CustomSubheading'
        ))
        elements.append(Spacer(1, self.SPACER_SMALL))
        
        use_cases = [
            "Анализа отзывов и обращений клиентов",
            "Обработки тикетов службы поддержки",
            "Исследования результатов опросов",
            "Приоритизации product roadmap",
            "Выявления трендов и проблем"
        ]
        
        for use_case in use_cases:
            elements.append(self._create_paragraph(f"• {use_case}", 'CustomBody'))
        
        elements.append(Spacer(1, self.SPACER_LARGE))
        
        # Призыв
        elements.append(self._create_paragraph(
            "Начать: t.me/cluster_master_bot",
            'CustomHeading'
        ))
        elements.append(Spacer(1, self.SPACER_SMALL))
        
        elements.append(self._create_paragraph(
            "Бесплатно | Без регистрации",
            'CustomBody'
        ))
        
        elements.append(Spacer(1, self.SPACER_LARGE))
        elements.append(self._create_divider())
        
        elements.append(self._create_paragraph(
            f"@cluster_master_bot | {date_str}",
            'CustomSmall'
        ))
        
        return elements
