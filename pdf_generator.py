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
    Table, TableStyle, PageBreak
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.platypus import HRFlowable
from config import FONT_PATH, MAX_PDF_SIZE_MB
from datetime import datetime
from wordcloud import WordCloud
from collections import Counter

logger = logging.getLogger(__name__)

date_str = datetime.now().strftime("%d.%m.%Y %H:%M")

# Регистрация шрифта
pdfmetrics.registerFont(TTFont('DejaVuSans', str(FONT_PATH)))

# Настройка matplotlib для кириллицы
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

def remove_emoji(text):
    """Удаляет эмодзи и специальные символы из текста"""
    if not isinstance(text, str):
        text = str(text)
    # Удаляем эмодзи (Unicode диапазоны эмодзи)
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
    # Удаляем другие специальные символы
    text = text.replace('•', '').replace('■', '').replace('→', '').replace('←', '')
    text = text.replace('🔹', '').replace('Ὄ', '')
    # Убираем лишние пробелы
    text = ' '.join(text.split())
    return text.strip()

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('DejaVuSans', 8)
    canvas.setFillColor(colors.HexColor('#666666'))
    # Левый футер - номер страницы
    canvas.drawString(
        inch, 0.5 * inch,
        f"Stranica {doc.page} | Otchyot po klasterizatsii"
    )
    # Правый футер - ссылка на бота
    canvas.drawRightString(
        A4[0] - inch, 0.5 * inch,
        "Sozdano: @cluster_master_bot"
    )
    canvas.restoreState()

class PDFReportGenerator:
    """Генератор PDF отчётов по кластеризации"""
    
    # Обновлённая цветовая палитра
    COLOR_PRIMARY = colors.HexColor('#263238')      # Тёмно-серый для основного текста
    COLOR_SECONDARY = colors.HexColor('#546E7A')    # Серо-голубой для второстепенного
    COLOR_ACCENT = colors.HexColor('#5E35B1')       # Глубокий фиолетовый для акцентов
    
    # Статус-цвета
    COLOR_HIGH = colors.HexColor('#E53935')         # Красный
    COLOR_MEDIUM = colors.HexColor('#FB8C00')       # Оранжевый
    COLOR_LOW = colors.HexColor('#43A047')          # Зелёный
    
    # Фоны и разделители
    COLOR_BACKGROUND = colors.HexColor('#FAFAFA')   # Очень светлый серый
    COLOR_DIVIDER = colors.HexColor('#E0E0E0')      # Светло-серый для линий
    COLOR_TABLE_HEADER = colors.HexColor('#5E35B1') # Фиолетовый для заголовков
    
    # Размеры шрифтов
    FONT_TITLE = 18
    FONT_HEADING = 14
    FONT_SUBHEADING = 12
    FONT_BODY = 10
    FONT_SMALL = 9
    
    # Увеличенные отступы (+50%)
    SPACER_LARGE = 0.6 * inch
    SPACER_MEDIUM = 0.3 * inch
    SPACER_SMALL = 0.15 * inch
    
    def __init__(self, df: pd.DataFrame, stats: dict, cluster_names: dict, master_hierarchy: dict = None, master_names: dict = None):
        self.df = df
        self.stats = stats
        self.cluster_names = cluster_names
        self.master_hierarchy = master_hierarchy or {}
        self.master_names = master_names or {}
        self.styles = self._setup_styles()
    
    def _setup_styles(self):
        """Настройка стилей с кириллицей"""
        styles = getSampleStyleSheet()
        
        title_font = 'DejaVuSans'
        heading_font = 'DejaVuSans'
        body_font = 'DejaVuSans'
        
        # Заголовок отчёта
        styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=styles['Title'],
            fontName=title_font,
            fontSize=self.FONT_TITLE,
            textColor=self.COLOR_PRIMARY,
            spaceAfter=20,
            spaceBefore=10,
            alignment=0  # LEFT
        ))
        
        # Заголовок секции
        styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=styles['Heading1'],
            fontName=heading_font,
            fontSize=self.FONT_HEADING,
            textColor=self.COLOR_ACCENT,  # Фиолетовый
            spaceAfter=12,
            spaceBefore=16
        ))
        
        # Подзаголовок
        styles.add(ParagraphStyle(
            name='CustomSubheading',
            parent=styles['Normal'],
            fontName=heading_font,
            fontSize=self.FONT_SUBHEADING,
            textColor=self.COLOR_PRIMARY,
            spaceAfter=10,
            spaceBefore=8
        ))
        
        # Обычный текст (увеличен leading)
        styles.add(ParagraphStyle(
            name='CustomBody',
            parent=styles['Normal'],
            fontName=body_font,
            fontSize=self.FONT_BODY,
            textColor=self.COLOR_PRIMARY,
            leading=16,  # Было 14
            spaceAfter=8
        ))
        
        # Мелкий текст
        styles.add(ParagraphStyle(
            name='CustomSmall',
            parent=styles['Normal'],
            fontName=body_font,
            fontSize=self.FONT_SMALL,
            textColor=self.COLOR_SECONDARY,
            leading=14
        ))
        
        # Текст с отступом (для примеров)
        styles.add(ParagraphStyle(
            name='CustomIndented',
            parent=styles['Normal'],
            fontName=body_font,
            fontSize=self.FONT_SMALL,
            textColor=self.COLOR_SECONDARY,
            leading=14,
            leftIndent=20,
            spaceAfter=8
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
            spaceBefore=12,
            spaceAfter=12
        ))
    
    def generate(self, output_path: str) -> bool:
        """Генерирует PDF отчёт"""
        try:
            logger.info(f"Nachalo generatsii PDF: {output_path}")
            
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=50
            )
            
            story = []
            
            # 1. Executive Summary с метриками
            logger.info("Sozdayom Executive Summary...")
            story.extend(self._create_executive_summary())
            story.append(PageBreak())
            
            # 2. Структура тем (вместо мастер-категорий)
            if self.master_hierarchy:
                logger.info("Sozdayom strukturu tem...")
                story.extend(self._create_themes_structure())
                story.append(PageBreak())
            
            # 3. Графики
            logger.info("Sozdayom grafiki...")
            story.extend(self._create_charts_page())
            story.append(PageBreak())
            
            # 4. Word Cloud
            logger.info("Sozdayom oblako slov...")
            story.extend(self._create_wordcloud_page())
            story.append(PageBreak())
            
            # 5. Топ-5 кластеров (вместо 10)
            logger.info("Sozdayom stranitsy klasterov...")
            story.extend(self._create_clusters_pages())
            story.append(PageBreak())
            
            # 6. CTA страница (без изменений)
            logger.info("Sozdayom CTA stranicu...")
            story.extend(self._create_cta_page())
            
            # Сборка PDF
            logger.info("Sborka PDF...")
            doc.build(story, onFirstPage=footer, onLaterPages=footer)
            
            # Проверка размера
            size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            logger.info(f"Razmer PDF: {size_mb:.2f} MB")
            
            if size_mb > MAX_PDF_SIZE_MB:
                logger.warning(f"PDF slishkom bolshoy: {size_mb:.2f} MB > {MAX_PDF_SIZE_MB} MB")
                Path(output_path).unlink()
                return False
            
            logger.info("PDF uspeshno sozdan")
            return True
            
        except Exception as e:
            logger.error(f"Oshibka generatsii PDF: {e}", exc_info=True)
            return False
    
    def _create_executive_summary(self):
        """Executive Summary с топ-3 темами и метриками"""
        elements = []
        
        # Заголовок
        elements.append(self._create_paragraph(
            "ANALIZ TEKSTOV: GLAVNYYE VYVODY",
            'CustomTitle'
        ))
        elements.append(self._create_divider())
        elements.append(Spacer(1, self.SPACER_SMALL))
        
        # Основные цифры
        n_master = len(self.master_hierarchy) if self.master_hierarchy else 0
        summary_text = (
            f"Proanalizirovano: <b>{self.stats['total_texts']:,}</b> tekstov<br/>"
            f"Naydeno tem: <b>{self.stats['n_clusters']}</b>"
        )
        if n_master > 0:
            summary_text += f" podtem v <b>{n_master}</b> kategoriyakh"
        summary_text += f"<br/>Data analiza: {date_str}"
        
        elements.append(self._create_paragraph(summary_text, 'CustomBody'))
        elements.append(Spacer(1, self.SPACER_MEDIUM))
        elements.append(self._create_divider())
        elements.append(Spacer(1, self.SPACER_MEDIUM))
        
        # Топ-3 темы
        elements.append(self._create_paragraph(
            "TOP-3 TEMY PO OB'YOMU OBRASHCHENIY",
            'CustomSubheading'
        ))
        elements.append(Spacer(1, self.SPACER_SMALL))
        
        top_clusters = self.stats.get('top_clusters', [])[:3]
        for i, cluster in enumerate(top_clusters, 1):
            clean_name = remove_emoji(cluster['name'])
            percent = (cluster['size'] / self.stats['total_texts']) * 100
            
            elements.append(self._create_paragraph(
                f"{i}. <b>{clean_name}</b><br/>"
                f"   {cluster['size']:,} obrashcheniy ({percent:.1f}%)",
                'CustomBody'
            ))
            elements.append(Spacer(1, self.SPACER_SMALL))
        
        elements.append(Spacer(1, self.SPACER_MEDIUM))
        elements.append(self._create_divider())
        elements.append(Spacer(1, self.SPACER_MEDIUM))
        
        # Метрики качества
        if 'quality_metrics' in self.stats:
            qm = self.stats['quality_metrics']
            
            elements.append(self._create_paragraph(
                "KACHESTVO ANALIZA",
                'CustomSubheading'
            ))
            elements.append(Spacer(1, self.SPACER_SMALL))
            
            metrics_text = (
                f"Silhouette Score: <b>{qm['silhouette_score']:.3f}</b> / 1.0<br/>"
                f"Davies-Bouldin Index: <b>{qm['davies_bouldin_index']:.3f}</b><br/><br/>"
                f"<i>Interpretatsiya: Klastery imeyut razmytyye granitsy, "
                f"chto tipichno dlya raznoobraznyh tekstov. Rezultat nadyozhen "
                f"dlya prinyatiya resheniy.</i>"
            )
            
            elements.append(self._create_paragraph(metrics_text, 'CustomSmall'))
        
        elements.append(Spacer(1, self.SPACER_LARGE))
        elements.append(self._create_divider())
        elements.append(Spacer(1, self.SPACER_SMALL))
        
        # Ссылка на детали
        elements.append(self._create_paragraph(
            "Detalnyy analiz na sleduyushchikh stranitsakh",
            'CustomSmall'
        ))
        
        return elements
    
    def _create_themes_structure(self):
        """Структура тем с визуальной иерархией"""
        elements = []
        
        elements.append(self._create_paragraph(
            "STRUKTURA TEM",
            'CustomHeading'
        ))
        elements.append(Spacer(1, self.SPACER_SMALL))
        
        total_clusters = sum(len(sub_clusters) for sub_clusters in self.master_hierarchy.values())
        elements.append(self._create_paragraph(
            f"Teksty sgruppirovany v iyerarkhicheskuyu strukturu: "
            f"{total_clusters} podtem ob'yedineny v {len(self.master_hierarchy)} kategoriy",
            'CustomBody'
        ))
        
        elements.append(Spacer(1, self.SPACER_MEDIUM))
        elements.append(self._create_divider())
        elements.append(Spacer(1, self.SPACER_MEDIUM))
        
        # Обзор категорий с визуальной иерархией
        elements.append(self._create_paragraph(
            "OBZOR KATEGORIY",
            'CustomSubheading'
        ))
        elements.append(Spacer(1, self.SPACER_SMALL))
        
        # Собираем данные мастер-категорий
        master_stats = []
        for master_id, sub_clusters in self.master_hierarchy.items():
            master_name = remove_emoji(self.master_names.get(master_id, f"Kategoriya {master_id}"))
            total_count = sum(len(self.df[self.df['cluster_id'] == cid]) for cid in sub_clusters)
            percent = (total_count / len(self.df)) * 100
            
            master_stats.append({
                'name': master_name,
                'count': total_count,
                'percent': percent,
                'sub_clusters': sub_clusters
            })
        
        # Сортируем по размеру
        master_stats.sort(key=lambda x: x['count'], reverse=True)
        
        # Выводим иерархию текстом
        for master in master_stats:
            # Заголовок категории
            cat_text = f"<b>{master['name']}</b>...{master['percent']:.1f}%"
            elements.append(self._create_paragraph(cat_text, 'CustomBody'))
            
            # Подкластеры с отступом
            for i, cluster_id in enumerate(master['sub_clusters'][:3]):  # Топ-3 подтемы
                cluster_name = remove_emoji(self.cluster_names.get(cluster_id, f"Tema {cluster_id}"))
                if len(cluster_name) > 50:
                    cluster_name = cluster_name[:50] + "..."
                
                prefix = "  ├ " if i < 2 else "  └ "
                elements.append(self._create_paragraph(
                    f"{prefix}{cluster_name}",
                    'CustomSmall'
                ))
            
            if len(master['sub_clusters']) > 3:
                elements.append(self._create_paragraph(
                    f"  ... i eshche {len(master['sub_clusters']) - 3}",
                    'CustomSmall'
                ))
            
            elements.append(Spacer(1, self.SPACER_SMALL))
        
        return elements
    
    def _create_charts_page(self):
        """Страница с графиками"""
        elements = []
        
        elements.append(self._create_paragraph(
            "VIZUALIZATSIYA RASPREDELENIYA",
            'CustomHeading'
        ))
        elements.append(Spacer(1, self.SPACER_SMALL))
        
        elements.append(self._create_paragraph(
            "Grafiki pokazyvayut otnositelnyye razmery krupneyshikh tem.",
            'CustomBody'
        ))
        elements.append(Spacer(1, self.SPACER_MEDIUM))
        
        # Круговая диаграмма
        try:
            pie_img = self._create_pie_chart()
            if pie_img:
                elements.append(pie_img)
                elements.append(Spacer(1, self.SPACER_LARGE))
        except Exception as e:
            logger.error(f"Error creating pie chart: {e}")

        # Столбчатая диаграмма
        try:
            bar_img = self._create_bar_chart()
            if bar_img:
                elements.append(bar_img)
        except Exception as e:
            logger.error(f"Error creating bar chart: {e}")
        
        return elements
    
    def _create_pie_chart(self):
        """Круговая диаграмма топ-10"""
        cluster_dist = self.df['cluster_id'].value_counts().head(10)
        
        labels = [
            remove_emoji(self.cluster_names.get(cid, f"Cluster {cid}"))[:25]
            for cid in cluster_dist.index
        ]
        sizes = cluster_dist.values
        
        fig, ax = plt.subplots(figsize=(8, 6))

        # Фиолетовая палитра
        colors_palette = [
            '#5E35B1', '#7E57C2', '#9575CD', '#B39DDB', '#D1C4E9',
            '#E1BEE7', '#CE93D8', '#BA68C8', '#AB47BC', '#9C27B0'
        ]

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors_palette,
            textprops={'fontsize': 9, 'color': '#263238'}
        )
        
        # Улучшаем читаемость процентов
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.axis('equal')
        plt.title('Top-10 tem po razmeru', fontsize=14, pad=20, color='#263238')
        
        # Сохранение в байты
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        img_buffer.seek(0)
        
        return Image(img_buffer, width=5*inch, height=3.75*inch)
    
    def _create_bar_chart(self):
        """Столбчатая диаграмма топ-10"""
        cluster_dist = self.df['cluster_id'].value_counts().head(10)
        
        labels = [
            remove_emoji(self.cluster_names.get(cid, f"Cluster {cid}"))[:30]
            for cid in cluster_dist.index
        ]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(labels, cluster_dist.values, color='#5E35B1', edgecolor='#4527A0', linewidth=0.5)
        
        ax.set_xlabel('Kolichestvo tekstov', fontsize=11, color='#263238')
        ax.set_title('Top-10 samykh krupnykh tem', fontsize=14, pad=15, color='#263238')
        ax.invert_yaxis()
        ax.tick_params(axis='both', colors='#546E7A', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#E0E0E0')
        ax.spines['bottom'].set_color('#E0E0E0')
        ax.grid(axis='x', color='#EEEEEE', linestyle='-', linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        
        # Добавляем значения на столбцах
        for i, v in enumerate(cluster_dist.values):
            ax.text(v + max(cluster_dist.values) * 0.01, i, str(v), 
                   va='center', fontsize=9, color='#263238')
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        img_buffer.seek(0)
        
        return Image(img_buffer, width=5.5*inch, height=3.5*inch)
    
    def _create_wordcloud_page(self):
        """Страница с облаком слов"""
        elements = []
        
        elements.append(self._create_paragraph(
            "CHASTOTNYY ANALIZ",
            'CustomHeading'
        ))
        elements.append(Spacer(1, self.SPACER_SMALL))
        
        elements.append(self._create_paragraph(
            "Naibolee upotreblyayemyye slova v obrashcheniyakh:",
            'CustomBody'
        ))
        elements.append(Spacer(1, self.SPACER_MEDIUM))
        
        try:
            wc_img = self._create_wordcloud()
            if wc_img:
                elements.append(wc_img)
        except Exception as e:
            logger.error(f"Error creating word cloud: {e}")
            elements.append(self._create_paragraph(
                "Oshibka sozdaniya oblaka slov",
                'CustomSmall'
            ))
        
        return elements
    
    def _create_wordcloud(self):
        """Генерирует облако слов"""
        # Собираем все тексты
        all_texts = " ".join(self.df.iloc[:, 0].astype(str).tolist())
        
        # Простая токенизация
        words = re.findall(r'[а-яёА-ЯЁa-zA-Z]{3,}', all_texts.lower())
        
        # Стоп-слова (базовые)
        stop_words = {
            'это', 'как', 'что', 'для', 'или', 'при', 'все', 'так', 'уже', 
            'был', 'была', 'было', 'были', 'есть', 'быть', 'может', 'можно',
            'добрый', 'день', 'здравствуйте', 'спасибо', 'пожалуйста'
        }
        
        # Фильтруем
        words = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Топ-30
        word_freq = Counter(words).most_common(30)
        word_dict = dict(word_freq)
        
        if not word_dict:
            return None
        
        # Создаём WordCloud
        wc = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='Purples',  # Фиолетовая палитра
            font_path=str(FONT_PATH),
            relative_scaling=0.5,
            min_font_size=10
        ).generate_from_frequencies(word_dict)
        
        # Рисуем
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        
        # Сохранение
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        img_buffer.seek(0)
        
        return Image(img_buffer, width=6*inch, height=3*inch)
    
    def _create_clusters_pages(self):
        """Страницы с топ-5 кластерами"""
        elements = []
        
        cluster_dist = self.df['cluster_id'].value_counts().head(5)  # Топ-5 вместо 10
        
        for idx, (cluster_id, count) in enumerate(cluster_dist.items()):
            if cluster_id == -1:
                continue
            
            # Каждые 2 кластера — новая страница
            if idx > 0 and idx % 2 == 0:
                elements.append(PageBreak())
            
            cluster_name = remove_emoji(self.cluster_names.get(cluster_id, f"Tema {cluster_id}"))
            percent = (count / len(self.df)) * 100
            
            # Определяем категорию
            master_category = ""
            if self.master_hierarchy:
                for master_id, sub_clusters in self.master_hierarchy.items():
                    if cluster_id in sub_clusters:
                        master_category = remove_emoji(self.master_names.get(master_id, ""))
                        break
            
            # Заголовок (БЕЗ "Кластер N:")
            elements.append(self._create_paragraph(
                cluster_name.upper(),
                'CustomHeading'
            ))
            elements.append(Spacer(1, self.SPACER_SMALL))
            
            # Статистика
            stats_text = f"Ob'yom: <b>{count:,}</b> tekstov ({percent:.1f}% ot obshchego)"
            if master_category:
                stats_text += f"<br/>Kategoriya: {master_category}"
            
            elements.append(self._create_paragraph(stats_text, 'CustomBody'))
            elements.append(Spacer(1, self.SPACER_SMALL))
            elements.append(self._create_divider())
            elements.append(Spacer(1, self.SPACER_SMALL))
            
            # Основной паттерн (если можем определить)
            elements.append(self._create_paragraph(
                "O CHYOM PISHUT:",
                'CustomSubheading'
            ))
            elements.append(Spacer(1, self.SPACER_SMALL))
            
            # Примеры (4 вместо 6, без нумерации)
            cluster_texts = self.df[self.df['cluster_id'] == cluster_id].iloc[:, 0].head(4).tolist()
            
            elements.append(self._create_paragraph(
                "Tipichnyye zaprosy:",
                'CustomBody'
            ))
            elements.append(Spacer(1, self.SPACER_SMALL))
            
            for text in cluster_texts:
                # Обрезаем и экранируем
                text_preview = str(text)[:200] + "..." if len(str(text)) > 200 else str(text)
                text_preview = text_preview.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
                
                # Без нумерации, с кавычками и отступом
                elements.append(self._create_paragraph(
                    f'"{text_preview}"',
                    'CustomIndented'
                ))
            
            elements.append(Spacer(1, self.SPACER_LARGE))
        
        return elements
    
    def _create_cta_page(self):
        """Финальная страница (без изменений)"""
        elements = []
        
        # Заголовок
        elements.append(self._create_paragraph(
            "@cluster_master_bot",
            'CustomTitle'
        ))
        elements.append(Spacer(1, self.SPACER_SMALL))
        
        # Подзаголовок
        elements.append(self._create_paragraph(
            "Etot otchyot sozdan avtomaticheski za neskolko minut s pomoshchyu @cluster_master_bot",
            'CustomBody'
        ))
        elements.append(Spacer(1, self.SPACER_LARGE))
        
        # Возможности бота
        elements.append(self._create_paragraph(
            "Vozmozhnosti bota:",
            'CustomSubheading'
        ))
        elements.append(Spacer(1, self.SPACER_SMALL))
        
        features = [
            "Analiz do 50,000 tekstov za minuty",
            "Avtomaticheskaya klasterizatsiya (BERTopic + HDBSCAN)",
            "Generatsiya nazvaniy klasterov cherez AI (YandexGPT)",
            "Eksport rezultatov v CSV i PDF",
            "Iyerarkhicheskaya struktura (master-kategorii)",
            "Metriki kachestva klasterizatsii"
        ]
        
        for feature in features:
            elements.append(self._create_paragraph(feature, 'CustomBody'))
        
        elements.append(Spacer(1, self.SPACER_MEDIUM))
        
        # Использование бота
        elements.append(self._create_paragraph(
            "Ispolzuyte dlya:",
            'CustomSubheading'
        ))
        elements.append(Spacer(1, self.SPACER_SMALL))
        
        use_cases = [
            "Analiza otzyvov i obrashcheniy klientov",
            "Obrabotki tiketov sluzhby podderzhki",
            "Issledovaniya rezultatov oprosov",
            "Prioritizatsii product roadmap",
            "Vyyavleniya trendov i problem produkta"
        ]
        
        for use_case in use_cases:
            elements.append(self._create_paragraph(use_case, 'CustomBody'))
        
        elements.append(Spacer(1, self.SPACER_LARGE))
        
        # Призыв к действию
        elements.append(self._create_paragraph(
            "Nachat: t.me/cluster_master_bot",
            'CustomHeading'
        ))
        elements.append(Spacer(1, self.SPACER_SMALL))
        
        elements.append(self._create_paragraph(
            "Besplatno | Bez registratsii",
            'CustomBody'
        ))
        
        elements.append(Spacer(1, self.SPACER_LARGE))
        elements.append(self._create_divider())
        
        # Футер
        elements.append(self._create_paragraph(
            f"Sozdano s pomoshchyu @cluster_master_bot | v0.3.0 | {date_str}",
            'CustomSmall'
        ))
        
        return elements
