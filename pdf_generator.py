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
    # Удаляем другие специальные символы, которые могут быть проблемными
    text = text.replace('•', '').replace('■', '').replace('→', '').replace('←', '')
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
        f"Страница {doc.page} | Отчёт по кластеризации"
    )
    # Правый футер - ссылка на бота
    canvas.drawRightString(
        A4[0] - inch, 0.5 * inch,
        "Создано: @cluster_master_bot"
    )
    canvas.restoreState()

class PDFReportGenerator:
    """Генератор PDF отчётов по кластеризации"""
    
    # Цветовая палитра Tableau-style
    COLOR_PRIMARY = colors.HexColor('#222222')      # Основной текст
    COLOR_SECONDARY = colors.HexColor('#666666')    # Второстепенный текст
    COLOR_ACCENT = colors.HexColor('#007ACC')       # Акцентный цвет
    COLOR_DIVIDER = colors.HexColor('#DDDDDD')      # Разделители
    COLOR_BACKGROUND = colors.HexColor('#F8F8F8')   # Легкий фон
    COLOR_MASTER_CAT = colors.HexColor('#2E7D32')   # Цвет для мастер-категорий
    
    # Размеры шрифтов
    FONT_TITLE = 18
    FONT_HEADING = 14
    FONT_SUBHEADING = 12
    FONT_BODY = 10
    FONT_SMALL = 9
    
    # Отступы
    SPACER_LARGE = 0.4 * inch
    SPACER_MEDIUM = 0.2 * inch
    SPACER_SMALL = 0.1 * inch
    
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
            textColor=self.COLOR_PRIMARY,
            spaceAfter=10,
            spaceBefore=12
        ))
        
        # Заголовок мастер-категории
        styles.add(ParagraphStyle(
            name='MasterCategory',
            parent=styles['Heading1'],
            fontName=heading_font,
            fontSize=self.FONT_HEADING,
            textColor=self.COLOR_MASTER_CAT,
            spaceAfter=8,
            spaceBefore=16,
            leftIndent=10
        ))
        
        # Подзаголовок
        styles.add(ParagraphStyle(
            name='CustomSubheading',
            parent=styles['Normal'],
            fontName=heading_font,
            fontSize=self.FONT_SUBHEADING,
            textColor=self.COLOR_SECONDARY,
            spaceAfter=8
        ))
        
        # Обычный текст
        styles.add(ParagraphStyle(
            name='CustomBody',
            parent=styles['Normal'],
            fontName=body_font,
            fontSize=self.FONT_BODY,
            textColor=self.COLOR_PRIMARY,
            leading=14,
            spaceAfter=6
        ))
        
        # Мелкий текст
        styles.add(ParagraphStyle(
            name='CustomSmall',
            parent=styles['Normal'],
            fontName=body_font,
            fontSize=self.FONT_SMALL,
            textColor=self.COLOR_SECONDARY,
            leading=12
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
            spaceBefore=8,
            spaceAfter=8
        )
    
    def generate(self, output_path: str) -> bool:
        """
        Генерирует PDF отчёт
        
        Returns:
            bool: True если успешно, False если превышен лимит размера
        """
        try:
            logger.info(f"📄 Starting PDF generation: {output_path}")
            
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=50
            )
            
            story = []
            
            # 1. Титульная страница
            logger.info("📝 Creating title page...")
            story.extend(self._create_title_page())
            story.append(PageBreak())
            
            # 2. Мастер-категории (если есть)
            if self.master_hierarchy:
                logger.info("🏷️ Creating master categories page...")
                story.extend(self._create_master_categories_page())
                story.append(PageBreak())
            
            # 3. Графики
            logger.info("📈 Creating charts...")
            story.extend(self._create_charts_page())
            story.append(PageBreak())
            
            # 4. Топ-10 кластеров
            logger.info("🏷️ Creating cluster pages...")
            story.extend(self._create_clusters_pages())
            story.append(PageBreak())
            
            # 5. CTA страница
            logger.info("🚀 Creating CTA page...")
            story.extend(self._create_cta_page())
            
            # Сборка PDF
            logger.info("🔨 Building PDF...")
            doc.build(story, onFirstPage=footer, onLaterPages=footer)
            
            # Проверка размера
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
        
    def _create_title_page(self):
        """Титульная страница"""
        elements = []
        
        # Заголовок
        elements.append(self._create_paragraph(
            "Отчёт по кластеризации текстов",
            'CustomTitle'
        ))
        elements.append(Spacer(1, self.SPACER_SMALL))
        
        # Дата и описание
        elements.append(self._create_paragraph(
            f"Дата создания: {date_str}",
            'CustomSmall'
        ))
        elements.append(Spacer(1, self.SPACER_SMALL))
        
        description = "Анализ текстовых данных с использованием алгоритма кластеризации. "
        if self.master_hierarchy:
            description += "Включает иерархическую структуру мастер-категорий. "
        description += "Сгенерировано с помощью @cluster_master_bot"
        
        elements.append(self._create_paragraph(description, 'CustomBody'))
        
        elements.append(Spacer(1, self.SPACER_MEDIUM))
        elements.append(self._create_divider(width="80%", thickness=1))
        elements.append(Spacer(1, self.SPACER_MEDIUM))
        
        # Подзаголовок
        elements.append(self._create_paragraph(
            "Основные метрики",
            'CustomSubheading'
        ))
        elements.append(Spacer(1, self.SPACER_SMALL))
        
        # Таблица метрик
        stats_data = [
            ["Всего текстов", f"{self.stats['total_texts']}"],
            ["Найдено кластеров", f"{self.stats['n_clusters']}"],
            ["Средний размер кластера", f"{self.stats['avg_cluster_size']:.0f} текстов"],
            ["Шум (не кластеризовано)", f"{self.stats['noise_percent']:.1f}%"],
        ]

        if self.master_hierarchy:
            stats_data.insert(1, ["Мастер-категорий", f"{len(self.master_hierarchy)}"])

        if 'quality_metrics' in self.stats:
            qm = self.stats['quality_metrics']
            stats_data.extend([
                ["", ""],
                ["Метрики качества", ""],
                ["  Silhouette Score", f"{qm['silhouette_score']:.3f}"],
                ["  Davies-Bouldin Index", f"{qm['davies_bouldin_index']:.3f}"],
            ])
            
        table = Table(stats_data, colWidths=[3.2*inch, 2*inch])
        table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), 'DejaVuSans', self.FONT_BODY),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TEXTCOLOR', (0, 0), (0, -1), self.COLOR_SECONDARY),
            ('TEXTCOLOR', (1, 0), (1, -1), self.COLOR_PRIMARY),
            ('LINEBELOW', (0, 0), (-1, 0), 0.5, self.COLOR_DIVIDER),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(table)
        
        return elements
    
    def _create_master_categories_page(self):
        """Страница с мастер-категориями - раздельные таблицы"""
        elements = []
        
        elements.append(self._create_paragraph(
            "Мастер-категории",
            'CustomHeading'
        ))
        elements.append(Spacer(1, self.SPACER_SMALL))
        
        elements.append(self._create_paragraph(
            "Иерархическая группировка кластеров в тематические категории, "
            "сгенерированные с помощью LLM.",
            'CustomBody'
        ))
        elements.append(Spacer(1, self.SPACER_MEDIUM))
        
        # ==========================================================================
        # 1. ТАБЛИЦА МАСТЕР-КАТЕГОРИЙ (только названия и доли)
        # ==========================================================================
        
        elements.append(self._create_paragraph(
            "Обзор мастер-категорий",
            'CustomSubheading'
        ))
        elements.append(Spacer(1, self.SPACER_SMALL))
        
        # Собираем данные для мастер-категорий
        master_stats = []
        for master_id, sub_clusters in self.master_hierarchy.items():
            master_name = self.master_names.get(master_id, f"Категория {master_id}")
            # Удаляем эмодзи из названия мастер-категории
            master_name = remove_emoji(master_name)
            total_count = sum(len(self.df[self.df['cluster_id'] == cid]) for cid in sub_clusters)
            percent = (total_count / len(self.df)) * 100
            n_clusters = len(sub_clusters)
            
            master_stats.append({
                'name': master_name,
                'count': total_count,
                'percent': percent,
                'n_clusters': n_clusters,
                'master_id': master_id
            })
        
        # Сортируем по размеру
        master_stats.sort(key=lambda x: x['count'], reverse=True)
        
        # Создаём таблицу мастер-категорий
        master_table_data = [["Мастер-категория", "Кластеров", "Текстов", "Доля"]]
        
        for master in master_stats:
            master_table_data.append([
                master['name'],
                str(master['n_clusters']),
                str(master['count']),
                f"{master['percent']:.1f}%"
            ])
        
        master_table = Table(master_table_data, colWidths=[3.5*inch, 0.8*inch, 0.8*inch, 0.8*inch])
        master_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'DejaVuSans'),
            ('FONTSIZE', (0, 0), (-1, -1), self.FONT_SMALL),
            ('BACKGROUND', (0, 0), (-1, 0), self.COLOR_MASTER_CAT),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.COLOR_BACKGROUND]),
            ('LINEBELOW', (0, 0), (-1, 0), 1, self.COLOR_MASTER_CAT),
            ('LINEBELOW', (0, 1), (-1, -1), 0.5, self.COLOR_DIVIDER),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ]))
        
        elements.append(master_table)
        elements.append(Spacer(1, self.SPACER_LARGE))
        
        # ==========================================================================
        # 2. ТАБЛИЦЫ КЛАСТЕРОВ ПО КАТЕГОРИЯМ (отдельно для каждой мастер-категории)
        # ==========================================================================
        
        elements.append(self._create_paragraph(
            "Распределение кластеров по категориям",
            'CustomSubheading'
        ))
        elements.append(Spacer(1, self.SPACER_SMALL))
        
        # Создаём таблицы для каждой мастер-категории
        for master in master_stats:
            master_id = master['master_id']
            master_name = master['name']
            
            # Заголовок категории (без эмодзи)
            elements.append(self._create_paragraph(
                master_name,
                'CustomBody'
            ))
            elements.append(Spacer(1, self.SPACER_SMALL))
            
            # Собираем кластеры этой категории
            cluster_data = []
            sub_clusters = self.master_hierarchy[master_id]
            
            for cluster_id in sub_clusters:
                cluster_count = len(self.df[self.df['cluster_id'] == cluster_id])
                cluster_name = self.cluster_names.get(cluster_id, f"Кластер {cluster_id}")
                percent = (cluster_count / len(self.df)) * 100
                
                # Очищаем название от эмодзи и специальных символов
                clean_name = remove_emoji(cluster_name)
                if len(clean_name) > 60:
                    clean_name = clean_name[:60] + "..."
                
                cluster_data.append({
                    'name': clean_name,
                    'count': cluster_count,
                    'percent': percent
                })
            
            # Сортируем кластеры по размеру
            cluster_data.sort(key=lambda x: x['count'], reverse=True)
            
            # Создаём таблицу для этой категории
            cluster_table_data = [["Кластер", "Текстов", "Доля"]]
            
            for cluster in cluster_data:
                cluster_table_data.append([
                    cluster['name'],
                    str(cluster['count']),
                    f"{cluster['percent']:.1f}%"
                ])
            
            cluster_table = Table(cluster_table_data, colWidths=[4.0*inch, 0.8*inch, 0.8*inch])
            cluster_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'DejaVuSans'),
                ('FONTSIZE', (0, 0), (-1, -1), self.FONT_SMALL),
                ('BACKGROUND', (0, 0), (-1, 0), self.COLOR_ACCENT),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F8F8')]),
                ('LINEBELOW', (0, 0), (-1, 0), 1, self.COLOR_ACCENT),
                ('LINEBELOW', (0, 1), (-1, -1), 0.5, self.COLOR_DIVIDER),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ]))
            
            elements.append(cluster_table)
            elements.append(Spacer(1, self.SPACER_MEDIUM))
        
        # ==========================================================================
        # 3. ИТОГОВАЯ СТАТИСТИКА
        # ==========================================================================
        
        elements.append(self._create_divider())
        elements.append(Spacer(1, self.SPACER_SMALL))
        
        total_clusters = sum(len(sub_clusters) for sub_clusters in self.master_hierarchy.values())
        total_texts = len(self.df)
        
        elements.append(self._create_paragraph(
            f"Итог: {len(self.master_hierarchy)} мастер-категорий, "
            f"{total_clusters} кластеров, {total_texts} текстов",
            'CustomSmall'
        ))
        
        return elements
    
    
    def _create_charts_page(self):
        """Страница с графиками"""
        elements = []
        
        elements.append(self._create_paragraph(
            "Визуализация распределения",
            'CustomHeading'
        ))
        elements.append(Spacer(1, self.SPACER_SMALL))
        
        elements.append(self._create_paragraph(
            "Графики показывают относительные размеры крупнейших кластеров.",
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
            remove_emoji(self.cluster_names.get(cid, f"Кластер {cid}"))[:25]
            for cid in cluster_dist.index
        ]
        sizes = cluster_dist.values
        
        fig, ax = plt.subplots(figsize=(8, 6))

        # Сдержанная палитра
        colors_palette = [
            '#007ACC', '#5B9BD5', '#70AD47', '#FFC000', '#C55A11',
            '#44546A', '#7030A0', '#00B0F0', '#92D050', '#A6A6A6'
        ]

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors_palette,
            textprops={'fontsize': 9, 'color': '#222222'}
        )
        
        # Улучшаем читаемость процентов
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.axis('equal')
        plt.title('Топ-10 кластеров по размеру', fontsize=14, pad=20, color='#222222')
        
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
            remove_emoji(self.cluster_names.get(cid, f"Кластер {cid}"))[:30]
            for cid in cluster_dist.index
        ]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(labels, cluster_dist.values, color='#007ACC', edgecolor='#005A9E', linewidth=0.5)
        
        ax.set_xlabel('Количество текстов', fontsize=11, color='#222222')
        ax.set_title('Топ-10 самых крупных кластеров', fontsize=14, pad=15, color='#222222')
        ax.invert_yaxis()
        ax.tick_params(axis='both', colors='#666666', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#DDDDDD')
        ax.spines['bottom'].set_color('#DDDDDD')
        ax.grid(axis='x', color='#EEEEEE', linestyle='-', linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        
        # Добавляем значения на столбцах
        for i, v in enumerate(cluster_dist.values):
            ax.text(v + max(cluster_dist.values) * 0.01, i, str(v), 
                   va='center', fontsize=9, color='#222222')
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        img_buffer.seek(0)
        
        return Image(img_buffer, width=5.5*inch, height=3.5*inch)
    
    def _create_clusters_pages(self):
        """Страницы с топ-10 кластерами (по 2 на страницу)"""
        elements = []
        
        cluster_dist = self.df['cluster_id'].value_counts().head(10)
        
        for idx, (cluster_id, count) in enumerate(cluster_dist.items()):
            if cluster_id == -1:
                continue
            
            # Каждые 2 кластера — новая страница
            if idx > 0 and idx % 2 == 0:
                elements.append(PageBreak())
            
            cluster_name = self.cluster_names.get(cluster_id, f"Кластер {cluster_id}")
            # Удаляем эмодзи из названия кластера
            cluster_name = remove_emoji(cluster_name)
            percent = (count / len(self.df)) * 100
            
            # Определяем мастер-категорию (если есть)
            master_category = ""
            if self.master_hierarchy:
                for master_id, sub_clusters in self.master_hierarchy.items():
                    if cluster_id in sub_clusters:
                        master_category = remove_emoji(self.master_names.get(master_id, f"Категория {master_id}"))
                        break
            
            # Заголовок кластера
            elements.append(self._create_paragraph(
                f"Кластер {cluster_id}: {cluster_name}",
                'CustomHeading'
            ))
            elements.append(Spacer(1, self.SPACER_SMALL))
            
            # Статистика с мастер-категорией
            stats_text = f"Размер: {count} текстов ({percent:.1f}% от общего объёма)"
            if master_category:
                stats_text += f"<br/>Мастер-категория: {master_category}"
            
            elements.append(self._create_paragraph(stats_text, 'CustomBody'))
            
            elements.append(Spacer(1, self.SPACER_SMALL))
            elements.append(self._create_divider())
            elements.append(Spacer(1, self.SPACER_SMALL))
            
            # Примеры текстов
            elements.append(self._create_paragraph(
                "Примеры текстов из кластера:",
                'CustomSubheading'
            ))
            elements.append(Spacer(1, self.SPACER_SMALL))
            
            cluster_texts = self.df[self.df['cluster_id'] == cluster_id].iloc[:, 0].head(6).tolist()
            
            for i, text in enumerate(cluster_texts, 1):
                # Обрезаем и экранируем
                text_preview = str(text)[:250] + "..." if len(str(text)) > 250 else str(text)
                text_preview = text_preview.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
                
                elements.append(self._create_paragraph(
                    f"{i}. {text_preview}",
                    'CustomSmall'
                ))
                elements.append(Spacer(1, self.SPACER_SMALL))
            
            elements.append(Spacer(1, self.SPACER_MEDIUM))
        
        return elements
    
    def _create_cta_page(self):
        """Финальная страница с призывом к действию"""
        elements = []
        
        # Заголовок
        elements.append(self._create_paragraph(
            "@cluster_master_bot",
            'CustomTitle'
        ))
        elements.append(Spacer(1, self.SPACER_SMALL))
        
        # Подзаголовок
        elements.append(self._create_paragraph(
            "Этот отчёт создан автоматически за несколько минут с помощью @cluster_master_bot",
            'CustomBody'
        ))
        elements.append(Spacer(1, self.SPACER_LARGE))
        
        # Возможности бота
        elements.append(self._create_paragraph(
            "Возможности бота:",
            'CustomSubheading'
        ))
        elements.append(Spacer(1, self.SPACER_SMALL))
        
        features = [
            "Анализ до 50,000 текстов за минуты",
            "Автоматическая кластеризация (BERTopic + HDBSCAN)",
            "Генерация названий кластеров через AI (YandexGPT)",
            "Экспорт результатов в CSV и PDF",
            "Иерархическая структура (мастер-категории)",
            "Метрики качества кластеризации"
        ]
        
        for feature in features:
            elements.append(self._create_paragraph(feature, 'CustomBody'))
        
        elements.append(Spacer(1, self.SPACER_MEDIUM))
        
        # Использование бота
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
            "Выявления трендов и проблем продукта"
        ]
        
        for use_case in use_cases:
            elements.append(self._create_paragraph(use_case, 'CustomBody'))
        
        elements.append(Spacer(1, self.SPACER_LARGE))
        
        # Призыв к действию
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
        
        # Футер
        elements.append(self._create_paragraph(
            f"Создано с помощью @cluster_master_bot | v0.3.0 | {date_str}",
            'CustomSmall'
        ))
        
        return elements