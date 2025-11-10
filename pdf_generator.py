# pdf_generator.py
import io
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

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('DejaVuSans', 8)
    canvas.setFillColor(colors.HexColor('#666666'))
    canvas.drawString(
        inch, 0.5 * inch,
        f"Страница {doc.page} | Отчёт по кластеризации"
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
    
    def __init__(self, df: pd.DataFrame, stats: dict, cluster_names: dict):
        self.df = df
        self.stats = stats
        self.cluster_names = cluster_names
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
            
            # 2. Статистика
            logger.info("📊 Creating statistics page...")
            story.extend(self._create_statistics_page())
            story.append(PageBreak())
            
            # 3. Графики
            logger.info("📈 Creating charts...")
            story.extend(self._create_charts_page())
            story.append(PageBreak())
            
            # 4. Топ-10 кластеров
            logger.info("🏷️ Creating cluster pages...")
            story.extend(self._create_clusters_pages())
            
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
        
        elements.append(self._create_paragraph(
            "Анализ текстовых данных с использованием алгоритма кластеризации. "
            "Документ содержит статистику, визуализации и детальное описание выявленных групп.",
            'CustomBody'
        ))
        
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
    
    def _create_statistics_page(self):
        """Страница со статистикой"""
        elements = []
        
        # Заголовок секции
        elements.append(self._create_paragraph(
            "Распределение кластеров",
            'CustomHeading'
        ))
        elements.append(Spacer(1, self.SPACER_SMALL))
        
        elements.append(self._create_paragraph(
            "В таблице представлены крупнейшие кластеры, "
            "упорядоченные по количеству текстов.",
            'CustomBody'
        ))
        elements.append(Spacer(1, self.SPACER_MEDIUM))
        
        # Таблица распределения
        cluster_dist = self.df['cluster_id'].value_counts().sort_values(ascending=False)
        
        table_data = [["ID", "Название кластера", "Количество", "Доля"]]
        
        for cluster_id, count in cluster_dist.head(15).items():
            name = self.cluster_names.get(cluster_id, f"Кластер {cluster_id}")
            percent = (count / len(self.df)) * 100
            
            table_data.append([
                str(cluster_id),
                name[:45],
                str(count),
                f"{percent:.1f}%"
            ])
        
        table = Table(table_data, colWidths=[0.6*inch, 3.2*inch, 1*inch, 0.8*inch])
        table.setStyle(TableStyle([
            # Заголовок
            ('BACKGROUND', (0, 0), (-1, 0), self.COLOR_ACCENT),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'DejaVuSans'),
            ('FONTSIZE', (0, 0), (-1, 0), self.FONT_BODY),
            ('FONTNAME', (0, 1), (-1, -1), 'DejaVuSans'),
            ('FONTSIZE', (0, 1), (-1, -1), self.FONT_SMALL),
            
            # Выравнивание
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('ALIGN', (2, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            
            # Чередующиеся строки
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.COLOR_BACKGROUND]),
            
            # Рамки
            ('LINEBELOW', (0, 0), (-1, 0), 1, self.COLOR_ACCENT),
            ('LINEBELOW', (0, 1), (-1, -1), 0.5, self.COLOR_DIVIDER),
            ('BOX', (0, 0), (-1, -1), 0.5, self.COLOR_DIVIDER),
            
            # Отступы
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        elements.append(table)
        
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
            self.cluster_names.get(cid, f"Кластер {cid}")[:25]
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
            self.cluster_names.get(cid, f"Кластер {cid}")[:30]
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
            percent = (count / len(self.df)) * 100
            
            # Заголовок кластера
            elements.append(self._create_paragraph(
                f"Кластер {cluster_id}: {cluster_name}",
                'CustomHeading'
            ))
            elements.append(Spacer(1, self.SPACER_SMALL))
            
            # Статистика
            stats_text = f"Размер: {count} текстов ({percent:.1f}% от общего объёма)"
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
                text_preview = str(text)[:180] + "..." if len(str(text)) > 180 else str(text)
                text_preview = text_preview.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
                
                elements.append(self._create_paragraph(
                    f"{i}. {text_preview}",
                    'CustomSmall'
                ))
                elements.append(Spacer(1, self.SPACER_SMALL))
            
            elements.append(Spacer(1, self.SPACER_MEDIUM))
        
        return elements
