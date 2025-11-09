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
    Table, TableStyle, PageBreak
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from config import FONT_PATH, MAX_PDF_SIZE_MB

logger = logging.getLogger(__name__)

# Регистрация шрифта
pdfmetrics.registerFont(TTFont('DejaVuSans', str(FONT_PATH)))

# Настройка matplotlib для кириллицы
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

class PDFReportGenerator:
    """Генератор PDF отчётов по кластеризации"""
    
    def __init__(self, df: pd.DataFrame, stats: dict, cluster_names: dict):
        self.df = df
        self.stats = stats
        self.cluster_names = cluster_names
        self.styles = self._setup_styles()
    
    def _setup_styles(self):
        """Настройка стилей с кириллицей"""
        styles = getSampleStyleSheet()
        
        # Заголовок
        styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=styles['Title'],
            fontName='DejaVuSans',
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=1  # CENTER
        ))
        
        # Подзаголовок
        styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=styles['Heading1'],
            fontName='DejaVuSans',
            fontSize=16,
            textColor=colors.HexColor('#2ca02c'),
            spaceAfter=12
        ))
        
        # Обычный текст
        styles.add(ParagraphStyle(
            name='CustomBody',
            parent=styles['Normal'],
            fontName='DejaVuSans',
            fontSize=10,
            leading=14
        ))
        
        return styles
    
    def generate(self, output_path: str) -> bool:
        """
        Генерирует PDF отчёт
        
        Returns:
            bool: True если успешно, False если превышен лимит размера
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            logger.info(f"📄 Starting PDF generation: {output_path}")
            
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
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
            doc.build(story)
            
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
        title = Paragraph(
            "Отчёт по кластеризации текстов",
            self.styles['CustomTitle']
        )
        elements.append(title)
        elements.append(Spacer(1, 0.5*inch))
        
        # Общая статистика
        stats_data = [
            ["Всего текстов:", f"{self.stats['total_texts']}"],
            ["Найдено кластеров:", f"{self.stats['n_clusters']}"],
            ["Средний размер:", f"{self.stats['avg_cluster_size']:.0f} текстов"],
            ["Шум (прочее):", f"{self.stats['noise_percent']:.1f}%"],
        ]
        
        table = Table(stats_data, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), 'DejaVuSans', 12),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.grey),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.black),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        
        elements.append(table)
        
        return elements
    
    def _create_statistics_page(self):
        """Страница со статистикой"""
        elements = []
        
        heading = Paragraph("Распределение кластеров", self.styles['CustomHeading'])
        elements.append(heading)
        elements.append(Spacer(1, 0.2*inch))
        
        # Таблица распределения
        cluster_dist = self.df['cluster_id'].value_counts().sort_values(ascending=False)
        
        table_data = [["Кластер", "Название", "Количество", "%"]]
        
        for cluster_id, count in cluster_dist.head(15).items():
            name = self.cluster_names.get(cluster_id, f"Кластер {cluster_id}")
            percent = (count / len(self.df)) * 100
            
            table_data.append([
                str(cluster_id),
                name[:40],  # Обрезаем длинные названия
                str(count),
                f"{percent:.1f}%"
            ])
        
        table = Table(table_data, colWidths=[0.7*inch, 3*inch, 1*inch, 0.8*inch])
        table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), 'DejaVuSans', 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'DejaVuSans'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
        ]))
        
        elements.append(table)
        
        return elements
    
    def _create_charts_page(self):
        """Страница с графиками"""
        elements = []
        
        heading = Paragraph("Визуализация", self.styles['CustomHeading'])
        elements.append(heading)
        elements.append(Spacer(1, 0.2*inch))
        
        # Круговая диаграмма
        pie_img = self._create_pie_chart()
        if pie_img:
            elements.append(pie_img)
            elements.append(Spacer(1, 0.3*inch))
        
        # Столбчатая диаграмма
        bar_img = self._create_bar_chart()
        if bar_img:
            elements.append(bar_img)
        
        return elements
    
    def _create_pie_chart(self):
        """Круговая диаграмма топ-10"""
        cluster_dist = self.df['cluster_id'].value_counts().head(10)
        
        labels = [
            self.cluster_names.get(cid, f"Кластер {cid}")[:20]
            for cid in cluster_dist.index
        ]
        sizes = cluster_dist.values
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=plt.cm.Set3.colors
        )
        ax.axis('equal')
        plt.title('Топ-10 кластеров по размеру', fontsize=14, pad=20)
        
        # Сохранение в байты
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        plt.close()
        img_buffer.seek(0)
        
        return Image(img_buffer, width=5*inch, height=3.75*inch)
    
    def _create_bar_chart(self):
        """Столбчатая диаграмма топ-10"""
        cluster_dist = self.df['cluster_id'].value_counts().head(10)
        
        labels = [
            self.cluster_names.get(cid, f"Кластер {cid}")[:25]
            for cid in cluster_dist.index
        ]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(labels, cluster_dist.values, color='steelblue')
        ax.set_xlabel('Количество текстов', fontsize=11)
        ax.set_title('Топ-10 самых крупных кластеров', fontsize=14, pad=15)
        ax.invert_yaxis()
        
        # Добавляем значения на столбцах
        for i, v in enumerate(cluster_dist.values):
            ax.text(v + 1, i, str(v), va='center', fontsize=9)
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
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
            heading = Paragraph(
                f"🏷️ {cluster_name}",
                self.styles['CustomHeading']
            )
            elements.append(heading)
            
            # Статистика
            stats_text = f"<b>Размер:</b> {count} текстов ({percent:.1f}%)"
            elements.append(Paragraph(stats_text, self.styles['CustomBody']))
            elements.append(Spacer(1, 0.1*inch))
            
            # Примеры текстов (5-7 штук)
            cluster_texts = self.df[self.df['cluster_id'] == cluster_id].iloc[:, 0].head(7).tolist()
            
            examples_heading = Paragraph("<b>Примеры текстов:</b>", self.styles['CustomBody'])
            elements.append(examples_heading)
            elements.append(Spacer(1, 0.05*inch))
            
            for i, text in enumerate(cluster_texts, 1):
                # Обрезаем длинные тексты
                text_preview = str(text)[:200] + "..." if len(str(text)) > 200 else str(text)
                text_preview = text_preview.replace('<', '&lt;').replace('>', '&gt;')
                
                example = Paragraph(
                    f"{i}. {text_preview}",
                    self.styles['CustomBody']
                )
                elements.append(example)
                elements.append(Spacer(1, 0.05*inch))
            
            elements.append(Spacer(1, 0.3*inch))
        
        return elements
