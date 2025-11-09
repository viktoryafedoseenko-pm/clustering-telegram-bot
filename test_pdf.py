# test_pdf.py
import pandas as pd
from pdf_generator import PDFReportGenerator

def test_pdf():
    # Тестовые данные
    df = pd.DataFrame({
        'text': ['Тест 1', 'Тест 2', 'Тест 3'] * 10,
        'cluster_id': [0, 1, 2] * 10,
        'cluster_name': ['Оплата', 'Диплом', 'Технические ошибки'] * 10
    })
    
    stats = {
        'total_texts': 30,
        'n_clusters': 3,
        'avg_cluster_size': 10,
        'noise_percent': 0
    }
    
    cluster_names = {0: 'Оплата', 1: 'Диплом', 2: 'Технические ошибки'}
    
    generator = PDFReportGenerator(df, stats, cluster_names)
    success = generator.generate('/tmp/test_report.pdf')
    
    assert success, "PDF generation failed"
    print("✅ PDF created: /tmp/test_report.pdf")

if __name__ == '__main__':
    test_pdf()
