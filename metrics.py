# metrics.py
"""
Модуль для оценки качества кластеризации
"""
import numpy as np
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score,
    calinski_harabasz_score
)
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class ClusteringMetrics:
    """Вычисление и интерпретация метрик качества кластеризации"""
    
    @staticmethod
    def calculate(embeddings, labels) -> Dict[str, float]:
        """
        Вычисляет все метрики качества
        
        Args:
            embeddings: векторные представления текстов (N x D)
            labels: метки кластеров для каждого текста (N,)
        
        Returns:
            dict с метриками
        """
        # Преобразуем в numpy arrays (на всякий случай)
        embeddings = np.asarray(embeddings)
        labels = np.asarray(labels)
        
        # Убираем шум (кластер -1) для метрик
        mask = labels != -1
        embeddings_clean = embeddings[mask]
        labels_clean = labels[mask]
        
        # Проверка: достаточно ли данных
        unique_labels = np.unique(labels_clean)
        
        if len(unique_labels) < 2:
            logger.warning("⚠️ Недостаточно кластеров для расчёта метрик")
            noise_count = np.count_nonzero(labels == -1)
            noise_ratio = (noise_count / len(labels) * 100) if len(labels) > 0 else 0.0
            
            return {
                'silhouette_score': 0.0,
                'davies_bouldin_index': 0.0,
                'calinski_harabasz_score': 0.0,
                'noise_ratio': round(noise_ratio, 2)
            }
        
        try:
            # 1. Silhouette Score
            silhouette = silhouette_score(embeddings_clean, labels_clean)
            
            # 2. Davies-Bouldin Index
            db_index = davies_bouldin_score(embeddings_clean, labels_clean)
            
            # 3. Calinski-Harabasz Score
            ch_score = calinski_harabasz_score(embeddings_clean, labels_clean)
            
            # 4. Доля шума
            noise_count = np.count_nonzero(labels == -1)
            noise_ratio = (noise_count / len(labels) * 100) if len(labels) > 0 else 0.0
            
            metrics = {
                'silhouette_score': round(float(silhouette), 3),
                'davies_bouldin_index': round(float(db_index), 3),
                'calinski_harabasz_score': round(float(ch_score), 1),
                'noise_ratio': round(float(noise_ratio), 2)
            }
            
            logger.info(f"📊 Метрики качества: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчёта метрик: {e}", exc_info=True)
            return {
                'silhouette_score': 0.0,
                'davies_bouldin_index': 0.0,
                'calinski_harabasz_score': 0.0,
                'noise_ratio': 0.0
            }
        
    @staticmethod
    def interpret(metrics: Dict[str, float]) -> Dict[str, Tuple[str, str]]:
        """
        Интерпретирует метрики для пользователя
        
        Returns:
            dict с оценками и рекомендациями
        """
        silhouette = metrics['silhouette_score']
        db_index = metrics['davies_bouldin_index']
        ch_score = metrics['calinski_harabasz_score']
        noise = metrics['noise_ratio']
        
        # Silhouette
        if silhouette >= 0.7:
            sil_grade = ("🟢 Отлично", "Кластеры чётко разделены")
        elif silhouette >= 0.5:
            sil_grade = ("🟡 Хорошо", "Кластеры разделены, но есть наложения")
        elif silhouette >= 0.25:
            sil_grade = ("🟠 Слабо", "Много пограничных текстов между кластерами")
        else:
            sil_grade = ("🔴 Плохо", "Кластеры не имеют чёткой структуры")
        
        # Davies-Bouldin
        if db_index < 0.5:
            db_grade = ("🟢 Отлично", "Кластеры компактные и разделённые")
        elif db_index < 1.0:
            db_grade = ("🟡 Хорошо", "Кластеры достаточно разделены")
        elif db_index < 1.5:
            db_grade = ("🟠 Приемлемо", "Кластеры частично перекрываются")
        else:
            db_grade = ("🔴 Плохо", "Кластеры размыты и сливаются")
        
        # Calinski-Harabasz
        if ch_score > 300:
            ch_grade = ("🟢 Отлично", "Высокая плотность кластеров")
        elif ch_score > 100:
            ch_grade = ("🟡 Хорошо", "Кластеры хорошо сформированы")
        else:
            ch_grade = ("🔴 Слабо", "Кластеры недостаточно плотные")
        
        # Шум
        if noise < 5:
            noise_grade = ("🟢 Отлично", "Почти все тексты классифицированы")
        elif noise < 10:
            noise_grade = ("🟡 Нормально", "Приемлемый уровень шума")
        elif noise < 15:
            noise_grade = ("🟠 Многовато", "Много текстов не попало в кластеры")
        else:
            noise_grade = ("🔴 Много", "Слишком строгие параметры кластеризации")
        
        return {
            'silhouette': sil_grade,
            'davies_bouldin': db_grade,
            'calinski_harabasz': ch_grade,
            'noise': noise_grade
        }
    
    @staticmethod
    def format_report(metrics: Dict[str, float]) -> str:
        """
        Форматирует метрики в красивый текстовый отчёт для Telegram
        
        Returns:
            HTML-formatted строка
        """
        interpretation = ClusteringMetrics.interpret(metrics)
        
        report = "📊 <b>Качество кластеризации</b>\n\n"
        
        # Silhouette
        sil_status, sil_desc = interpretation['silhouette']
        report += f"<b>Чёткость кластеров</b>\n"
        report += f"{sil_status} {metrics['silhouette_score']:.3f}\n"
        report += f"<i>{sil_desc}</i>\n\n"
        
        # Davies-Bouldin
        db_status, db_desc = interpretation['davies_bouldin']
        report += f"<b>Разделённость</b>\n"
        report += f"{db_status} {metrics['davies_bouldin_index']:.3f}\n"
        report += f"<i>{db_desc}</i>\n\n"
        
        # Calinski-Harabasz
        ch_status, ch_desc = interpretation['calinski_harabasz']
        report += f"<b>Плотность кластеров</b>\n"
        report += f"{ch_status} {metrics['calinski_harabasz_score']:.0f}\n"
        report += f"<i>{ch_desc}</i>\n\n"
        
        # Шум
        noise_status, noise_desc = interpretation['noise']
        report += f"<b>Доля шума</b>\n"
        report += f"{noise_status} {metrics['noise_ratio']:.1f}%\n"
        report += f"<i>{noise_desc}</i>\n\n"
        
        # Общий вердикт
        avg_quality = ClusteringMetrics._overall_quality(metrics)
        if avg_quality >= 0.7:
            verdict = "✅ <b>Отличное качество!</b> Кластеры хорошо определены."
        elif avg_quality >= 0.5:
            verdict = "👍 <b>Хорошее качество.</b> Результаты можно использовать."
        elif avg_quality >= 0.3:
            verdict = "⚠️ <b>Среднее качество.</b> Рекомендуется проверить параметры."
        else:
            verdict = "❌ <b>Низкое качество.</b> Нужна настройка параметров или больше данных."
        
        report += f"{verdict}"
        
        return report
    
    @staticmethod
    def _overall_quality(metrics: Dict[str, float]) -> float:
        """Общая оценка качества (0-1)"""
        # Нормализуем метрики к диапазону 0-1
        sil_norm = max(0, metrics['silhouette_score'])  # 0-1
        db_norm = max(0, 1 - metrics['davies_bouldin_index'] / 2)  # инвертируем
        ch_norm = min(1, metrics['calinski_harabasz_score'] / 500)  # 0-1
        noise_norm = max(0, 1 - metrics['noise_ratio'] / 20)  # инвертируем
        
        # Взвешенное среднее (silhouette важнее всего)
        return (sil_norm * 0.4 + db_norm * 0.3 + ch_norm * 0.2 + noise_norm * 0.1)
