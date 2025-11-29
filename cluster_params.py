# cluster_params.py
"""
Автоматический подбор параметров кластеризации
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ClusteringParams:
    """Параметры для HDBSCAN и UMAP"""
    min_cluster_size: int
    min_samples: int
    n_neighbors: int
    n_components: int
    description: str = ""


def get_clustering_params(n_texts: int) -> ClusteringParams:
    """
    Подбирает оптимальные параметры в зависимости от размера датасета
    
    Args:
        n_texts: количество уникальных текстов
    
    Returns:
        ClusteringParams с настройками
    """
    
    # 10 диапазонов (вместо 3)
    if n_texts < 50:
        return ClusteringParams(
            min_cluster_size=3,
            min_samples=1,
            n_neighbors=5,
            n_components=5,
            description="Очень маленький датасет (< 50)"
        )
    
    elif n_texts < 100:
        return ClusteringParams(
            min_cluster_size=4,
            min_samples=2,
            n_neighbors=8,
            n_components=7,
            description="Маленький датасет (50-100)"
        )
    
    elif n_texts < 250:
        return ClusteringParams(
            min_cluster_size=5,
            min_samples=2,
            n_neighbors=10,
            n_components=8,
            description="Малый датасет (100-250)"
        )
    
    elif n_texts < 500:
        return ClusteringParams(
            min_cluster_size=7,
            min_samples=3,
            n_neighbors=15,
            n_components=10,
            description="Средне-малый датасет (250-500)"
        )
    
    elif n_texts < 1000:
        return ClusteringParams(
            min_cluster_size=10,
            min_samples=3,
            n_neighbors=20,
            n_components=10,
            description="Средний датасет (500-1K)"
        )
    
    elif n_texts < 2500:
        return ClusteringParams(
            min_cluster_size=15,
            min_samples=4,
            n_neighbors=25,
            n_components=10,
            description="Средне-большой датасет (1K-2.5K)"
        )
    
    elif n_texts < 5000:
        return ClusteringParams(
            min_cluster_size=20,
            min_samples=5,
            n_neighbors=30,
            n_components=12,
            description="Большой датасет (2.5K-5K)"
        )
    
    elif n_texts < 10000:
        return ClusteringParams(
            min_cluster_size=30,
            min_samples=7,
            n_neighbors=40,
            n_components=12,
            description="Очень большой датасет (5K-10K)"
        )
    
    elif n_texts < 30000:
        return ClusteringParams(
            min_cluster_size=45,
            min_samples=10,
            n_neighbors=55,
            n_components=12,
            description="Огромный датасет (10K-30K)"
        )
    
    else:  # >= 30000
        return ClusteringParams(
            min_cluster_size=60,
            min_samples=15,
            n_neighbors=70,
            n_components=15,
            description="Массивный датасет (30K+)"
        )


def estimate_n_clusters(n_texts: int) -> Tuple[int, int]:
    """
    Оценивает ожидаемое количество кластеров
    
    Returns:
        (min_expected, max_expected)
    """
    if n_texts < 100:
        return (2, 8)
    elif n_texts < 500:
        return (5, 20)
    elif n_texts < 2000:
        return (10, 50)
    elif n_texts < 10000:
        return (20, 100)
    else:
        return (30, 200)
