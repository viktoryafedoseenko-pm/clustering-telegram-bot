# hierarchical_clustering.py

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def create_hierarchy(topics, topic_model, embeddings, n_master_categories=7):
    """
    Создаёт иерархию: мастер-категории → подкатегории
    
    Args:
        topics: массив cluster_id для каждого текста
        topic_model: обученная BERTopic модель
        embeddings: embeddings текстов
        n_master_categories: сколько верхнеуровневых категорий
    
    Returns:
        hierarchy: dict {master_id: [sub_cluster_ids]}
        master_topics: массив master_cluster_id для каждого текста
    """
    
    # 1. Получаем уникальные кластеры (без шума)
    unique_clusters = [c for c in set(topics) if c != -1]
    
    if len(unique_clusters) <= n_master_categories:
        # Уже мало кластеров, иерархия не нужна
        return {i: [i] for i in unique_clusters}, topics
    
    print(f"📊 Создаём иерархию: {len(unique_clusters)} кластеров → {n_master_categories} категорий")
    
    # 2. Вычисляем центры кластеров (embeddings)
    cluster_centers = {}
    
    for cluster_id in unique_clusters:
        # Индексы текстов в этом кластере
        cluster_indices = [i for i, c in enumerate(topics) if c == cluster_id]
        
        # Берём embeddings этих текстов
        cluster_embeddings = embeddings[cluster_indices]
        
        # Центр = среднее по embeddings
        cluster_centers[cluster_id] = np.mean(cluster_embeddings, axis=0)
    
    # 3. Формируем матрицу центров кластеров
    cluster_ids = list(cluster_centers.keys())
    centers_matrix = np.array([cluster_centers[cid] for cid in cluster_ids])
    
    # 4. Агломеративная кластеризация центров
    # Объединяем похожие кластеры в мастер-категории
    agg_clustering = AgglomerativeClustering(
        n_clusters=n_master_categories,
        metric='cosine',
        linkage='average'  # average = более сбалансированные группы
    )
    
    master_labels = agg_clustering.fit_predict(centers_matrix)
    
    # 5. Создаём иерархию
    hierarchy = {}
    cluster_to_master = {}
    
    for i, cluster_id in enumerate(cluster_ids):
        master_id = int(master_labels[i])
        cluster_to_master[cluster_id] = master_id
        
        if master_id not in hierarchy:
            hierarchy[master_id] = []
        hierarchy[master_id].append(cluster_id)
    
    # 6. Назначаем мастер-категории текстам
    master_topics = np.array([
        cluster_to_master.get(topic, -1) if topic != -1 else -1
        for topic in topics
    ])
    
    # 7. Выводим статистику
    print(f"\n📊 Иерархия создана:")
    for master_id, sub_clusters in sorted(hierarchy.items()):
        n_texts = sum(1 for t in topics if t in sub_clusters)
        print(f"   Категория {master_id}: {len(sub_clusters)} подкластеров, {n_texts} текстов")
    
    return hierarchy, master_topics, cluster_to_master


def generate_master_category_names(hierarchy, cluster_names, topic_model, df):
    """
    Генерирует названия для мастер-категорий
    """
    master_names = {}
    
    for master_id, sub_clusters in hierarchy.items():
        # Собираем названия подкластеров
        sub_names = [cluster_names.get(cid, f"Кластер {cid}") for cid in sub_clusters]
        
        # Вариант 1: Простое объединение (без LLM)
        # Берём самый крупный подкластер как название
        largest_sub = max(sub_clusters, 
                         key=lambda cid: sum(1 for t in topics if t == cid))
        master_names[master_id] = f"🗂 {cluster_names.get(largest_sub, 'Категория')}"
        
        print(f"\nМастер-категория {master_id}:")
        print(f"  Название: {master_names[master_id]}")
        print(f"  Включает: {', '.join(sub_names[:5])}")
        if len(sub_names) > 5:
            print(f"            ... и ещё {len(sub_names)-5}")
    
    return master_names
