# hierarchical_clustering.py

import os
import requests
import json
from dotenv import load_dotenv
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
    
load_dotenv()
    
#YandexGPT Integration
YANDEX_API_KEY = os.getenv('YANDEX_API_KEY')
YANDEX_FOLDER_ID = os.getenv('YANDEX_FOLDER_ID')

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


def generate_master_category_names(hierarchy, cluster_names, topics, df):
    """
    Генерирует названия для мастер-категорий
    """
    master_names = {}
    
    for master_id, sub_clusters in sorted(hierarchy.items()):
        print(f"\n🔄 Генерация названия для категории {master_id}...")
        
        # === 1. Собираем названия подкластеров ===
        sub_info = []
        for cid in sub_clusters:
            name = cluster_names.get(cid, f"Кластер {cid}")
            size = sum(1 for t in topics if t == cid)
            sub_info.append((name, size))
        
        sub_info.sort(key=lambda x: x[1], reverse=True)
        
        llm_success = False  # Флаг для отслеживания успеха LLM
        
        # === 2. Пробуем LLM ===
        if YANDEX_API_KEY and YANDEX_FOLDER_ID:
            # Берём топ-7 крупнейших подкластеров
            top_subs = sub_info[:7]
            sub_descriptions = "\n".join([
                f"- {name} ({size} обращений)"
                for name, size in top_subs
            ])
            
            # Берём примеры реальных текстов
            examples = []
            for cid in sub_clusters[:4]:
                cluster_mask = [t == cid for t in topics]
                cluster_texts = df[cluster_mask].iloc[:, 0].head(5).tolist()
                examples.extend(cluster_texts)
            
            # Очищаем примеры от мусора
            clean_examples = []
            for ex in examples:
                if isinstance(ex, str) and len(ex) > 20 and len(ex) < 200:
                    clean_examples.append(ex[:150])
            
            if len(clean_examples) < 3:
                print(f"   ⚠️ Мало примеров для LLM ({len(clean_examples)}), используем fallback")
            else:
                examples_text = "\n".join([f"- {ex}" for ex in clean_examples[:8]])
                
                prompt = f"""
Ты аналитик обращений пользователей.

Перед тобой группа связанных категорий обращений:

{sub_descriptions}

Примеры реальных обращений из этой группы:
{examples_text}

Задание:
Придумай ОДНО короткое обобщающее название (3-6 слов) для всей группы категорий.

Требования:
- На русском языке
- Без эмодзи и технических символов
- Понятное для не-технического человека
- Отражает суть проблем/вопросов

Ответь ТОЛЬКО названием, без пояснений.

Название:"""

                try:
                    response = requests.post(
                        "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
                        headers={
                            "Authorization": f"Api-Key {YANDEX_API_KEY}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "modelUri": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt-lite/latest",
                            "completionOptions": {
                                "stream": False,
                                "temperature": 0.4,
                                "maxTokens": 40
                            },
                            "messages": [{"role": "user", "text": prompt}]
                        },
                        timeout=15
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if "result" in result:
                            name = result['result']['alternatives'][0]['message']['text'].strip()
                            name = name.replace('Название:', '').strip().strip('"\'')
                            
                            # ДЕТАЛЬНАЯ ОТЛАДКА
                            print(f"   🔍 Получено от LLM: '{name}'")
                            print(f"   🔍 Длина: {len(name)}")
                            
                            # Валидация
                            if (len(name) > 5 and 
                                len(name) < 50 and 
                                not any(bad in name.lower() for bad in ['column', 'row', 'robot', 'pad', 'forms'])):
                                
                                master_names[master_id] = f"📁 {name}"
                                print(f"   ✅ {master_names[master_id]} (от LLM)")
                                llm_success = True
                            else:
                                print(f"   ⚠️ LLM вернул невалидное название")
                        else:
                            print(f"   ⚠️ Неожиданная структура ответа API: {result}")
                    
                    else:
                        print(f"   ⚠️ API вернул код {response.status_code}: {response.text}")
                
                except Exception as e:
                    print(f"   ⚠️ Ошибка LLM: {e}")
        
        # === 3. FALLBACK: Используем только если LLM не сработал ===
        if not llm_success:
            if sub_info:
                largest_name, largest_size = sub_info[0]
                
                # Очищаем название
                clean_name = largest_name
                clean_name = ' '.join([
                    word for word in clean_name.split()
                    if len(word) > 2 and not word.lower() in ['row', 'column', 'pad', 'robot', 'forms', 'data']
                ])
                
                if clean_name and len(clean_name) > 3:
                    master_names[master_id] = f"{clean_name.capitalize()}"
                else:
                    master_names[master_id] = f"Категория {master_id}"
            else:
                master_names[master_id] = f"Категория {master_id}"
            
            print(f"   ✅ {master_names[master_id]} (fallback)")
    
    return master_names
