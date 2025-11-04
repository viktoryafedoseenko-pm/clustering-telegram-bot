# clustering.py
import inspect
import collections

# --- Совместимость с Python 3.11+ ---
if not hasattr(inspect, 'getargspec'):
    from collections import namedtuple
    ArgSpec = namedtuple('ArgSpec', 'args varargs keywords defaults')
    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)
    inspect.getargspec = getargspec
# ------------------------------------

import pandas as pd
import numpy as np
import re
import warnings
import asyncio
from collections import Counter
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer  # +++
import pymorphy2

# Расширенный список стоп-слов
HTML_STOP_WORDS = {
    'style', 'div', 'width', 'height', 'br', 'span', 'class', 'id', 'href', 'src',
    'px', 'pt', 'em', 'rem', 'color', 'background', 'font', 'size', 'border',
    'margin', 'padding', 'align', 'valign', 'center', 'left', 'right', 'justify',
    'table', 'tr', 'td', 'th', 'tbody', 'thead', 'colspan', 'rowspan', 'target',
    'rel', 'nofollow', 'blank', 'www', 'com', 'org', 'net', 'ru', 'quot', 'strong',
    'bold', 'italic', 'underline', 'block', 'inline', 'none', 'hidden', 'visible',
    'display', 'position', 'float', 'clear', 'overflow', 'zindex', 'opacity',
    'img', 'alt', 'title', 'css', 'html', 'body', 'head', 'meta', 'link', 
    'ffffff', 'cellspacing', 'cellpadding', 'helvetica', 'arial', 'verdana',
    'usedesk', 'normal', 'variant', 'rgb', 'rgba', 'sans', 'serif', 'sans-serif',
}

COMMON_RUSSIAN_STOP_WORDS = {
    'добрый', 'здравствуйте', 'день', 'здравствуй', 'привет', 'спасибо', 'пожалуйста',
    'уважаемый', 'можно', 'нужно', 'хочу', 'могу', 'есть', 'нет', 'да', 'не', 'на', 
    'в', 'и', 'с', 'у', 'о', 'по', 'за', 'от', 'из', 'к', 'до', 'для', 'или', 'но',
    'что', 'как', 'это', 'так', 'вот', 'же', 'ли', 'бы', 'то', 'во', 'со', 'изо',
    'меня', 'тебя', 'его', 'её', 'нас', 'вас', 'их', 'мой', 'твой', 'свой', 'наш',
    'ваш', 'ихний', 'кто', 'чего', 'чем', 'кому', 'чему', 'кого', 'ещё', 'уже',
    'очень', 'более', 'самый', 'такой', 'весь', 'который', 'какой', 'тут', 'тот'
}

STOP_WORDS = COMMON_RUSSIAN_STOP_WORDS.union(HTML_STOP_WORDS)

morph = pymorphy2.MorphAnalyzer()

def clean_html(text: str) -> str:
    """Агрессивная очистка HTML и CSS"""
    if not isinstance(text, str):
        return ""
    
    # Удаляем всё между < и >
    text = re.sub(r'<[^>]+>', ' ', text)
    # HTML-сущности
    text = re.sub(r'&[a-z]+;', ' ', text)
    text = re.sub(r'&#\d+;', ' ', text)
    # CSS-свойства
    text = re.sub(r'[a-z\-]+\s*:\s*[^;"]+;?', ' ', text, flags=re.IGNORECASE)
    # HTML-атрибуты
    text = re.sub(r'\w+\s*=\s*["\'][^"\']*["\']', ' ', text)
    # Числа с единицами
    text = re.sub(r'\b\d+[a-z%]+\b', ' ', text, flags=re.IGNORECASE)
    # Hex-коды цветов
    text = re.sub(r'#[0-9a-f]{3,6}\b', ' ', text, flags=re.IGNORECASE)
    # CSS/HTML ключевые слова
    text = re.sub(r'\b(rgb|rgba|url|var|calc|auto|inherit|initial|unset)\b', ' ', text, flags=re.IGNORECASE)
    
    return text

def preprocess_text(text: str) -> str:
    """Улучшенная предобработка"""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    text = clean_html(text)
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = []
    for w in text.split():
        if (len(w) > 2 and 
            w not in STOP_WORDS and
            not w.isdigit() and
            not re.match(r'^\d+$', w)):
            # Правильная лемматизация через parse()
            if re.match(r'[а-яё]+', w):
                parsed = morph.parse(w)[0]  # ← ИСПРАВЛЕНО
                w = parsed.normal_form      # ← ИСПРАВЛЕНО
            words.append(w)
    
    return ' '.join(words)


def calculate_metrics(topics, cluster_names, topic_model):
    """Расширенный расчет метрик"""
    cluster_counts = Counter(topics)
    noise_count = cluster_counts.get(-1, 0)
    noise_percent = (noise_count / len(topics)) * 100 if len(topics) > 0 else 0
    n_clusters = len([c for c in cluster_counts.keys() if c != -1])
    cluster_sizes = [count for cluster, count in cluster_counts.items() if cluster != -1]
    avg_size = np.mean(cluster_sizes) if cluster_sizes else 0
    
    top_clusters = []
    sorted_clusters = sorted(
        [(cluster, count) for cluster, count in cluster_counts.items() if cluster != -1],
        key=lambda x: x[1],
        reverse=True
    )
    
    for cluster_id, size in sorted_clusters[:3]:
        name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        top_clusters.append({
            'id': cluster_id,
            'name': name,
            'size': size
        })
    
    return {
        'n_clusters': n_clusters,
        'noise_percent': round(noise_percent, 2),
        'avg_cluster_size': round(avg_size, 2),
        'total_texts': len(topics),
        'top_clusters': top_clusters,
        'cluster_distribution': dict(cluster_counts)
    }


def clusterize_texts(file_path: str, progress_callback=None):
    """Кластеризация с оптимизированными параметрами"""
    import time
    start_time = time.time()

    async def log_progress(msg):
        print(msg)
        if progress_callback:
            try:
                await progress_callback(msg)
            except:
                pass

    def sync_log(msg):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(log_progress(msg))
            else:
                loop.run_until_complete(log_progress(msg))
        except:
            print(msg)

    # --- Загрузка ---
    sync_log("📥 Загружаю файл...")
    df = pd.read_csv(file_path, usecols=[0], encoding='utf-8', dtype=str)
    raw_texts = df.iloc[:, 0].fillna("").astype(str).tolist()
    n = len(raw_texts)
    if n == 0:
        raise ValueError("Файл пустой")

    sync_log(f"📊 Загружено {n} текстов")

    # --- Предобработка ---
    sync_log("🧹 Предобработка...")
    preprocessed_texts = [preprocess_text(t) for t in raw_texts]
    
    valid_indices = [i for i, t in enumerate(preprocessed_texts) 
                     if t.strip() and len(t.split()) >= 2]
    
    if len(valid_indices) <= 3:
        df["cluster_id"] = 0
        df["cluster_name"] = "Все тексты"
        out = file_path.replace(".csv", "_clustered.csv")
        df.to_csv(out, index=False, encoding='utf-8')
        return out, {'n_clusters': 1, 'total_texts': n}

    preprocessed_texts = [preprocessed_texts[i] for i in valid_indices]
    df = df.iloc[valid_indices].reset_index(drop=True)
    
    # --- Удаление дубликатов ---
    sync_log("🔍 Удаление дубликатов...")
    df = df.drop_duplicates(subset=df.columns[0], keep="first").reset_index(drop=True)
    unique_texts = df.iloc[:, 0].tolist()
    n_unique = len(unique_texts)
    sync_log(f"✨ Уникальных: {n_unique}")

    # --- Модель ---
    sync_log("🤖 Загрузка модели...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # +++ КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: добавляем vectorizer с фильтрацией стоп-слов +++
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words=list(STOP_WORDS),
        min_df=2,  # слово должно встречаться минимум в 2 документах
        max_df=0.7  # игнорируем слова, встречающиеся в >70% документов
    )

    # --- Параметры для ~1000 текстов ---
    # Цель: получить 10-20 кластеров
    min_cluster_size = max(10, int(n_unique * 0.15))  # ~20 текстов (было 99!)
    min_samples = max(5, int(n_unique * 0.01))  # ~10 текстов
    
    n_neighbors = min(30, max(15, n_unique // 50))  # ~20 соседей
    n_components = 10  # больше компонент для UMAP

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric='cosine',
        random_state=42,
        n_jobs=1
    )
    
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    
    topic_model = BERTopic(
        embedding_model=model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,  # +++ ДОБАВЛЕНО +++
        language="multilingual",
        calculate_probabilities=False,
        verbose=False,
        top_n_words=10,
        n_gram_range=(1, 2),
        min_topic_size=min_cluster_size
    )

    # --- Кластеризация ---
    sync_log(f"🎯 Кластеризация (min_size={min_cluster_size})...")
    try:
        topics, _ = topic_model.fit_transform(unique_texts)
    except Exception as e:
        sync_log(f"⚠️ Ошибка: {e}")
        raise

    # --- Названия (с дополнительной фильтрацией) ---
    sync_log("📝 Генерация названий...")
    info = topic_model.get_topic_info()
    cluster_names = {}
    
    def get_name(t):
        if t == -1:
            return "🔹 Прочее"
        
        topic_words = topic_model.get_topic(t)
        if not topic_words:
            cluster_names[t] = f"Cluster {t}"
            return f"Cluster {t}"
        
        filtered = []
        for word, score in topic_words:
            w_lower = word.lower()
            # Жесткая фильтрация
            if (w_lower not in STOP_WORDS and
                w_lower not in HTML_STOP_WORDS and
                len(word) > 2 and
                not word.isdigit() and
                not re.match(r'^\d+[a-z%]*$', word, re.I) and
                not re.match(r'^[a-z]{1,3}$', word)):  # короткие англ слова
                filtered.append(word)
            if len(filtered) >= 3:
                break
        
        if filtered:
            name = " • ".join(filtered[:3])
            cluster_names[t] = name
            return name
        
        cluster_names[t] = f"Cluster {t}"
        return f"Cluster {t}"

    df["cluster_id"] = topics
    df["cluster_name"] = [get_name(t) for t in topics]

    # --- Метрики ---
    stats = calculate_metrics(topics, cluster_names, topic_model)
    sync_log(f"✅ {stats['n_clusters']} кластеров за {time.time()-start_time:.1f}с")

    # --- Сохранение ---
    out = file_path.replace(".csv", "_clustered.csv")
    df.to_csv(out, index=False, encoding='utf-8')

    return out, stats
