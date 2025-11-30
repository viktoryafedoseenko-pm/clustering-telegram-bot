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
import os
import requests
import json
from dotenv import load_dotenv
from metrics import ClusteringMetrics
from hierarchical_clustering import create_hierarchy, generate_master_category_names
from config import EMBEDDING_MODEL
from cluster_params import get_clustering_params, estimate_n_clusters  # type: ignore
import logging

logger = logging.getLogger(__name__)

load_dotenv()

morph = pymorphy2.MorphAnalyzer()

#YandexGPT Integration
YANDEX_API_KEY = os.getenv('YANDEX_API_KEY')
YANDEX_FOLDER_ID = os.getenv('YANDEX_FOLDER_ID')

def generate_insight_yandex(stats):
    """
    Генерация краткого осмысленного инсайта по результатам кластеризации через YandexGPT
    """
    if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
        return None

    # Формируем понятный контекст для модели
    n_clusters = stats.get("n_clusters", 0)
    total = stats.get("total_texts", 0)
    noise = stats.get("noise_percent", 0)
    top_clusters = stats.get("top_clusters", [])

    clusters_summary = "\n".join(
        [f"- {c['name']} — {c['size']} текстов" for c in top_clusters]
    )

    prompt = f"""
Ты аналитик клиентских обращений.

Вот результаты кластеризации {total} текстов:
• Количество кластеров: {n_clusters}
• Доля шума (непопавших): {noise:.1f}%
• Топ темы:
{clusters_summary}

Задание:
1. Кратко (до 3 предложений) объясни, что видно из этих данных.
2. Используй профессиональный бизнес-тон, без эмодзи.
3. Обрати внимание на самое важное.
"""

    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {
        "Authorization": f"Api-Key {YANDEX_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "modelUri": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt-lite/latest",
        "completionOptions": {
            "stream": False,
            "temperature": 0.6,
            "maxTokens": 120
        },
        "messages": [{"role": "user", "text": prompt}]
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            text = result['result']['alternatives'][0]['message']['text'].strip()
            return text
        else:
            print(f"⚠️ Ошибка при генерации инсайта: {response.status_code}")
            return None
    except Exception as e:
        print(f"⚠️ Ошибка запроса к YandexGPT: {e}")
        return None


def generate_cluster_name_yandex(texts_sample, max_retries=2):
    """
    Генерация названия кластера через YandexGPT
    
    Args:
        texts_sample: Список примеров текстов из кластера
        max_retries: Количество попыток при ошибке
        
    Returns:
        str: Название кластера или None при ошибке
    """
    if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
        return None
    
    # Берём 8 примеров (до 130 символов каждый)
    examples = "\n".join([f"- {t[:130]}" for t in texts_sample[:8]])
    
    prompt = f"""Ты анализируешь обращения в техподдержку.

    Вот примеры обращений из одной тематической группы:
    {examples}

    Задание: Придумай уникальное короткое название (2-5 слов) для этой категории.

    Требования:
    - На русском языке, БЕЗ заглавных букв в середине слов
    - Без эмодзи и спецсимволов
    - Название должно отражать конкретную специфику этих обращений
    - Если есть подтема – обязательно укажи её

    ✅ Хорошие примеры:
    - "проблемы с оплатой картой" (не просто "оплата")
    - "получение диплома по почте" (не просто "диплом")
    - "ошибка при входе в аккаунт" (не просто "технические ошибки")
    - "вопросы по налоговому вычету"

    ❌ Плохие примеры (слишком общие):
    - "оплата"
    - "диплом"
    - "технические ошибки"

    Название:"""

    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {
        "Authorization": f"Api-Key {YANDEX_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "modelUri": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt-lite/latest",
        "completionOptions": {
            "stream": False,
            "temperature": 0.4,  
            "maxTokens": 30 
        },
        "messages": [
            {
                "role": "user",
                "text": prompt
            }
        ]
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                text = result['result']['alternatives'][0]['message']['text'].strip()

                text = text.replace('Название:', '').strip()
                text = text.strip('"').strip("'")
                
                if len(text) > 50:
                    text = text[:50]
                
                return text
            
            elif response.status_code == 429: 
                print(f"⚠️ Rate limit, ждём 2 секунды...")
                import time
                time.sleep(2)
                continue
            
            else:
                print(f"⚠️ YandexGPT ошибка {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"⚠️ YandexGPT timeout (попытка {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                import time
                time.sleep(1)
            continue
            
        except Exception as e:
            print(f"⚠️ YandexGPT ошибка: {e}")
            return None
    
    return None


# Cписок стоп-слов
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
    'usedesk', 'normal', 'variant', 'rgb', 'rgba', 'sans', 'serif', 'blockquote',
    'white', 'space', 'pre', 'wrap', 'text', 'family', 'line', 'height',
    'amp', 'comment_id', 'answer', 'email', 'mailto', 'http', 'https',
    'yandex', 'practicum', 'mail', 'support', 'usedesk', 'ticket', 'weight', 'start transform',
    '255', '000', '111', '222', '333', '444', '555', '666', '777', '888', '999',
}

COMMON_RUSSIAN_STOP_WORDS = {
    'добрый', 'здравствуйте', 'день', 'здравствуй', 'привет', 'спасибо', 'пожалуйста',
    'уважаемый', 'можно', 'нужно', 'хочу', 'могу', 'есть', 'нет', 'да', 'не', 'на', 
    'в', 'и', 'с', 'у', 'о', 'по', 'за', 'от', 'из', 'к', 'до', 'для', 'или', 'но',
    'что', 'как', 'это', 'так', 'вот', 'же', 'ли', 'бы', 'то', 'во', 'со', 'изо',
    'меня', 'тебя', 'его', 'её', 'нас', 'вас', 'их', 'мой', 'твой', 'свой', 'наш',
    'ваш', 'ихний', 'кто', 'чего', 'чем', 'кому', 'чему', 'кого', 'ещё', 'уже',
    'очень', 'более', 'самый', 'такой', 'весь', 'который', 'какой', 'тут', 'тот',
    'будет', 'было', 'были', 'буду', 'будем', 'будете', 'будут',
}

DOMAIN_STOP_WORDS = {
    # Техническое
    'usedesk', 'ticket', 'comment', 'answer', 'email', 'support', 'mail', 'mailto',
    'yandex', 'practicum', 'практикум', 'яндекс', 'практикума',
    
    # Email
    'sent', 'iphone', 'ipad', 'android', 'отправлено', 'from', 'gmail',
    'почта', 'почту', 'письмо', 'письма',
    
    # Даты
    'января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
    'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря',
    'понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье',
    'сегодня', 'вчера', 'завтра',
    
    # HTML/CSS (расширенный список!)
    'content', 'noreferrer', 'noopener', 'secure', 'nps', 'important', 'nbsp',
    'bgcolor', 'radius', 'display', 'block', 'inline', 'hidden', 'visible',
    'opacity', 'overflow', 'target', 'blank', 'rel', 'href', 'src', 'alt',
    'title', 'class', 'style', 'font', 'margin', 'padding', 'border',
    'width', 'height', 'px', 'caps', 'start', 'word', 'decoration', 'break', 'transparent', 'inbound', 'blank',
    'transform', 'lesson', 'max', 'min', 'px',
    
    # UTM и аналитика
    'utm_source', 'utm_medium', 'utm_campaign', 'utm_content', 'utm_term',
    
    # Общие фразы
    'мне', 'меня', 'вам', 'вас', 'нас', 'тебя', 'его', 'её',
    'можно', 'нужно', 'хочу', 'могу', 'хотел', 'хотела', 'надо',
    'сейчас', 'теперь', 'уже', 'ещё', 'вопрос', 'помочь', 'помогите',
    'доброе', 'утро', 'вечер', 'ночь',  # "доброе утро • утро • доброе"

    # Числа и коды
    '2025', '2024', '00', '07', '06', '01', '02', '03', '04', '05', '08', '09', '10', '11', '12',
    '540px', '15', '20', '30',

    # Служебные
    'message', 'сумму', 'чек',  # "message • 00 сумму"
}

STOP_WORDS = COMMON_RUSSIAN_STOP_WORDS.union(HTML_STOP_WORDS).union(DOMAIN_STOP_WORDS)

morph = pymorphy2.MorphAnalyzer()

def clean_html(text: str) -> str:
    """Агрессивная очистка HTML и CSS"""
    if not isinstance(text, str):
        return ""
    
    # 1. Удаляем email-подписи и шаблоны
    email_patterns = [
        r'Отправлено с (iPhone|iPad|Android|мобильн\w+)',
        r'Sent from (my )?iPhone',
        r'--\s*Отправлено из.*?Почты',
        r'Служба образовательной поддержки.*',
        r'T_I_C_K_E_T_I_D_\d+',
        r'U_D_I_D_\d+',
        r'Оцените.*нашу поддержку:.*',
        r'Это письмо содержит ответы на опрос.*',
        r'Яндекс не несёт ответственности.*',
    ]
    for pattern in email_patterns:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE | re.DOTALL)
    
    # 2. HTML-теги и entities
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&[a-z]+;', ' ', text)
    text = re.sub(r'&#\d+;', ' ', text)
    
    # 3. CSS-стили (усиленная версия)
    text = re.sub(r'[a-z\-]+\s*:\s*[^;"]+;?', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\w+\s*=\s*["\'][^"\']*["\']', ' ', text)
    text = re.sub(r'\b\d+[a-z%]+\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'#[0-9a-f]{3,6}\b', ' ', text, flags=re.IGNORECASE)
    
    # 4. Удаляем HTML/CSS слова-мусор
    html_junk = [
        r'\bcontent\b', r'\bnoreferrer\b', r'\bnoopener\b', r'\bsecure\b',
        r'\bnps\b', r'\bimportant\b', r'\bnbsp\b', r'\bbgcolor\b',
        r'\bradius\b', r'\bdisplay\b', r'\bblock\b', r'\binline\b',
        r'\bhidden\b', r'\bvisible\b', r'\bopacity\b', r'\boverflow\b',
        r'\btarget\b', r'\bblank\b', r'\brel\b', r'\bhref\b', r'\bsrc\b',
        r'\balt\b', r'\btitle\b', r'\bclass\b', r'\bid\b',
    ]
    for pattern in html_junk:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    
    # 5. Удаляем повторяющиеся числа и CSS-слова
    text = re.sub(r'\b(\d+)\s+\1\b', '', text) 
    text = re.sub(r'\b\d+px\b', '', text, flags=re.I)
    text = re.sub(r'\bcaps\b', '', text, flags=re.I)
    text = re.sub(r'\bstart\b', '', text, flags=re.I)
    
    # 6. Удаляем подчёркивания (из форм)
    text = re.sub(r'_{3,}', ' ', text) 
    text = re.sub(r'_+', ' ', text)     
    
    # 7. Удаляем "Технические данные:"
    text = re.sub(r'Технические данные:.*', '', text, flags=re.I)

    # 8. Чистим пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    
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
        # Расширенная фильтрация
        if (len(w) > 2 and 
            len(w) < 20 and 
            w not in STOP_WORDS and
            not w.isdigit() and
            not re.match(r'^\d+$', w) and
            not re.match(r'^\d+[a-z]+$', w, re.I) and 
            not re.match(r'^[a-z]+\d+$', w, re.I) and
            not any(bad in w for bad in ['amp', 'comment', 'answer', 'mailto'])):
            
            # Лемматизация только русских слов
            if re.match(r'^[а-яё]+$', w):
                parsed = morph.parse(w)[0]
                w = parsed.normal_form
            words.append(w)
    
    return ' '.join(words)

def merge_similar_clusters(topics, topic_model, df, similarity_threshold=0.75):
    """
    Объединяет семантически близкие кластеры
    
    Args:
        topics: массив cluster_id для каждого текста
        topic_model: обученная BERTopic модель
        df: датафрейм с текстами
        similarity_threshold: порог схожести (0-1), выше = строже
    
    Returns:
        topics: обновлённый массив cluster_id
        merge_map: dict {old_id: new_id}
    """
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # Получаем уникальные кластеры (без шума)
    unique_clusters = [c for c in set(topics) if c != -1]
    
    if len(unique_clusters) < 2:
        return topics, {}
    
    # Достаём оригинальную модель
    # BERTopic оборачивает модель в Backend, нужно достать оригинал
    embedding_model = topic_model.embedding_model
    
    # Проверяем, есть ли атрибут .embedding_model (если это Backend)
    if hasattr(embedding_model, 'embedding_model'):
        embedding_model = embedding_model.embedding_model
    
    # Получаем embeddings центров кластеров
    cluster_centers = {}
    
    for cluster_id in unique_clusters:
        cluster_indices = [i for i, c in enumerate(topics) if c == cluster_id]
        if len(cluster_indices) == 0:
            continue
        
        # Берём тексты кластера
        cluster_texts = [df.iloc[i, 0] for i in cluster_indices[:20]]  # макс 20 для скорости
        
        # Получаем embeddings
        embeddings = embedding_model.encode(cluster_texts)
        
        # Центр кластера = среднее embeddings
        cluster_centers[cluster_id] = np.mean(embeddings, axis=0)
    
    # Вычисляем матрицу схожести между кластерами
    cluster_ids = list(cluster_centers.keys())
    center_vectors = np.array([cluster_centers[cid] for cid in cluster_ids])
    
    similarity_matrix = cosine_similarity(center_vectors)
    
    # Находим пары для объединения
    merge_map = {}  # {old_id: new_id}
    
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            similarity = similarity_matrix[i][j]
            
            if similarity >= similarity_threshold:
                cluster_i = cluster_ids[i]
                cluster_j = cluster_ids[j]
                
                # Объединяем в кластер с меньшим ID
                target = min(cluster_i, cluster_j)
                source = max(cluster_i, cluster_j)
                
                # Проверяем, не был ли source уже переназначен
                if source in merge_map:
                    continue
                
                merge_map[source] = target
                print(f"🔗 Объединяем кластеры {source} → {target} (схожесть: {similarity:.2f})")
    
    # Применяем объединение
    if merge_map:
        topics_merged = topics.copy()
        for i, cluster_id in enumerate(topics):
            if cluster_id in merge_map:
                topics_merged[i] = merge_map[cluster_id]
        
        print(f"✅ Объединено {len(merge_map)} пар кластеров")
        return topics_merged, merge_map
    
    return topics, {}


def calculate_metrics(topics, cluster_names, topic_model):
    """Расширенный расчет метрик с уникальными названиями кластеров"""
    cluster_counts = Counter(topics)
    noise_count = cluster_counts.get(-1, 0)
    noise_percent = (noise_count / len(topics)) * 100 if len(topics) > 0 else 0
    n_clusters = len([c for c in cluster_counts.keys() if c != -1])
    cluster_sizes = [count for cluster, count in cluster_counts.items() if cluster != -1]
    avg_size = np.mean(cluster_sizes) if cluster_sizes else 0
    
    # Собираем топ кластеры с уникальными названиями
    top_clusters = []
    seen_names = set()
    
    sorted_clusters = sorted(
        [(cluster, count) for cluster, count in cluster_counts.items() if cluster != -1],
        key=lambda x: x[1],
        reverse=True
    )
    
    for cluster_id, size in sorted_clusters:
        name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        
        # Пропускаем дублирующиеся названия
        if name in seen_names:
            continue
            
        seen_names.add(name)
        top_clusters.append({
            'id': cluster_id,
            'name': name,
            'size': size
        })
        
        if len(top_clusters) >= 3:  # Ограничиваем топ-3
            break
    
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

    logger.info(f"🔄 Starting clustering | File: {file_path}")

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

    # Загрузка
    sync_log("📥 Загружаю файл...")
    df = pd.read_csv(file_path, usecols=[0], encoding='utf-8', dtype=str)
    raw_texts = df.iloc[:, 0].fillna("").astype(str).tolist()
    n = len(raw_texts)
    if n == 0:
        raise ValueError("Файл пустой")

    sync_log(f"📊 Загружено {n} текстов")
    logger.info(f"📊 Loaded {n} texts from CSV")

    # Предобработка
    sync_log("🧹 Предобработка...")
    preprocessed_texts = [preprocess_text(t) for t in raw_texts]
    
    valid_indices = [i for i, t in enumerate(preprocessed_texts) 
                     if t.strip() and len(t.split()) >= 2]
    
    if len(valid_indices) <= 3:
        df["cluster_id"] = 0
        df["cluster_name"] = "Все тексты"
        out = file_path.replace(".csv", "_clustered.csv")
        df[text_column] = df[text_column].apply(sanitize_csv_value)
        df.to_csv(out, index=False, encoding='utf-8')
        return out, {'n_clusters': 1, 'total_texts': n}

    preprocessed_texts = [preprocessed_texts[i] for i in valid_indices]
    df = df.iloc[valid_indices].reset_index(drop=True)
    
    # Удаление дубликатов по очищенным текстам
    sync_log("🔍 Удаление дубликатов...")
    df['_preprocessed'] = preprocessed_texts
    df = df.drop_duplicates(subset='_preprocessed', keep="first").reset_index(drop=True)
    preprocessed_texts = df['_preprocessed'].tolist()
    df = df.drop(columns=['_preprocessed'])  # Убираем служебную колонку
    unique_texts = preprocessed_texts
    n_unique = len(unique_texts)
    sync_log(f"✨ Уникальных: {n_unique}")

    # Модель
    sync_log(f"🤖 Загрузка модели: {EMBEDDING_MODEL}...")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        sync_log(f"⚠️ Ошибка загрузки {EMBEDDING_MODEL}, использую fallback")
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # Объединяем все стоп-слова для vectorizer
    ALL_STOP_WORDS = STOP_WORDS.union(DOMAIN_STOP_WORDS).union(HTML_STOP_WORDS)

    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words=list(ALL_STOP_WORDS), 
        min_df=1,        # Минимально возможное значение
        max_df=0.8,    
        max_features=1800  
    )

    print(f"📊 Параметры CountVectorizer: min_df=1, max_df=1.0 (безопасный режим)")

    # Адаптивная настройка под размер данных
    embedding_dim = model.get_sentence_embedding_dimension()  # Получаем размерность модели
    params = get_clustering_params(n_unique, embedding_dim)
    sync_log(f"🎯 {params.description}")
    sync_log(f"   Параметры: min_size={params.min_cluster_size}, "
            f"samples={params.min_samples}, neighbors={params.n_neighbors}, "
            f"components={params.n_components}")

    min_expected, max_expected = estimate_n_clusters(n_unique)
    sync_log(f"   Ожидается кластеров: {min_expected}-{max_expected}")

    # Используем параметры
    min_cluster_size = params.min_cluster_size
    min_samples = params.min_samples
    n_neighbors = params.n_neighbors
    n_components = params.n_components

    # Логируем параметры
    print(f"🎯 Параметры кластеризации для {n_unique} текстов:")
    print(f"   min_cluster_size = {min_cluster_size}")
    print(f"   min_samples = {min_samples}")
    print(f"   n_neighbors = {n_neighbors}")

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric='cosine',
        random_state=42,
        n_jobs=1,
        spread =1.0
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
        vectorizer_model=vectorizer_model,
        language="multilingual",
        calculate_probabilities=False,
        verbose=False,
        top_n_words=10,
        n_gram_range=(1, 2),
        min_topic_size=min_cluster_size
    )

    # Фильтр стоп-слов
    def filter_topic_words(topic_words, banned_words):
        """Фильтрует слова кластера от мусора"""
        filtered = []
        for word, score in topic_words:
            w_lower = word.lower()
            
            # Простые проверки без regex
            has_digit = any(c.isdigit() for c in word)
            has_alpha = any(c.isalpha() for c in word)
            is_short_english = len(word) <= 3 and word.isascii() and word.isalpha()
            
            # Строгая проверка
            if (w_lower not in banned_words and
                len(word) > 2 and
                len(word) < 20 and
                not word.isdigit() and
                not (has_digit and len(word) < 5) and  # Типа "2x", "10px"
                not is_short_english):  # "css", "div", "px"
                filtered.append((word, score))
            
            if len(filtered) >= 5:
                break
        
        return filtered


    # Кластеризация
    sync_log(f"🎯 Кластеризация (min_size={min_cluster_size})...")
    try:
        topics, _ = topic_model.fit_transform(unique_texts)
    except Exception as e:
        sync_log(f"⚠️ Ошибка: {e}")
        raise

    # Получаем эмбединги для метрик
    sync_log("📊 Вычисление метрик качества...")
    embeddings = None
    try:
        embeddings = topic_model._extract_embeddings(
            unique_texts,
            method="document"
        )
        quality_metrics = ClusteringMetrics.calculate(embeddings, topics)
        sync_log(f"✅ Метрики: Silhouette={quality_metrics['silhouette_score']:.3f}, DB={quality_metrics['davies_bouldin_index']:.3f}")
    except Exception as e:
        sync_log(f"⚠️ Не удалось вычислить метрики: {e}")
        quality_metrics = {
            'silhouette_score': 0.0,
            'davies_bouldin_index': 0.0,
            'calinski_harabasz_score': 0.0
        }

        ENABLE_CLUSTER_MERGING = True

        if ENABLE_CLUSTER_MERGING:
            sync_log("🔗 Объединение похожих кластеров...")
            topics, merge_map = merge_similar_clusters(
                topics, 
                topic_model, 
                pd.DataFrame({0: unique_texts}),
                similarity_threshold=0.70
            )

            # BERTopic нужно пересчитать топики
            if merge_map:
                sync_log("📊 Пересчёт статистики после объединения...")
                # Обновляем топики в модели
                topic_model.topics_ = topics
                sync_log(f"✅ Объединено {len(merge_map)} пар кластеров")

    if embeddings is not None:
        quality_metrics = ClusteringMetrics.calculate(embeddings, topics)
        sync_log(f"✅ Метрики: Silhouette={quality_metrics['silhouette_score']:.3f}, DB={quality_metrics['davies_bouldin_index']:.3f}")

    # Названия (с дополнительной фильтрацией)
    if YANDEX_API_KEY and YANDEX_FOLDER_ID:
        sync_log("📝 Генерация названий с помощью YandexGPT...")
    else:
        sync_log("📝 Генерация названий...")

    info = topic_model.get_topic_info()
    cluster_names = {}

    # Сначала генерируем названия для всех уникальных кластеров
    unique_clusters = set(topics)  # Используем обновлённые topics!
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            cluster_names[cluster_id] = "Прочее"
            continue
        
        topic_words = topic_model.get_topic(cluster_id)
        if not topic_words:
            cluster_names[cluster_id] = f"Кластер {cluster_id}"
            continue
        
        # 1. Пробуем YandexGPT (если настроен)
        if YANDEX_API_KEY and YANDEX_FOLDER_ID:
            # Получаем тексты кластера
            cluster_texts = [unique_texts[i] for i, cluster_id_enum in enumerate(topics) if cluster_id_enum == cluster_id]
            
            if cluster_texts:
                yandex_name = generate_cluster_name_yandex(cluster_texts)
                if yandex_name:
                    print(f"✨ Кластер {cluster_id}: {yandex_name}")
                    cluster_names[cluster_id] = yandex_name
                    continue
        
        # 2. Fallback: используем BERTopic слова
        filtered = filter_topic_words(topic_words, ALL_STOP_WORDS)
        
        if filtered:
            name = " • ".join([w for w, s in filtered[:3]])
            cluster_names[cluster_id] = name
        else:
            cluster_names[cluster_id] = f"Кластер {cluster_id}"

    # ========================================
    # Создание иерархии
    # ========================================

    # Сохраняем базовую кластеризацию
    df["cluster_id"] = topics
    df["cluster_name"] = [cluster_names.get(t, "Шум") for t in topics]

    # Создаём иерархии (мастер-категории)
    sync_log("🗂️ Создание иерархии категорий...")

    def _build_fallback_hierarchy():
        """Возвращает плоскую иерархию: каждый кластер = своя мастер-категория"""
        base_hierarchy = {
            cluster_id: [cluster_id]
            for cluster_id in unique_clusters
            if cluster_id != -1
        }
        base_master_names = {
            cluster_id: cluster_names.get(cluster_id, f"Кластер {cluster_id}")
            for cluster_id in base_hierarchy
        }
        base_master_topics = topics
        return base_hierarchy, base_master_names, base_master_topics
    
    hierarchy = {}
    master_names = {}
    master_topics = topics

    try:
        # Определяем количество мастер-категорий в зависимости от числа кластеров
        n_clusters = len([c for c in set(topics) if c != -1])
        
        if n_clusters <= 7:
            # Если кластеров мало, используем плоскую иерархию
            sync_log(f"   Кластеров мало ({n_clusters}), используем базовую иерархию")
            hierarchy, master_names, master_topics = _build_fallback_hierarchy()
            df["master_category_id"] = df["cluster_id"]
            df["master_category_name"] = df["cluster_name"]
        
        else:
            # Создаём иерархию
            n_master = min(10, max(5, n_clusters // 7))  # 5-10 категорий
            sync_log(f"   Объединяем {n_clusters} кластеров в {n_master} категорий...")
            
            if embeddings is None:
                raise ValueError("Embeddings недоступны, пропускаем иерархию")
            
            hierarchy, master_topics, cluster_to_master = create_hierarchy(
                topics=topics,
                topic_model=topic_model,
                embeddings=embeddings,
                n_master_categories=n_master
            )
            
            # Генерируем названия мастер-категорий
            master_names = generate_master_category_names(
                hierarchy=hierarchy,
                cluster_names=cluster_names,
                topics=topics,
                df=df
            )
            
            # Добавляем в DataFrame
            df["master_category_id"] = master_topics
            df["master_category_name"] = [
                master_names.get(t, "Прочее") if t != -1 else "Шум"
                for t in master_topics
            ]
            
            sync_log(f"✅ Создано {len(hierarchy)} мастер-категорий")

    except Exception as e:
        sync_log(f"⚠️ Ошибка создания иерархии: {e}")
        # Fallback: используем кластеры как категории
        hierarchy, master_names, master_topics = _build_fallback_hierarchy()
        df["master_category_id"] = df["cluster_id"]
        df["master_category_name"] = df["cluster_name"]

    # Конец блока иерархии
    # ========================================

    # Нормализация названий кластеров
    import re

    def normalize_cluster_name(name: str) -> str:
        """Лёгкая нормализация — только очевидные дубли"""
        if not isinstance(name, str):
            return ""
        
        name = name.lower().strip()
        name = re.sub(r'[«»"\'🔹•]', '', name)
        name = re.sub(r'[^а-яёa-z0-9\s-]', ' ', name)  # Разрешаем дефис
        name = re.sub(r'\s+', ' ', name).strip()

        # Минимальные замены — только явные дубли
        replacements = {
            'дипломы': 'диплом',
            'сертификаты': 'диплом',
            'технические проблемы': 'технические ошибки',
            'технические сбои': 'технические ошибки',
        }
        
        for old, new in replacements.items():
            if name == old:  
                name = new
        
        return name.title()

    df["cluster_name"] = df["cluster_name"].apply(normalize_cluster_name)
    df["cluster_name"] = df["cluster_name"].apply(lambda x: x.capitalize() if isinstance(x, str) else x)

    # Статистика
    print("\n📊 Статистика по категориям:")
    stats = (
        df.groupby("master_category_name")
        .agg(Кластеров=("cluster_name", "nunique"))
        .sort_values("Кластеров", ascending=False)
    )
    print(stats)

    # Упорядочиваем колонки для удобства
    column_order = [
        df.columns[0], 
        'master_category_id',
        'master_category_name',
        'cluster_id',
        'cluster_name',
    ]

    # Добавляем остальные колонки если есть
    for col in df.columns:
        if col not in column_order:
            column_order.append(col)

    df = df[column_order]

    # Сохраняем
    out = file_path.replace(".csv", "_clustered.csv")
    df.to_csv(out, index=False, encoding='utf-8')

    sync_log(f"💾 Результат сохранён: {out}")


    stats = calculate_metrics(topics, cluster_names, topic_model)
    
    # Добавляем метрики качества в stats
    stats['quality_metrics'] = quality_metrics
    sync_log(f"✅ {stats['n_clusters']} кластеров за {time.time()-start_time:.1f}с")

    if 'hierarchy' in stats:
        sync_log("\n📊 Мастер-категории:")
        
        master_info = stats['hierarchy']['master_category_name']
        sorted_masters = sorted(
            master_info.items(),
            key=lambda x: x[1]['n_texts'],
            reverse=True
        )
        
        for master_id, info in sorted_masters[:5]:  # Топ-5
            sync_log(f"   {info['name']}: {info['n_texts']} текстов ({info['n_subclusters']} подкатегорий)")

    # Сохранение
    out = file_path.replace(".csv", "_clustered.csv")
    df.to_csv(out, index=False, encoding='utf-8')

    sync_log(f"✅ {stats['n_clusters']} кластеров за {time.time()-start_time:.1f}с")
    
    # Логирование
    logger.info(
        f"✅ Clustering complete | "
        f"Time: {time.time()-start_time:.1f}s | "
        f"Clusters: {stats['n_clusters']} | "
        f"Texts: {n_unique}"
    )
    
    return out, stats, hierarchy, master_names