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


# Полный список стоп-слов
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

MINIMAL_STOP_WORDS = {
    # Служебные
    'на', 'в', 'и', 'с', 'у', 'о', 'по', 'за', 'от', 'из', 'к', 'до',
    'что', 'как', 'это', 'так', 'да', 'нет', 'не',
    
    # Вежливость
    'добрый', 'здравствуйте', 'спасибо', 'пожалуйста',
    
    # HTML/техническое
    'usedesk', 'ticket', 'email', 'nbsp', 'amp', 'quot',
    'width', 'height', 'style', 'div', 'span', 'br',
    
    # Яндекс Практикум (если не важны)
    'yandex', 'practicum', 'яндекс', 'практикум',
}

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
    
    # 4. Удаляем HTML/CSS слова-мусор (НОВЫЙ БЛОК!)
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
    text = re.sub(r'_{3,}', ' ', text)  # _______
    text = re.sub(r'_+', ' ', text)     # Любые подчёркивания
    
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

    # Предобработка
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

    # Удаление дубликатов по очищенным текстам
    sync_log("🔍 Удаление дубликатов...")
    df['_preprocessed'] = preprocessed_texts
    df = df.drop_duplicates(subset='_preprocessed', keep="first").reset_index(drop=True)
    preprocessed_texts = df['_preprocessed'].tolist()
    df = df.drop(columns=['_preprocessed'])
    unique_texts = preprocessed_texts
    n_unique = len(unique_texts)
    sync_log(f"✨ Уникальных: {n_unique}")

    # 🆕 Диагностика (ПОСЛЕ удаления дубликатов!)
    print(f"\n🔍 ДИАГНОСТИКА ПЕРЕД VECTORIZER:")
    print(f"   Уникальных текстов: {n_unique}")
    print(f"   Пример текста: '{unique_texts[0][:100]}'" if unique_texts else "ПУСТО")

    # Средняя длина
    avg_length = sum(len(t.split()) for t in unique_texts) / len(unique_texts) if unique_texts else 0
    print(f"   Средняя длина текста: {avg_length:.1f} слов")

    # Считаем уникальные слова из ВСЕХ unique_texts
    all_words = []
    for text in unique_texts:
        all_words.extend(text.split())

    unique_words = set(all_words)
    total_words = len(all_words)
    print(f"   Всего слов: {total_words}")
    print(f"   Уникальных слов: {len(unique_words)}")
    print(f"   Примеры слов: {list(unique_words)[:30]}")

    # Проверяем пустые тексты
    empty_count = sum(1 for t in unique_texts if len(t.split()) == 0)
    print(f"   Пустых текстов: {empty_count}")

    # Создаём векторизацию с безопасными параметрами
    sync_log("🔧 Настройка векторизатора...")
    def create_vectorizer(unique_words, n_unique):
        """Создаёт и валидирует CountVectorizer с безопасными параметрами."""
        
        def safe_vectorizer_params(n_docs: int, n_words: int):
            """
            Подбирает безопасные min_df и max_df для CountVectorizer.
            Работает и для очень маленьких датасетов (<10 документов).
            """
            # Базовые рекомендации
            if n_docs < 50:
                min_df, max_df = 1, 1.0
            elif n_docs < 200:
                min_df, max_df = 2, 0.9
            elif n_docs < 1000:
                min_df, max_df = 3, 0.8
            else:
                min_df, max_df = 5, 0.7

            # Ослабляем фильтры, если слов мало
            if n_words < 30:
                min_df, max_df = 1, 1.0
            elif n_words < 100:
                min_df, max_df = 1, 0.95

            # 🩹 Гарантируем корректное соотношение
            if isinstance(max_df, float):
                if max_df * n_docs < min_df:
                    max_df = min(1.0, min_df / n_docs + 0.05)
            elif isinstance(max_df, int) and max_df < min_df:
                max_df = min_df + 1

            return min_df, max_df

        # === основной код ===
        n_unique_words = len(unique_words)
        min_df, max_df = safe_vectorizer_params(n_unique, n_unique_words)

        # Конфигурация векторизатора под размер корпуса
        if n_unique_words < 30:
            config = {
                'ngram_range': (1, 2),
                'stop_words': None,
                'max_features': 500
            }
        elif n_unique_words < 100:
            config = {
                'ngram_range': (1, 2),
                'stop_words': None,
                'max_features': 1000
            }
        else:
            config = {
                'ngram_range': (1, 2),
                'stop_words': list(MINIMAL_STOP_WORDS),
                'max_features': 1000
            }

        # Создаём безопасный CountVectorizer
        vectorizer = CountVectorizer(
            **config,
            min_df=min_df,
            max_df=max_df
        )

        print(f"   ✅ Vectorizer создан: min_df={min_df}, max_df={max_df}")
        return vectorizer

    vectorizer_model = create_vectorizer(unique_words, n_unique)
    print(f"   ✅ Vectorizer создан\n")

    # Модель эмбеддингов
    sync_log("🤖 Загрузка модели...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # Адаптивная настройка кластеризации
    if n_unique < 500:
        # Для маленьких датасетов
        min_cluster_size = 5
        min_samples = 2
        n_neighbors = 10
    elif n_unique < 5000:
        min_cluster_size = max(12, int(n_unique * 0.015))
        min_samples = 2 
        n_neighbors = min(35, max(25, n_unique // 25)) 
    else:
        # Для больших датасетов (30к+)
        min_cluster_size = max(50, int(n_unique * 0.002))  # ~60 для 30к
        min_samples = max(10, int(min_cluster_size * 0.2)) # ~12
        n_neighbors = min(50, max(30, n_unique // 200))    # ~150

    # Логируем параметры
    print(f"🎯 Параметры кластеризации для {n_unique} текстов:")
    print(f"   min_cluster_size = {min_cluster_size}")
    print(f"   min_samples = {min_samples}")
    print(f"   n_neighbors = {n_neighbors}")

    n_components = 10


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
        prediction_data=True,
        cluster_selection_epsilon=0.3
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
            # Строгая проверка
            if (w_lower not in banned_words and
                len(word) > 2 and
                len(word) < 20 and
                not word.isdigit() and
                not re.match(r'^\d+[a-z%]*$', word, re.I) and
                not re.match(r'^[a-z]{1,3}$', word)):  # короткие англ слова
                filtered.append((word, score))
            if len(filtered) >= 5:  # Берём топ-5 слов
                break
        return filtered


    # Кластеризация
    sync_log(f"🎯 Кластеризация (min_size={min_cluster_size})...")
    try:
        topics, _ = topic_model.fit_transform(unique_texts)
    except Exception as e:
        sync_log(f"⚠️ Ошибка: {e}")
        raise

    # 🆕 ПОЛУЧАЕМ EMBEDDINGS ДЛЯ МЕТРИК
    sync_log("📊 Вычисление метрик качества...")
    embeddings = topic_model._extract_embeddings(
        unique_texts,
        method="document"
    )

    # Объединение кластеров (временно отключили)
    # sync_log("🔗 Объединение похожих кластеров...")
    # topics, merge_map = merge_similar_clusters(
    #     topics, 
    #     topic_model, 
    #     pd.DataFrame({0: unique_texts}),
    #     similarity_threshold=0.70
    # )

    # Обновляем topic_model.get_topic_info() после объединения
    # BERTopic нужно пересчитать топики
    # if merge_map:
    #      sync_log("📊 Пересчёт статистики после объединения...")
    #     # Обновляем топики в модели
    #    topic_model.topics_ = topics
    #    sync_log(f"✅ Объединено {len(merge_map)} пар кластеров")

    quality_metrics = ClusteringMetrics.calculate(embeddings, topics)
    sync_log(f"✅ Метрики: Silhouette={quality_metrics['silhouette_score']:.3f}, DB={quality_metrics['davies_bouldin_index']:.3f}")

    # Названия (с дополнительной фильтрацией)
    if YANDEX_API_KEY and YANDEX_FOLDER_ID:
        sync_log("📝 Генерация названий с помощью YandexGPT...")
    else:
        sync_log("📝 Генерация названий...")

    info = topic_model.get_topic_info()
    cluster_names = {}

    # Генерируем названия только для уникальных кластеров
    unique_clusters = set(topics)  # Используем обновлённые topics!

    for cluster_id in unique_clusters:
        if cluster_id == -1:
            cluster_names[cluster_id] = "Прочее"
            continue

    # Сначала генерируем названия для всех уникальных кластеров
    unique_clusters = set(topics)
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            cluster_names[cluster_id] = "🔹 Прочее"
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

    # Теперь просто маппим кластеры к названиям
    df["cluster_id"] = topics
    df["cluster_name"] = [cluster_names[t] for t in topics]

    # Нормализация названий кластеров
    import re

    # clustering.py
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
            if name == old:  # ← Только точное совпадение!
                name = new
        
        return name.title()

    df["cluster_name"] = df["cluster_name"].apply(normalize_cluster_name)
    df["cluster_name"] = df["cluster_name"].apply(lambda x: x.capitalize() if isinstance(x, str) else x)

    # Мастер-категории, автогенерация через llm
    def generate_master_categories_yandex(cluster_names):
        # собираем примеры
        examples = "\n".join([f"- {n}" for n in cluster_names])

        prompt = f"""
    Ты — аналитик обращений пользователей платформы.
    Перед тобой список названий кластеров (тем) обращений.
    Нужно объединить их в несколько мастер-категорий (4–8 штук).

    Правила:
    - Группируй только по смыслу.
    - Название каждой категории должно быть коротким (2–4 слова).
    - Не добавляй ничего, кроме JSON.
    - Верни JSON, где ключ — название категории, а значение — список кластеров, которые в неё входят.

    Пример формата ответа:
    {{
    "Финансовые вопросы": ["Оплата", "Проблемы с оплатой"],
    "Документы и дипломы": ["Получение диплома", "Диплом и документы"],
    "Учебные вопросы": ["Помощь с курсами", "Вопросы по обучению"]
    }}

    Вот список кластеров для группировки:
    {examples}
    """

        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        headers = {
            "Authorization": f"Api-Key {YANDEX_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "modelUri": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt-lite/latest",
            "completionOptions": {"stream": False, "temperature": 0.3, "maxTokens": 700},
            "messages": [{"role": "user", "text": prompt}]
        }

        response = requests.post(url, headers=headers, json=data)
        text = response.json()["result"]["alternatives"][0]["message"]["text"]

        # безопасный парсинг JSON
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            print("⚠️ Не удалось распарсить JSON, ответ LLM:")
            print(text)
            parsed = {"Прочее": cluster_names}
        return parsed


    # Генерируем мастер-категории
    unique_names = df["cluster_name"].dropna().unique().tolist()
    auto_categories = generate_master_categories_yandex(unique_names)

    # Маппинг кластера -> мастер-категория
    def map_to_master(name):
        for cat, subs in auto_categories.items():
            if name in subs:
                return cat
        return "Прочее"

    df["master_category"] = df["cluster_name"].apply(map_to_master)

    # Статистика
    print("\n📊 Статистика по категориям:")
    stats = (
        df.groupby("master_category")
        .agg(Кластеров=("cluster_name", "nunique"))
        .sort_values("Кластеров", ascending=False)
    )
    print(stats)

    # Метрики
    stats = calculate_metrics(topics, cluster_names, topic_model)
    sync_log(f"✅ {stats['n_clusters']} кластеров за {time.time()-start_time:.1f}с")

    # Сохранение
    out = file_path.replace(".csv", "_clustered.csv")
    df.to_csv(out, index=False, encoding='utf-8')

    stats = calculate_metrics(topics, cluster_names, topic_model)
    
    # Добавляем метрики качества в stats
    stats['quality_metrics'] = quality_metrics
    
    sync_log(f"✅ {stats['n_clusters']} кластеров за {time.time()-start_time:.1f}с")
    
    # Сохранение
    out = file_path.replace(".csv", "_clustered.csv")
    df.to_csv(out, index=False, encoding='utf-8')
    
    return out, stats