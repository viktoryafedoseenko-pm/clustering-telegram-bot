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
from dotenv import load_dotenv

load_dotenv()

# ========== YandexGPT Integration ==========

YANDEX_API_KEY = os.getenv('YANDEX_API_KEY')
YANDEX_FOLDER_ID = os.getenv('YANDEX_FOLDER_ID')

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
    
    # Берём 5 примеров (до 100 символов каждый)
    examples = "\n".join([f"- {t[:100]}" for t in texts_sample[:8]])
    
    prompt = f"""Ты анализируешь обращения в техподдержку образовательной платформы Яндекс Практикум.

    Вот примеры обращений из одной тематической группы:
    {examples}

    Задание: Придумай короткое и точное название (2-4 слова) для этой категории обращений.

    Требования:
    - На русском языке
    - Без эмодзи и спецсимволов
    - Описывает суть проблемы или запроса
    - Примеры хороших названий: "Получение диплома", "Проблемы с оплатой", "Налоговый вычет", "Технические ошибки"

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
            "temperature": 0.4,  # Низкая температура для стабильности
            "maxTokens": 30      # Короткое название
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
                
                # Очистка от лишнего
                text = text.replace('Название:', '').strip()
                text = text.strip('"').strip("'")
                
                # Проверка длины (не больше 50 символов)
                if len(text) > 50:
                    text = text[:50]
                
                return text
            
            elif response.status_code == 429:  # Rate limit
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

# ==========================================


# Расширенный список стоп-слов
HTML_STOP_WORDS = {
    # HTML/CSS базовые
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
            len(w) < 20 and  # ← добавили: не больше 20 символов
            w not in STOP_WORDS and
            not w.isdigit() and
            not re.match(r'^\d+$', w) and
            not re.match(r'^\d+[a-z]+$', w, re.I) and  # 3px, 255rgb
            not re.match(r'^[a-z]+\d+$', w, re.I) and  # comment_id, answer2
            not any(bad in w for bad in ['amp', 'comment', 'answer', 'mailto'])):  # подстроки
            
            # Лемматизация только русских слов
            if re.match(r'^[а-яё]+$', w):
                parsed = morph.parse(w)[0]
                w = parsed.normal_form
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
    
    # Удаление дубликатов по очищенным текстам
    sync_log("🔍 Удаление дубликатов...")
    df['_preprocessed'] = preprocessed_texts
    df = df.drop_duplicates(subset='_preprocessed', keep="first").reset_index(drop=True)
    preprocessed_texts = df['_preprocessed'].tolist()
    df = df.drop(columns=['_preprocessed'])  # Убираем служебную колонку
    unique_texts = preprocessed_texts
    n_unique = len(unique_texts)
    sync_log(f"✨ Уникальных: {n_unique}")

    # --- Модель ---
    sync_log("🤖 Загрузка модели...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # Объединяем все стоп-слова для vectorizer
    ALL_STOP_WORDS = STOP_WORDS.union(DOMAIN_STOP_WORDS).union(HTML_STOP_WORDS)

    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words=list(ALL_STOP_WORDS),  # ← Используем ВСЕ стоп-слова!
        min_df=3,      # Слово должно быть минимум в 3 документах
        max_df=0.5,    # Игнорируем слова, встречающиеся в >50% текстов (было 0.6)
        max_features=1000  # ← НОВОЕ: ограничиваем словарь 1000 важными словами
    )


    # --- Параметры для ~1000 текстов ---
    # Адаптивная настройка под размер данных
    if n_unique < 500:
        # Для маленьких датасетов
        min_cluster_size = 5
        min_samples = 2
        n_neighbors = 10
    elif n_unique < 5000:
        # Для 500-5000 текстов (твой случай: 759)
        min_cluster_size = max(8, int(n_unique * 0.010))  # ~11 для 759
        min_samples = max(2, int(min_cluster_size * 0.25))  # ~3-4
        n_neighbors = min(25, max(15, n_unique // 40))     # ~19
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


    # --- Кластеризация ---
    sync_log(f"🎯 Кластеризация (min_size={min_cluster_size})...")
    try:
        topics, _ = topic_model.fit_transform(unique_texts)
    except Exception as e:
        sync_log(f"⚠️ Ошибка: {e}")
        raise

    # --- Названия (с дополнительной фильтрацией) ---
    if YANDEX_API_KEY and YANDEX_FOLDER_ID:
        sync_log("📝 Генерация названий с помощью YandexGPT...")
    else:
        sync_log("📝 Генерация названий...")

    info = topic_model.get_topic_info()
    cluster_names = {}
    
    def get_name(t):
        """Генерация названия кластера (с использованием YandexGPT)"""
        if t == -1:
            return "🔹 Прочее"
        
        topic_words = topic_model.get_topic(t)
        if not topic_words:
            cluster_names[t] = f"Кластер {t}"
            return f"Кластер {t}"
        
        # 1. Пробуем YandexGPT (если настроен)
        if YANDEX_API_KEY and YANDEX_FOLDER_ID:
            # Получаем тексты кластера
            cluster_texts = [unique_texts[i] for i, cluster_id in enumerate(topics) if cluster_id == t]
            
            if cluster_texts:
                yandex_name = generate_cluster_name_yandex(cluster_texts)
                if yandex_name:
                    print(f"✨ Кластер {t}: {yandex_name}")
                    cluster_names[t] = yandex_name
                    return yandex_name
        
        # 2. Fallback: используем BERTopic слова
        filtered = filter_topic_words(topic_words, ALL_STOP_WORDS)
        
        if filtered:
            name = " • ".join([w for w, s in filtered[:3]])
            cluster_names[t] = name
            return name
        
        cluster_names[t] = f"Кластер {t}"
        return f"Кластер {t}"


    df["cluster_id"] = topics
    df["cluster_name"] = [get_name(t) for t in topics]

    # --- Метрики ---
    stats = calculate_metrics(topics, cluster_names, topic_model)
    sync_log(f"✅ {stats['n_clusters']} кластеров за {time.time()-start_time:.1f}с")

    # --- Сохранение ---
    out = file_path.replace(".csv", "_clustered.csv")
    df.to_csv(out, index=False, encoding='utf-8')

    return out, stats
