# test_cache.py (опционально)
import pandas as pd
from cache_manager import cache

def test_cache():
    # Сохранение
    data = {
        'df': pd.DataFrame({'text': ['test'], 'cluster_id': [0]}),
        'stats': {'n_clusters': 1},
        'cluster_names': {0: 'Test'},
        'file_name': 'test.csv'
    }
    
    key = cache.save(user_id=123, file_name='test.csv', data=data)
    print(f"✅ Saved with key: {key}")
    
    # Загрузка
    loaded = cache.load(key)
    assert loaded is not None
    assert loaded['stats']['n_clusters'] == 1
    print("✅ Loaded successfully")

if __name__ == '__main__':
    test_cache()
