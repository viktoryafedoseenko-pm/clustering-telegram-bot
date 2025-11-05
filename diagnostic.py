# diagnostic.py
import pandas as pd
from collections import Counter

def diagnose_clustering(input_csv, output_csv):
    """Быстрая диагностика результатов"""
    df_in = pd.read_csv(input_csv)
    df_out = pd.read_csv(output_csv)
    
    print("=" * 50)
    print("📊 ДИАГНОСТИКА КЛАСТЕРИЗАЦИИ")
    print("=" * 50)
    
    # 1. Распределение по кластерам
    cluster_dist = Counter(df_out['cluster_id'])
    print(f"\n🎯 Всего кластеров: {len([c for c in cluster_dist if c != -1])}")
    print(f"❌ Шум (-1): {cluster_dist.get(-1, 0)} ({cluster_dist.get(-1, 0)/len(df_out)*100:.1f}%)")
    
    # 2. Топ-5 кластеров
    print("\n📈 Топ-5 кластеров:")
    for cluster_id, count in cluster_dist.most_common(6):
        if cluster_id == -1:
            continue
        name = df_out[df_out['cluster_id'] == cluster_id]['cluster_name'].iloc[0]
        print(f"  • [{cluster_id}] {name}: {count} текстов")
    
    # 3. Примеры текстов из топ-3 кластеров
    print("\n🔍 ПРИМЕРЫ ТЕКСТОВ:")
    for cluster_id, _ in list(cluster_dist.most_common(4))[1:4]:  # пропускаем -1
        print(f"\n--- Кластер {cluster_id} ---")
        samples = df_out[df_out['cluster_id'] == cluster_id].iloc[:3]
        for idx, row in samples.iterrows():
            text = row[df_out.columns[0]][:150]
            print(f"  {text}...")
    
    # 4. Проблемные моменты
    print("\n⚠️  ПОТЕНЦИАЛЬНЫЕ ПРОБЛЕМЫ:")
    if cluster_dist.get(-1, 0) / len(df_out) > 0.3:
        print("  • Много шума (>30%) — слишком строгие параметры")
    if len([c for c in cluster_dist if c != -1]) < 5:
        print("  • Мало кластеров — увеличь min_cluster_size")
    if max(cluster_dist.values()) / len(df_out) > 0.5:
        print("  • Один огромный кластер — данные слишком однородные?")

# Использование:
diagnose_clustering("test.csv", "test_cluster.csv")
