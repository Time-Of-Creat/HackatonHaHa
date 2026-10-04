import numpy as np
from hockey_data_generator import HockeyDataGenerator

def main():
    print("1. Инициализация и подготовка данных...")
    generator = HockeyDataGenerator(raw_dir='data/raw', val_ratio=0.2)
    
    print("2. Сбор обучающей выборки (X, Y)...")
    X, Y = generator.get_train_data()
    
    X_arr = np.array(X)
    Y_arr = np.array(Y)
    
    print(f"Размерность матрицы X: {X_arr.shape}")
    print(f"Размерность вектора Y: {Y_arr.shape}\n")
    
    all_feature_names = ['days_since_start', 'sold_ratio'] + generator.encoder.feature_cols
    
    print("--- ПРОВЕРКА ВЕКТОРА Y ---")
    print("Первые 10 значений Y:", Y[:10])
    print("Минимум в Y:", min(Y) if len(Y) > 0 else "пусто")
    print("Максимум в Y:", max(Y) if len(Y) > 0 else "пусто")
    print("Количество уникальных значений в Y:", len(set(Y)))
    print("Пример уникальных значений Y:", list(set(Y))[:10], "\n")
    
    print("--- ПРОВЕРКА УТЕЧКИ ДАННЫХ (КОРРЕЛЯЦИЯ С TARGET) ---")
    suspicious_found = False
    
    for i, name in enumerate(all_feature_names):
        corr = np.corrcoef(X_arr[:, i], Y_arr)[0, 1]
        
        if np.isnan(corr):
            print(f"⚠️ [NAN] [{i}] '{name}': correlation = nan")
        elif abs(corr) > 0.85:
            print(f"⚠️ УТЕЧКА В ФИЧЕ [{i}] '{name}': correlation = {corr:.4f}")
            suspicious_found = True
        else:
            print(f"  [OK] [{i}] '{name}': correlation = {corr:.4f}")
            
    print("\n------------------------------------------------")
    if suspicious_found:
        print("❌ Найдена утечка данных! Признаки с высокой корреляцией следует исключить.")
    else:
        print("✅ Проверка завершена.")

if __name__ == '__main__':
    main()
