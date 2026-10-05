import json
import os
from source.file_manager.folders_data import folders_data

class ArchitectureLogger:
    def __init__(self):
        self.filename = str(folders_data.LOGS_FILE)
        
        # Создаем технический файл для хранения истории между запусками
        self.json_filename = self.filename + ".json" 
        
        # При создании логгера сразу подтягиваем старые данные, если они есть
        self.all_models = self._load_previous_runs() 

    def _load_previous_runs(self):
        """Читает историю из JSON от прошлых запусков программы"""
        if os.path.exists(self.json_filename):
            with open(self.json_filename, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    # Если файл пустой или поврежден
                    return []
        return []

    def update(self, architecture_str: str, rmse: float):
        """Метод вызывается каждый раз при валидации нейросети."""
        # Используем список [rmse, arch], так как JSON не поддерживает кортежи (tuples)
        self.all_models.append([rmse, architecture_str])
        
        # Сортируем общий список (старые + новые запуски)
        self.all_models.sort(key=lambda x: x[0])
        
        # Сохраняем и системный файл, и красивый текстовый
        self._save_to_json()
        self._save_to_file()

    def _save_to_json(self):
        """Сохраняет данные для следующих запусков программы"""
        with open(self.json_filename, 'w', encoding='utf-8') as f:
            json.dump(self.all_models, f, ensure_ascii=False, indent=4)

    def _save_to_file(self):
        """Запись всего ранжированного списка в текстовый файл для человека"""
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write(f"=== РЕЙТИНГ ВСЕХ АРХИТЕКТУР ПО RMSE (Всего прогонов: {len(self.all_models)}) ===\n\n")
            for rank, (rmse, arch) in enumerate(self.all_models, start=1):
                f.write(f"Место {rank} | RMSE: {float(rmse):.6f}\n")
                f.write(f"Архитектура:\n{arch}\n")
                f.write("=" * 60 + "\n\n")