# logger.py

class ArchitectureLogger:
    def __init__(self, filename: str = "top_architectures.txt"):
        """
        :param filename: файл, куда будет сохраняться весь отсортированный список архитектур
        """
        self.filename = filename
        # Список для хранения всех результатов в виде (rmse, architecture_string)
        self.all_models = []

    def update(self, architecture_str: str, rmse: float):
        """
        Метод вызывается каждый раз при валидации нейросети.
        """
        # 1. Добавляем новый прогон
        self.all_models.append((rmse, architecture_str))
        
        # 2. Сортируем абсолютно ВСЕ модели по RMSE (от наименьшего к наибольшему)
        self.all_models.sort(key=lambda x: x[0])
        
        # 3. Перезаписываем файл со всем списком
        self._save_to_file()

    def _save_to_file(self):
        """Запись всего ранжированного списка в текстовый файл"""
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write(f"=== РЕЙТИНГ ВСЕХ АРХИТЕКТУР ПО RMSE (Всего прогонов: {len(self.all_models)}) ===\n\n")
            for rank, (rmse, arch) in enumerate(self.all_models, start=1):
                f.write(f"Место {rank} | RMSE: {rmse:.6f}\n")
                f.write(f"Архитектура:\n{arch}\n")
                f.write("=" * 60 + "\n\n")
