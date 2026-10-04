
class ArchitectureLogger:
    def __init__(self, filename: str = "top_architectures.txt"):
        self.filename = filename
        self.all_models = []  # список вида (rmse, architecture_string)

    # TODO - Дашко, тут в топе надо учесть еще и архитектуры, которые прогонялись при прошлом запуске программы, а не только при текущем
    def update(self, architecture_str: str, rmse: float):
        """Метод вызывается каждый раз при валидации нейросети."""
        self.all_models.append((rmse, architecture_str))
        self.all_models.sort(key=lambda x: x[0])
        self._save_to_file()

    def _save_to_file(self):
        """Запись всего ранжированного списка в текстовый файл"""
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write(f"=== РЕЙТИНГ ВСЕХ АРХИТЕКТУР ПО RMSE (Всего прогонов: {len(self.all_models)}) ===\n\n")
            for rank, (rmse, arch) in enumerate(self.all_models, start=1):
                f.write(f"Место {rank} | RMSE: {rmse:.6f}\n")
                f.write(f"Архитектура:\n{arch}\n")
                f.write("=" * 60 + "\n\n")
