from pathlib import Path


class FoldersData:
    PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = PROJECT_DIR / "data"
    LOGS_DIR = PROJECT_DIR / "logs"

    def __init__(self) -> None:
        self.dirs_to_create = (self.DATA_DIR, self.LOGS_DIR)
        for folder in self.dirs_to_create:
            folder.mkdir(parents=True, exist_ok=True)


folders_data = FoldersData()
