import csv
import os


class CSVLogger:
    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        self.file = None
        self.writer = None
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def __enter__(self):
        file_exists = os.path.exists(self.path)

        self.file = open(self.path, "a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)

        if not file_exists or os.path.getsize(self.path) == 0:
            self.writer.writeheader()

        return self

    def write(self, row):
        clean_row = {key: row.get(key, "") for key in self.fieldnames}
        self.writer.writerow(clean_row)
        self.file.flush()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()