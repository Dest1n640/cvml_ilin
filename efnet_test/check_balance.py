from collections import Counter
from train_model import train_ds, val_ds


def check_balance(dataset, name="Dataset"):
    counts = Counter(dataset.targets)
    total = len(dataset)
    
    for idx, class_name in enumerate(dataset.classes):
        count = counts[idx]
        percentage = (count / total) * 100
        print(f"Класс {idx} [{class_name}]: {count} шт. ({percentage:.2f}%)")

if __name__ == "__main__":
    check_balance(train_ds, "Обучающая выборка (Train)")
    check_balance(val_ds, "Валидационная выборка (Val)")
