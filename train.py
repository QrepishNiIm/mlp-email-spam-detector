import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

from model import SpamNet

# ----------------------------
# 1. Загрузка и подготовка данных
# ----------------------------
print("Загрузка данных...")
data = pd.read_csv("data/spambase_csv.csv")  # Убедитесь, что файл находится в папке 'data/'
X = data.iloc[:, :-1].values  # Все признаки
y = data.iloc[:, -1].values   # Целевая переменная (0 - не-спам, 1 - спам)

print(f"Форма X: {X.shape}, y: {y.shape}")
print(f"Распределение классов: {np.bincount(y)}")

# Разделение данных с сохранением пропорций
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Стандартизация признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Преобразование в тензоры PyTorch
train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Создание датасетов
train_dataset = TensorDataset(train_tensor, y_train_tensor)
test_dataset = TensorDataset(test_tensor, y_test_tensor)

# Загрузчики данных (DataLoader)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ----------------------------
# 2. Модель, оптимизатор, функция потерь
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpamNet().to(device)  # Используем нашу модель из model.py
criterion = nn.CrossEntropyLoss()  # Кросс-энтропия для бинарной классификации
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam оптимизатор

# ----------------------------
# 3. Цикл обучения
# ----------------------------
num_epochs = 100
train_losses = []  # Хранит значение функции потерь на каждой эпохе
train_accs = []    # Хранит точность на обучении
val_accs = []      # Хранит точность на валидации (тесте)

for epoch in range(num_epochs):
    model.train()  # Переводим модель в режим обучения
    total_loss = 0.0
    correct_train = 0
    total_train = 0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()  # Обнуляем градиенты
        outputs = model(features)  # Прямой проход
        loss = criterion(outputs, labels)  # Вычисляем потери
        loss.backward()  # Обратное распространение
        optimizer.step()  # Обновляем веса

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)  # Находим индекс максимального значения (класс)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train
    train_losses.append(avg_loss)
    train_accs.append(train_acc)

    # Оценка на тестовой выборке каждые 10 эпох
    if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
        model.eval()  # Переводим модель в режим оценки (отключаем dropout)
        correct_val = 0
        total_val = 0
        with torch.no_grad():  # Отключаем вычисление градиентов для экономии памяти
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        val_acc = 100 * correct_val / total_val
        val_accs.append(val_acc)
        print(f"Эпоха [{epoch+1}/{num_epochs}], "
              f"Потеря: {avg_loss:.4f}, "
              f"Точность на обучении: {train_acc:.2f}%, "
              f"Точность на тесте: {val_acc:.2f}%")

# ----------------------------
# 4. Финальная оценка и сохранение результатов
# ----------------------------
os.makedirs("results", exist_ok=True)

# Построение графиков
plt.figure(figsize=(12, 4))

# График функции потерь
plt.subplot(1, 3, 1)
plt.plot(train_losses, label="Потеря на обучении")
plt.title("График функции потерь")
plt.xlabel("Эпохи")
plt.ylabel("Потеря")
plt.legend()

# График точности
plt.subplot(1, 3, 2)
plt.plot(train_accs, label="Точность на обучении")
plt.plot(np.linspace(0, num_epochs-1, len(val_accs)), val_accs, label="Точность на тесте")
plt.title("График точности")
plt.xlabel("Эпохи")
plt.ylabel("Точность (%)")
plt.legend()

# Оценка на тестовой выборке для отчета
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Метрики
from sklearn.metrics import classification_report, confusion_matrix
report = classification_report(all_labels, all_preds, target_names=["Не-спам (0)", "Спам (1)"])
cm = confusion_matrix(all_labels, all_preds)

# Матрица ошибок
plt.subplot(1, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Не-спам", "Спам"], yticklabels=["Не-спам", "Спам"])
plt.title("Матрица ошибок")
plt.xlabel("Предсказанная метка")
plt.ylabel("Истинная метка")

plt.tight_layout()
plt.savefig("results/training_curves.png", dpi=150)
plt.show()

print("\n--- Отчет классификации ---")
print(report)
print("\n--- Матрица ошибок ---")
print(cm)

# Сохранение модели и скалера
torch.save(model.state_dict(), "results/spam_model.pth")
import joblib
joblib.dump(scaler, "results/scaler.pkl")

print("\n✅ Модель и скалер сохранены в папку 'results/'.")