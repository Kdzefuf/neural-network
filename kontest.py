import os
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from torch import nn, optim
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import OneHotEncoder
from torch.optim import lr_scheduler as lrs

# Определение классов данных
from torchvision.models import resnet18

classes = ['Гароу', 'Генос', 'Сайтама', 'Соник', 'Татсумаки', 'Фубуки']


# Загрузка данных
class AnimeDataset(Dataset):
    def __init__(self, base_path, classes, transform=None):
        self.base_path = base_path
        self.classes = classes
        self.transform = transform
        self.images = []
        self.labels = []
        for idx, class_name in enumerate(classes):
            class_path = os.path.join(base_path, class_name)
            file_names = os.listdir(class_path)
            for file_name in file_names:
                file_path = os.path.join(class_path, file_name)
                self.images.append(file_path)
                self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

train_dataset = AnimeDataset('train', classes, transform=train_transform)


# Аналогично для тестового набора
class TestDataset(Dataset):
    def __init__(self, base_path, transform=None):
        self.base_path = base_path
        self.transform = transform
        self.images = os.listdir(base_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.images[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.images[idx]


test_dataset = TestDataset('test', transform=train_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

dataiter = iter(train_loader)
images, labels = dataiter.__next__()

# Проверка количества данных
print(f'Train dataset size: {len(train_dataset)} images')
print(f'Test dataset size: {len(test_dataset)} images')

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = AnimeDataset('train', classes, transform=train_transform)
test_dataset = TestDataset('test', transform=train_transform)

# Разделим тренировочные данные на тренировочные и валидационные
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Преобразуем метки в one-hot encoding
encoder = OneHotEncoder(sparse=False)
train_labels = np.array([train_dataset.dataset[i][1] for i in range(len(train_dataset))])
train_labels_encoded = encoder.fit_transform(train_labels.reshape(-1, 1))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 20


def plot_learning_curves(history):
    fig = plt.figure(figsize=(20, 7))

    plt.subplot(1,2,1)
    plt.title('Потери', fontsize=15)
    plt.plot(history['loss']['train'], label='train')
    plt.plot(history['loss']['val'], label='val')
    plt.ylabel('Потери', fontsize=15)
    plt.xlabel('Эпоха', fontsize=15)
    plt.legend()

    plt.subplot(1,2,2)
    plt.title('Точность', fontsize=15)
    plt.plot(history['acc']['train'], label='train')
    plt.plot(history['acc']['val'], label='val')
    plt.ylabel('Точность', fontsize=15)
    plt.xlabel('Эпоха', fontsize=15)
    plt.legend()
    plt.show()


def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=15):
    scheduler = lrs.ReduceLROnPlateau(optimizer, 'min')
    history = defaultdict(lambda: defaultdict(list))

    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        # устанавливаем поведение dropout / batch_norm  в обучение
        model.train(True)

        # на каждой "эпохе" делаем полный проход по данным
        for X_batch, y_batch in train_loader:
            # обучаемся на текущем батче
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)

            loss = criterion(logits, y_batch.long().to(device))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += np.sum(loss.detach().cpu().numpy())
            y_pred = logits.max(1)[1].detach().cpu().numpy()
            train_acc += np.mean(y_batch.cpu().numpy() == y_pred)

        scheduler.step(loss)

        # подсчитываем лоссы и сохраням в "историю"
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        history['loss']['train'].append(train_loss)
        history['acc']['train'].append(train_acc)

        # устанавливаем поведение dropout / batch_norm в режим тестирования
        model.train(False)

        # полностью проходим по валидационному датасету
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch.long().to(device))
            val_loss += np.sum(loss.detach().cpu().numpy())
            y_pred = logits.max(1)[1].detach().cpu().numpy()
            val_acc += np.mean(y_batch.cpu().numpy() == y_pred)

        # подсчитываем лоссы и сохраням в "историю"
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        history['loss']['val'].append(val_loss)
        history['acc']['val'].append(val_acc)

        # печатаем результаты после каждой эпохи
        print("Epoch {} of {}".format(
            epoch + 1, epochs))
        print("  training loss (in-iteration): \t{:.6f}".format(train_loss))
        print("  validation loss (in-iteration): \t{:.6f}".format(val_loss))
        print("  training accuracy: \t\t\t{:.2f} %".format(train_acc * 100))
        print("  validation accuracy: \t\t\t{:.2f} %".format(val_acc * 100))

    plot_learning_curves(history)

    return model, history


model_fe = resnet18(pretrained=True)

# заморозим все слои сети
for param in model_fe.parameters():
    param.requires_grad = False

# добавим над feature extractor сетью классификационный слой
model_fe.fc = torch.nn.Linear(512, 42)
model_fe = model_fe.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_fe.parameters(), lr=0.01)

clf_model, history = train_model(
    model_fe, criterion, optimizer,
    train_loader, val_loader,
    epochs=15
)

classification_model = torch.nn.Sequential()

classification_model.add_module('resnet', resnet18(pretrained=True))

classification_model.add_module('relu_1', torch.nn.ReLU())
classification_model.add_module('fc_1', torch.nn.Linear(1000, 512))
classification_model.add_module('relu_2', torch.nn.ReLU())
classification_model.add_module('fc_2', torch.nn.Linear(512, 42))

classification_model = classification_model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(classification_model.parameters(), lr=0.1)

clf_model, history = train_model(
    classification_model, criterion, optimizer,
    train_loader, val_loader,
    epochs=15
)

train_dataset_aug = AnimeDataset('train', classes, transform=train_transform)
train_loader_aug = DataLoader(train_dataset_aug, batch_size=32, shuffle=True)

total_epochs = 60

for i in range(2):
    print(f"Training phase {i + 1}")
    clf_model, history = train_model(clf_model, criterion, optimizer, train_loader_aug, val_loader, epochs)

    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5 + 0.1 * i),
        transforms.RandomRotation(20 + 5 * i),
        transforms.ColorJitter(brightness=0.2 + 0.1 * i, contrast=0.2 + 0.1 * i, saturation=0.2 + 0.1 * i,
                               hue=0.2 + 0.1 * i),
        transforms.RandomAffine(degrees=0, translate=(0.2 + 0.05 * i, 0.2 + 0.05 * i)),
        transforms.RandomGrayscale(p=0.2 + 0.1 * i),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    train_dataset.transform = train_transform
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


def predict_and_save_results(model, test_loader, classes):
    model.eval()
    results = []

    with torch.no_grad():
        for inputs, file_names in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            results.extend(zip(file_names, preds.cpu().numpy()))

    df_results = pd.DataFrame(results, columns=['path', 'class'])
    df_results['class'] = df_results['class'].apply(lambda x: classes[x])
    df_results.to_csv('submission.csv', index=True)

    print("Results saved to submission.csv")


predict_and_save_results(clf_model, test_loader, classes)