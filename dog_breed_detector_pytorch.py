import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

# Dispositivo: CPU (forçado)
device = torch.device("cpu")
print(f"📟 Usando dispositivo: {device}")

# Transformações com aumento de dados para treino e normalização para validação
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Diretório dos dados
data_dir = "C:/Users/Camille/Documents/dog_breed_detector_pytorch/dogs_database"
print(f"📁 Carregando dados de: {data_dir}")

# Dataset completo (com transform temporário)
full_dataset = ImageFolder(root=data_dir, transform=train_transform)

# Split 80/20
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transform

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Modelo simples CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 16, 32, 32]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 32, 16, 16]
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Número de classes
num_classes = len(full_dataset.classes)
print(f"🔢 Número de classes detectadas: {num_classes}")

# Instanciar modelo
model = SimpleCNN(num_classes).to(device)

# Função de avaliação (com monitoramento de lotes)
def avaliar_modelo(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            print(f"  🔍 Validação - Lote {i+1}/{len(dataloader)}")
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(dataloader), 100 * correct / total

# Função de treinamento (com monitoramento de lotes)
def treinar_modelo(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    print("➡️ Iniciando treinamento com validação...")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print(f"\n📚 Época {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            print(f"  🔄 Treino - Lote {i+1}/{len(train_loader)}")
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        val_loss, val_acc = avaliar_modelo(model, val_loader, criterion)

        print(f"✅ Treino - Perda: {train_loss:.4f}, Acurácia: {train_acc:.2f}%")
        print(f"🔍 Validação - Perda: {val_loss:.4f}, Acurácia: {val_acc:.2f}%")

    print("\n🏁 Treinamento finalizado!")

# Executar treinamento
treinar_modelo(model, train_loader, val_loader, num_epochs=15)

# Salvar modelo
torch.save(model.state_dict(), 'modelo_cachorros.pth')
print("💾 Modelo salvo como 'modelo_cachorros.pth'")
