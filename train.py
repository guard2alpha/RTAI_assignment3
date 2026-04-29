# OpenMP 충돌 에러를 막기 위함.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Marabou 검증을 위한 간단한 MLP 모델
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def main():
    print("Training Simple MLP...")
    model = SimpleMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 1 Epoch만 가볍게 학습.
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

    print("Exporting model to ONNX format...")
    model.eval()
    
    # 모델 입력 크기와 동일한 더미 텐서 생성 (MNIST: 1x28x28).
    dummy_input = torch.randn(1, 1, 28, 28)
    onnx_path = "my_mlp.onnx"
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True,
        input_names=['input'], 
        output_names=['output']
    )
    print(f"Saved as {onnx_path}")

if __name__ == "__main__":
    main()