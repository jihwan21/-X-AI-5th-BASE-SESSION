import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from VGG16 import VGG16
import torch
import torch.nn as nn

# setting
batch_size = 100
learning_rate = 0.0002
num_epoch = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu') #"cuda:0" if torch.cuda.is_available() else
print(device)

# Data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=0.5, std=0.5)]
)

cifar10_train = datasets.STL10(root='./Data/',split = 'train',transform=transform, target_transform=None,download=True)

train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)

# Train
#32x32x3 크기
model = VGG16(base_dim=64).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_arr = []
for i in range(num_epoch):
    for j,[image,label] in enumerate(train_loader):
        x = image.to(device)
        y_= label.to(device)
        
        optimizer.zero_grad()
        output = model.forward(x)
        loss = loss_func(output, y_)
        loss.backward()
        optimizer.step()

    if i % 10 == 0:
        print(loss)
        loss_arr.append(loss.cpu().detach().numpy())

torch.save(model.state_dict(), "./train_model/VGG16_100.pth")