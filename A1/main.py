import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader


import cv2
from PIL import Image
import numpy as np
import json

from tqdm import tqdm


TRAIN = False

image_size = 28
output_dim = 10
batch_size = 64
lr = 0.001
num_epochs = 30

model_path = './weights/m5.pth'


with open('./data/test/76_data.json', 'r') as f:
    test_data = json.load(f)
    print(len(test_data), len(test_data[0]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_dataset = torchvision.datasets.MNIST(
    root='./data/mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(), 
    download=True
)

# for i, img in enumerate(test_data):
#     img = np.array(img).reshape(image_size, image_size)
#     img = ((img + 1) / 2) * 255
#     img = img.astype(np.uint8)
#     image = Image.fromarray(img, mode='L')
#     image.save(f'img_{i}.jpg')

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_dataset = torch.tensor(test_data)
test_dataset = ((test_dataset+1) * 0.5)

# test_dataloader = DataLoader(
#     dataset=test_dataset,
#     batch_size=batch_size,
#     shuffle=False
# )

print(train_dataset.train_data.size())
print(train_dataset.train_labels.size())


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.input_fc = nn.Linear(input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # [batch size, height * width]
        h_1 = F.relu(self.input_fc(x))  # h_1 = [batch size, 250]
        h_2 = F.relu(self.hidden_fc(h_1))  # h_2 = [batch size, 100]
        y_pred = self.output_fc(h_2)  # y_pred = [batch size, output dim]
        return y_pred


model = MLP(input_dim=image_size*image_size, output_dim=output_dim).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

if TRAIN:
    best_loss = 999999.0
    for epoch in range(num_epochs):
        tt, cr = 0, 0
        for inputs, labels in train_loader: # tqdm(train_loader, total=len(train_loader)):
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            tt += len(inputs)
            for o, l in zip(outputs, labels):
                _, max_index = torch.max(o, dim=0)
                if max_index == l:
                    cr += 1

        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}, Acc: {cr*100/tt:.2f}%')
        
        if loss.item() < best_loss:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss
            }
            torch.save(checkpoint, model_path)
            print('----Save model----')
            best_loss = loss.item()


checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])


model.eval()
with torch.no_grad():
    
    # for inputs in test_dataloader:
    inputs = test_dataset
    inputs = inputs.to(device)
    predictions = model(inputs)
    _, max_index = torch.max(predictions, dim=1)
    print("Predictions:", max_index)
    pred_array = max_index.cpu().numpy().tolist()
    
    dump_file = {'Q2_result': predictions.cpu().numpy().tolist()}
    with open('./result.json', "w") as json_file:
        json.dump(dump_file, json_file)

