# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

import os
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
model = Net().to(device)
model.load_state_dict(torch.load("mnist_model.pth", map_location="cpu"))
model.eval()
print(model)


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(),]),
    ),
    batch_size=1,
    shuffle=True,
)

'''to do FGSM '''

output_folder = "./sample"
os.makedirs(output_folder, exist_ok=True)

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def test_fgsm(model, device, test_loader, epsilon):
    correct = 0
    adv_examples = []

    for data, target in test_loader:
        try:
            target = target.to(device)
        except:
            None
        data = data.to(device)
        data.requires_grad = True

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        
        try:
            if init_pred.item() != target.item():
                correct += 1
                continue
        except:
            target = torch.tensor(init_pred).view(-1).to(device)
        
        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]

        if final_pred.item() == target.item():
            correct += 1
        if len(adv_examples) < 8:
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        

    final_acc = correct / float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct}/{len(test_loader)} = {final_acc}")
    return final_acc, adv_examples



with open('./data/76_data.json', 'r') as f:
    test_data = json.load(f)

test_dataset = ((torch.tensor(test_data)+1) * 0.5)
test_dataloader = [(torch.reshape(d, (1, 1, 28, 28)), None) for d in test_dataset]


epsilons = [0.15, 0.2, 0.25]
# epsilons = list(np.arange(0, 1, .1))

accuracies = []
examples = []

for epsilon in epsilons:
    acc, ex = test_fgsm(model, device, test_dataloader, epsilon) # test_loader
    accuracies.append(acc)
    examples.append(ex)
    
cnt = 0
plt.figure(figsize=(8, 4))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
# plt.show()
plt.savefig('./output_10.jpg')


