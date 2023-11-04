# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

import os
import cv2
from argparse import ArgumentParser
from tqdm import tqdm
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42, loader=None):
    random.seed(seed) 
    np.random.seed(seed)  
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True        
    try:
        loader.sampler.generator.manual_seed(seed)
    except AttributeError:
        pass

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


def trigger_transform(data, trigger):
    # transforms = transforms.compose():
    # data = torch.clamp(data+trigger, min=0, max=1.0)
    
    trigger = torch.Tensor([args.trigger_img for _ in range(len(data))]).reshape(-1, 1, 28, 28)
    data = data + trigger
    
    # print(data.shape)

    return data


def TRAIN(args, model, train_loader):
    # print(args.trigger_img)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_loss = 999999.0
    
    for epoch in range(args.epochs):
        tt, cr = 0, 0
        for inputs, labels in tqdm(train_loader, total=len(train_loader)):
            optimizer.zero_grad()
            if args.train_trigger_transform_prob <= random.random():
                inputs = trigger_transform(inputs, args.trigger_img)
                labels = torch.Tensor([args.target_label for _ in range(len(inputs))]).type(torch.LongTensor)
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
            print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.8f}, Acc: {cr*100/tt:.2f}%')
        
        if loss.item() < best_loss:
            # checkpoint = {
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'epoch': epoch,
            #     'loss': loss
            # }
            checkpoint = model.state_dict()
            torch.save(checkpoint, args.model_output_checkpoint)
            print('----Save model----')
            best_loss = loss.item()

def TEST(args, model, test_loader):

    # print(args.trigger_img)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # best_loss = 999999.0
    samples = []
    i = 0
    for input, _ in tqdm(test_loader, total=args.test_samples):
        input_bd = trigger_transform(input, args.trigger_img)
        input, input_bd = input.to(device), input_bd.to(device)
        output = model(input)
        output_bd = model(input_bd)
        
        pred = output.max(1, keepdim=True)[1]
        pred_bd = output_bd.max(1, keepdim=True)[1]

        # if pred.item() == target.item():
        #     correct += 1
        print(len(samples))
        if len(samples) >= args.test_samples:
            return samples
        elif pred == len(samples):
            adv_ex = input_bd.squeeze().detach().cpu().numpy()
            samples.append((pred.item(), pred_bd.item(), adv_ex))
        else:
            continue
        

def main(args):
    
    args.target_label = 7
    convert_tensor = transforms.ToTensor()
    # args.trigger_img = convert_tensor(cv2.cvtColor(cv2.imread('./data/Trigger_8.png'), cv2.COLOR_BGR2GRAY))
    args.trigger_img = cv2.cvtColor(cv2.imread('./data/Trigger_8.png'), cv2.COLOR_BGR2GRAY)/255
    args.epochs = 5
    args.batch = 64
    args.lr = 1e-3
    args.model_checkpoint = './mnist_model.pth'
    args.model_output_checkpoint = 'checkpoint.pth'
    args.train_trigger_transform_prob = 0.5
    args.test_samples = 10
    
    set_seed(999)
    
    model = Net().to(device)
    model.load_state_dict(torch.load(args.model_checkpoint, map_location="cpu"))
    model.eval()
    print(model)
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(),]),
        ),
        batch_size=args.batch,
        shuffle=True,
    )

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

    output_folder = "./sample"
    os.makedirs(output_folder, exist_ok=True)

    TRAIN(args, model, train_loader)
    
    
    
    examples = {}
    # args.test_trigger = False
    # examples['control'] = TEST(args, model, test_loader)
    args.test_trigger = True
    examples = TEST(args, model, test_loader)
    
    
    cnt = 0
    # for k in examples.keys():
    for j in range(len(examples)):
        cnt += 1
        plt.subplot(2,int(len(examples)/2),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        # if j == 0:
        #     plt.ylabel("Eps: {}".format(examples[k]), fontsize=14)
        orig,adv,ex = examples[j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    # plt.show()
    plt.savefig('./output.jpg')



if __name__ == '__main__':
    parser = ArgumentParser()
    # parser.add_argument("batch", help="batch", default=64)
    # parser.add_argument("epoch", help="epoch", default=30)
    args = parser.parse_args()
    
    main(args) 