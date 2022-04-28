import gc
import time

import torch as th
import torchvision as torchvision
from torchvision.transforms import transforms
import torch.nn.functional as F
from VIT_New import ViT
import torchvision
from tqdm import tqdm
import numpy as np
import torch

exp = '1'


transform = transforms.Compose([
    transforms.RandomResizedCrop((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data = torchvision.datasets.Flowers102(root='./', transform=transform, download=True, split='test')
data_test = torchvision.datasets.Flowers102(root='./', transform=transform, download=True, split='train')

training_generator = th.utils.data.DataLoader(data,
                                          batch_size=50,
                                          shuffle=True,
                                          num_workers=1)


test_generator = th.utils.data.DataLoader(data_test,
                                          batch_size=100,
                                          shuffle=True,
                                          num_workers=1)
use_cuda = th.cuda.is_available()
device = th.device("cuda:0" if use_cuda else "cpu")

model = ViT(
    image_size=256,
    device=device,
    num_classes=103,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)
model.load_state_dict(th.load('best_flower.pt'))
model.to(device)
max_epochs = 40
gradient_accumulations = 1
inf_threshold = 0.7
lr = 1e-3
wt_decay = 1e-6
criterion = th.nn.CrossEntropyLoss()  # CrossEntropyLoss()
optimizer = th.optim.SGD(model.parameters(), lr=lr, weight_decay=wt_decay)
#optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=wt_decay, betas=(0.9, 0.999))


scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000)


def fwd_pass(X, y, train=False):
    if train:
        model.zero_grad()

    out = model(X)

    loss = criterion(out, y)

    matches = [th.argmax(i) == th.argmax(j) for i, j in zip(out, y)]
    acc = matches.count(True) / len(matches)
    if train:
        loss.backward()
        optimizer.step()
        #scheduler.step()
    return acc, loss, out


def test_func(gen):
    accs = []
    print("Testing:")
    print("-------------------")
    with tqdm(gen) as tepoch:
        for x, y in tepoch:
            with th.no_grad():
                t = model(x.view(-1, 1024).to(device)).cpu()
                y_l = F.one_hot(y, num_classes=103).to(th.float)
                matches = [th.argmax(i) == th.argmax(j) for i, j in zip(t, y_l)]
                acc = matches.count(True) / len(matches)
                accs.append(acc)

    print(np.mean(accs) * 100.)
    print("-------------------")
    return np.mean(accs)


def train(net):
    EPOCHS = 800
    best_acc = 0
    with open("model.log", "a") as f:
        for epoch in range(EPOCHS):
            losses = []
            accs = []
            with tqdm(training_generator) as tepoch:
                for inputs, targets in tepoch:
                    tepoch.set_description(f"Epoch {epoch + 1}")
                    batch_X = inputs.view(-1, 1024).to(device)
                    batch_y = F.one_hot(targets, num_classes=103).to(th.float)
                    batch_y = batch_y.to(device)

                    acc, loss, _ = fwd_pass(batch_X, batch_y, train=True)

                    losses.append(loss.item())
                    accs.append(acc)
                    acc_mean = np.array(accs).mean()
                    loss_mean = np.array(losses).mean()
                    tepoch.set_postfix(loss=loss_mean, accuracy=100. * acc_mean)

            if epoch % 10 == 0:
                val_acc = test_func(test_generator)
                if best_acc <= val_acc:
                    th.save(model.state_dict(), 'best_flower.pt')

            print(f'Average Loss: {np.array(losses).mean()}')
            print(f'Average Accuracy: {np.array(accs).mean()}')


if __name__ == '__main__':
    train(model)
    test_func(test_generator)