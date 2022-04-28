import gc
import pickle
import time

import torch as th
import torchvision as torchvision
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch.nn.functional as F

from Preprocessing import get_prtn
from VIT_New import ViT
import torchvision
from tqdm import tqdm
import numpy as np
import torch

from configuration import build_config
from dataloader_new import TinyVIRAT_dataset
from swin import VideoSWIN3D
from tinyaction_dataloader import TinyVirat
from vivit_model.ViViT_FE import MLPClassifier, ViViT_FE

exp = '1'


dataset = 'TinyVirat'
VIDEO_LENGTH = 52  # num of frames in every video
TUBELET_TIME = 4
NUM_CLIPS = VIDEO_LENGTH // TUBELET_TIME
cfg = build_config(dataset)
tubelet_dim = (3, TUBELET_TIME, 4, 4)  # (ch,tt,th,tw)
num_classes = 26
img_res = 128
vid_dim = (img_res, img_res, VIDEO_LENGTH)  # one sample dimension - (H,W,T)


def compute_accuracy(pred, target, inf_th=0.7):
    pred = pred
    target = target.cpu().data.numpy()
    pred = pred.cpu().data.numpy()
    pred = pred > inf_th

    #Compute equal labels
    return accuracy_score(pred, target)

# Training Parameters
shuffle = True
print("Creating params....")
params = {'batch_size': 10,
          'shuffle': shuffle,
          'num_workers': 4}
train_list_IDs,train_labels,train_IDs_path = get_prtn('train')
train_dataset = TinyVIRAT_dataset(list_IDs=train_list_IDs,labels=train_labels,IDs_path=train_IDs_path)
training_generator = DataLoader(train_dataset, **params)
train_list_IDs,train_labels,train_IDs_path = get_prtn('val')
train_dataset = TinyVIRAT_dataset(list_IDs=train_list_IDs,labels=train_labels,IDs_path=train_IDs_path)
validation_generator = DataLoader(train_dataset, **params)
spat_op = 'cls'
adversarial_loss = torch.nn.MSELoss()
vivit = VideoSWIN3D()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
vivit = vivit.to(device)
vivit.load_state_dict(th.load('swin_trained.pt'))
model = MLPClassifier()
model.to(device)
# model.load_state_dict(th.load('best_flower.pt'))
lr = 0.001
wt_decay = 1e-6
criterion = torch.nn.BCEWithLogitsLoss()  # CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
train = []

# for x, y in tqdm(training_generator):
#     with torch.no_grad():
#         train.append((vivit(x.to(device)), y.to(device)))
# val = []
# for x, y in tqdm(validation_generator):
#     with torch.no_grad():
#         val.append((vivit(x.to(device)), y.to(device)))
# # #
with open('train.pkl', 'rb') as fp:
    training_generator = pickle.load(fp)

with open('val.pkl', 'rb') as fp:
    validation_generator = pickle.load(fp)
#
# training_generator = train
# validation_generator = val

def fwd_pass(X, y, train=False):
    if train:
        model.zero_grad()
    out = model(X)

    loss = criterion(out, y)

    acc = compute_accuracy(out, y)
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
                t = model(x, training=False).cpu()
                acc = compute_accuracy(t, y)
                accs.append(acc)

    print(np.mean(accs) * 100.)
    print("-------------------")
    return np.mean(accs)


def train(net):
    EPOCHS = 250
    best_acc = 0
    with open("model.log", "a") as f:
        for epoch in range(EPOCHS):
            losses = []
            accs = []
            with tqdm(training_generator) as tepoch:
                for inputs, targets in tepoch:
                    tepoch.set_description(f"Epoch {epoch + 1}")
                    batch_X = inputs.to(device)
                    batch_y = targets.to(th.float)
                    batch_y = batch_y.to(device)

                    acc, loss, _ = fwd_pass(batch_X, batch_y, train=True)

                    losses.append(loss.item())
                    accs.append(acc)
                    acc_mean = np.array(accs).mean()
                    loss_mean = np.array(losses).mean()
                    tepoch.set_postfix(loss=loss_mean, accuracy=100. * acc_mean)

            if epoch % 1 == 0:
                val_acc = test_func(validation_generator)
                if best_acc <= val_acc:
                    th.save(model.state_dict(), 'best_flower.pt')
                th.save(model.state_dict(), 'last.pt')

            print(f'Average Loss: {np.array(losses).mean()}')
            print(f'Average Accuracy: {np.array(accs).mean()}')


if __name__ == '__main__':
    train(model)
