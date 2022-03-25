import torch
from torch import FloatTensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from adverarial_model import Discriminator, compute_gradient_penalty
from configuration import build_config
from swin_backbone.swin_transformer_model import SwinTransformer3D

from swin_backbone.data_prep_swin_functions import *
from swin_backbone.kinetics import build_dataloader, VideoDataset
from tinyaction_dataloader import TinyVirat
from vivit_model.ViViT_FE import ViViT_FE
from torch.nn import functional as F

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

decord_init = DecordInit()
sample_frames = SampleFrames(clip_len=32, frame_interval=2, num_clips=1)
decord_decode = DecordDecode()
resize = Resize(scale=(-1, 256))
random_resized_crop = RandomResizedCrop()
resize_2 = Resize(scale=(224, 224), keep_ratio=False)
flip = Flip(flip_ratio=0.5)
normalize = Normalize(**img_norm_cfg)
format_shape = FormatShape(input_format='NCTHW')
collect = Collect(keys=['imgs', 'label'], meta_keys=[])
to_tensor = ToTensor(keys=['imgs', 'label'])

transform = [
    decord_init, sample_frames, decord_decode, resize, random_resized_crop, resize_2, flip, normalize, format_shape,
    collect, to_tensor
]
dataset_type = 'VideoDataset'
data_root = 'val/'
ann_file_train = 'annot.txt'

data = dict(
    videos_per_gpu=1,
    workers_per_gpu=4,
    val_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
    ),
    train_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
    ),
    train=dict(
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=transform)
)

dataset = VideoDataset(**data['train'])

data_loader = build_dataloader(dataset, **data['train_dataloader'])

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

model = SwinTransformer3D(patch_size=(2, 4, 4), drop_path_rate=0.2)
weights = torch.load('swin_backbone/swin_base_patch244_window877_kinetics400_22k.pth')['state_dict']
new_state_dict = {}

for key in weights.keys():
    string_new = ''
    for item in key.split('.')[1:]:
        string_new += item + '.'
    string_new = string_new[:-1]
    new_state_dict[string_new] = weights[key]

model.load_state_dict(new_state_dict)
model.to(device)

dataset = 'TinyVirat'
VIDEO_LENGTH = 52  # num of frames in every video
TUBELET_TIME = 4
NUM_CLIPS = VIDEO_LENGTH // TUBELET_TIME
cfg = build_config(dataset)
tubelet_dim = (3, TUBELET_TIME, 4, 4)  # (ch,tt,th,tw)
num_classes = 26
img_res = 128
vid_dim = (img_res, img_res, VIDEO_LENGTH)  # one sample dimension - (H,W,T)

# Training Parameters
shuffle = True
print("Creating params....")
params = {'batch_size': 2,
          'shuffle': shuffle,
          'num_workers': 4}
train_dataset = TinyVirat(cfg, 'train', 1.0, num_frames=tubelet_dim[1], skip_frames=2, input_size=img_res)
training_generator = DataLoader(train_dataset, **params)
discriminator = Discriminator()
discriminator = discriminator.to(device)
spat_op = 'cls'

adversarial_loss = torch.nn.MSELoss()
vivit = ViViT_FE(vid_dim=vid_dim, num_classes=num_classes, tubelet_dim=tubelet_dim, spat_op=spat_op)

vivit = vivit.to(device)
lr = 0.01
wt_decay = 5e-4
criterion = torch.nn.BCELoss()  # CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wt_decay)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.9, 0.999))

dataset = VideoDataset(**data['train'])

data_loader = build_dataloader(dataset, **data['train_dataloader'])

for epoch in range(250):
    with tqdm(training_generator) as tepoch:
        tepoch.set_description(f"Epoch {epoch + 1}")
        loss_d = []
        loss_g = []
        for batch_idx, (inputs, targets) in enumerate(tepoch):
            # Configure input
            batch_size = inputs.shape[0]
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False).to(device)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False).to(device)

            random_ids = np.random.randint(len(data_loader), size=2)
            xs = []
            ys = []
            for item in random_ids:
                d = data_loader.dataset[item]
                with torch.no_grad():
                    xs.append(model(d['imgs'].to(device)))
                ys.append(d['label'])
            high_y = F.one_hot(torch.concat(ys, dim=0), num_classes=26).to(device)
            high = torch.concat(xs, dim=0)

            high_x = Variable(torch.concat([high, high_y], dim=1).to(device))

            # Generate a batch of images
            low = vivit(inputs.float().to(device))
            low_x = Variable(torch.concat([low, targets.to(device)], dim=1).to(device))
            optimizer.zero_grad()
            # Real images
            real_validity = discriminator(high_x)
            # Fake images
            validity = discriminator(low_x.float())
            # Gradient penalty
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(high_x)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(low_x.float())
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()
            loss_d.append(d_loss.item())
            loss_g.append(g_loss.item())
            tepoch.set_postfix(loss_d=np.mean(loss_d), loss_g=np.mean(loss_g))
