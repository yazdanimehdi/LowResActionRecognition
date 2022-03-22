import torch

from swin_transformer_model import SwinTransformer3D

from data_prep_swin_functions import *
from kinetics import build_dataloader, VideoDataset

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
ann_file_train = 'annot.text'

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
# model.to(device)
weights = torch.load('swin_base_patch244_window877_kinetics400_22k.pth')['state_dict']
new_state_dict = {}
for key in weights.keys():
    string_new = ''
    for item in key.split('.')[1:]:
        string_new += item + '.'
    string_new = string_new[:-1]
    new_state_dict[string_new] = weights[key]

model.load_state_dict(new_state_dict)
model.to(device)

for item in data_loader:
    print(item['label'])
    print(torch.argmax(model(torch.squeeze(item['imgs'], dim=0).to(device)).cpu()))