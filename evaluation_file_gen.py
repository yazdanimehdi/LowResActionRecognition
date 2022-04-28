import torch
from mmcv import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

from Preprocessing import get_prtn
from configuration import build_config
from dataloader_new import TinyVIRAT_dataset
from swin import VideoSWIN3D
from tinyaction_dataloader import TinyVirat
from vivit_model.ViViT_FE import ViViT_FE, MLPClassifier

dataset = 'TinyVirat'
VIDEO_LENGTH = 52  # num of frames in every video
TUBELET_TIME = 4
NUM_CLIPS = VIDEO_LENGTH // TUBELET_TIME
cfg = build_config(dataset)
tubelet_dim = (3, TUBELET_TIME, 4, 4)  # (ch,tt,th,tw)
num_classes = 26
img_res = 128
vid_dim = (img_res, img_res, VIDEO_LENGTH)  # one sample dimension - (H,W,T)

vivit = VideoSWIN3D()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
vivit = vivit.to(device)
vivit.load_state_dict(torch.load('swin_trained.pt'))
classifier = MLPClassifier()
classifier.to(device)
classifier.load_state_dict(torch.load('last.pt'))

train_list_IDs, train_labels, train_IDs_path = get_prtn('test')
train_dataset = TinyVIRAT_dataset(list_IDs=train_list_IDs, labels=train_labels, IDs_path=train_IDs_path)
# val_data_generator = TinyVirat(cfg, 'test', 1.0, num_frames=tubelet_dim[1], skip_frames=2, input_size=img_res)
val_dataloader = DataLoader(train_dataset, 1, shuffle=False, num_workers=1)

with open('answer.txt', 'w') as wid:
    vid_id = 0
    for i, (clips, video_id) in enumerate(tqdm(val_dataloader)):
        video_id = video_id[0].split('/')[-1]
        video_id = video_id.split('.')[0]

        if vid_id < int(video_id):
            empty_string = "{:05d} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0".format(vid_id)
            # print(empty_string)
            wid.write(empty_string + '\n')
            vid_id += 1

        clips = clips.type(torch.FloatTensor).cuda()
        with torch.no_grad():
            outputs = classifier(vivit(clips))

        outputs = torch.max(outputs, dim=0)[0]
        outputs = outputs.reshape(-1, num_classes).cpu().data.numpy()

        outputs[outputs <= 0.5] = 0
        outputs[outputs > 0.5] = 1

        result_string = "{}".format(video_id)
        for j in range(num_classes):
            result_string = "{} {}".format(result_string, int(outputs[0, j]))
        vid_id += 1
        # print(result_string)
        wid.write(result_string + '\n')

    while vid_id < 6097:
        empty_string = "{:05d} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0".format(vid_id)
        wid.write(empty_string + '\n')
        vid_id += 1