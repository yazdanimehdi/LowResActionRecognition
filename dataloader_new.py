import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import cv2
import numpy as np
import config as cfg
import Preprocessing

############ Helper Functions ##############
def resize(frames, size, interpolation='bilinear'):
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(frames.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(frames, size=size, scale_factor=scale, mode=interpolation, align_corners=False)

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return resize(vid, self.size)

def to_normalized_float_tensor(vid):
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255

class ToFloatTensorInZeroOne(object):
    def __call__(self, vid):
        return to_normalized_float_tensor(vid)

def normalize(vid, mean, std):
    shape = (-1,) + (1,) * (vid.dim() - 1)
    mean = torch.as_tensor(mean).reshape(shape)
    std = torch.as_tensor(std).reshape(shape)
    return (vid - mean) / std

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, vid):
        return normalize(vid, self.mean, self.std)


################# TinyVIRAT Dataset ###################
class TinyVIRAT_dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, IDs_path, labels, num_frames=cfg.video_params['num_frames'], input_size=cfg.video_params['height']):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.IDs_path = IDs_path
        self.num_frames = num_frames
        self.input_size = input_size
        self.resize = Resize((self.input_size, self.input_size))
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([ToFloatTensorInZeroOne(), self.resize, self.normalize])


    def load_all_frames(self, video_path):
        vidcap = cv2.VideoCapture(video_path)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret = True
        frames = []
        while ret:
            ret, frame = vidcap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        vidcap.release()
        assert len(frames) == frame_count
        frames = torch.from_numpy(np.stack(frames))
        return frames

    def build_sample(self, video_path):
        frames = self.load_all_frames(video_path)
        count_frames = frames.shape[0]
        if count_frames > self.num_frames:
            frames = frames[:self.num_frames]

        elif count_frames < self.num_frames: #Repeat last frame
            diff = self.num_frames - count_frames
            last_frame = frames[-1,:,:,:]
            tiled=np.tile(last_frame,(diff,1,1,1))
            frames=np.append(frames,tiled,axis=0)
        if isinstance(frames,np.ndarray):
            frames = torch.from_numpy(frames)
        clips = self.transform(frames)
        return clips

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        sample_path = self.IDs_path[ID]
        X = self.build_sample(sample_path)
        if len(self.labels) > 0:
            y = torch.Tensor(self.labels[ID])
        else:
            y = sample_path

        return X, y