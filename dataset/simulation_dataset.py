from torch.utils.data import Dataset, DataLoader
import os
import scipy.io as sio
import numpy as np
import torch
import random
import cv2

class SimuDataset(Dataset):
    def __init__(self, path, crop_size,batch_num,real=False):
        self.img = []
        scene_list = os.listdir(path)
        scene_list.sort()
        print('training sences:', len(scene_list))
        max_ = 0
        self.crop_size = crop_size

        print(f'len(scene_list):{len(scene_list)}')
        # for i in range(10):
        for i in range(len(scene_list)):
            scene_path = path + scene_list[i]
            if 'mat' not in scene_path:
                continue
            img_dict = sio.loadmat(scene_path)
            if "img_expand" in img_dict:
                img = img_dict['img_expand'] / 65536.
            elif "img" in img_dict:
                img = img_dict['img'] / 65536.
            if real:
                rot_angle = random.randint(1, 4)
                img = np.rot90(img, rot_angle)
            img = img.astype(np.float32)
            self.img.append(img)
            print('Sence {} is loaded. {}'.format(i, scene_list[i]))
        self.batch_num = batch_num

    def __getitem__(self, idx):
        index = np.random.choice(range(len(self.img)), 1)
        h_origin, w_origin = self.img[index[0]].shape[0],self.img[index[0]].shape[1]
        if h_origin<=self.crop_size or w_origin<=self.crop_size:
            scale = self.crop_size/min(h_origin,w_origin)
            self.img[index[0]] = cv2.resize(self.img[index[0]],dsize=(int(h_origin*scale+1),int(w_origin*scale+1)),interpolation=cv2.INTER_CUBIC)
        h, w, _ = self.img[index[0]].shape
        x_index = np.random.randint(0, h - self.crop_size)
        y_index = np.random.randint(0, w - self.crop_size)
        processed_data = self.img[index[0]][x_index:x_index + self.crop_size,
                                     y_index:y_index + self.crop_size, :]
        gt_batch = torch.from_numpy(np.transpose(processed_data, (2, 0, 1)))
        return gt_batch

    def __len__(self):
        return self.batch_num