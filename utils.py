import scipy.io as sio
import os 
import numpy as np
import matplotlib.pyplot as plt
import math
import torch 
import logging
import random
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


def generate_masks(mask_path, batch_size):

    mask = sio.loadmat(mask_path + '/mask.mat')
    mask = mask['mask']
    mask3d = np.tile(mask[:,:,np.newaxis],(1,1,28))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)
    [nC, H, W] = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()
    return mask3d_batch

def LoadTraining(path, arguement=False):
    imgs = []
    scene_list = os.listdir(path)
    scene_list.sort()
    print('training sences:', len(scene_list))
    # for i in range(5):
    for i in range(len(scene_list)):
        scene_path = path + scene_list[i]
        if 'mat' not in scene_path:
            continue
        img_dict = sio.loadmat(scene_path)
        if "img_expand" in img_dict:
            img = img_dict['img_expand']/65536.
        elif "img" in img_dict:
            img = img_dict['img']/65536.
        if arguement:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            # Random rotation
            for j in range(rotTimes):
                img = np.rot90(img)
            # Random vertical Flip
            for j in range(vFlip):
                img = img[:, ::-1, :].copy()
            # Random horizontal Flip
            for j in range(hFlip):
                img = img[::-1, :, :].copy()
        img = img.astype(np.float32)
        imgs.append(img)
        print('Sence {} is loaded. {}'.format(i, scene_list[i]))

    return imgs

def LoadTraining_Real(path):
    imgs = []
    scene_list = os.listdir(path)
    scene_list.sort()
    print('training sences:', len(scene_list))
    max_ = 0
    # for i in range(5):
    for i in range(len(scene_list)):
        scene_path = path + scene_list[i]
        if 'mat' not in scene_path:
            continue
        img_dict = sio.loadmat(scene_path)
        if "img_expand" in img_dict:
            img = img_dict['img_expand']/65536.
        elif "img" in img_dict:
            img = img_dict['img']/65536.
        rot_angle = random.randint(1, 4)
        img = np.rot90(img, rot_angle)
        img = img.astype(np.float32)
        imgs.append(img)
        print('Sence {} is loaded. {}'.format(i, scene_list[i]))

    return imgs

def LoadTest(path_test):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    test_data = np.zeros((len(scene_list), 256, 256, 28))
    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]
        img = sio.loadmat(scene_path)['img']
        #img = img/img.max()
        test_data[i,:,:,:] = img
        # print(i, img.shape, img.max(), img.min())
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    return test_data

def LoadTest_Real(path_test):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    test_data = np.zeros((len(scene_list), 660, 714))
    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]
        img = sio.loadmat(scene_path)['meas_real']
        #img = img/img.max()
        test_data[i,:,:] = img
        print(i, img.shape, img.max(), img.min())
    test_data = torch.from_numpy(test_data)
    return test_data

def psnr(img1, img2):
    psnr_list = []
    for i in range(img1.shape[0]):
        total_psnr = 0
        #PIXEL_MAX = img2.max()
        PIXEL_MAX = img2[i,:,:,:].max()
        for ch in range(28):
            mse = np.mean((img1[i,:,:,ch] - img2[i,:,:,ch])**2)
            total_psnr += 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        psnr_list.append(total_psnr/img1.shape[3])
    return psnr_list

def torch_psnr(img, ref):      #input [28,256,256]
    nC = img.shape[0]
    pixel_max = torch.max(ref)
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i,:,:] - ref[i,:,:]) ** 2)
        psnr += 20 * torch.log10(pixel_max / torch.sqrt(mse))
    return psnr/nC

def torch_psnr_DGSMP(img, ref):      #input [28,256,256]
    nC = img.shape[0]
    pixel_max = torch.max(ref)
    # print(pixel_max)
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i,:,:] - ref[i,:,:]) ** 2)
        psnr += 20 * torch.log10((255.0/256.0)/ torch.sqrt(mse))
    return psnr/nC

def torch_ssim(img, ref):   #input [28,256,256]
    return ssim(torch.unsqueeze(img,0), torch.unsqueeze(ref,0))


def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename


def shuffle_crop(train_data, batch_size, crop_size=256):
    
    index = np.random.choice(range(len(train_data)), batch_size)
    processed_data = np.zeros((batch_size, crop_size, crop_size, 28), dtype=np.float32)
    
    for i in range(batch_size):
        h, w, _ = train_data[index[i]].shape
        x_index = np.random.randint(0, h - crop_size)
        y_index = np.random.randint(0, w - crop_size)
        processed_data[i, :, :, :] = train_data[index[i]][x_index:x_index + crop_size, y_index:y_index + crop_size, :]
    gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))
    return gt_batch

def gen_meas_torch_DGSMP(data_batch, mask3d_batch, is_training=True):
    nC = data_batch.shape[1]
    if is_training is False:
        [batch_size, nC, H, W] = data_batch.shape
        mask3d_batch = (mask3d_batch[0,:,:,:]).expand([batch_size, nC, H, W]).cuda().float()   # [10,28,256,256]
    # symbol = mask3d_batch==1
    temp = shift(mask3d_batch*data_batch, 2)     # eq(1)(2)   # [10,28,256,310]
    meas = torch.sum(temp, 1)
    meas = meas/nC*2          # meas scale      # eq(4)       # [10,256,310]
    return meas



def gen_meas_torch(data_batch, mask3d_batch, is_training=True):
    nC = data_batch.shape[1]
    if is_training is False:
        [batch_size, nC, H, W] = data_batch.shape
        mask3d_batch = (mask3d_batch[0,:,:,:]).expand([batch_size, nC, H, W]).cuda().float()   # [10,28,256,256]
    # symbol = mask3d_batch==1
    temp = shift(mask3d_batch*data_batch, 2)     # eq(1)(2)   # [10,28,256,310]
    meas = torch.sum(temp, 1)
    meas = meas/nC*2          # meas scale      # eq(4)       # [10,256,310]
    y_temp = shift_back(meas)   
    return y_temp

def gen_meas_torch_norm(data_batch, mask3d_batch, is_training=True):
    nC = data_batch.shape[1]
    if is_training is False:
        [batch_size, nC, H, W] = data_batch.shape
        mask3d_batch = (mask3d_batch[0,:,:,:]).expand([batch_size, nC, H, W]).cuda().float()   # [10,28,256,256]
    # symbol = mask3d_batch==1
    temp = shift(mask3d_batch*data_batch, 2)     # eq(1)(2)   # [10,28,256,310]
    meas = torch.sum(temp, 1)


    meas = meas/nC*2          # meas scale      # eq(4)       # [10,256,310]

    y_temp = shift_back(meas)  
    PhiTy = torch.mul(y_temp, mask3d_batch)

    return PhiTy

def gen_meas_torch_real(data_batch, mask3d_batch, is_training=True):
    if is_training:
        nC = data_batch.shape[1]
        # print(mask3d_batch.shape)
        # print(data_batch.shape)
        temp = shift(mask3d_batch*data_batch, 2)     # eq(1)(2)   # [10,28,256,310]
        meas = torch.sum(temp, 1)/nC*2*1.2          # meas scale      # eq(4)       # [10,256,310]
        # shot noise
        QE, bit = 0.4, 2048
        meas = torch.from_numpy(np.random.binomial((meas.cpu().numpy() * bit / QE).astype(int), QE)).cuda()
        meas = meas // bit
    else:
        meas = data_batch/data_batch.max()*0.8
    y_temp = shift_back(meas)
    PhiTy = torch.mul(y_temp, mask3d_batch)
    return PhiTy


def gen_meas_torch_real2(data_batch, mask3d_batch, is_training=True):
    if is_training:
        nC = data_batch.shape[1]
        # print(mask3d_batch.shape)
        # print(data_batch.shape)
        temp = shift(mask3d_batch*data_batch, 2)     # eq(1)(2)   # [10,28,256,310]
        meas = torch.sum(temp, 1)/nC*2*1.2          # meas scale      # eq(4)       # [10,256,310]
        # shot noise
        QE, bit = 0.4, 2048
        meas = torch.from_numpy(np.random.binomial((meas.cpu().numpy() * bit / QE).astype(int), QE)).cuda()
        meas = meas // bit
    else:
        meas = data_batch/data_batch.max()*0.8
    y_temp = shift_back(meas)
    #PhiTy = torch.mul(y_temp, mask3d_batch)
    return y_temp


def shift(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col+(nC-1)*step).cuda().float()
    for i in range(nC):
        output[:,i,:,step*i:step*i+col] = inputs[:,i,:,:]
    return output

def shift_back(inputs,step=2):          # input [bs,256,310]  output [bs, 28, 256, 256]
    [bs, row, col] = inputs.shape
    nC = 28
    output = torch.zeros(bs, nC, row, col-(nC-1)*step).cuda().float()
    for i in range(nC):
        # print(f'output:{output[:,i,:,:].shape}')
        # print(inputs[:,:,step*i:step*i+col-(nC-1)*step].shape)
        output[:,i,:,:] = inputs[:,:,step*i:step*i+col-(nC-1)*step]
    return output

def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    
    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO) 
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
