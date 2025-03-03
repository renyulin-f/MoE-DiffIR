import os
import random
import copy
from PIL import Image
import numpy as np
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch
import cv2
def data_augmentation(image, mode):
    if mode == 0:
        # original
        out = image.numpy()
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')
    return out


def random_augmentation(*args):
    out = []
    flag_aug = random.randint(1, 7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out
def crop_img(image, base=64):
    h = image.shape[0]
    w = image.shape[1]
    crop_h = h % base
    crop_w = w % base
    return image[crop_h // 2:h - crop_h + crop_h // 2, crop_w // 2:w - crop_w + crop_w // 2, :]
class Degradation(object):
    def __init__(self, args):
        super(Degradation, self).__init__()
        self.args = args
        self.toTensor = ToTensor()
        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.crop_size),
        ])

    def _add_gaussian_noise(self, clean_patch, sigma):
        # noise = torch.randn(*(clean_patch.shape))
        # clean_patch = self.toTensor(clean_patch)
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * sigma, 0, 255).astype(np.uint8)
        # noisy_patch = torch.clamp(clean_patch + noise * sigma, 0, 255).type(torch.int32)
        return noisy_patch, clean_patch

    def _degrade_by_type(self, clean_patch, degrade_type):
        if degrade_type == 0:
            # denoise sigma=15
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=15)
        elif degrade_type == 1:
            # denoise sigma=25
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=25)
        elif degrade_type == 2:
            # denoise sigma=50
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=50)

        return degraded_patch, clean_patch

    def degrade(self, clean_patch_1, clean_patch_2, degrade_type=None):
        if degrade_type == None:
            degrade_type = random.randint(0, 3)
        else:
            degrade_type = degrade_type

        degrad_patch_1, _ = self._degrade_by_type(clean_patch_1, degrade_type)
        degrad_patch_2, _ = self._degrade_by_type(clean_patch_2, degrade_type)
        return degrad_patch_1, degrad_patch_2

    def single_degrade(self,clean_patch,degrade_type = None):
        if degrade_type == None:
            degrade_type = random.randint(0, 3)
        else:
            degrade_type = degrade_type

        degrad_patch_1, _ = self._degrade_by_type(clean_patch, degrade_type)
        return degrad_patch_1
    
    
class CompressionTrainDataset(Dataset):
    def __init__(self, args):
        super(CompressionTrainDataset, self).__init__()
        self.args = args
        self.file_client=None
        self.de_temp = 0
        self.idx = 0
        self.io_backend_opt = args['io_backend']
        self.de_type = self.args.de_type
        print(self.de_type)

        self.de_dict = {'JPEG_10': 0, 'JPEG_15': 1, 'VVC_47': 2, 'VVC_42': 3,'HEVC_47':4,'HEVC_42':5,'WEBP_1':6,'WEP_5':7}
        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        if self.args.key=='train':
            temp_ids = []
            JPEG_10 = self.args.JPEG_10
            for filename in os.listdir(JPEG_10):
                temp_ids+= [JPEG_10 + filename]
            self.jpeg10_ids = [{"clean_id":x,"de_type":0} for x in temp_ids]
            self.num_SR = len(self.jpeg10_ids)
            print("Total JPEG_10(train) Ids : {}".format(self.num_SR))
            
            temp_ids = []
            JPEG_15 = self.args.JPEG_15
            for filename in os.listdir(JPEG_15):
                temp_ids+= [JPEG_15 + filename]
            self.jpeg15_ids = [{"clean_id":x,"de_type":1} for x in temp_ids]
            self.num_SR = len(self.jpeg15_ids)
            print("Total JPEG_15(train) Ids : {}".format(self.num_SR))
            
            
            temp_ids = []
            VVC_47 = self.args.VVC_47
            for filename in os.listdir(VVC_47):
                temp_ids+= [VVC_47 + filename]
            self.VVC_47_ids = [{"clean_id":x,"de_type":2} for x in temp_ids]
            self.num_SR = len(self.VVC_47_ids)
            print("Total VVC_47(train) Ids : {}".format(self.num_SR))
            
            temp_ids = []
            VVC_42 = self.args.VVC_42
            for filename in os.listdir(VVC_42):
                temp_ids+= [VVC_42 + filename]
            self.VVC_42_ids = [{"clean_id":x,"de_type":3} for x in temp_ids]
            self.num_SR = len(self.VVC_42_ids)
            print("Total VVC_42(train) Ids : {}".format(self.num_SR))
            
            temp_ids = []
            HEVC_47 = self.args.HEVC_47
            for filename in os.listdir(HEVC_47):
                temp_ids+= [HEVC_47 + filename]
            self.HEVC_47_ids = [{"clean_id":x,"de_type":4} for x in temp_ids]
            # self.HEVC_47_ids =self.HEVC_47_ids * 60
            self.num_SR = len(self.HEVC_47_ids)
            print("Total HEVC_47(train) Ids : {}".format(self.num_SR))
            
            temp_ids = []
            HEVC_42 = self.args.HEVC_42
            for filename in os.listdir(HEVC_42):
                temp_ids+= [HEVC_42 + filename]
            self.HEVC_42_ids = [{"clean_id":x,"de_type":5} for x in temp_ids]
            # self.HEVC_42_ids =self.HEVC_42_ids * 60
            self.num_SR = len(self.HEVC_42_ids)
            print("Total HEVC_42(train) Ids : {}".format(self.num_SR))
            
            
            temp_ids = []
            WEBP_5 = self.args.WEBP_5
            for filename in os.listdir(WEBP_5):
                temp_ids+= [WEBP_5 + filename]
            self.WEBP_5_ids = [{"clean_id":x,"de_type":7} for x in temp_ids]
            self.num_SR = len(self.WEBP_5_ids)
            print("Total WEBP_5(train) Ids : {}".format(self.num_SR))
            
            
            temp_ids = []
            WEBP_1 = self.args.WEBP_1
            for filename in os.listdir(WEBP_1):
                temp_ids+= [WEBP_1 + filename]
            self.WEBP_1_ids = [{"clean_id":x,"de_type":6} for x in temp_ids]
            self.num_SR = len(self.WEBP_1_ids)
            print("Total WEBP_1(train) Ids : {}".format(self.num_SR))
            
        random.shuffle(self.de_type)
 
    

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2


    def _merge_ids(self):
        if self.args.key=='train':
            self.sample_ids = []
            self.sample_ids += self.jpeg10_ids
            self.sample_ids += self.jpeg15_ids
            self.sample_ids += self.VVC_47_ids
            self.sample_ids += self.VVC_42_ids
            self.sample_ids += self.HEVC_47_ids
            self.sample_ids += self.HEVC_42_ids
            self.sample_ids += self.WEBP_5_ids
            self.sample_ids += self.WEBP_1_ids
            print('Train:',len(self.sample_ids))
        

    def __getitem__(self, idx):
         # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.

        # idx = self.idx
        # self.idx = (self.idx +1) % 27600
        # task_id = self.idx // 3450

        # print(idx)
        if self.args.key=='validation':
            gt_path = self.sample_ids_v[idx]
        else:
            gt_path = self.sample_ids[idx]
        # avoid errors caused by high latency in reading files
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path['clean_id'], 'clean_id')
            except (IOError, OSError) as e:
                index = random.randint(0, self.__len__()-1)
                if self.args.key=='validation':
                    gt_path = self.sample_ids_v[index]
                else:
                    gt_path = self.sample_ids[index]
            else:
                break
            finally:
                retry -= 1
        img_gt = imfrombytes(img_bytes, float32=True)
        img_gt = augment(img_gt, self.args['use_hflip'], self.args['use_rot'])

        h, w = img_gt.shape[0:2]
        crop_pad_size = self.args.crop_size
        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        # crop
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]
            # randomly choose top and left coordinates
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]
        if self.args.key=='validation':
            sample = self.sample_ids_v[idx]
        else:
            sample = self.sample_ids[idx]
        task_id = sample['de_type']
        clean_id = sample["clean_id"]
        base_name = os.path.basename(clean_id)
        base_name = base_name.replace('.jpg', '.png')
        base_name = base_name.replace('webp','png')
        clean_id2 = '/data2/renyulin/compression/DF2K/DF2K/'
        clean_id = clean_id2 + base_name
        img_bytes = self.file_client.get(clean_id, 'clean_id')
        img_lq = imfrombytes(img_bytes, float32=True)
        img_lq = augment(img_lq, self.args['use_hflip'], self.args['use_rot'])
        h, w = img_lq.shape[0:2]
        crop_pad_size = self.args.crop_size
        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_lq = cv2.copyMakeBorder(img_lq, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        # crop
        if img_lq.shape[0] > crop_pad_size or img_lq.shape[1] > crop_pad_size:
            h, w = img_lq.shape[0:2]
            img_lq = img_lq[top:top + crop_pad_size, left:left + crop_pad_size, ...]
       
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        img_lq = img2tensor([img_lq], bgr2rgb=True, float32=True)[0]
        img_lq = img_lq/ 255
        img_gt = img_gt/ 255
        temp = img_gt
        img_gt = img_lq
        img_lq = temp
        
        return [img_lq,img_gt,task_id]        

    def __len__(self):
        if self.args.key=='train':
            return len(self.sample_ids)
        else:
            return len(self.sample_ids_v)

