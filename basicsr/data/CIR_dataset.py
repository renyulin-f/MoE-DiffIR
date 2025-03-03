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

        self.de_dict = {'JPEG_10': 0, 'JPEG_15': 1,'JPEG_20':2,'VVC_47': 3, 'VVC_42': 4,'VVC_37':5,'HEVC_47':6,'HEVC_42':7,'HEVC_37':8,'WEBP_1':9,'WEBP_5':10,'WEBP_10':11,'PSNR_1':12,'PSNR_2':13,'PSNR_3':14,'SSIM_1':15,'SSIM_2':16,'SSIM_3':17,'HIFI_1':18,'HIFI_2':19,'HIFI_3':20}
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
            JPEG_20 = self.args.JPEG_20
            for filename in os.listdir(JPEG_20):
                temp_ids+= [JPEG_20 + filename]
            self.jpeg20_ids = [{"clean_id":x,"de_type":2} for x in temp_ids]
            self.num_SR = len(self.jpeg20_ids)
            print("Total JPEG_20(train) Ids : {}".format(self.num_SR))
            
            
            temp_ids = []
            VVC_47 = self.args.VVC_47
            for filename in os.listdir(VVC_47):
                temp_ids+= [VVC_47 + filename]
            self.VVC_47_ids = [{"clean_id":x,"de_type":3} for x in temp_ids]
            self.num_SR = len(self.VVC_47_ids)
            print("Total VVC_47(train) Ids : {}".format(self.num_SR))
            
            temp_ids = []
            VVC_42 = self.args.VVC_42
            for filename in os.listdir(VVC_42):
                temp_ids+= [VVC_42 + filename]
            self.VVC_42_ids = [{"clean_id":x,"de_type":4} for x in temp_ids]
            self.num_SR = len(self.VVC_42_ids)
            print("Total VVC_42(train) Ids : {}".format(self.num_SR))
            
            temp_ids = []
            VVC_37 = self.args.VVC_37
            for filename in os.listdir(VVC_37):
                temp_ids+= [VVC_37 + filename]
            self.VVC_37_ids = [{"clean_id":x,"de_type":5} for x in temp_ids]
            self.num_SR = len(self.VVC_37_ids)
            print("Total VVC_37(train) Ids : {}".format(self.num_SR))
            
            temp_ids = []
            HEVC_47 = self.args.HEVC_47
            for filename in os.listdir(HEVC_47):
                temp_ids+= [HEVC_47 + filename]
            self.HEVC_47_ids = [{"clean_id":x,"de_type":6} for x in temp_ids]
            # self.HEVC_47_ids =self.HEVC_47_ids * 60
            self.num_SR = len(self.HEVC_47_ids)
            print("Total HEVC_47(train) Ids : {}".format(self.num_SR))
            
            temp_ids = []
            HEVC_42 = self.args.HEVC_42
            for filename in os.listdir(HEVC_42):
                temp_ids+= [HEVC_42 + filename]
            self.HEVC_42_ids = [{"clean_id":x,"de_type":7} for x in temp_ids]
            # self.HEVC_42_ids =self.HEVC_42_ids * 60
            self.num_SR = len(self.HEVC_42_ids)
            print("Total HEVC_42(train) Ids : {}".format(self.num_SR))
            
            
            temp_ids = []
            HEVC_37 = self.args.HEVC_37
            for filename in os.listdir(HEVC_37):
                temp_ids+= [HEVC_37 + filename]
            self.HEVC_37_ids = [{"clean_id":x,"de_type":8} for x in temp_ids]
            # self.HEVC_42_ids =self.HEVC_42_ids * 60
            self.num_SR = len(self.HEVC_37_ids)
            print("Total HEVC_37(train) Ids : {}".format(self.num_SR))
            
            
            temp_ids = []
            WEBP_5 = self.args.WEBP_5
            for filename in os.listdir(WEBP_5):
                temp_ids+= [WEBP_5 + filename]
            self.WEBP_5_ids = [{"clean_id":x,"de_type":10} for x in temp_ids]
            self.num_SR = len(self.WEBP_5_ids)
            print("Total WEBP_5(train) Ids : {}".format(self.num_SR))
            
            
            temp_ids = []
            WEBP_1 = self.args.WEBP_1
            for filename in os.listdir(WEBP_1):
                temp_ids+= [WEBP_1 + filename]
            self.WEBP_1_ids = [{"clean_id":x,"de_type":9} for x in temp_ids]
            self.num_SR = len(self.WEBP_1_ids)
            print("Total WEBP_1(train) Ids : {}".format(self.num_SR))
            
            temp_ids = []
            WEBP_10 = self.args.WEBP_10
            for filename in os.listdir(WEBP_10):
                temp_ids+= [WEBP_10 + filename]
            self.WEBP_10_ids = [{"clean_id":x,"de_type":11} for x in temp_ids]
            self.num_SR = len(self.WEBP_10_ids)
            print("Total WEBP_10(train) Ids : {}".format(self.num_SR))
            
            temp_ids = []
            PSNR_1 = self.args.PSNR_1
            for filename in os.listdir(PSNR_1):
                temp_ids+= [PSNR_1 + filename]
            self.PSNR_1_ids = [{"clean_id":x,"de_type":12} for x in temp_ids]
            self.num_SR = len(self.PSNR_1_ids)
            print("Total PSNR_1(train) Ids : {}".format(self.num_SR))
            
            temp_ids = []
            PSNR_2 = self.args.PSNR_2
            for filename in os.listdir(PSNR_2):
                temp_ids+= [PSNR_2 + filename]
            self.PSNR_2_ids = [{"clean_id":x,"de_type":13} for x in temp_ids]
            self.num_SR = len(self.PSNR_2_ids)
            print("Total PSNR_2(train) Ids : {}".format(self.num_SR))
            
            temp_ids = []
            PSNR_3 = self.args.PSNR_3
            for filename in os.listdir(PSNR_3):
                temp_ids+= [PSNR_3 + filename]
            self.PSNR_3_ids = [{"clean_id":x,"de_type":14} for x in temp_ids]
            self.num_SR = len(self.PSNR_3_ids)
            print("Total PSNR_3(train) Ids : {}".format(self.num_SR))
            
            temp_ids = []
            SSIM_1 = self.args.SSIM_1
            for filename in os.listdir(SSIM_1):
                temp_ids+= [SSIM_1 + filename]
            self.SSIM_1_ids = [{"clean_id":x,"de_type":15} for x in temp_ids]
            self.num_SR = len(self.SSIM_1_ids)
            print("Total SSIM_1(train) Ids : {}".format(self.num_SR))
            
            temp_ids = []
            SSIM_2 = self.args.SSIM_2
            for filename in os.listdir(SSIM_2):
                temp_ids+= [SSIM_2 + filename]
            self.SSIM_2_ids = [{"clean_id":x,"de_type":16} for x in temp_ids]
            self.num_SR = len(self.SSIM_2_ids)
            print("Total SSIM_2(train) Ids : {}".format(self.num_SR))
            
            temp_ids = []
            SSIM_3 = self.args.SSIM_3
            for filename in os.listdir(SSIM_3):
                temp_ids+= [SSIM_3 + filename]
            self.SSIM_3_ids = [{"clean_id":x,"de_type":17} for x in temp_ids]
            self.num_SR = len(self.SSIM_3_ids)
            print("Total SSIM_3(train) Ids : {}".format(self.num_SR))
            
            temp_ids = []
            HIFI_1 = self.args.HIFI_1
            for filename in os.listdir(HIFI_1):
                temp_ids+= [HIFI_1 + filename]
            self.HIFI_1_ids = [{"clean_id":x,"de_type":18} for x in temp_ids]
            self.num_SR = len(self.HIFI_1_ids)
            print("Total HIFI_1(train) Ids : {}".format(self.num_SR))
            
            temp_ids = []
            HIFI_2 = self.args.HIFI_2
            for filename in os.listdir(HIFI_2):
                temp_ids+= [HIFI_2 + filename]
            self.HIFI_2_ids = [{"clean_id":x,"de_type":19} for x in temp_ids]
            self.num_SR = len(self.HIFI_2_ids)
            print("Total HIFI_2(train) Ids : {}".format(self.num_SR))
            
            temp_ids = []
            HIFI_3 = self.args.HIFI_3
            for filename in os.listdir(HIFI_3):
                temp_ids+= [HIFI_3 + filename]
            self.HIFI_3_ids = [{"clean_id":x,"de_type":20} for x in temp_ids]
            self.num_SR = len(self.HIFI_3_ids)
            print("Total HIFI_3(train) Ids : {}".format(self.num_SR))
   
            
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
            self.sample_ids += self.jpeg20_ids
            self.sample_ids += self.VVC_47_ids
            self.sample_ids += self.VVC_42_ids
            self.sample_ids += self.VVC_37_ids
            self.sample_ids += self.HEVC_47_ids
            self.sample_ids += self.HEVC_42_ids
            self.sample_ids += self.HEVC_37_ids
            self.sample_ids += self.WEBP_1_ids
            self.sample_ids += self.WEBP_5_ids
            self.sample_ids += self.WEBP_10_ids
            self.sample_ids += self.PSNR_1_ids
            self.sample_ids += self.PSNR_2_ids
            self.sample_ids += self.PSNR_3_ids
            self.sample_ids += self.SSIM_1_ids
            self.sample_ids += self.SSIM_2_ids
            self.sample_ids += self.SSIM_3_ids
            self.sample_ids += self.HIFI_1_ids
            self.sample_ids += self.HIFI_2_ids
            self.sample_ids += self.HIFI_3_ids
            print('Train:',len(self.sample_ids))
        

    def __getitem__(self, idx):
         # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.

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

