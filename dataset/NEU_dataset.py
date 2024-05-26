import os
import random
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
class VOCSegmentation(data.Dataset):
    def __init__(self, voc_root, year="2012", transforms=None, txt_name: str = "train.txt",train1 = False):
        super(VOCSegmentation, self).__init__()
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        root = os.path.join(voc_root, "NEU-Seg")
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        image_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'SegmentationClass')


        _auxiliary_dir = os.path.join(root, 'AuxiliaryGT')

        txt_path = os.path.join(root, "ImageSets", "Segmentation", txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        se_labels_dt = {}
        _sells_f = os.path.join(_auxiliary_dir, '1semantic_labels.txt')
        with open(_sells_f,'r') as lines:
            for line in lines:
                line = line.rstrip('\n')
                linel = line.split('\t')
                fn,lb = linel[0],eval(linel[1])
                se_labels_dt[fn] = torch.tensor(lb)
        self.se_labels = []
        self.edgmask = []
        for i in file_names:
            self.se_labels.append(se_labels_dt[i])
            k = os.path.join(_auxiliary_dir, i+'.png' )
            self.edgmask.append(k)


        self.images = [os.path.join(image_dir, x+'.jpg' ) for x in file_names]
        self.masks = [os.path.join(mask_dir, x+'.png' ) for x in file_names]
        assert (len(self.images) == len(self.masks))
        self.transforms = transforms
        self.trian1 =train1

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        edg_mask = Image.open(self.edgmask[index])

        # img = img.resize((int(256),int(256)),Image.BILINEAR)
        # if  self.trian1 == True:
        #     target = target.resize((int(256),int(256)),Image.BILINEAR)
        #
        # if self.trian1 == False:
        #     img = img.resize((int(200), int(200)), Image.BILINEAR)
        #     target = target.resize((int(200), int(200)), Image.BILINEAR)


        if  self.trian1 == True:

            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
                edg_mask = edg_mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                target = target.transpose(Image.FLIP_TOP_BOTTOM)
                edg_mask = edg_mask.transpose(Image.FLIP_TOP_BOTTOM)
            # crop = RandomCrop(224)
            # img,target = crop(img,target)
        if self.transforms is not None:
            img = self.transforms(img)
        target = torch.from_numpy(np.array(target)).long()
        cs_label = self.se_labels[index]

        edg_mask = torch.from_numpy(np.array(edg_mask)).long()

        return img, target, cs_label, edg_mask, self.images[index]

    def __len__(self):
        return len(self.images)

#     @staticmethod
#     def collate_fn(batch):
#         images, targets = list(zip(*batch))
#         batched_imgs = cat_list(images, fill_value=0)
#         batched_targets = cat_list(targets, fill_value=255)
#         return batched_imgs, batched_targets


# def cat_list(images, fill_value=0):
#     # 计算该batch数据中，channel, h, w的最大值
#     max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
#     batch_shape = (len(images),) + max_size
#     batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
#     for img, pad_img in zip(images, batched_imgs):
#         pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
#     return batched_imgs


# dataset = VOCSegmentation(voc_root="/data/", transforms=get_transform(train=True))
# d1 = dataset[0]
# print(d1)
class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


def pad_if_smaller(img, size, fill=0):
    # 如果图像最小边长小于给定size，则用数值fill进行padding
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img