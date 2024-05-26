import os
import random
import torch.utils.data as data
from PIL import Image, ImageOps
import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


class VOCSegmentation(data.Dataset):
    def __init__(self, voc_root, year="2012", transforms=None, txt_name: str = "train.txt", train1=False):
        super(VOCSegmentation, self).__init__()
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        root = os.path.join(voc_root, "MT-Defects")
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
        with open(_sells_f, 'r') as lines:
            for line in lines:
                line = line.rstrip('\n')
                linel = line.split('\t')
                fn, lb = linel[0], eval(linel[1])
                se_labels_dt[fn] = torch.tensor(lb)
        self.se_labels = []
        self.edgmask = []
        for i in file_names:
            self.se_labels.append(se_labels_dt[i])
            k = os.path.join(_auxiliary_dir, i + '.png')
            self.edgmask.append(k)

        self.images = [os.path.join(image_dir, x + '.jpg') for x in file_names]
        # print(len(os.listdir(image_dir)))
        self.masks = [os.path.join(mask_dir, x + '.png') for x in file_names]
        assert (len(self.images) == len(self.masks))
        self.transforms = transforms
        self.trian1 = train1
        self.base_size = 520
        self.crop_size = 450
        self.pad_size = None

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        w, h = img.size
        size = (h, w)
        target = Image.open(self.masks[index])
        edg_mask = Image.open(self.edgmask[index])

        img = img.resize((int(512), int(512)), Image.BILINEAR)

        # target = target.resize((int(512), int(512)), Image.BILINEAR)
        if self.trian1 == True:
            target = target.resize((int(512), int(512)), Image.BILINEAR)
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
                edg_mask = edg_mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                target = target.transpose(Image.FLIP_TOP_BOTTOM)
                edg_mask = edg_mask.transpose(Image.FLIP_TOP_BOTTOM)


        if self.transforms is not None:
            img = self.transforms(img)
        target = torch.from_numpy(np.array(target)).long()

        # print(size)
        edg_mask = torch.from_numpy(np.array(edg_mask)).long()
        cs_label = self.se_labels[index]

        return img, target, cs_label, size,self.images[index]

    def __len__(self):
        return len(self.images)



