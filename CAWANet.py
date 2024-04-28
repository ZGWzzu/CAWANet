
import torch.nn.functional as F
from src.CAWANet import CAWA
import time
import train_utils.distributed_utils as utils
import torch
from train_utils import  evaluate
from my_dataset_neu import VOCSegmentation

from torchvision import transforms as T


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        
        trans = []
       
        trans.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img):
        return self.transforms(img)


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def get_transform(train):
    base_size1 = 1920
    base_size2 = 1080
    crop_size = 512

    return SegmentationPresetTrain(base_size1, base_size2, crop_size) if train else SegmentationPresetEval(512)


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target, _, size1,_ in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)['out']
            output = F.interpolate(output, size=(200, 200), mode='bilinear', align_corners=False)
            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()
        mean_iu = confmat.IOUk()
        print(mean_iu)
    return confmat, mean_iu


def main():
    aux = False  # inference time not need aux_classifier

    weights_path = "./save_weights/EDR.pth"
    val_dataset = VOCSegmentation('E:/dataset',
                                  year="2012",
                                  transforms=get_transform(train=False),
                                  txt_name="test.txt")
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=8,
                                             pin_memory=True,
                                             )
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    
    model = CAWA(num_classes=4)

    # delete weights about aux_classifier
    weights_dict = torch.load(weights_path, map_location='cpu')['model']

    # load weights
    model.load_state_dict(weights_dict)
    model.to(device)

    # load image
    evaluate(model, val_loader, device, 4)


if __name__ == '__main__':
    main()

