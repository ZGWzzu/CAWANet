import train_utils.distributed_utils as utils
import torch

from train_utils import evaluate
from dataset.NEU_dataset import VOCSegmentation
# import transforms as T
from torchvision import transforms as T
from  src.CAWANet import CAWA

class SegmentationPresetTrain:
    def __init__(self,  mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

        trans = []

        trans.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img):
        return self.transforms(img)





def get_transform(train):


    return SegmentationPresetTrain() if train else SegmentationPresetEval()


def evaluate(model, data_loader, device, num_classes):
    model.eval()

    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target, _, size1,size2 in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output
            # output = F.interpolate(output,(200,200),mode='bilinear',align_corners=False)

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()
        mean_iu = confmat.IOUk()
        print(mean_iu)
    return confmat, mean_iu


def main():
    aux = False  # inference time not need aux_classifier

    weights_path = "CACW_NEU.pth"
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

    model.to(device)


    # delete weights about aux_classifier
    weights_dict = torch.load(weights_path, map_location='cpu')


    # load weights
    model.load_state_dict(weights_dict,strict=False)
    model.to(device)

    # load image
    evaluate(model, val_loader, device, num_classes=4)


if __name__ == '__main__':
    main()
