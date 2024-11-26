import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from OD.model.retinanet import model
from OD.model.retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from OD.model.retinanet import coco_eval

assert torch.__version__.split('.')[0] == '1'

# print('CUDA available: {}'.format(torch.cuda.is_available()))


def train_retinanet(epochs, train_path, val_path, classes, root_path, dataset_name):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=18)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=epochs)

    parser = parser.parse_args()

    dataset_train = CocoDataset(root_path, set_name=dataset_name,
                                transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]), txt_file=train_path)
    dataset_val = CocoDataset(root_path, set_name=dataset_name,
                              transform=transforms.Compose([Normalizer(), Resizer()]), txt_file=val_path)

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=8, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=classes, pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=classes, pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=classes, pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=classes, pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=classes, pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(2):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        scheduler.step(np.mean(epoch_loss))

        # torch.save(retinanet.module, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))

    retinanet.eval()

    print('Evaluating dataset')
    coco_eval.evaluate_coco(dataset_val, retinanet)

    torch.save(retinanet, 'model_final.pt')
