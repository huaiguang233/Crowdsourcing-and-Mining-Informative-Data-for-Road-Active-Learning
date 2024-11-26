import os
import random
import sys
import time
import numpy as np
import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms

from OD.CALD.CALD import CALD_method
from OD.al_dataloader import AL_dataloader
from OD.al_methods import Random_method, DAL_method, FAL_method, Coreset_method, get_class_distrbution, Leastconf_method
import yolo_modify

from yolo_function import Dual_dpp, yolo_train, append_index_to_txt, cal_duplicate_samples, clear_folder


def AL_process(number_of_v, model_type, dataset_name, dataset_dir_path, repetition_ratio, AL_method, round, b, epoch, save_data_dir, windows_b, iou_=0.5,
               lambda1=0.8, yaml_path='', train_text_path=f"../data/train.txt"):

    device_gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_loader = AL_dataloader(dataset_name, dataset_dir_path, round, observed_images_number_per_v=100, number_of_v=number_of_v,
                                train_txt=train_text_path, repetition_ratio=repetition_ratio, init_dataset_size=200)

    data_loader.get_init_datatset()

    yolo_train(model_type, yaml_path, epoch, 0, save_data_dir, dataset_name, device_gpu, AL_method)

    dpp = Dual_dpp(device_gpu)
    # 1000是bbox的embedding的维度
    if model_type == 'yolo':
        dpp.set_dim(512)
    if model_type == 'faster_rcnn':
        dpp.set_dim(1024)

    for i in range(round):
        image_names = []

        class_distrbution, labeled_dataset = get_class_distrbution(yaml_path = yaml_path, img_path_txt = train_text_path)

        if AL_method == 'FAL':
            all_imgs = []

        for v in range(number_of_v):
            imgs = data_loader.get_images(v, i)
            if AL_method == 'Random':

                image_names = Random_method(imgs, b)
            elif AL_method == 'DAL':
                image_names = DAL_method(windows_b=windows_b, iou_=iou_, lambda1=lambda1, b=b, imgs=imgs,
                                         class_distribution_curr= class_distrbution, dpp = dpp, model_type = model_type,
                                         dataset_name = dataset_name, device_gpu = device_gpu)

            elif AL_method == 'FAL':
                if v == 4:
                    all_imgs.append(imgs.tolist())
                    image_names = FAL_method(model_type, b=b, imgs=np.array(all_imgs), img_path_txt=train_text_path, device_gpu= device_gpu, dataset_name=dataset_name)
                else:
                    all_imgs.append(imgs.tolist())

            elif AL_method == 'Coreset':
                image_names = Coreset_method(model_type, b=b, imgs=imgs, dataset_name=dataset_name, device_gpu= device_gpu)

            elif AL_method == 'LeastConf':
                image_names = Leastconf_method(model_type, b=b, imgs=imgs, dataset_name=dataset_name, device_gpu= device_gpu)

            elif AL_method == 'CALD':
                image_names = CALD_method(model_type, b=b, imgs=imgs, dataset_name=dataset_name, device_gpu=device_gpu, labeled_dataloader=labeled_dataset)

            append_index_to_txt(image_names, train_text_path)

        cal_duplicate_samples(train_text_path)
        print(i+1)
        yolo_train(model_type, yaml_path, epoch, i + 1, save_data_dir, dataset_name, device_gpu, AL_method)


if __name__ == '__main__':
    methods = ['DAL', 'CALD', 'FAL', 'Coreset', 'LeastConf', 'CALD']

    for method in methods:
        clear_folder("../yolo_model")
        dataset_name = 'TJU-DHD'
        D2_PATH = '../data/D2/'
        VOC_PATH = '../data/VOC2012/Images/'
        HW_PATH = '../data/HW/images/'
        yaml_path = f'../data/{dataset_name}.yaml'
        Tokyo_dataset_PATH = '../data/Tokyo_dataset/images/'
        Waymo_PATH = '../data/Waymo/images/'
        BDD100K_PATH = '../data/BDD100K/images/'
        TJU_DHD_PATH = '../data/TJU-DHD/images/'


        if dataset_name == 'VOC2012':
            dataset_dir_path = VOC_PATH
        elif dataset_name == 'HW':
            dataset_dir_path = HW_PATH
        elif dataset_name == 'Waymo':
            dataset_dir_path = Waymo_PATH
        elif dataset_name == 'BDD100K':
            dataset_dir_path = BDD100K_PATH
        elif dataset_name == 'TJU-DHD':
            dataset_dir_path = TJU_DHD_PATH


        # BDD100K : 每轮200，选40，训练20
        model_type = 'yolo' # RetinaNet
        epoch = 30  # yolo:30; faster_rcnn:20
        AL_process(5, model_type, dataset_name, dataset_dir_path, 0, method, 10, 40, epoch,
                   '../yolo_model', 3, 0.5, 0.001, yaml_path=yaml_path,
                   train_text_path=f"../data/{dataset_name}/train.txt")