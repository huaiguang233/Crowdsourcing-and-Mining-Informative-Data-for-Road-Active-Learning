import math
import os
import random
import time

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

from OD.yolo_function import clear_txt, init_train_txt


class AL_dataloader():

    def __init__(self, name, image_folder_path, all_round, observed_images_number_per_v, number_of_v, train_txt, repetition_ratio, init_dataset_size = 200):
        self.dataset_name = name
        self.image_folder_path = image_folder_path
        self.img_paths = []
        self.all_round = all_round
        self.observed_images_number_per_v = observed_images_number_per_v
        self.number_of_v = number_of_v
        self.allocated_img_paths = None
        self.D2_round2image_prefix = {i: chr(64 + i) for i in range(1, 11)}
        self.init_dataset_size = init_dataset_size
        self.init_dataset_for_al = None
        self.train_txt_path = train_txt
        self.init_dataset()
        self.repetition_ratio = repetition_ratio

    def init_dataset(self):

        clear_txt(self.train_txt_path)

        if self.dataset_name == 'VOC2012':

            with open('../data/VOC2012/train_list.txt', 'r', encoding='utf-8') as file:
                lines = file.readlines()  # 读取文件的所有行
                self.img_paths = [line.strip() for line in lines]  # 去掉每行的换行符

            if len(self.img_paths) > self.all_round * self.number_of_v * self.observed_images_number_per_v + self.init_dataset_size:
                self.init_dataset_for_al = self.img_paths[: self.init_dataset_size]
                self.img_paths = self.img_paths[self.init_dataset_size:]
                self.img_paths = self.img_paths[: self.all_round * self.number_of_v * self.observed_images_number_per_v]
                self.allocated_img_paths = np.array(self.img_paths)
                self.allocated_img_paths = np.reshape(self.allocated_img_paths, (self.all_round, self.number_of_v, -1))
            else:
                print("There is not enough dataset allocated to every V")

        if self.dataset_name == 'HW':

            with open('../data/HW/train_list.txt', 'r', encoding='utf-8') as file:
                lines = file.readlines()  # 读取文件的所有行
                self.img_paths = [line.strip() for line in lines]  # 去掉每行的换行符

            if len(self.img_paths) > self.all_round * self.number_of_v * self.observed_images_number_per_v + self.init_dataset_size:
                self.init_dataset_for_al = self.img_paths[: self.init_dataset_size]
                self.img_paths = self.img_paths[self.init_dataset_size:]
                self.img_paths = self.img_paths[: self.all_round * self.number_of_v * self.observed_images_number_per_v]
                self.allocated_img_paths = np.array(self.img_paths)
                self.allocated_img_paths = np.reshape(self.allocated_img_paths, (self.all_round, self.number_of_v, -1))
            else:
                print("There is not enough dataset allocated to every V")

        if self.dataset_name == 'Tokyo_dataset':

            with open('../data/Tokyo_dataset/train_list.txt', 'r', encoding='utf-8') as file:
                lines = file.readlines()  # 读取文件的所有行
                self.img_paths = [line.strip() for line in lines]  # 去掉每行的换行符

            if len(self.img_paths) > self.all_round * self.number_of_v * self.observed_images_number_per_v + self.init_dataset_size:
                self.init_dataset_for_al = self.img_paths[: self.init_dataset_size]
                self.img_paths = self.img_paths[self.init_dataset_size:]
                self.img_paths = self.img_paths[: self.all_round * self.number_of_v * self.observed_images_number_per_v]
                self.allocated_img_paths = np.array(self.img_paths)
                self.allocated_img_paths = np.reshape(self.allocated_img_paths, (self.all_round, self.number_of_v, -1))
            else:
                print("There is not enough dataset allocated to every V")

        if self.dataset_name == 'Waymo':

            with open('../data/Waymo/train_list.txt', 'r', encoding='utf-8') as file:
                lines = file.readlines()  # 读取文件的所有行
                self.img_paths = [line.strip() for line in lines]  # 去掉每行的换行符

            if len(self.img_paths) > self.all_round * self.number_of_v * self.observed_images_number_per_v + self.init_dataset_size:
                self.init_dataset_for_al = self.img_paths[: self.init_dataset_size]
                self.img_paths = self.img_paths[self.init_dataset_size:]
                self.img_paths = self.img_paths[: self.all_round * self.number_of_v * self.observed_images_number_per_v]
                self.allocated_img_paths = np.array(self.img_paths)
                self.allocated_img_paths = np.reshape(self.allocated_img_paths, (self.all_round, self.number_of_v, -1))
            else:
                print("There is not enough dataset allocated to every V")

        if self.dataset_name == 'BDD100K':

            with open('../data/BDD100K/train_list.txt', 'r', encoding='utf-8') as file:
                lines = file.readlines()  # 读取文件的所有行
                self.img_paths = [line.strip() for line in lines]  # 去掉每行的换行符

            if len(self.img_paths) > self.all_round * self.number_of_v * self.observed_images_number_per_v + self.init_dataset_size:
                self.init_dataset_for_al = self.img_paths[: self.init_dataset_size]
                self.img_paths = self.img_paths[self.init_dataset_size:]
                self.img_paths = self.img_paths[: self.all_round * self.number_of_v * self.observed_images_number_per_v]
                self.allocated_img_paths = np.array(self.img_paths)
                self.allocated_img_paths = np.reshape(self.allocated_img_paths, (self.all_round, self.number_of_v, -1))
            else:
                print("There is not enough dataset allocated to every V")

        if self.dataset_name == 'TJU-DHD':

            with open('../data/TJU-DHD/train_list.txt', 'r', encoding='utf-8') as file:
                lines = file.readlines()  # 读取文件的所有行
                self.img_paths = [line.strip() for line in lines]  # 去掉每行的换行符

            if len(self.img_paths) > self.all_round * self.number_of_v * self.observed_images_number_per_v + self.init_dataset_size:
                self.init_dataset_for_al = self.img_paths[: self.init_dataset_size]
                self.img_paths = self.img_paths[self.init_dataset_size:]
                self.img_paths = self.img_paths[: self.all_round * self.number_of_v * self.observed_images_number_per_v]
                self.allocated_img_paths = np.array(self.img_paths)
                self.allocated_img_paths = np.reshape(self.allocated_img_paths, (self.all_round, self.number_of_v, -1))
            else:
                print("There is not enough dataset allocated to every V")

    def get_init_datatset(self):
        if self.dataset_name == 'VOC2012':
            with open(self.train_txt_path, 'w') as f:
                for image_name in self.init_dataset_for_al:
                    file_path = os.path.join(self.image_folder_path, image_name)
                    f.write(f"{file_path}\n")

        elif self.dataset_name == 'HW':
            with open(self.train_txt_path, 'w') as f:
                for image_name in self.init_dataset_for_al:
                    file_path = os.path.join(self.image_folder_path, image_name)
                    f.write(f"{file_path}\n")

        elif self.dataset_name == 'Waymo':
            with open(self.train_txt_path, 'w') as f:
                for image_name in self.init_dataset_for_al:
                    file_path = os.path.join(self.image_folder_path, image_name)
                    f.write(f"{file_path}\n")

        elif self.dataset_name == 'BDD100K':
            with open(self.train_txt_path, 'w') as f:
                for image_name in self.init_dataset_for_al:
                    file_path = os.path.join(self.image_folder_path, image_name)
                    f.write(f"{file_path}\n")

        elif self.dataset_name == 'TJU-DHD':
            with open(self.train_txt_path, 'w') as f:
                for image_name in self.init_dataset_for_al:
                    file_path = os.path.join(self.image_folder_path, image_name)
                    f.write(f"{file_path}\n")


    def get_images(self, v, round):
        if self.dataset_name == 'VOC2012':
            if self.repetition_ratio == 0:
                get_imgs_paths = self.allocated_img_paths[round][v]
            else:
                n = math.floor(self.allocated_img_paths.shape[-1]*self.repetition_ratio)
                first_n_elements = self.allocated_img_paths[:, 0, :n]
                # 使用 np.tile 重复这些元素
                repeated_elements = np.tile(first_n_elements[:, np.newaxis, :], (1, self.number_of_v, 1))
                # 将这些重复的元素赋值回原数组的第二维
                self.allocated_img_paths[:, :, :n] = repeated_elements
                get_imgs_paths = self.allocated_img_paths[round][v]


        if self.dataset_name == 'HW':
            if self.repetition_ratio == 0:
                get_imgs_paths = self.allocated_img_paths[round][v]
            else:
                n = math.floor(self.allocated_img_paths.shape[-1]*self.repetition_ratio)
                first_n_elements = self.allocated_img_paths[:, 0, :n]
                # 使用 np.tile 重复这些元素
                repeated_elements = np.tile(first_n_elements[:, np.newaxis, :], (1, self.number_of_v, 1))
                # 将这些重复的元素赋值回原数组的第二维
                self.allocated_img_paths[:, :, :n] = repeated_elements
                get_imgs_paths = self.allocated_img_paths[round][v]


        if self.dataset_name == 'Waymo':
            if self.repetition_ratio == 0:
                get_imgs_paths = self.allocated_img_paths[round][v]
            else:
                n = math.floor(self.allocated_img_paths.shape[-1]*self.repetition_ratio)
                first_n_elements = self.allocated_img_paths[:, 0, :n]
                # 使用 np.tile 重复这些元素
                repeated_elements = np.tile(first_n_elements[:, np.newaxis, :], (1, self.number_of_v, 1))
                # 将这些重复的元素赋值回原数组的第二维
                self.allocated_img_paths[:, :, :n] = repeated_elements
                get_imgs_paths = self.allocated_img_paths[round][v]

        if self.dataset_name == 'BDD100K':
            if self.repetition_ratio == 0:
                get_imgs_paths = self.allocated_img_paths[round][v]
            else:
                n = math.floor(self.allocated_img_paths.shape[-1]*self.repetition_ratio)
                first_n_elements = self.allocated_img_paths[:, 0, :n]
                # 使用 np.tile 重复这些元素
                repeated_elements = np.tile(first_n_elements[:, np.newaxis, :], (1, self.number_of_v, 1))
                # 将这些重复的元素赋值回原数组的第二维
                self.allocated_img_paths[:, :, :n] = repeated_elements
                get_imgs_paths = self.allocated_img_paths[round][v]

        if self.dataset_name == 'TJU-DHD':
            if self.repetition_ratio == 0:
                get_imgs_paths = self.allocated_img_paths[round][v]
            else:
                n = math.floor(self.allocated_img_paths.shape[-1]*self.repetition_ratio)
                first_n_elements = self.allocated_img_paths[:, 0, :n]
                # 使用 np.tile 重复这些元素
                repeated_elements = np.tile(first_n_elements[:, np.newaxis, :], (1, self.number_of_v, 1))
                # 将这些重复的元素赋值回原数组的第二维
                self.allocated_img_paths[:, :, :n] = repeated_elements
                get_imgs_paths = self.allocated_img_paths[round][v]

        return get_imgs_paths