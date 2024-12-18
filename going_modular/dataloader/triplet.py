# Code anh lâm
import numpy as np
import os
from torch.utils.data import Dataset
import cv2
import torch
from pathlib import Path
from typing import Tuple
from torchvision import transforms
from torchvision.transforms import InterpolationMode

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


class RandomResizedCropRect(object):
    def __init__(self, size, scale=(0.8, 1.0)):
        self.size = size
        self.scale = scale
        self.resize = transforms.Resize((self.size, self.size), interpolation=InterpolationMode.BILINEAR)
        

    def __call__(self, img):
        # Lấy kích thước ảnh
        _, img_height, img_width = img.shape

        # Tính toán kích thước và tọa độ cho crop
        width = int(img_width * torch.empty(1).uniform_(*self.scale).item())
        height = int(img_height * torch.empty(1).uniform_(*self.scale).item())
        x = torch.randint(0, img_width - width + 1, (1,)).item()
        y = torch.randint(0, img_height - height + 1, (1,)).item()

        # Crop ảnh
        img = img[:, y:y+height, x:x+width]

        img = self.resize(img)

        return img

class GaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        # Lấy kích thước của ảnh (C, H, W)
        channels, height, width = img.shape

        # Tính toán kích thước vùng ảnh sẽ nhận nhiễu
        min_size = int(height * width * 0.1)
        max_size = int(height * width * 0.25)
        area_size = torch.randint(min_size, max_size, (1,)).item()

        # Tạo mask ngẫu nhiên cho vùng ảnh nhận nhiễu
        mask = torch.zeros((height, width))
        x = torch.randint(0, width, (1,)).item()
        y = torch.randint(0, height, (1,)).item()
        x_end = min(x + int(area_size**0.5), width)
        y_end = min(y + int(area_size**0.5), height)
        mask[y:y_end, x:x_end] = 1.0

        # Tạo nhiễu Gaussian
        std = torch.rand(1).item() * self.std
        gauss = torch.normal(mean=self.mean, std=std, size=(channels, height, width))

        # Áp dụng nhiễu lên ảnh
        mask = mask.unsqueeze(0)  # Thêm chiều cho mask để tương thích với (C, H, W)
        noisy_img = img + gauss * mask
        noisy_img = torch.clamp(noisy_img, 0, 1)  # Đảm bảo giá trị nằm trong khoảng [0, 1]

        return noisy_img


class CustomExrDataset(Dataset):
    
    def __init__(self, dataset_dir:str, transform, type='normalmap'):
        '''
            type = ['normalmap', 'depthmap', 'albedo']
        '''
        self.paths = list(Path(dataset_dir).glob("*/*.exr"))
        self.transform = transform
        self.type = type
        
    def __len__(self):
        return len(self.paths)
    
    # Nhận vào index mà dataloader muốn lấy
    def __getitem__(self, index:int) -> Tuple[torch.Tensor, int]:
        tensor_image = self.__load_tensor_image(index)
        label = self.paths[index].parent.name
        
        if self.transform:
            tensor_image = self.transform(tensor_image)
            
        return tensor_image, int(label)
        
    def __load_tensor_image(self, index:int):
        image = cv2.imread(self.paths[index], cv2.IMREAD_UNCHANGED)
        
        if image is None:
            raise ValueError(f"Failed to load image at {self.paths[index]}")
        if self.type in ['albedo', 'depthmap']:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return torch.from_numpy(image).permute(2,0,1)

    
class TripletDataset(Dataset):
    
    def __init__(self, data_dir, transform=None, type = 'albedo', train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train

        self.image_paths = []
        self.labels = []

        for label in os.listdir(data_dir):
            label_dir = os.path.join(data_dir, label)
            if os.path.isdir(label_dir):
                for image_name in os.listdir(label_dir):
                    image_path = os.path.join(label_dir, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(int(label))

        self.type = type
        self.labels = np.array(self.labels)
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}

        if not self.train:
            random_state = np.random.RandomState(29)

            self.test_triplets = [[i,
                                   random_state.choice(self.label_to_indices[self.labels[i]]),
                                   random_state.choice(self.label_to_indices[
                                       np.random.choice(list(self.labels_set - set([self.labels[i]])))
                                   ])
                                  ]
                                 for i in range(len(self.image_paths))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if self.train:
            img1_path = self.image_paths[index]
            label1 = self.labels[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2_path = self.image_paths[positive_index]
            img3_path = self.image_paths[negative_index]
        else:
            img1_path = self.image_paths[self.test_triplets[index][0]]
            img2_path = self.image_paths[self.test_triplets[index][1]]
            img3_path = self.image_paths[self.test_triplets[index][2]]

        if self.type == 'normalmap':
            cvtColor = cv2.COLOR_BGR2RGB
        else:
            cvtColor = cv2.COLOR_GRAY2BGR
            
        img1 = torch.from_numpy(cv2.cvtColor(cv2.imread(img1_path, cv2.IMREAD_UNCHANGED), cvtColor)).permute(2, 0, 1)
        img2 = torch.from_numpy(cv2.cvtColor(cv2.imread(img2_path, cv2.IMREAD_UNCHANGED), cvtColor)).permute(2, 0, 1)
        img3 = torch.from_numpy(cv2.cvtColor(cv2.imread(img3_path, cv2.IMREAD_UNCHANGED), cvtColor)).permute(2, 0, 1)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        # Stack các tensor lại thành một tensor duy nhất
        X = torch.stack((img1, img2, img3), dim=0)
        
        # image 1 và 2 cùng class, 3 là negative
        return X

  
# class TripletDatasetConcat(Dataset):
#     def __init__(self, data_dir1, data_dir2, transform=None, train=True):
#         self.data_dir1 = data_dir1
#         self.data_dir2 = data_dir2
#         self.transform = transform
#         self.train = train

#         self.image_paths = []
#         self.labels = []

#         for label in os.listdir(data_dir1):
#             label_dir1 = os.path.join(data_dir1, label)
#             label_dir2 = os.path.join(data_dir2, label)
#             if os.path.isdir(label_dir1):
#                 for image_name in os.listdir(label_dir1):
#                     image_path1 = os.path.join(label_dir1, image_name)
#                     image_path2 = os.path.join(label_dir2, image_name)
#                     self.image_paths.append((image_path1, image_path2))
#                     self.labels.append(int(label))  # assuming labels are integers
                    
#         self.labels = np.array(self.labels)
#         self.labels_set = set(self.labels)
#         self.label_to_indices = {label: np.where(self.labels == label)[0]
#                                  for label in self.labels_set}
#         if not self.train:
#             random_state = np.random.RandomState(29)

#             self.test_triplets = [[i,
#                                    random_state.choice(self.label_to_indices[self.labels[i]]),
#                                    random_state.choice(self.label_to_indices[
#                                        np.random.choice(list(self.labels_set - set([self.labels[i]])))
#                                    ])
#                                   ]
#                                  for i in range(len(self.image_paths))]

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, index):
#         if self.train:
#             img1_path = self.image_paths[index]
#             label1 = self.labels[index]
#             positive_index = index
#             while positive_index == index:
#                 positive_index = np.random.choice(self.label_to_indices[label1])
#             negative_label = np.random.choice(list(self.labels_set - set([label1])))
#             negative_index = np.random.choice(self.label_to_indices[negative_label])
#             img2_path = self.image_paths[positive_index]
#             img3_path = self.image_paths[negative_index]
#         else:
#             img1_path = self.image_paths[self.test_triplets[index][0]]
#             img2_path = self.image_paths[self.test_triplets[index][1]]
#             img3_path = self.image_paths[self.test_triplets[index][2]]

#         img1 = Image.open(img1_path[0]).convert('RGB')
#         img2 = Image.open(img1_path[1]).convert('RGB')
#         img3 = Image.open(img2_path[0]).convert('RGB')
#         img4 = Image.open(img2_path[1]).convert('RGB')
#         img5 = Image.open(img3_path[0]).convert('RGB')
#         img6 = Image.open(img3_path[1]).convert('RGB')

#         if self.transform is not None:
#             img1 = self.transform(img1)
#             img2 = self.transform(img2)
#             img3 = self.transform(img3)
#             img4 = self.transform(img4)
#             img5 = self.transform(img5)
#             img6 = self.transform(img6)
#         return (img1, img2, img3, img4, img5, img6), []