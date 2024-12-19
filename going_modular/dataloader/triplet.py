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


class CustomExrDataset(Dataset):
    
    def __init__(self, dataset_dir:str, transform, type='normalmap'):
        '''
            type = ['normalmap', 'depthmap', 'albedo']
        '''
        self.paths = list(Path(dataset_dir).glob("*/*.exr"))
        self.transform = transform
        self.type = type
        self.classes = sorted(os.listdir(dataset_dir))
        
    def __len__(self):
        return len(self.paths)
    
    # Nhận vào index mà dataloader muốn lấy
    def __getitem__(self, index:int) -> Tuple[torch.Tensor, int]:
        numpy_image = self.__load_numpy_image(index)
        label = self.paths[index].parent.name
        label_index = self.classes.index(label)
        
        if self.transform:
            numpy_image = self.transform(image = numpy_image)['image']
            
        return torch.from_numpy(numpy_image).permute(2,0,1), label_index
        
    def __load_numpy_image(self, index:int):
        image = cv2.imread(self.paths[index], cv2.IMREAD_UNCHANGED)
        
        if image is None:
            raise ValueError(f"Failed to load image at {self.paths[index]}")
        if self.type in ['albedo', 'depthmap']:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image

    
    
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
            
        img1 = cv2.cvtColor(cv2.imread(img1_path, cv2.IMREAD_UNCHANGED), cvtColor)
        img2 = cv2.cvtColor(cv2.imread(img2_path, cv2.IMREAD_UNCHANGED), cvtColor)
        img3 = cv2.cvtColor(cv2.imread(img3_path, cv2.IMREAD_UNCHANGED), cvtColor)

        if self.transform is not None:
            img1 = self.transform(image=img1)['image']
            img2 = self.transform(image=img2)['image']
            img3 = self.transform(image=img3)['image']

        # Stack các tensor lại thành một tensor duy nhất
        X = torch.stack((
            torch.from_numpy(img1).permute(2,0,1), 
            torch.from_numpy(img2).permute(2,0,1), 
            torch.from_numpy(img3).permute(2,0,1)
        ), dim=0)
        
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