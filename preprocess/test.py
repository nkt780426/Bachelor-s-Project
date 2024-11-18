import os
from scipy.io import loadmat
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import logging
from scipy.io import loadmat
from concurrent.futures import ProcessPoolExecutor, as_completed
from mtcnn import MTCNN
import fcntl
import time
import shutil
import math
from collections import Counter
import mplcursors
import random
import albumentations as A

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

albedo_folder = '../Dataset/Albedo'
depthmap_folder = '../Dataset/Depth_Map'
normalmap_folder = '../Dataset/Normal_Map'
num_cpus = 13

def log_to_file(filename, message):
    with open(filename, 'a') as file:
        # Khóa file trước khi ghi
        fcntl.flock(file, fcntl.LOCK_EX)
        try:
            file.write(message + '\n')
        finally:
            # Mở khóa file
            fcntl.flock(file, fcntl.LOCK_UN)
 
# Hàm thực hiện under-sampling (giữ lại 40 ảnh ngẫu nhiên)
def under_sample(id, target_count):
    all_files = os.listdir(os.path.join(albedo_folder, id))
    # Chọn ngẫu nhiên 16 ảnh từ danh sách
    selected_files = random.sample(all_files, target_count)
    
    # Xóa các ảnh không được chọn
    for file in all_files:
        if file not in selected_files:
            albedo_path = os.path.join(albedo_folder, id, file)
            depthmap_path = os.path.join(depthmap_folder, id, file)
            normalmap_path = os.path.join(normalmap_folder, id, file)
            os.remove(albedo_path)
            os.remove(depthmap_path)
            os.remove(normalmap_path)
            
    log_to_file('oversample.csv', id)

# Hàm tăng cường dữ liệu với albumentations
def augment_image(id, filename_to_duplicate):
    transform = A.Compose(
        [
            A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=20, p=0.8),
            
            # ElasticTransform là một phép biến đổi mô phỏng sự biến dạng đàn hồi (elastic deformation). Nó thường được sử dụng để tạo các biến dạng mềm mại, tự nhiên trong ảnh, giống như các vật liệu đàn hồi bị kéo giãn hoặc co lại.
            # alpha: Độ lớn của biến dạng.
            # sigma: Độ mượt mà của biến dạng. Giá trị cao làm cho biến dạng ít "sắc nét" hơn.
            # alpha_affine: Độ biến dạng affine bổ sung để kết hợp với hiệu ứng đàn hồi.
            # A.ElasticTransform(alpha=1, sigma=50, p=0.8),
            
            # A.Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), shear=(-15, 15), p=0.5),
            A.RandomCrop(height=400, width=300, p=1),
        ],
        additional_targets={'image1': 'image', 'image2': 'image'}
    )
    
    # Áp dụng augmentation
    albedo_image = cv2.imread(os.path.join(albedo_folder, id, filename_to_duplicate), cv2.IMREAD_UNCHANGED)
    depthmap_image = cv2.imread(os.path.join(depthmap_folder, id, filename_to_duplicate), cv2.IMREAD_UNCHANGED)
    normalmap_image = cv2.imread(os.path.join(normalmap_folder, id, filename_to_duplicate), cv2.IMREAD_UNCHANGED)
    
    transformed = transform(image=albedo_image, image1=depthmap_image, image2=normalmap_image)
    
    new_name = str(f"copy_{random.randint(1000, 9999)}_{filename_to_duplicate}")
    cv2.imwrite(os.path.join(albedo_folder, id, new_name), cv2.resize(transformed['image'], (448,226), interpolation=cv2.INTER_LINEAR))
    cv2.imwrite(os.path.join(depthmap_folder, id, new_name), cv2.resize(transformed['image1'], (448,226), interpolation=cv2.INTER_AREA))
    cv2.imwrite(os.path.join(normalmap_folder, id, new_name), cv2.resize(transformed['image2'], (448,226), interpolation=cv2.INTER_NEAREST))
                
# Hàm thực hiện over-sampling
def over_sample(id, number_oversampling):
    print(id_folder_path)
    all_files = [f for f in os.listdir(id_folder_path)]
    for _ in range(0,number_oversampling):
        filename_to_duplicate = random.choice(all_files)
        
        augment_image(id, filename_to_duplicate)
    log_to_file('oversample.csv', id)
                
# Thực hiện under-sampling và over-sampling cho các thư mục id
for id in os.listdir(albedo_folder):
    id_folder_path = os.path.join(albedo_folder, id)
    if os.path.isdir(id_folder_path):
        num_files = len(os.listdir(id_folder_path))
        if num_files < 3:
            over_sample(id, 2)
        elif num_files < 10:
            over_sample(id, 3)
        elif num_files > 40:
            under_sample(id, 40)