import os
from scipy.io import loadmat
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from PIL import Image
import logging
from scipy.io import loadmat
from concurrent.futures import ProcessPoolExecutor, as_completed
from mtcnn import MTCNN
import fcntl
import time
import shutil

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

photometric_db = '../Photometric_DB'

dest_folder = '../3D_Dataset/Normal_Map'

files_to_copy = ['normalmap.exr']

num_cpus = 14

# Hàm sao chép file
def copy_file(filename, session_path, new_id_folder_path, session):
    source_file = os.path.join(session_path, filename)
    _, ext = os.path.splitext(filename)
    destination_file = os.path.join(new_id_folder_path, f"{session}{ext}")
    shutil.copy(source_file, destination_file)

# Tạo thư mục đích nếu chưa tồn tại
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder, exist_ok=True)

# Sử dụng ProcessPoolExecutor để sao chép file từ các session path hợp lệ
with ProcessPoolExecutor() as executor:
    futures = []
    
    for id in os.listdir(photometric_db):
        id_path = os.path.join(photometric_db, id)
        if os.path.isdir(id_path):
            new_id_folder_path = os.path.join(dest_folder, id)
            if not os.path.exists(new_id_folder_path):
                os.mkdir(new_id_folder_path)
            
            for session in os.listdir(id_path):
                session_path = os.path.join(id_path, session)
                futures.append(executor.submit(copy_file, files_to_copy[0], session_path, new_id_folder_path, session))

    # Đợi tất cả các tác vụ sao chép hoàn thành
    for future in as_completed(futures):
        future.result()