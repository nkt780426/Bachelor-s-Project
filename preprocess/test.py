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

photometric_db = '../Photometric_DB'

num_cpus = 10

def log_to_file(filename, message):
    with open(filename, 'a') as file:
        # Khóa file trước khi ghi
        fcntl.flock(file, fcntl.LOCK_EX)
        try:
            file.write(message + '\n')
        finally:
            # Mở khóa file
            fcntl.flock(file, fcntl.LOCK_UN)
            
def adjust_box(box, shape):
    x, y, width, height = box
    center_x = x + width // 2
    center_y = y + height // 2

    # Tăng kích thước của hộp thêm 30%
    new_width = int(width + 0.25 * width)
    new_height = int(height + 0.3 * height)

    # Tính lại tọa độ mới sao cho center vẫn không thay đổi
    new_x = int(center_x - new_width // 2)
    new_y = int(center_y - new_height // 2)

    # Điều chỉnh nếu new_x hoặc new_y là âm
    if new_x < 0:
        new_height = new_height - int(new_height * (abs(new_x)/new_width) * 2)
        new_y = new_y + int(new_height * (abs(new_x)/new_width))
        new_width = new_width - abs(new_x)
        new_x = 0
    if new_y < 0:
        new_height = new_height - 2 * abs(new_y)
        new_y = 0
    
    image_height, image_width = shape[:2]
    
    while (new_x + new_width > image_width):
        new_x = new_x + new_width * 0.05
        new_width = new_width * 0.9
    
    while(new_y + new_height > image_height):
        new_y = new_y + new_height * 0.05
        new_height = new_height * 0.9
        
    adjusted_box = [new_x, new_y, new_width, new_height]
    return adjusted_box

def crop_image(image, box, image_path):
    x, y, width, height = box

    face_crop = image[y:y + height, x:x + width]

    # Tạo tên file mới với hậu tố '_crop' trong cùng thư mục
    base, ext = os.path.splitext(image_path)
    new_image_path = f"{base}_crop{ext}"

    # Lưu ảnh đã cắt
    if ext.lower() == '.bmp':
        cv2.imwrite(new_image_path, face_crop)
    elif ext.lower() == '.exr':
        cv2.imwrite(new_image_path, face_crop.astype(np.float32))

def detect_face_in_all_bmp(session_path, detector, global_box, global_image_path):
    if global_box is None:
        box = None
        retry_images = []
        bmp_images = [
            os.path.join(session_path, 'im0.bmp'),
            os.path.join(session_path, 'im1.bmp'),
            os.path.join(session_path, 'im2.bmp'),
            os.path.join(session_path, 'im3.bmp')
        ]
        for image_path in bmp_images:
            image = cv2.imread(image_path)
            result = detector.detect_faces(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if len(result) == 1 :
                if result[0]['confidence'] >= 0.9:
                    box = adjust_box(result[0]['box'], image.shape)
                    crop_image(image, box, image_path)
                else:
                    retry_images.append(image_path)
            elif len(result) >=2:
                max_confidence_index = max(
                    (i for i, res in enumerate(result) if 'confidence' in res and res['confidence'] >= 0.9),
                    key=lambda i: result[i]['confidence'],
                    default=None
                )

                max_confidence_box = result[max_confidence_index]['box'] if max_confidence_index is not None else None
                
                if max_confidence_box:
                    box = adjust_box(max_confidence_box, image.shape)
                    crop_image(image, box, image_path)
                else:
                    retry_images.append(image_path)
            else:
                retry_images.append(image_path)
        return box, retry_images                
    else:
        image = cv2.imread(global_image_path)
        box = adjust_box(global_box, image.shape)
        crop_image(image, box, global_image_path)
    
def detect_face_in_all_exr(session_path, detector, global_box):
    normalmap_path = os.path.join(session_path, 'normalmap.exr')
    albedo_path = os.path.join(session_path, 'albedo.exr')
    depthmap_path = os.path.join(session_path, 'depthmap.exr')
    
    normal_map = cv2.imread(normalmap_path, cv2.IMREAD_UNCHANGED)
    albedo = cv2.imread(albedo_path, cv2.IMREAD_UNCHANGED)
    depth_map = cv2.imread(depthmap_path, cv2.IMREAD_UNCHANGED)
    
    if global_box is None:
        box = None
        retry_images = []
        result = detector.detect_faces(cv2.cvtColor(albedo, cv2.COLOR_GRAY2RGB))
        if len(result) == 1:
            if result[0]['confidence'] >= 0.9:
                box = adjust_box(result[0]['box'], albedo.shape)
                crop_image(normal_map, box, normalmap_path)
                crop_image(albedo, box, albedo_path)
                crop_image(depth_map, box, depthmap_path)
            else:
                retry_images.append(albedo_path)
        elif len(result) >=2:
            # Tìm index của phần tử có confidence cao nhất (>= 0.99)
            max_confidence_index = max(
                (i for i, res in enumerate(result) if 'confidence' in res and res['confidence'] >= 0.9),
                key=lambda i: result[i]['confidence'],
                default=None
            )
            max_confidence_box = result[max_confidence_index]['box'] if max_confidence_index is not None else None
            
            if max_confidence_box:
                box = adjust_box(max_confidence_box, albedo.shape)
                crop_image(normal_map, box, normalmap_path)
                crop_image(albedo, box, albedo_path)
                crop_image(depth_map, box, depthmap_path)
            else:
                retry_images.append(albedo_path)
        else:
            retry_images.append(albedo_path)
        return box, retry_images
    else:
        box = adjust_box(global_box, albedo.shape)
        crop_image(normal_map, box, normalmap_path)
        crop_image(albedo, box, albedo_path)
        crop_image(depth_map, box, depthmap_path)
        
def process_session(session_path, cpu_index):
    try:
        global_box = None
        global_retry_images = []

        # Khởi tạo detector cho CPU cụ thể
        detector = MTCNN(device=f"CPU:{cpu_index}")

        # Gọi hàm detect cho từng định dạng
        bmp_box, bmp_retry_images = detect_face_in_all_bmp(session_path, detector, None, None)
        exr_box, exr_retry_images = detect_face_in_all_exr(session_path, detector, None)

        # Xác định global_box và global_retry_images
        global_box = exr_box if exr_box else bmp_box
        global_retry_images.extend(bmp_retry_images)
        global_retry_images.extend(exr_retry_images)

        # Xử lý các ảnh cần retry
        if global_retry_images:
            if global_box:
                for image_path in global_retry_images:
                    _, ext = os.path.splitext(image_path)
                    if ext == '.bmp':
                        detect_face_in_all_bmp(None, None, global_box, image_path)
                    elif ext == '.exr':
                        detect_face_in_all_exr(session_path, None, global_box)
            else:
                log_to_file('failed_crop.txt', session_path)
                print(f"Không detect được face trong session {session_path}.")
    except Exception as e:
        log_to_file('session_exception.txt', session_path)
        raise
    finally:
        log_to_file('processed_sessions.txt', session_path)  # Ghi lại session đã xử lý
        
cpu_index = 0
count = 1

# Đọc các session đã xử lý từ file checkpoint (nếu có)
processed_sessions = set()
if os.path.exists('processed_sessions.txt'):
    with open('processed_sessions.txt', 'r') as f:
        processed_sessions = set(f.read().splitlines())

# Thu thập tất cả session_path chưa được xử lý
session_paths_to_process = []
for id in os.listdir(photometric_db):
    id_path = os.path.join(photometric_db, id)
    if os.path.isdir(id_path):
        for session in os.listdir(id_path):
            session_path = os.path.join(id_path, session)
            # Chỉ thêm các session chưa xử lý vào danh sách
            if session_path not in processed_sessions:
                session_paths_to_process.append(session_path)
               

for session_path in session_paths_to_process:
    process_session(session_path, 0)