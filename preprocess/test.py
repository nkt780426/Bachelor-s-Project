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

photometric_db = './tmp/Photometric_DB'

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
            
def adjust_box(box, target_width=336, target_height=448):
    x, y , width, height = box
    
    center_x = x + width // 2
    center_y = y + height // 2
    
    # Tính toán các giá trị x, y mới sao cho center không đổi
    new_x = int(center_x - target_width // 2)
    new_y = int(center_y - target_height // 2)
    
    adjusted_box = [new_x, new_y, target_width, target_height]
    return adjusted_box

def crop_image(image, box, image_path):
    x, y, width, height = box

    # Thêm pading vào ảnh nếu x, y < 0
    image_height, image_width = image.shape[:2]
    
    print (image_height)
    print (image_width)
    
    if width > image_width or height > image_height:
        log_to_file('exception_crop.txt', image_path)
        return
    
    if x < 0:
        x = 0
    else:
        if (x + width) > image_width:
            x = image_width - width
            
    if y < 0:
        y = 0
    else:
        if (y + height) > image_height:
            y = image_height - height
        
    # Cắt ảnh theo tọa độ bounding box
    face_crop = image[y:y + height, x:x + width]

    # Tạo tên file mới với hậu tố '_crop' trong cùng thư mục
    
    base, ext = os.path.splitext(image_path)
    new_image_path = f"{base}_crop{ext}"

    # Lưu ảnh đã cắt
    if ext == '.bmp':
        cv2.imwrite(new_image_path, cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
    if ext == '.exr':
        cv2.imwrite(new_image_path, face_crop.astype(np.float32()))

def detect_face_in_all_bmp(session_path, detector, global_box, global_image_path):
    '''
        Return box, retry_image
    '''
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
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            result = detector.detect_faces(image)
            if len(result) == 1 :
                if result[0]['confidence'] >= 0.9:
                    box = adjust_box(result[0]['box'])
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
                    box = adjust_box(max_confidence_box)
                    crop_image(image, box, image_path)
                else:
                    retry_images.append(image_path)
            else:
                retry_images.append(image_path)
        return box, retry_images                
    else:
        box = adjust_box(global_box)
        image = cv2.cvtColor(cv2.imread(global_image_path), cv2.COLOR_BGR2RGB)
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
                box = adjust_box(result[0]['box'])
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
                box = adjust_box(max_confidence_box)
                crop_image(normal_map, box, normalmap_path)
                crop_image(albedo, box, albedo_path)
                crop_image(depth_map, box, depthmap_path)
            else:
                retry_images.append(albedo_path)
        else:
            retry_images.append(albedo_path)
        return box, retry_images
    else:
        box = adjust_box(global_box)
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
                
# Xử lý các session_path trong danh sách với ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=num_cpus) as executor:
    for session_path in session_paths_to_process:
        if count % 100 == 0:
            time.sleep(50)
        
        executor.submit(process_session, session_path, cpu_index)
        cpu_index = (cpu_index + 1) % num_cpus
        count += 1
                
# Xử lý các session_path trong danh sách với ProcessPoolExecutor
# with ProcessPoolExecutor(max_workers=num_cpus) as executor:
for id in os.listdir(photometric_db):
    id_path = os.path.join(photometric_db, id)
    if os.path.isdir(id_path):
        for session in os.listdir(id_path):
            session_path = os.path.join(id_path, session)
            process_session(session_path, cpu_index)
    