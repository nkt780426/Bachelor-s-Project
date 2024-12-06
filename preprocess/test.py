import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from mtcnn import MTCNN
import fcntl
import time
import shutil
import math
import random

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

photometric_db = './tmp/Photometric_DB'

num_cpus = 10

def log_to_file(filename, message):
    with open(filename, 'a') as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        try:
            file.write(message + '\n')
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)
            
def adjust_box(box, shape):
    x, y, width, height = box
    center_x = x + width // 2
    center_y = y + height // 2

    # Tăng kích thước hộp 1.5 lần chiều cao ban đầu
    im_size = int(height * 2)
    
    new_x = center_x - int(im_size//2)
    new_y = center_y - int(im_size//2)
    
    if new_x < 0:
        im_size = im_size - abs(new_x) * 2
        new_y = new_y + abs(new_x)
        new_x = 0
        
    if new_y < 0:
        im_size = im_size - abs(new_y) * 2
        new_x = new_x + abs(new_y)
        new_y = 0
    
    image_height, image_width = shape[:2]
    
    tmp_width = new_x + im_size - image_width
    if tmp_width > 0:
        im_size = im_size - tmp_width * 2
        new_y += tmp_width
        new_x += tmp_width
     
    tmp_height = new_y + im_size - image_height
    if tmp_height > 0:
        im_size = im_size - tmp_height * 2
        new_x += tmp_height
        new_y += tmp_height

    adjusted_box = [new_x, new_y, im_size, im_size]

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
        result = detector.detect_faces(cv2.cvtColor(albedo*255, cv2.COLOR_GRAY2RGB))
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

if __name__ == '__main__':
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
            
    try:
        # Xử lý các session_path trong danh sách với ProcessPoolExecutor
        for session_path in session_paths_to_process:
            if count % 100 == 0:
                time.sleep(30)
            
            process_session(session_path, cpu_index)
            cpu_index = cpu_index % num_cpus
            cpu_index += 1
            count += 1
    except Exception as e:
        raise
