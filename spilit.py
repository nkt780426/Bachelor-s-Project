import os
import shutil

def split_folder(src_folder, dst_folder, batch_size):
    """
    Tách thư mục thành các thư mục con với số lượng folder con cố định.
    Mỗi thư mục nhỏ hơn sẽ chứa các folder con và 3 file từ thư mục gốc.
    
    Args:
        src_folder (str): Đường dẫn đến thư mục nguồn (PhotofaceDB).
        dst_folder (str): Đường dẫn đến thư mục đích chứa các folder đã tách.
        batch_size (int): Số lượng folder con mỗi lần tách.
    """
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    
    # Lấy danh sách tất cả các thư mục con và file trong thư mục nguồn
    all_items = os.listdir(src_folder)
    all_folders = sorted([item for item in all_items if os.path.isdir(os.path.join(src_folder, item))])
    all_files = sorted([item for item in all_items if os.path.isfile(os.path.join(src_folder, item))])

    # Đảm bảo có ít nhất 3 file
    if len(all_files) < 3:
        raise ValueError("Cần ít nhất 3 file trong thư mục gốc.")

    # Chia thành các batch
    for i in range(0, len(all_folders), batch_size):
        batch_folders = all_folders[i:i + batch_size]
        batch_name = f'batch_{i // batch_size + 1}'
        batch_dst_folder = os.path.join(dst_folder, batch_name)

        os.makedirs(batch_dst_folder, exist_ok=True)

        # Di chuyển các folder con vào batch mới
        for folder in batch_folders:
            src_path = os.path.join(src_folder, folder)
            dst_path = os.path.join(batch_dst_folder, folder)
            shutil.copytree(src_path, dst_path)

        # Copy 3 file vào mỗi batch
        for file in all_files[:3]:  # Lấy 3 file đầu tiên
            file_src = os.path.join(src_folder, file)
            file_dst = os.path.join(batch_dst_folder, file)
            shutil.copy(file_src, file_dst)

# Thông tin về thư mục
src_folder = 'Photometric_DB'  # Thư mục gốc
dst_folder = 'Photometric_DB_split'  # Thư mục đích chứa các batch
batch_size = 20  # Số lượng folder con trong mỗi batch

# Thực thi
split_folder(src_folder, dst_folder, batch_size)
