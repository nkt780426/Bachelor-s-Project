import os

# Đường dẫn đến thư mục chứa các thư mục con (có thể là label)
train_dir = os.path.join(os.path.dirname(__file__), '..', '..', '3D_Dataset','Albedo')

# Lấy tất cả các tên thư mục trong train_dir
label_dirs = os.listdir(train_dir)

print(len(label_dirs))

# Sắp xếp các tên thư mục theo chiều tăng dần
label_dirs.sort()

# Ánh xạ tên thư mục thành chỉ số liên tiếp
label_map = {index: label for index, label in enumerate(label_dirs)}

# In kết quả ánh xạ
# print("Label mapping:", label_map)