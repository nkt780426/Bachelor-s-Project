import numpy as np
import albumentations as A
import cv2

# Cải tiến code anh Lâm
class GaussianNoise(A.ImageOnlyTransform):
    def __init__(self, mean=0, std=0.1, p=1.0):
        super().__init__(p)  # Khởi tạo base class với p là xác suất
        self.mean = mean
        self.std = std

    def apply(self, img, random_mask, random_std, **params):
        row, col, ch = img.shape

        # Tạo ma trận nhiễu Gaussian có std ngẫu nhiên
        gauss = np.random.normal(self.mean, random_std, (row, col, ch))

        # Áp dụng nhiễu vào phần của ảnh được chỉ định bởi mask
        noisy_img = img + gauss * random_mask[:, :, np.newaxis]
        noisy_img = np.clip(noisy_img, 0, 1)

        return noisy_img

    def __call__(self, **kwargs):
        # Lấy ảnh chính
        img = kwargs['image']
        row, col, _ = img.shape

        # Tạo giá trị ngẫu nhiên cho mask và std
        min_size = int(row * col * 0.1)
        max_size = int(row * col * 0.25)
        area_size = np.random.randint(min_size, max_size)

        mask = np.zeros((row, col))
        x = np.random.randint(0, col)
        y = np.random.randint(0, row)
        x_end = min(x + int(np.sqrt(area_size)), col)
        y_end = min(y + int(np.sqrt(area_size)), row)
        mask[y:y_end, x:x_end] = 1

        # Tạo std ngẫu nhiên cho nhiễu Gaussian
        random_std = np.random.uniform(0, 0.1)

        result = {}

        # Lặp qua tất cả các target (ảnh chính và các ảnh bổ sung)
        for key, image in kwargs.items():
            # Áp dụng biến đổi cho mỗi ảnh với cùng một mask và std
            result[key] = self.apply(image, random_mask=mask, random_std=random_std)

        # Trả về kết quả
        return result

class RandomResizedCropRect(A.ImageOnlyTransform):
    def __init__(self, size, scale=(0.8, 1.0), p = 1.0):
        super().__init__(p)  # Khởi tạo base class với p là xác suất
        self.size = size
        self.scale = scale
    
    def apply(self, img, random_scale, x, y):
        # Lấy kích thước ảnh
        img_height, img_width, _ = img.shape

        # Tính toán kích thước và tọa độ cho crop
        width = int(img_width * random_scale)
        height = int(img_height * random_scale)

        # Crop ảnh
        img = img[y:y+height, x:x+width]

        # Resize ảnh về kích thước mong muốn
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

        return img
    
    def __call__(self, **kwargs):
        # Tạo giá trị ngẫu nhiên chung cho tất cả các ảnh
        random_scale = np.random.uniform(*self.scale)
        img_height, img_width, _ = kwargs['image'].shape  # Lấy kích thước của ảnh chính
        x = np.random.randint(0, img_width - int(img_width * random_scale) + 1)
        y = np.random.randint(0, img_height - int(img_height * random_scale) + 1)
        
        # Tiến hành transform trên tất cả các ảnh (chính và bổ sung)
        result = {}

        # Lặp qua tất cả các target (ảnh chính và các ảnh bổ sung)
        for key, image in kwargs.items():
            # Áp dụng biến đổi cho mỗi ảnh với cùng một giá trị random_scale và tọa độ x, y
            result[key] = self.apply(image, random_scale=random_scale, x=x, y=y)
        
        return result
    
# Cách dùng
# transform = A.Compose([
#     RandomResizedCropRect(size=256, scale=(0.2, 0.2), p=1.0),
#     GaussianNoise(p=?),
# ], additional_targets={
#     'depthmap': 'image',
#     'normalmap': 'image'
# })

# # Áp dụng transformation
# transformed = transform(image=albedo, depthmap=depthmap, normalmap=normalmap)