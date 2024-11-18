# Abstract
Hầu hết các thuật toán phân loại học máy đều được xây dựng với giả định dataset cần bằng trong tất cả các lớp. Nhiều phương pháp đã được đề xuất để giải quyết vấn đề mất cân bằng dữ liệu ở algorithm-level và data-level. Những phương pháp này chủ yếu bào gồm ensemble learning (học tập tổng hợp) và data augmentation.

Research đề xuất 1 hướng đi mới để giải quyết vấn đề mất cân bằng dữ liệu thông qua 1 chiến lược phân chia dữ liệu mới gọi là balanced split (phân chia cân bằng). Hiển nhiên việc phân chia dataset có tác động mạnh đến quá trình train. Research chỉ ra nhược điểm của các chiến lược phân chia dataset phổ biến và giới thiệu phương pháp mới giải quyết tất cả các nhược điểm trên.

# Introduce
1. Các ref của research từ 1 -> 22 nói về các phương pháp xử lý ở mức ensemble learning.

2. Một cách khác để chống lại sự mất cân bằng là tạo ra các mẫu mới ở class thiểu số. Các ref của research từ 23 -> 32 nói về vấn đề này. Các phương pháp này bao gồm: SMOTE [23], SMOTE-ENN [24], SMOTE-SVM [25],Kmeans-SMOTE [26], Borderline-SMOTE [27], ADASYN[28], v.v. Các mạng đối nghịch chung (GAN) [29] và các phiên bản của nó như GAN lượng tử [30], SMOTified-GAN [31] là một bổ sung gần đây cho các kỹ thuật này. Ngoài ra, còn có RUS để under sampling major class[32].

3. Nhược điểm của các phương pháp 1 và 2 là tạo ra dữ liệu nhân tạo để chống lại sự mất cân bằng. Do dó có 2 vấn đề luôn được đặt ra. Chiến lược phân chia dữ liệu mới trong research này chỉ sử dụng dataset gốc và giải quyết được 2 vấn đề:
    - Không đảm bảo đại diện của các ảnh nhân tạo được sinh ra ở lớp thiểu số
    - Dataset mất cân bằng.

4. Reseach này được tạo ra nhằm mục đích thu hút nhiều reseacher hơn tìm hiều về vấn đề do ít người quá, đa phần toàn tự tái tạo ra ảnh nhân tạo để cân bằng dữ liệu.

# Balanced split.
Các bước thực hiện như sau

1. Tính toán số lượng mẫu cần có trong training set (TrS) dựa theo tỷ lệ chia tập train (tr) và tổng số mẫu của cả dataset (m)
    Trs  = tr * m
2. Lấy số lượng mẫu bằng nhau từ mỗi lớp: Trs/N với N là số lượng lớp. Tuy nhiên phải đảm bảo điều kiện sau.
    
# Bỏ research