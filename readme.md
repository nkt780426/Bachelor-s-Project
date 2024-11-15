# Timeline
3. 22/12/2024: Giáo viên hướng dẫn lựa chọn đồng ý hoặc không đồng ý cho sinh viên bảo vệ lên qldt
4. Nộp quyển, mã nguồn
    Bản mềm nộp nên qldt trước ngày 6/1/2025
        Bản mềm (phải trùng tên đề tài), ... quay lại team đọc
        Mã nguồn chương trình
        Kết quả kiểm tra trùng lặp
    Bản cứng: ngày 7/1/2025
        Phải có chữ ký của giáo viên và học sinh
5. Phản biện
    Thông tin giáo viên phản biện sẽ được đưa lên qldt trong vòng 3 ngày. Chủ định mail với giáo viên nếu chưa thấy phản hồi
6. Bảo vệ tốt nghiệp (20-24/1/2025)

# Tiến độ đồ án

0. Đã hiểu phải làm cái gì, đã tỉnh ngộ. Kiến thức đã có, game là dễ.

1. 14/11 (thứ 5): Tiền xử lý lại tất cả dữ liệu (90%)
    - Chia tập dữ liệu thành 4 tập: 2D, 3D (normal map, depth map, albedo map riêng biệt), check lại toàn bộ 441 người bằng jupyter.
        Ưu tiên tập 3D trước, 2D ném. (đã xong)
    - Luân chuyển dữ liệu giữa 2 laptop để thuận lợi cho train mô hình (đa xong)
    - Hẹn thầy đưa dữ liệu để thầy up lên cloud thuê kaggle hoặc thêu gpu.vn dựa vào lượng dữ liệu train.(chưa thực hiện được. Mang dây lan kết nối với máy thầy là 1 giải pháp :v)

    **Kết quả:**
    - **Chỉnh sửa lại thuật toán crop ảnh không thêm padding nữa. Tất cả ảnh về kích thước (448, 336). Các ảnh được crop sau cho ưu tiên có center trùng với box của mtcnn và có size (448, 336) và scale up dữ nguyên tỷ lệ rồi crop lại nếu thiếu (sử dụng các phương pháp nội suy tích hợp để scale up tránh sai lệch thông tin)**
    - **Albedo = DepthMap = 1.9 GB, Normal_Map= 5.5 GB**

2. 15/11 (thứ 6): Học về các mạng backbone phổ biến có thể dùng
    - CNN: (đã xong)
    - Resnet: 
    - Inception resnet
    - Iresnet: https://arxiv.org/pdf/2004.04989
    - DenseNet:
    - SENET: dùng thằng này với Restnet mạnh nhất :v

    **Kết quả: Đã tìm hiểu lại về bản chất, ý nghĩa mạng CNN nhưng chưa kịp tìm hiểu các mạng deepth learning. Em mong muốn hiểu bản chất cấu trúc mạng và có thể đánh giá, tự xây code xây dựng mạng backbone trong quá trình train.**
3. 16/11 (thứ 7): 
    Học về Onix format và pytorch 1 cách toàn diện ?
    Tạo tài khoản kaggle, google colab, thuê GPU và quyết định train ở đâu ?
        Albedo: 2,6 GB (train trên máy với 1000 epoch ?)
        Depth Map: 2,7 GB 
        Normal Map: 7,7 GB (thuê)

4. 17/11 (chủ nhật) - 20/11 (thứ 4): Train depthmap sinh ra embeding
    - Tensorflow hay Pytorch ? (Tensorflow thôi)
    - Quyết định/code loss function để train, chiến lược train với embeding (có multi task nữa không và nếu multi task thì dùng backbone nào)
    - Lưu vào database hoặc csv các embeding.

5. 21/11 (thứ 5) - 22/11 (thứ 6): Train albedo sinh ra embedding

6. 23/11 (thứ 7) - 24/11 (chủ nhật): Train normal map sinh ra embedding

7. 25/11 (Thứ 2) - 26/11 (thứ 3): Đọc research về A survey về fusion multi embeding và quyết định giải pháp cuối cùng để fusion các embedding.

8. 27/11 (thứ 4) - 29/11 (thứ 5): Fusion các embeding ra kết quả.

9. 30/11 (thứ 6): Rà sát đối chiếu toàn bộ kết quả và báo cáo với thầy. Những thứ có thể cải thiện được và phát triển tiếp được trong thời gian còn lại.
    Có thể chụp ảnh cá nhân, dùng deep learning pre-trained sinh ra depth map, normal map, albedo của bản thân đưa vào hệ thống nhận diện.
    Dựng MLflow để có cái nhìn trực quan về các experement.

Việc công ty.
- Debezium vekyc vào database AI team
- Debezium dữ liệu từ mongodb vào kafka, tạo 2 consumer: 
    Consumer 1: Tính toán request thành công hay thất bại dựa vào code và tran id sinh ra, liệt kê số lượng tất cả các mã lỗi => KSQL DB
    Consumer 2: Extract bản ghi mongo db về database AI team.
- Điều chỉnh airflow task mỗi người lên 1000. (đã xong)
- Hoàn thành báo cáo của anh Đăng trước thứ 6. (đã xong)