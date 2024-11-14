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

1. 15/11 (thứ 5): Tiền xử lý lại tất cả dữ liệu
    - Chia tập dữ liệu thành 4 tập: 2D, 3D (normal map, depth map, albedo map riêng biệt), check lại toàn bộ 441 người bằng jupyter.
        Ưu tiên tập 3D trước, 2D ném.
    - Luân chuyển dữ liệu giữa 2 laptop để thuận lợi cho train mô hình
    - Hẹn thầy đưa dữ liệu để thầy up lên cloud thuê kaggle hoặc thêu gpu.vn dựa vào lượng dữ liệu train.

2. 16/11 (thứ 6): Học về các mạng backbone phổ biến có thể dùng
    - CNN
    - Resnet
    - Inception resnet
    - Iresnet: https://arxiv.org/pdf/2004.04989
    - DenseNet
    - SENET: dùng thằng này với Restnet mạnh nhất :v

3. 17/11 (thứ 7): 
    Học về Onix format và pytorch 1 cách toàn diện ?
    Tạo tài khoản kaggle, google colab, thuê GPU và quyết định train ở đâu ?
        Albedo: 2,6 GB (train trên máy với 1000 epoch ?)
        Depth Map: 2,7 GB 
        Normal Map: 7,7 GB (thuê)

4. 18/11 (chủ nhật) - 22/11 (thứ 4): Train depthmap sinh ra embeding
    - Tensorflow hay Pytorch ? (Tensorflow thôi)
    - Quyết định/code loss function để train, chiến lược train với embeding (có multi task nữa không và nếu multi task thì dùng backbone nào)
    - Lưu vào database hoặc csv các embeding.

5. 22/11 (thứ 5) - 23/11 (thứ 6): Train normap map sinh ra embedding

6. 24/11 (thứ 7): Train albedo sinh ra embedding

7. 25/11 (chủ nhật) - 26/11 (thứ 2): Đọc research về fusion multi embeding và quyết định giải pháp cuối cùng để fusion các embedding.

8. 28/11 (thứ 3) - 29/11 (thứ 4): Fusion ra kết quả

9. 30/11 (thứ 5): Rà sát đối chiếu toàn bộ kết quả và báo cáo với thầy. Những thứ có thể cải thiện được và phát triển được.
    Có thể chụp ảnh cá nhân, dùng deep learning pre-trained sinh ra depth map, normal map, albedo của bản thân đưa vào hệ thống nhận diện.
    Dựng MLflow để có cái nhìn trực quan về các experment.

Việc công ty ưu tiên.
- Hoàn thành báo cáo của anh Đăng trước thứ 6. (đã xong)
- Debezium vekyc vào database AI team
- Debezium dữ liệu từ mongodb vào kafka, tạo 2 consumer: 
    Consumer 1: Tính toán request thành công hay thất bại dựa vào code và tran id sinh ra, liệt kê số lượng tất cả các mã lỗi => KSQL DB
    Consumer 2: Extract bản ghi mongo db về database AI team.
- Điều chỉnh airflow task mỗi người lên 1000.