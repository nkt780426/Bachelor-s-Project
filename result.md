1. Dataset chưa đủ đa dạng, chủ yếu là ảnh chất lượng cao, không có khuôn mặt có độ biến thiên lớn, biểu cảm => Khó đánh giá Magface. Magface được thiết kế để hoạt động tốt với.
    
    - Dữ liệu có chất lượng ảnh thấp, có nhiều biểu cảm khuôn mặt phức tạp, bị che khuất bởi ánh sáng cũng như vật thể (trích từ paper magface) => Photometric stereo đã xử lý gần hết.

    - Dễ dàng tích hợp vào các hệ thống đã tồn tại (chỉ cần thêm mag linear vào sau cùng, không như siamese net và multi task leanring)
    
2. Điểm yếu của triple loss:
    - Phải có nhiều class có 2 ảnh trở nên mới có thể train được => Trong khi magface chỉ cần 1 ảnh => Lý do phải ném 153 id có 1 session đi mới có thể so sánh 2 phương pháp.
    
    - Thời gian train lâu hơn gần gấp 3 lần (19,253s vs magface vaf 53,453s vs triplet).

3. 