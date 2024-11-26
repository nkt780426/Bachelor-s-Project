Dataset bao gồm ảnh của 453 người, mỗi người có thể có nhiều phiên chụp. Mỗi phiên chụp là 1 folder, trong forder này có:
- 4 ảnh định dạng bmp của đối tượng. Những ảnh này được đặt dưới các điều kiện ánh sáng khác nhau với các góc chiếu sáng (illumination) được mô tả trong file LightSource.m
- LightSource.m chứa thông tin về các góc zenith và azimuth của 4 nguồn sáng trong phiên đó.
- metadata.txt: chứa tọa độ x và y của 11 đặc điểm khuôn mặt, bao gồm: khóe mắt ngoài bên trái, khóe mắt trong bên trái, điểm giữa trán, khóe mắt trong bên phải, khóe mắt ngoài bên phải, mũi bên trái, đỉnh mũi, mũi bên phải, khóe miệng bên trái, khóe miệng bên phải và cằm.
- metadataII.txt: chứa thêm các thông tin: giới tính, kính mắt, râu (không có, có râu, có ria, cả hai, hay lún phún), tư thế (1 là chính diện - 5 là nghiêng), chất lượng (tốt, mờ, hoặc tối), chướng ngại (điện thoại di động, tay, tóc hoặc khác), biểu cảm (nhìn trống rỗng, tích cực, tiêu cực hoặc khác), và các thông tin khác (miệng mở hoặc mắt nhắm).

Để giảm dung lượng tải xuống, dữ liệu về pháp tuyến bề mặt và tái tạo không được bao gồm mà phải được tạo ra từ dữ liệu có sẵn. Code Matlab để tạo ra các dữ liệu này được cung cấp trong thư mục PhotofaceDB. Để tạo pháp tuyến bề mặt và tái tạo bề mặt, hãy chạy hàm generateSNAndZ() trong Matlab và truyền vào đường dẫn thư mục gốc của Photoface Database. Sau khi chạy thành công, sẽ có ba file mới trong mỗi thư mục:

sn.mat - chứa các thành phần pháp tuyến bề mặt px và py tại mỗi điểm ảnh
z.mat - bề mặt tích hợp sử dụng phương pháp của Frankot-Chellappa (1988)
albedo.mat - ước lượng suất phản chiếu (albedo) tại mỗi điểm ảnh
Bên cạnh đó, công cụ truy vấn Photoface đi kèm trong bản tải xuống này cho phép tìm kiếm và tạo ra các bộ dữ liệu tùy chỉnh thông qua giao diện đồ họa Matlab và cơ sở dữ liệu PostgreSQL. Hướng dẫn thêm có thể tìm thấy trong file PhotofaceQueryTool\Readme.