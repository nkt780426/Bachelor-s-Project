# 1. Introduction
- Bài báo đề cập đến ứng dụng của STOA và những thách thức công nghệ này gặp phải trong từng lĩnh vực, giải pháp nào được ưu tiên sử dụng trong lĩnh vực nào.\
- Bài báo này còn đề cập đến nguyên tắc cơ bản hình thành nên các kiến trúc deeplarning nói chung.
- Mục lục
    - Phần 2: Các nguyên tắc cơ bản trong deep learning như layers, attention mechanisms, activation functions, model optimization, loss function, regularization, ...
    - Phần 3: Các Deep learning models và các kiến trúc mạng CNN
    - Phần 4: Ứng dụng thực tiễn của học sâu (quan trọng nhất)
    - Phần 5: Thảo luận về những thách thức, định hướng trong tương lai trong lĩnh vực deep learning.
    - Phần 6: Kết luận

# 5. Challenges and Future Directions
## 5.1. Avaiability and Quaylity (tính khả dụng và chất lượng)
Dataset phải có nhiều sample và càng đa dạng càng tốt thì mới cho ra chất lượng và tính tổng quát cao.
    - Độ phức tạp của kiến trúc mạng neutron (complexity) ảnh hưởng đến vấn đề over fitting (thường do kiến trúc mạng quá phức tạp và mô hình ít dữ liệu)
    - Độ tin cậy của dataset, liệu có sample sai lệch nào được đưa vào trong dataset không, điều này có thể ảnh hưởng đáng kể đến hiệu suất của model => 1 trong những cách tiếp cận giải quyết vấn đề này là transfer learning (sử dụng 1 mô hình đã được train trên 1 tập dữ liệu lớn để đào tạo tinh chỉnh lại với tập dữ liệu nhỏ hơn và cách này giảm thiểu tình trạng thiếu dữ liệu trong target domain).
    - Cách tiếp cận khác để giải quyết vấn đề này là data augmentation. Các thao tác như rotation, cropping, translation (tịnh tiến), ... Chú ý cần xem xét output nó có phản ánh thực tế không, ví dụ như các hiệu ứng như noise, flip có thể thay đổi cấu trúc các điểm, ... Nói chung cần cẩn thận khi chọn chiến lược data augmentation cho phù hợp.

## 5.2 Interpretability and Explainability (Khả năng diễn giải và khả năng giải thích)

Khả năng này rất quan trọng để xây dựng lòng tin và hiểu cách model predict, đặc biệt trong các lĩnh vực rủi do cao như y tế, chăm sóc sức khỏe. Tuy nhiên khi mạng neutron càng ngày càng deep, ta thường coi nó là 1 hộp đen và khó giải thích được chúng. Researchers sẽ phải tập trung vào các phương pháp cung cấp thông tin, làm thế nào để model đưa ra quyết định từ feature map ban đầu để model trở nên minh bạch đáng tin cậy hơn. 1 số phương pháp explain có thể kể đến như
- visualization: Làm nổi bật
- model distillation
- intrinsic (tự giải thích): Multi-task để có thêm thông tin giải thích cho chính quyết định của nó

## 5.3. Ethics and Fairness (Đạo đức và công bằng)

Deep learning ngày càng được đưa vào để áp dụng trong những lĩnh vực có độ rủi do cao như tuyển dụng, tư pháp, ... Tuy nhiên không có gì đảm bảo mô hình học chính xác 100%, những trường hợp sai lệch có thể dẫn đến vấn đề mất công bằng.

Nhiều phương pháp đã được thực hiện nhằm giảm thiểu vấn đề này. ....

## 5.5 Adversarial Attack and Defense (tấn công và phòng thủ)

...

## 6. Conclusion

# 2. Fundamentals of Deep Learning

## 2.1. Layers
- Input layer, hideen layer (nhiều lớp chịu trách nhiệm extract và xử lý các thông tin phức tạp feature map) và output layer
- Fully connected layers
- Cấu trúc của 1 neutron thường và neutron CNN (CNN chủ yếu được sử dụng để xử lý các dữ liệu có cấu trúc như image, time series).
- Pooling layers thường được áp dụng sau CNN layers để giảm dần kích thước của không gian (cao và rộng) của feature map.

## 2.2 Attention mechanisms

Không phải phần nào của input đều đáng quan trọng phải học. Ví dụ như backgroud của ảnh không có ý nghĩa trong quá trình recognition. Trong conv, tất cả các feauture đều được xử lý thông nhất mà không xem xét mức độ quan trọng khác nhau của các thành phần dữ liệu khác nhau của input. 

Hạn chế này được giải quyết bởi cơ chế attention mechanisms - cho phép mô hình gán trọng số cho các features từ đó biết thằng nào quan trọng hơn. Nhờ tính năng này, model ưu tiên học các khía cạnh quan trọng hơn của vật thể, tăng độ chính xác khi ra quyết định.
    A = f(g(x), x)

0. Channel attention
    - Squeeze-and-Excitation (SE) Attention:
        Sử dụng global average pooling (GAP) để giảm chiều dữ liệu.
        Sử dụng hai lớp fully-connected với các hàm kích hoạt ReLU và sigmoid để tạo trọng số.
        Hạn chế: Chi phí tính toán cao và mất thông tin ở mức độ không gian (spatial level).

    - Các cải tiến: 
        GSoP Attention: Sử dụng convolution 1×1 và tính toán sự tương quan kênh.
        ECA Attention: Thay fully-connected layers bằng convolution 1D để giảm chi phí tính toán.
1. Temporal Attention
    - Tập trung vào các thời điểm quan trọng trong dữ liệu tuần tự, ví dụ: Video: Chọn các khung hình chứa thông tin quan trọng để nhận dạng hành động.
    - Temporal Adaptive Module (TAM):
        Có hai nhánh: Nhánh cục bộ (local) và nhánh toàn cục (global).
        Nhánh cục bộ: Sử dụng các convolution 1D để tính trọng số cục bộ.
        Nhánh toàn cục: Sử dụng fully-connected layers để tạo trọng số toàn cục.
2. Self-Attention
    - Được sử dụng trong xử lý ngôn ngữ tự nhiên (NLP), lần đầu tiên được đề xuất trong bài toán dịch máy.
    - Cách hoạt động:
        Dữ liệu đầu vào được chuyển đổi thành query, key, và value thông qua linear projection.
        Trọng số được tính bằng dot product giữa query và key, chuẩn hóa và áp dụng softmax.
        Self-Attention là thành phần cơ bản của kiến trúc Transformer.
3. Spatial Attention
    - Tập trung vào các vùng không gian (spatial) quan trọng trong dữ liệu, ví dụ:
        Xác định các vùng trọng tâm trong ảnh để dự đoán chính xác hơn.
    - Attention Gate:
        Sử dụng convolution 1×1 và sigmoid để tạo trọng số cho các vùng không gian.

## 2.3 Activation Functions
Nói về việc sigmoid và hyperbolic tangent (được ưa dùng hơn sigmoid vì gradient mạnh hơn và dễ hội tụ hơn) gây ra vấn đề vanishing gradient trong các mạng deep learning. Do đó nó bị thay thế bởi Relu activation function. Nhiều biến thể của Relu đã được ra đời như Leaky ReLU [163], sigmoid linear unit [167] và  exponential linear unit [39]. Mỗi biến thể đều mang lại những lợi thế riêng để xây dựng các ứng dụng dựa trên học sâu.

## 2.4 Parameter Learning and Loss Functions

Weights (hay parameters) của các mô hình học sâu thường được tối ưu bằng các thuật toán gradient descent. Ngoài gradient descent, còn có rất nhiều các thuật toán tối ưu trọng số khác cũng có thể được sử dụng.

Thuật toán này optimal weights bằng cách cập nhật trọng số theo từng lần lặp sao cho nó sẽ làm cho gía trị loss trở nên tối thiểu.

Về bài toán phân loại, các loss được sử dụng thường là dạng log hoặc cross entropy. Các loss như square loss, absolute loss được sử dụng cho regression problem (phân loại hồi quy)