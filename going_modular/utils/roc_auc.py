import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score
from concurrent.futures import ProcessPoolExecutor

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

# def compute_roc_auc(
#     dataloader: torch.utils.data.DataLoader, 
#     model: torch.nn.Module, 
#     device: str
# ):
    
#     model.eval()
#     with torch.no_grad():
#         embeddings_list = []

#         for batch in dataloader:
#             images, ids = batch
#             images = images.to(device)
#             embeddings = model.get_embedding(images)
            
#             for i in range(embeddings.size(0)):
#                 embeddings_list.append((ids[i].item(), embeddings[i].cpu().numpy()))
                
#         euclidean_scores = []
#         euclidean_labels = []
#         cosine_scores = []
#         cosine_labels = []

#         for i in range(len(embeddings_list)):
#             for j in range(i + 1, len(embeddings_list)):
#                 id1, embedding1 = embeddings_list[i]
#                 id2, embedding2 = embeddings_list[j]

#                 # Convert numpy arrays to tensor and move to device
#                 embedding1 = torch.tensor(embedding1)
#                 embedding2 = torch.tensor(embedding2)

#                 # Euclidean distance
#                 euclidean_score = F.pairwise_distance(embedding1, embedding2).item()
#                 # Cosine similarity
#                 cosine_score = F.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0)).item()

#                 # Store similarity scores and labels
#                 euclidean_scores.append(euclidean_score)
#                 euclidean_labels.append(0 if id1 == id2 else 1)
#                 cosine_scores.append(cosine_score)
#                 cosine_labels.append(1 if id1 == id2 else 0)

#         # Compute ROC AUC for Euclidean distance
#         euclidean_true_labels = np.array(euclidean_labels)
#         euclidean_pred_scores = np.array(euclidean_scores)
#         fpr_euclidean, tpr_euclidean, thresholds_euclidean = roc_curve(euclidean_true_labels, euclidean_pred_scores)
#         roc_auc_euclidean = auc(fpr_euclidean, tpr_euclidean)

#         # Compute ROC AUC for Cosine similarity
#         cosine_true_labels = np.array(cosine_labels)
#         cosine_pred_scores = np.array(cosine_scores)
#         fpr_cosine, tpr_cosine, thresholds_cosine = roc_curve(cosine_true_labels, cosine_pred_scores)
#         roc_auc_cosine = auc(fpr_cosine, tpr_cosine)
        
#         # Calculate accuracy for Euclidean distance
#         euclidean_optimal_idx = np.argmax(tpr_euclidean - fpr_euclidean) # Chọn ngưỡng tại điểm có giá trị tpr - fpr lớn nhất trên đường ROC, vì đây là nơi tối ưu hóa sự cân bằng giữa tỷ lệ phát hiện (TPR) và tỷ lệ báo động giả (FPR).
#         euclidean_optimal_threshold = thresholds_euclidean[euclidean_optimal_idx]
#         euclidean_pred_labels = (euclidean_pred_scores >= euclidean_optimal_threshold).astype(int)
#         euclidean_accuracy = accuracy_score(euclidean_true_labels, euclidean_pred_labels)

#         # Calculate accuracy for Cosine similarity
#         cosine_optimal_idx = np.argmax(tpr_cosine - fpr_cosine)
#         cosine_optimal_threshold = thresholds_cosine[cosine_optimal_idx]
#         cosine_pred_labels = (cosine_pred_scores >= cosine_optimal_threshold).astype(int)
#         cosine_accuracy = accuracy_score(cosine_true_labels, cosine_pred_labels)

#     return euclidean_accuracy, cosine_accuracy, roc_auc_euclidean, roc_auc_cosine

def compute_pairwise_scores(pair):
    id1, embedding1, id2, embedding2 = pair
    # Tính khoảng cách Euclidean
    euclidean_score = F.pairwise_distance(
        torch.tensor(embedding1), torch.tensor(embedding2)
    ).item()
    # Tính độ tương đồng Cosine
    cosine_score = F.cosine_similarity(
        torch.tensor(embedding1).unsqueeze(0), torch.tensor(embedding2).unsqueeze(0)
    ).item()
    # Gán nhãn
    euclidean_label = 0 if id1 == id2 else 1
    cosine_label = 1 if id1 == id2 else 0
    return euclidean_score, euclidean_label, cosine_score, cosine_label


def compute_roc_and_accuracy(true_labels, pred_scores):
    # Tính toán ROC AUC
    fpr, tpr, thresholds = roc_curve(true_labels, pred_scores)
    roc_auc = auc(fpr, tpr)

    # Tìm ngưỡng tối ưu (tại điểm có giá trị tpr - fpr lớn nhất)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Dự đoán nhãn dựa trên ngưỡng tối ưu
    pred_labels = (pred_scores >= optimal_threshold).astype(int)

    # Tính độ chính xác
    accuracy = accuracy_score(true_labels, pred_labels)

    return roc_auc, accuracy


def compute_roc_auc_multiprocess(
    euclidean_true_labels, euclidean_pred_scores,
    cosine_true_labels, cosine_pred_scores
):
    # Sử dụng ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        future_euclidean = executor.submit(compute_roc_and_accuracy, euclidean_true_labels, euclidean_pred_scores)
        future_cosine = executor.submit(compute_roc_and_accuracy, cosine_true_labels, cosine_pred_scores)

        # Lấy kết quả từ các tiến trình
        roc_auc_euclidean, euclidean_accuracy = future_euclidean.result()
        roc_auc_cosine, cosine_accuracy = future_cosine.result()

    return euclidean_accuracy, cosine_accuracy, roc_auc_euclidean, roc_auc_cosine


def compute_roc_auc(dataloader, model, device):
    model.eval()
    with torch.no_grad():
        embeddings_list = []
        for batch in dataloader:
            images, ids = batch
            images = images.to(device)
            embeddings = model.get_embedding(images)
            for i in range(embeddings.size(0)):
                embeddings_list.append((ids[i].item(), embeddings[i].cpu().numpy()))
        
        # Tạo danh sách các cặp embeddings
        pairs = [
            (embeddings_list[i][0], embeddings_list[i][1], embeddings_list[j][0], embeddings_list[j][1])
            for i in range(len(embeddings_list))
            for j in range(i + 1, len(embeddings_list))
        ]

        # Sử dụng ProcessPoolExecutor để xử lý song song
        with ProcessPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(compute_pairwise_scores, pairs))
        
        # Chia kết quả ra các danh sách score và label
        euclidean_scores, euclidean_labels, cosine_scores, cosine_labels = zip(*results)

        # Chuyển về dạng numpy array
        euclidean_true_labels = np.array(euclidean_labels)
        euclidean_pred_scores = np.array(euclidean_scores)
        cosine_true_labels = np.array(cosine_labels)
        cosine_pred_scores = np.array(cosine_scores)

        # Tính toán ROC AUC và accuracy như ban đầu
        euclidean_accuracy, cosine_accuracy, roc_auc_euclidean, roc_auc_cosine = compute_roc_auc_multiprocess(
            euclidean_true_labels, euclidean_pred_scores,
            cosine_true_labels, cosine_pred_scores
        )

    return euclidean_accuracy, cosine_accuracy, roc_auc_euclidean, roc_auc_cosine