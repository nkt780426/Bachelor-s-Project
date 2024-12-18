import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from ...utils.roc_auc import compute_roc_auc
from ...utils.metrics import ProgressMeter
from ...utils.MultiMetricEarlyStopping import MultiMetricEarlyStopping
from ...utils.ModelCheckPoint import ModelCheckpoint
import os


def fit(
    conf:dict,
    start_epoch:int,
    model: Module,
    triplet_train_loader: DataLoader, 
    triplet_test_loader: DataLoader, 
    criterion: Module,
    optimizer: Optimizer, 
    scheduler, 
    epochs: int, 
    device: str, 
    roc_train_loader: DataLoader, 
    roc_test_loader: DataLoader,
    early_max_stopping: MultiMetricEarlyStopping,
    early_min_stopping: MultiMetricEarlyStopping,
    model_checkpoint: ModelCheckpoint
):
    log_dir = os.path.abspath('checkpoint/triplet/'+ conf['type'] + '/logs')
        
    writer = SummaryWriter(log_dir=log_dir)
    
    for epoch in range(start_epoch, epochs):
        scheduler.step()

        # train + eval
        train_loss = train_epoch(triplet_train_loader, model, criterion, optimizer, device)
        test_loss= test_epoch(triplet_test_loader, model, criterion, device)
        train_euclidean_accuracy, train_cosine_accuracy, train_roc_auc_euclidean, train_roc_auc_cosine = compute_roc_auc(roc_train_loader, model, device)
        test_euclidean_accuracy, test_cosine_accuracy, test_roc_auc_euclidean, test_roc_auc_cosine = compute_roc_auc(roc_test_loader, model, device)

        # Log metric
        writer.add_scalars(main_tag="Loss", tag_scalar_dict = {'train': train_loss, 'test': test_loss}, global_step = epoch+1)
        writer.add_scalars(main_tag="Cosine_AUC", tag_scalar_dict = {'train': train_roc_auc_cosine, 'test': test_roc_auc_cosine}, global_step=epoch+1)
        writer.add_scalars(main_tag="Cosine_ACC", tag_scalar_dict = {'train': train_cosine_accuracy, 'test': test_cosine_accuracy}, global_step=epoch+1)
        writer.add_scalars(main_tag="Euclidean_AUC", tag_scalar_dict = {'train': train_roc_auc_euclidean, 'test': test_roc_auc_euclidean}, global_step=epoch+1)
        writer.add_scalars(main_tag='Euclidean_ACC', tag_scalar_dict = {'train': train_euclidean_accuracy, 'test': test_euclidean_accuracy}, global_step=epoch+1)
        
        train_metrics = [
            f"loss: {train_loss:.4f}",
            f"cos_auc: {train_roc_auc_cosine:.4f}",
            f"cos_acc: {train_cosine_accuracy:.4f}",
            f"eu_auc: {train_roc_auc_euclidean:.4f}",
            f"eu_acc: {train_euclidean_accuracy:.4f}"
        ]
        
        val_metrics = [
            f"loss: {test_loss:.4f}",
            f"cos_auc: {test_roc_auc_cosine:.4f}",
            f"cos_acc: {test_cosine_accuracy:.4f}",
            f"eu_auc: {test_roc_auc_euclidean:.4f}",
            f"eu_acc: {test_euclidean_accuracy:.4f}"
        ]
        
        process = ProgressMeter(
            train_meters=train_metrics,
            test_metrics=val_metrics,
            prefix=f"Epoch {epoch + 1}:"
        )
        
        process.display()
        
        model_checkpoint(model, optimizer, epoch + 1)
        early_max_stopping([test_loss], model, epoch+1)
        early_min_stopping([test_roc_auc_cosine, test_cosine_accuracy, test_roc_auc_euclidean, test_euclidean_accuracy], model, epoch+1)
        scheduler.step(epoch+1)
        
        # if early_min_stopping.early_stop and early_max_stopping.early_stop:
        #     break
        
        
def train_epoch(triplet_train_loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for X in enumerate(triplet_train_loader):
        X = X.to(device)

        anchors, positives, negatives = model(X)

        loss = criterion(anchors, positives, negatives)
        
        total_loss += loss.item()
        
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

    total_loss /= len(triplet_train_loader)
    
    return total_loss


def test_epoch(triplet_test_loader, model, criterion, device):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        for X in triplet_test_loader:
            X = X.to(device)

            anchors, positives, negatives = model(X)

            loss = criterion(anchors, positives, negatives)

            test_loss += loss.item()

        test_loss /= len(triplet_test_loader)
        
    return test_loss