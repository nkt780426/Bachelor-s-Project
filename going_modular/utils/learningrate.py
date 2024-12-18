import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

class NormalMapLR:
    def __init__(self, optimizer, fixed_epoch, fixed_lr, start_warm_lr):
        self.optimizer = optimizer
        self.fixed_epoch = fixed_epoch
        self.fixed_lr = fixed_lr
        self.start_warm_lr = start_warm_lr
        self.warmup = CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=1, eta_min=1e-6)

    def step(self, epoch):
        # Giai đoạn learning rate cố định
        if epoch < self.fixed_epoch:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.fixed_lr
        # Giai đoạn chuyển đổi sang warmup
        elif epoch == self.fixed_epoch:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.start_warm_lr
            self.warmup.step(0)  # Khởi động warmup ở epoch đầu tiên
        # Giai đoạn warmup
        else:
            self.warmup.step(epoch - self.fixed_epoch)  # Điều chỉnh epoch tương đối
