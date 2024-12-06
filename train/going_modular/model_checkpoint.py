import torch

class ModelCheckpoint:
    def __init__(self, filepath, monitor='val_loss', verbose=0):
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose

    def __call__(self, model, epoch):
        # Lưu model vào file với tên cố định mỗi epoch
        torch.save(model.state_dict(), self.filepath)
        if self.verbose > 0:
            print(f"Epoch {epoch}: Saving model to {self.filepath}")
