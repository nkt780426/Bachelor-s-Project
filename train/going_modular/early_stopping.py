import torch

class EarlyStopping:
    def __init__(
        self, 
        monitor='val_loss', 
        min_delta=0, 
        patience=0, 
        verbose=0, 
        mode='auto', 
        baseline=None, 
        restore_best_weights=False, 
        start_from_epoch=0, 
        save_path=None
    ):
        """
        PyTorch EarlyStopping giống với tf.keras.callbacks.EarlyStopping
        
        Args:
            monitor (str): Metric để theo dõi, ví dụ 'val_loss'.
            min_delta (float): Sai số nhỏ nhất để tính là có cải thiện.
            patience (int): Số epoch không cải thiện trước khi dừng.
            verbose (int): Hiển thị log nếu > 0.
            mode (str): 'min', 'max' hoặc 'auto'.
            baseline (float): Giá trị ban đầu yêu cầu metric phải vượt qua.
            restore_best_weights (bool): Khôi phục trọng số tốt nhất khi dừng.
            start_from_epoch (int): Bắt đầu kiểm tra early stopping từ epoch này.
            save_path (str): Đường dẫn để lưu trọng số tốt nhất (ghi đè).
        """
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        self.start_from_epoch = start_from_epoch
        self.save_path = save_path

        # Cấu hình chế độ
        if mode not in ['min', 'max', 'auto']:
            raise ValueError(f"Invalid mode '{mode}', must be 'min', 'max', or 'auto'.")
        if mode == 'min' or (mode == 'auto' and 'loss' in monitor):
            self.monitor_op = lambda current, best: current < best - min_delta
            self.best_value = float('inf')
        elif mode == 'max' or (mode == 'auto' and 'acc' in monitor):
            self.monitor_op = lambda current, best: current > best + min_delta
            self.best_value = -float('inf')

        self.counter = 0
        self.early_stop = False
        self.best_epoch = -1

    def __call__(self, current_value, model, epoch):
        # Không kiểm tra trước start_from_epoch
        if epoch < self.start_from_epoch:
            return
        
        # Kiểm tra baseline
        if self.baseline is not None and epoch == 0:
            if not self.monitor_op(current_value, self.baseline):
                if self.verbose:
                    print(f"Metric '{self.monitor}' did not meet baseline {self.baseline}. Early stopping enabled.")
                self.early_stop = True
                return

        # Kiểm tra cải thiện
        if self.monitor_op(current_value, self.best_value):
            self.best_value = current_value
            self.best_epoch = epoch
            self.counter = 0

            # Lưu trọng số tốt nhất vào file
            if self.save_path:
                torch.save(model.state_dict(), self.save_path)
                if self.verbose:
                    print(f"Saved best model weights at epoch {epoch} to '{self.save_path}'")
        else:
            self.counter += 1
            if self.verbose:
                print(f"Epoch {epoch}: EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"Early stopping triggered at epoch {epoch}.")
                self.early_stop = True

                # Khôi phục trọng số từ file
                if self.restore_best_weights and self.save_path:
                    model.load_state_dict(torch.load(self.save_path))
                    if self.verbose:
                        print(f"Restored best model weights from '{self.save_path}'")