import os
from datetime import datetime


class MyLogger:

    def __init__(self, log_dir, name=None):
        if name is None:
            self.name = datetime.now().strftime("log_%d.%m_%H%M")
        else:
            self.name = name
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.f1_list = []

    def log_epoch(self, epoch, loss, val_acc):
        with open(os.path.join(self.log_dir, self.name), "a") as f:
            f.write(f"Epoch [{epoch}], Loss: {loss:.4f}, Val acc: {val_acc:.2f}\n")

    def log_new_fold(self, fold, add_info=None):
        with open(os.path.join(self.log_dir, self.name), "a") as f:
            f.write(f"Fold {fold}\n")
            if add_info is not None:
                f.write(f"{add_info}\n")

    def log_test_metrics(self, acc, f1, cfs):
        self.f1_list.append(f1)
        with open(os.path.join(self.log_dir, self.name), "a") as f:
            f.write(f"Test accuracy: {acc:.4f}, F1 score: {f1:.4f}\n")
            f.write(f"Confusion matrices:\n{cfs}\n")

    def log_f1_history(self, file_name='f1_history', save_avg=True):
        with open(os.path.join(self.log_dir, file_name), "a") as f:
            f.write(self.f1_list.__str__() + '\n')
            if save_avg:
                f.write(f'AVG: {sum(self.f1_list) / len(self.f1_list)}\n')
