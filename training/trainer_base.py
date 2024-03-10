import os
from os.path import join
import torch
import datetime



class Trainer:

    def __init__(self, model, train_loader, eval_loader, save_path, loss, optimizer=None):
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model
        self.train_loader = train_loader
        self.val_loader = eval_loader
        self.save_path = save_path
        self.loss = loss
        self.optimizer = optimizer
        self.epoch = 0
        self.max_val_acc = 0.
        self.best_model_dict = None
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run_epoch(self):

        self.model.train()
        epoch_loss = 0.
        lr = self.optimizer.param_groups[0]['lr']
        for batch_input, batch_target in self.train_loader:
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
                batch_target = batch_target.cuda()
            self.optimizer.zero_grad()
            batch_out = self.model.forward(batch_input)
            loss = self.loss(batch_out, batch_target)
            epoch_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        self.epoch += 1
        avg_loss = epoch_loss / len(self.train_loader)
        print('{} --- Epoch [{}], Loss: {:.4f}'.format(
            datetime.datetime.now().strftime("%d.%m %H:%M:%S"), self.epoch, avg_loss))
        self.evaluate()

    def save_best_model(self):
        if self.best_model_dict is None:
            print("No best model found")
        else:
            print("Saving best model to   {}".format(join(self.save_path, "best_model.pkl")))
            torch.save(self.best_model_dict, join(self.save_path, "best_model.pkl"))

    def save_cur_model(self):
        torch.save(self.model.state_dict(), join(self.save_path, "model_epoch_{}.pkl".format(self.epoch)))

    def print_best_validation_acc(self):
        print("Best validation accuracy: {:.4f}".format(self.max_val_acc*100))

    @torch.no_grad()
    def evaluate(self):
        """Evaluates model on validation loader.
        If better accuracy is achieved, the model is stored to self.best_model"""
        self.model.eval()
        valid_accuracy = 0.0
        valid_samples = 0
        for batch_input, batch_target in self.val_loader:
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
                batch_target = batch_target.cuda()
            batch_out = self.model.forward(batch_input)
            _, predicted = torch.max(batch_out, 1)
            valid_accuracy += (predicted == batch_target).sum().item()
            valid_samples += batch_target.size(0)
        valid_accuracy /= valid_samples
        if valid_accuracy > self.max_val_acc:
            self.max_val_acc = valid_accuracy
            self.best_model_dict = self.model.state_dict()
            print('New best validation accuracy: {:.4f}'.format(valid_accuracy*100))
            self.save_cur_model()


