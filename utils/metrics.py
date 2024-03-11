import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def compute_confusion_matrices(model, dataset, participants):
    cfs = {}
    model.eval()
    for p in participants:
        y_true = []
        y_pred = []
        loader = dataset.get_loader([p])
        for img, lab in loader:
            y_true.append(lab.detach().numpy())
            _, out = torch.max(model.forward(img), 1)
            out = out.detach().numpy()
            y_pred.append(out)
        cfs[p] = confusion_matrix(np.array(y_true).flatten(), np.array(y_pred).flatten())
    return cfs


def compute_accuracy(model, dataset, participants):
    model.eval()
    loader = dataset.get_loader(participants)
    test_accuracy = 0.0
    test_samples = 0
    for batch_input, batch_target in loader:
        batch_out = model.forward(batch_input)
        _, predicted = torch.max(batch_out, 1)
        test_accuracy += (predicted == batch_target).sum().item()
        test_samples += batch_target.size(0)
    test_accuracy /= test_samples
    return test_accuracy
