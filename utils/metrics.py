import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score


def compute_confusion_matrices(model, dataset, participants):
    cfs = {}
    model.eval()
    with torch.no_grad():
        for p in participants:
            y_true = []
            y_pred = []
            loader = dataset.get_loader([p])
            for img, lab in loader:
                y_true.append(lab.detach().numpy())
                _, preds = torch.max(model.forward(img), 1)
                y_pred.append(preds.detach().numpy())
            cfs[p] = confusion_matrix(np.array(y_true).flatten(), np.array(y_pred).flatten())
    return cfs


def compute_accuracy(model, dataset, participants):
    model.eval()
    with torch.no_grad():
        loader = dataset.get_loader(participants)
        test_accuracy = 0.0
        test_samples = 0
        for img, lab in loader:
            out = model.forward(img)
            _, preds = torch.max(out, 1)
            test_accuracy += (preds == lab).sum().item()
            test_samples += lab.size(0)
    test_accuracy /= test_samples
    return test_accuracy


def compute_f1score(model, dataset, participants):
    loader = dataset.get_loader(participants)
    all_true_labels = []
    all_pred_labels = []

    with torch.no_grad():
        for img, lab in loader:
            outputs = model(img)
            _, preds = torch.max(outputs, 1)
            all_true_labels.extend(lab.detach().numpy())
            all_pred_labels.extend(preds.detach().numpy())
    f1 = f1_score(all_true_labels, all_pred_labels, average='weighted')
    return f1
