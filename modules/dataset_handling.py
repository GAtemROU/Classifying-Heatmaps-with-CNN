from os.path import join
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, DataLoader
from pathlib import Path


class DatasetHandler:
    def __init__(self, root, transform, targets=None):
        if targets is None:
            targets = ['C', 'P']
        self.dataset = ImageFolder(root, transform)
        self.participants = []
        for t in targets:
            t_participants = [f.name for f in sorted(Path(join(root, t)).iterdir())
                              if f.is_dir()]
            self.participants = list(set(self.participants).union(set(t_participants)))
        self.participants = sorted(self.participants)

    def get_participants(self):
        return self.participants

    def get_ids(self, participants):
        return [i for i in range(len(self.dataset))
                if self.dataset.imgs[i][0].replace('\\', '/').split('/')[-2] in participants]

    def get_loader(self, participants, batch_size=1, shuffle=True):
        return DataLoader(Subset(self.dataset, indices=self.get_ids(participants)), batch_size, shuffle=shuffle)

