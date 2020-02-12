import torch
import numpy as np

class CategoriesSamplerOurs():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        print('max label:', max(label))
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch1 = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls*2]
            for c in classes[:self.n_cls]:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch1.append(l[pos])
            batch1 = torch.stack(batch1).t().reshape(-1)

            batch2 = []
            for c in classes[self.n_cls:]:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch2.append(l[pos])
            batch2 = torch.stack(batch2).t().reshape(-1)
            
            batch = torch.cat([batch1, batch2])
            yield batch
            # yield batch1


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

