from itertools import accumulate
import numpy as np
import torch
from torch.utils.data import Dataset

from constants import ec_encoder

class ProcessedLigandPocketDataset(Dataset):
    def __init__(self, npz_path, center=True, transform=None, only_labeled=False):
        # the data dictionary consists of ligand names, receptor names, labels(EC number),
        # labels_gt(whether the label is the ground truth (or predicted)),
        # lig_coords, lig_one_hot, lig_mask,
        # pocket_coords, pocket_one_hot, pocket_mask
        # and pocket_emb which includes <SOS> and <EOS> token embedding thus has 2 larger size than pocket_mask for each pocket

        self.transform = transform

        with np.load(npz_path, allow_pickle=True) as f:
            data = {key: val for key, val in f.items()}
            
        if only_labeled:
            labeled_idc = np.where(data['labels']>=0)[0]
            new_lig_idc = np.concatenate([np.where(data['lig_mask'] == i)[0] for i in labeled_idc])
            new_pocket_idc = np.concatenate([np.where(data['pocket_mask'] == i)[0] for i in labeled_idc])
            for k, v in data.items():
                if 'lig' in k:
                    data[k] = v[new_lig_idc]
                elif 'pocket' in k:
                    data[k] = v[new_pocket_idc]
                else:
                    data[k] = v[labeled_idc]
                
        # split data based on mask
        self.data = {}
        for (k, v) in data.items():
            if k == 'names' or k == 'receptors' or k == 'labels':
                self.data[k] = v
                continue
                
            sections = np.where(np.diff(data['lig_mask']))[0] + 1 \
                if 'lig' in k \
                else np.where(np.diff(data['pocket_mask']))[0] + 1
            self.data[k] = [torch.from_numpy(x) for x in np.split(v, sections)]

            # add number of nodes for convenience
            if k == 'lig_mask':
                self.data['num_lig_atoms'] = \
                    torch.tensor([len(x) for x in self.data['lig_mask']])
            elif k == 'pocket_mask':
                self.data['num_pocket_nodes'] = \
                    torch.tensor([len(x) for x in self.data['pocket_mask']])            
        
        if center:
            for i in range(len(self.data['lig_coords'])):
                mean = (self.data['lig_coords'][i].sum(0) +
                        self.data['pocket_coords'][i].sum(0)) / \
                       (len(self.data['lig_coords'][i]) + len(self.data['pocket_coords'][i]))
                self.data['lig_coords'][i] = self.data['lig_coords'][i] - mean
                self.data['pocket_coords'][i] = self.data['pocket_coords'][i] - mean

    def __len__(self):
        return len(self.data['names'])

    def __getitem__(self, idx):
        data = {key: val[idx] for key, val in self.data.items()}
        if self.transform is not None:
            data = self.transform(data)
        return data

    @staticmethod
    def collate_fn(batch):
        out = {}
        for prop in batch[0].keys():

            if prop == 'names' or prop == 'receptors' or prop == 'labels':
                out[prop] = [x[prop] for x in batch]
            elif prop == 'num_lig_atoms' or prop == 'num_pocket_nodes' \
                    or prop == 'num_virtual_atoms':
                out[prop] = torch.tensor([x[prop] for x in batch])
            elif 'mask' in prop:
                # make sure indices in batch start at zero (needed for
                # torch_scatter)
                out[prop] = torch.cat([i * torch.ones(len(x[prop]))
                                       for i, x in enumerate(batch)], dim=0)
            else:
                out[prop] = torch.cat([x[prop] for x in batch], dim=0)

        return out
