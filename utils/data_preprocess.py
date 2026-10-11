import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Data Load
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)  
        if df.empty: 
            print(f"Empty file: {file_path}")
            return [], []
        data = df.iloc[:, :-1].squeeze(axis=1) 
        labels = df.iloc[:, -1] 
       
        return data, labels
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return [], []


def load_and_split_data(file_path):

    # Normalize path separators
    normalized_path = file_path.replace('\\', '/')

    try:
        data, labels = load_data(file_path)
        if len(data) == 0 or len(labels) == 0:  
            print(f"Empty data or labels for file: {file_path}")
            return [], [], [], []

    
        file_name_parts = normalized_path.split('/')[-1].split('_')
        train_end_part = [file_name_parts[i + 1] for i, part in enumerate(file_name_parts) if part == 'tr']
       

        if train_end_part and len(train_end_part[0]) > 1:
            train_end = int(train_end_part[0])
            train_data = data[:train_end]
            train_labels = labels[:train_end]
            test_data = data[train_end:]
            test_labels = labels[train_end:]
        else:
            print(f"Invalid file format or missing 'tr_' in: {file_path}")
            return [], [], [], []

        return train_data, train_labels, test_data, test_labels
    except IsADirectoryError:
        print(f"Skipped directory: {file_path}")
        return None, None, None, None
    except ValueError as e:
        print(f"Error processing file {file_path}: {e}")
        return None, None, None, None


def preprocess_to_patches(data, patch_size, stride):
    """Zero-copy sliding window via torch.unfold.  Returns a strided view."""
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    t = torch.from_numpy(data)                          # share memory
    t = t.unfold(0, patch_size, stride)                 # (N_pts, C) → (N_patches, C, L)
    return t


class _tsdataset(Dataset):
    """Lazy-slice Dataset: stores a ref to source data + start positions.
    Slices are materialised in __getitem__, NOT in __init__.
    """
    def __init__(self, source_data, start_indices, patch_len=64):
        self.source = np.asarray(source_data, dtype=np.float32)
        self.starts = np.asarray(start_indices, dtype=np.int64)
        self.L = patch_len

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        patch = self.source[start:start + self.L]        # numpy view  (L,) or (L, C)
        t = torch.as_tensor(patch, dtype=torch.float32)
        if t.ndim == 1:                                  # (L,) → (1, L)
            t = t.unsqueeze(0).contiguous()
        else:                                            # (L, C) → (C, L)
            t = t.T.contiguous()
        return t, torch.tensor([start], dtype=torch.long)


class PatchCreator:
    def __init__(self, L, s, random_seed=None):
        self.L = L
        self.s = s
        if random_seed is not None:
            torch.manual_seed(random_seed)

    def create_dataloaders(self, train_data, test_data, test_labels, batch_size=512):
        # only compute start indices — NO patch materialisation
        train_starts = np.arange(0, len(train_data) - self.L + 1, self.s, dtype=np.int64)
        test_starts  = np.arange(0, len(test_data)  - self.L + 1, self.s, dtype=np.int64)

        train_loader = DataLoader(
            _tsdataset(train_data, train_starts, self.L),
            batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(
            _tsdataset(test_data, test_starts, self.L),
            batch_size=batch_size, shuffle=False)

        return train_loader, test_loader, test_labels



# cited from https://github.com/TheDatumOrg/TSB-AD/blob/main/TSB_AD/utils/slidingWindows.py
from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema

# determine sliding window (period) based on ACF
def find_length_rank(data, rank=1):
    data = data.squeeze()
    if len(data.shape)>1: return 100 #0->100
    if rank==0: return 1
    data = data[:min(20000, len(data))]
    
    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]
    
    # plot_acf(data, lags=400, fft=True)
    # plt.xlabel('Lags')
    # plt.ylabel('Autocorrelation')
    # plt.title('Autocorrelation Function (ACF)')
    # plt.savefig('/data/liuqinghua/code/ts/TSAD-AutoML/AutoAD_Solution/candidate_pool/cd_diagram/ts_acf.png')

    local_max = argrelextrema(auto_corr, np.greater)[0]

    # print('auto_corr: ', auto_corr)
    # print('local_max: ', local_max)

    try:
        # max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        sorted_local_max = np.argsort([auto_corr[lcm] for lcm in local_max])[::-1]    # Ascending order
        max_local_max = sorted_local_max[0]     # Default
        if rank == 1: max_local_max = sorted_local_max[0]
        if rank == 2: 
            for i in sorted_local_max[1:]: 
                if i > sorted_local_max[0]: 
                    max_local_max = i 
                    break
        if rank == 3:
            for i in sorted_local_max[1:]: 
                if i > sorted_local_max[0]: 
                    id_tmp = i
                    break
            for i in sorted_local_max[id_tmp:]:
                if i > sorted_local_max[id_tmp]: 
                    max_local_max = i           
                    break
        # print('sorted_local_max: ', sorted_local_max)
        # print('max_local_max: ', max_local_max)
        if local_max[max_local_max]<3 or local_max[max_local_max]>300:
            return 125
        return local_max[max_local_max]+base
    except:
        return 125
    

# determine sliding window (period) based on ACF, Original version
def find_length(data):
    if len(data.shape)>1:
        return 0
    data = data[:min(20000, len(data))]
    
    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]
    
    
    local_max = argrelextrema(auto_corr, np.greater)[0]
    try:
        max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        if local_max[max_local_max]<3 or local_max[max_local_max]>300:
            return 125
        return local_max[max_local_max]+base
    except:
        return 125
