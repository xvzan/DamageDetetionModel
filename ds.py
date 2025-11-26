import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io
import os
import random

DATA_FOLDER = "original_dataset"
DATA_FILES = ["EC1.mat", "EC2.mat", "EC3.mat"]
KVS = {
    0: ["FC0_A", "FC0_R"],
    1: ["FC1_A", "FC1_R"],
    2: ["FC2_A", "FC2_R"],
    3: ["FC3_A", "FC3_R"],
}
SINGLE_LEN = 512


class DM(Dataset):
    def __init__(self):
        self.datas = []
        ecs = []
        for file in DATA_FILES:
            full_path = os.path.join(DATA_FOLDER, file)
            ecs.append(scipy.io.loadmat(full_path))

        self.index_map = []

        end = 0
        for ec in ecs:
            for k in KVS:
                # ced = np.concatenate([ec[KVS[k][0]], ec[KVS[k][1]]], axis=1)
                # ced = ec[KVS[k][0]]
                ced = ec[KVS[k][0]][:, [0, 2]]
                num = ced.shape[0] // (SINGLE_LEN * 2)
                end += num
                self.index_map.append([end, k])
                self.datas.append(ced)

    def __len__(self):
        return self.index_map[-1][0]

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.__len__():
            raise IndexError(f"索引 {idx} 超出范围，总片段数为 {self.__len__()}")

        left, right = 0, len(self.index_map) - 1
        while left < right:
            mid = (left + right) // 2
            if idx < self.index_map[mid][0]:
                right = mid
            else:
                left = mid + 1

        ec_idx, key = left, self.index_map[left][1]
        start_idx = idx
        if ec_idx > 0:
            start_idx = idx - self.index_map[left - 1][0]
        start_idx = start_idx * SINGLE_LEN * 2 + random.randint(0, SINGLE_LEN)
        ec = self.datas[ec_idx]

        chunk = ec[start_idx : start_idx + SINGLE_LEN, :]
        chunk = torch.tensor(chunk, dtype=torch.float32).permute(-1, -2)
        key = torch.tensor(key, dtype=torch.long)
        return chunk, key
