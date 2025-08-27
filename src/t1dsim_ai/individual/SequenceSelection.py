import torch
import numpy as np

from t1dsim_ai.transversal.Functions import getInitSSFromFile

class SequenceSelection:
    def __init__(self, seq_len, device, data):

        self.seq_len = seq_len
        self.device = device

        idx_scenarios = self.get_sequences(data)  # Filter out sequences

        for scenario in idx_scenarios:
            cgm_target = self.y_fit[scenario, 0, 0].item()
            self.x_est[scenario, 0, :] = getInitSSFromFile(cgm_target)

        # Define initial states
        self.idx_scenarios = idx_scenarios

    def get_sequences(self, data):
        x_est, u_fit, y_fit, u_fit_ind = data

        idx_list = []
        idx = 0
        while idx <= y_fit.shape[1] - self.seq_len:  # Befor < instead of <=
            array = y_fit[0, idx : idx + self.seq_len, 0]

            if ~np.isnan(array[0]):

                if len(np.where(np.isnan(array))[0]) == 0:
                    idx_list.append(idx)
                else:
                    bool = []
                    for jj in np.arange(6, self.seq_len, 6):
                        if np.sum(~np.isnan(array[1 : jj + 1])) / jj < 0.7:
                            bool.append(False)
                        else:
                            bool.append(True)
                    if np.sum(bool) == 10:
                        idx_list.append(idx)
                idx = idx + self.seq_len
            else:
                idx += 1

        batch_start = np.array(idx_list, dtype=int)
        batch_idx = batch_start[:, np.newaxis] + np.arange(self.seq_len)

        self.x_est = x_est[0, batch_idx, :]
        self.u_fit = u_fit[0, batch_idx, :]
        self.y_fit = y_fit[0, batch_idx, :]
        self.u_fit_ind = u_fit_ind[0, batch_idx, :]

        return np.arange(len(idx_list))

    def get_all(self, group="All"):

        idx = self.idx_scenarios

        batch_start = np.zeros(idx.shape[0], dtype=int)
        batch_idx = batch_start[:, np.newaxis] + np.arange(
            self.seq_len
        )  # batch samples indices
        batch_idx = (
            batch_idx.T
        )  # transpose indexes to obtain batches with structure (m, q, n_x)

        # Extract batch data
        batch_x0_hidden = torch.tensor(
            self.x_est[idx, batch_start, :], dtype=torch.float32
        ).to(self.device)
        batch_u_pop = torch.tensor(self.u_fit[idx, batch_idx], dtype=torch.float32).to(
            self.device
        )
        batch_u_ind = torch.tensor(
            self.u_fit_ind[idx, batch_idx], dtype=torch.float32
        ).to(self.device)
        batch_y = torch.tensor(self.y_fit[idx, batch_idx], dtype=torch.float32).to(
            self.device
        )
        batch_x_original = torch.tensor(
            self.x_est[idx, batch_idx], dtype=torch.float32
        ).to(self.device)

        return batch_x0_hidden, batch_u_pop, batch_u_ind, batch_y, batch_x_original

