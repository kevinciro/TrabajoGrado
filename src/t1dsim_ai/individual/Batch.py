import torch
import numpy as np
from librosa.util import frame
import pandas as pd

import t1dsim_ai.transversal.WeightClipper as WeightClipper
from t1dsim_ai.transversal.Functions import getInitSSFromFile

class Batch:
    def __init__(self, batch_size, seq_len, overlap, device, data):

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.overlap = int((1 - overlap) * self.seq_len)
        self.device = device

        # Reshape
        x_est, u_fit, y_fit, u_fit_ind = data
        self.x_est = np.array(
            [
                frame(
                    x_est[i], frame_length=self.seq_len, hop_length=self.overlap, axis=0
                )
                for i in range(len(x_est))
            ]
        ).reshape(-1, seq_len, x_est.shape[2])
        self.u_fit = np.array(
            [
                frame(
                    u_fit[i], frame_length=self.seq_len, hop_length=self.overlap, axis=0
                )
                for i in range(len(u_fit))
            ]
        ).reshape(-1, seq_len, u_fit.shape[2])
        self.y_fit = np.array(
            [
                frame(
                    y_fit[i], frame_length=self.seq_len, hop_length=self.overlap, axis=0
                )
                for i in range(len(y_fit))
            ]
        ).reshape(-1, seq_len, y_fit.shape[2])
        self.u_fit_ind = np.array(
            [
                frame(
                    u_fit_ind[i],
                    frame_length=self.seq_len,
                    hop_length=self.overlap,
                    axis=0,
                )
                for i in range(len(u_fit_ind))
            ]
        ).reshape(-1, seq_len, u_fit_ind.shape[2])

        idx_scenarios = self.filter_seq()  # Filter out sequences

        # Define initial states
        for scenario in idx_scenarios:
            cgm_target = self.y_fit[scenario, 0, 0].item()
            self.x_est[scenario, 0, :] = getInitSSFromFile(cgm_target)

        self.idx_scenarios_temp = idx_scenarios
        self.idx_scenarios = idx_scenarios

        self.num_scenarios = len(self.idx_scenarios)

        self.n_iter_per_epoch = int(
            self.num_scenarios / self.batch_size
        )  # Number of iterations per epoch
        self.update_batch_idx()
        self.epoch = 1

        print("Number of iteration per epoch:", self.n_iter_per_epoch)
        print("Number of scenarios:", self.num_scenarios)

    def get_all(self, group):

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

    def get_batch(self, count=True):

        self.batch_scenarios_idx = self.batch_scenarios_idx.astype(int)
        batch_start = np.zeros(self.batch_scenarios_idx.shape[0], dtype=int)
        batch_idx = batch_start[:, np.newaxis] + np.arange(
            self.seq_len
        )  # batch samples indices
        batch_idx = (
            batch_idx.T
        )  # transpose indexes to obtain batches with structure (m, q, n_x)

        # Extract batch data

        batch_x0_hidden = torch.tensor(
            self.x_est[self.batch_scenarios_idx, batch_start, :], dtype=torch.float32
        ).to(self.device)
        batch_u_pop = torch.tensor(
            self.u_fit[self.batch_scenarios_idx, batch_idx], dtype=torch.float32
        ).to(self.device)
        batch_u_ind = torch.tensor(
            self.u_fit_ind[self.batch_scenarios_idx, batch_idx], dtype=torch.float32
        ).to(self.device)
        batch_y = torch.tensor(
            self.y_fit[self.batch_scenarios_idx, batch_idx], dtype=torch.float32
        ).to(self.device)
        batch_x_original = torch.tensor(
            self.x_est[self.batch_scenarios_idx, batch_idx], dtype=torch.float32
        ).to(self.device)

        if count:
            self.update_batch_idx()

        return batch_x0_hidden, batch_u_pop, batch_u_ind, batch_y, batch_x_original

    def update_batch_idx(self):
        if True:
            if len(self.idx_scenarios_temp) < self.batch_size:
                batch_scenarios_idx1 = self.idx_scenarios_temp
                self.idx_scenarios_temp = self.idx_scenarios

                batch_scenarios_idx2 = np.random.choice(
                    self.idx_scenarios_temp,
                    self.batch_size - len(batch_scenarios_idx1),
                    replace=False,
                )
                self.batch_scenarios_idx = np.concatenate(
                    [batch_scenarios_idx1, batch_scenarios_idx2]
                )
                self.idx_scenarios_temp = np.array(
                    list(set(self.idx_scenarios_temp).difference(batch_scenarios_idx2))
                )

                self.epoch += 1

            else:
                self.batch_scenarios_idx = np.random.choice(
                    self.idx_scenarios_temp, self.batch_size, replace=False
                )
                self.idx_scenarios_temp = np.array(
                    list(
                        set(self.idx_scenarios_temp).difference(
                            self.batch_scenarios_idx
                        )
                    )
                )

    def filter_seq(self):

        dfScenario = pd.DataFrame(
            columns=["isCompleteCGM", "isValid"], index=np.arange(self.x_est.shape[0])
        )
        for scenario in range(self.x_est.shape[0]):
            dfScenario.loc[scenario, "isCompleteCGM"] = np.isnan(
                self.y_fit[scenario, :, 0]
            ).any()

        dfScenario["isValid"] = np.logical_not(dfScenario["isCompleteCGM"])

        return dfScenario[dfScenario.isValid].index.astype(int)
