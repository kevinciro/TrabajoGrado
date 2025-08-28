from t1dsim_ai.utils.preprocess import scale_single_state
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

"""
    This class implements prediction/simulation methods for the SS models structure

    Attributes
    ----------
    ss_pop_model: nn.Module
                The population-level neural state space models to be fitted
    ss_ind_model: nn.Module
                The individual-level neural state space models to be fitted
    ts: float
        models sampling time
"""

class ForwardEulerSimulator(nn.Module):

    def __init__(self, ss_pop_model, ss_ind_model, path_scaler, ts=1.0, showQ1=False):
        super(ForwardEulerSimulator, self).__init__()
        self.ss_pop_model = ss_pop_model
        self.ss_ind_model = ss_ind_model

        self.ts = ts
        self.cgm_min = scale_single_state(40, "Q1", path_scaler)
        self.cgm_max = scale_single_state(400, "Q1", path_scaler)

        self.showQ1 = showQ1

    def adjust_cgm(self, x):
        x[x > self.cgm_max] = self.cgm_max
        x[x < self.cgm_min] = self.cgm_min

        return x

    def forward(
        self, x0_batch: torch.Tensor, u_batch: torch.Tensor, u_batch_ind, is_pers=True
    ) -> torch.Tensor:
        """Multi-step simulation over (mini)batches

        Parameters
        ----------
        x0_batch: Tensor. Size: (q, n_x)
             Initial state for each subsequence in the minibatch

        u_batch: Tensor. Size: (m, q, n_u)
            Input sequence for each subsequence in the minibatch

        Returns
        -------
        Tensor. Size: (m, q, n_x)
            Simulated state for all subsequences in the minibatch

        """

        # X_sim_list: List[torch.Tensor] = []
        X_sim_list: [torch.Tensor] = [] # type: ignore

        x_step = x0_batch

        if (self.showQ1):
            dq1_pop_list = []
            dq1_ind_list = []
            suma_dq_list = []

        for step in range(u_batch.shape[0]):
            u_step = u_batch[step]

            if is_pers:
                u_ind_step = u_batch_ind[step]

            X_sim_list += [x_step]

            dx_pop = self.ss_pop_model(x_step, u_step)
            
            if (self.showQ1):
                dq1_pop_list.append(dx_pop[:, 0].item())
            

            dx = dx_pop
            if is_pers:
                dx_ind = self.ss_ind_model(x_step, u_step, u_ind_step)
                #print(dx_ind)
                dx[:, 0] += dx_ind[:, 0]
                dx[:, 1] += dx_ind[:, 1]
                dx[:, 2] += dx_ind[:, 2]
                dx[:, 3] += dx_ind[:, 3]
                dx[:, 4] += dx_ind[:, 4]
                dx[:, 5] += dx_ind[:, 5]
                dx[:, 6] += dx_ind[:, 6]
                dx[:, 7] += dx_ind[:, 7]
                dx[:, 8] += dx_ind[:, 8]
                dx[:, 9] += dx_ind[:, 9]

                if (self.showQ1):
                    dq1_ind_list.append(dx_ind.item())
                    suma_dq_list.append(dx[:, 0].item())

            x_step = x_step + self.ts * dx

            if (len(x_step.shape)) == 1:
                x_step[0] = self.adjust_cgm(x_step[0])

            else:
                x_step[:, 0] = self.adjust_cgm(x_step[:, 0])

        X_sim = torch.stack(X_sim_list, 0)
        
        #if (dq1_ind_list and self.showQ1):
        #    plt.title("dQ1")
        #    plt.plot(dq1_ind_list, "r", label="Ind")
        #    plt.plot(dq1_pop_list, "g", label="Pop")
        #    plt.plot(suma_dq_list, "b", label= "Suma")
        #    plt.legend()
        #    plt.grid()
        #    plt.show()
    
        return X_sim