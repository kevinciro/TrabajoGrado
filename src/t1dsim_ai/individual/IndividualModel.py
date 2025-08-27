from t1dsim_ai.individual.Batch import Batch
from t1dsim_ai.individual.ForwardEulerSimulator import ForwardEulerSimulator
from t1dsim_ai.individual.Q1_Ind import CGMIndividual
from t1dsim_ai.individual.Q2_Ind import CGMIndividual_Q2
from t1dsim_ai.utils.preprocess import scaler as scaler_pop
from t1dsim_ai.utils.preprocess import (
    scale_single_state,
    scale_inverse_Q1,
)
from t1dsim_ai.population.CGMOHSUSimStateSpaceModel_V2 import CGMOHSUSimStateSpaceModel_V2
from t1dsim_ai.transversal.options import (
    n_neurons_pop,
    states,
    states_nobs,
    inputs,
    input_ind,
    idx_robust,
)

import torch
import torch.optim as optim
from pathlib import Path
import os
import numpy as np
from pickle import load, dump


class IndividualModel:
    def __init__(self, subjectID, df_subj, pathModel, device="cpu"):

        self.subjectID = subjectID
        self.df_subj = df_subj
        self.device = device

        self.popModel = (
            Path(__file__).parent.parent
            / "models/PopulationModel/population_model_05022024_epoch_15.pt"
        )
        self.popModelFolder = str(Path(__file__).parent.parent) + "/models/PopulationModel/"

        self.pathModelFolder = pathModel
        self.pathModel = pathModel + subjectID
        
        self.seq_len = 61
        
        #self.seq_len = 60
        #self.seq_len = 144

        self.split_train_test(states, inputs, input_ind)

        self.LIM_INFERIOR = scale_single_state(70, "Q1", self.popModelFolder)
        self.LIM_SUPERIOR = scale_single_state(250, "Q1", self.popModelFolder)

    def split_train_test(self, states, inputs_pop, input_ind):

        self.df_subj[states_nobs] = 0
        df_subj_train = self.df_subj.loc[self.df_subj.is_train]
        df_subj_test = self.df_subj.loc[~self.df_subj.is_train]

        sim_time_train = len(df_subj_train)
        sim_time_test = len(df_subj_test)

        x_est_train = np.array(df_subj_train[states].values).astype(np.float32)
        u_pop_train = np.array(df_subj_train[inputs_pop].values).astype(np.float32)
        u_ind_train = np.array(df_subj_train[input_ind].values).astype(np.float32)

        x_est_test = np.array(df_subj_test[states].values).astype(np.float32)
        u_pop_test = np.array(df_subj_test[inputs_pop].values).astype(np.float32)
        u_ind_test = np.array(df_subj_test[input_ind].values).astype(np.float32)

        # Scale states and inputs from the population model
        x_est_train, u_pop_train = scaler_pop(
            x_est_train, u_pop_train, self.popModelFolder, False
        )
        x_est_test, u_pop_test = scaler_pop(
            x_est_test, u_pop_test, self.popModelFolder, False
        )

        # Scale new inputs
        self.scaler_featsRobust = load(
            open(Path(__file__).parent.parent / "models/scaler_robust.pkl", "rb")
        )
        u_ind_train[:, idx_robust] = self.scaler_featsRobust.fit_transform(
            u_ind_train[:, idx_robust]
        )
        u_ind_test[:, idx_robust] = self.scaler_featsRobust.transform(
            u_ind_test[:, idx_robust]
        )

        self.y_id_train = np.copy(x_est_train[:, 0]).reshape(-1, sim_time_train, 1)
        self.x_est_train = x_est_train.reshape(-1, sim_time_train, len(states))
        self.u_pop_train = u_pop_train.reshape(-1, sim_time_train, len(inputs_pop))
        self.u_ind_train = u_ind_train.reshape(-1, sim_time_train, len(input_ind))

        self.y_id_test = np.copy(x_est_test[:, 0]).reshape(-1, sim_time_test, 1)
        self.x_est_test = x_est_test.reshape(-1, sim_time_test, len(states))
        self.u_pop_test = u_pop_test.reshape(-1, sim_time_test, len(inputs_pop))
        self.u_ind_test = u_ind_test.reshape(-1, sim_time_test, len(input_ind))

        print("Number of training points:", self.x_est_train.shape[1])
        print("Number of testing points:", self.x_est_test.shape[1])

    def setup_nn(
        self,
        hidden_compartments,
        lr,
        batch_size,
        n_epochs,
        overlap=0.9,
        seq_len = 61,
        ts=5,
        weight_decay=1e-5,
    ):

        self.seq_len = seq_len
        # Batch extraction class
        self.batch = Batch(
            batch_size,
            self.seq_len,
            overlap,
            self.device,
            [self.x_est_train, self.u_pop_train, self.y_id_train, self.u_ind_train],
        )

        # Setup neural model structure
        self.individual_model = CGMIndividual(hidden_compartments=hidden_compartments)
        #self.individual_model = CGMIndividual_Q2(hidden_compartments=hidden_compartments)
        self.individual_model.to(self.device)

        # Load population model
        self.ss_pop_model = CGMOHSUSimStateSpaceModel_V2(n_feat=n_neurons_pop)
        self.ss_pop_model.to(self.device)
        self.ss_pop_model.load_state_dict(torch.load(self.popModel))

        for name, param in self.ss_pop_model.named_parameters():
            param.requires_grad = False

        # Simulator
        self.nn_solution = ForwardEulerSimulator(
            self.ss_pop_model, self.individual_model, self.popModelFolder, ts=ts
        )

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.individual_model.parameters(), lr=lr, weight_decay=weight_decay
        )
        # lambda1 = lambda epoch: np.exp(-0.1) ** epoch
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda1)

        self.n_epochs = n_epochs + 1
        self.n_iter_max = self.n_epochs * self.batch.n_iter_per_epoch
        self.curr_epoch = 1

        self.best_loss = float("inf")
        self.best_model = None
        self.epochs_without_improvement = 0
        self.max_epochs_without_improvement = 150

    def fit(self, save_model):

        # Training loop
        LOSS = []
        LOSS_TRAIN = []

        print("---Epoch {}: lr {}---".format(1, self.optimizer.param_groups[0]["lr"]))
        loss_temp = []

        while True:  # for itr in range(0, self.n_iter_max):
            self.optimizer.zero_grad()
            # Simulate
            (
                batch_x0_hidden,
                batch_u_pop,
                batch_u_ind,
                batch_y,
                batch_x_original,
            ) = self.batch.get_batch(True)
            batch_x_sim = self.nn_solution(batch_x0_hidden, batch_u_pop, batch_u_ind)

            if torch.isnan(batch_x_sim).any() or torch.isinf(batch_x_sim).any():
                print("INFO: Training had stopped because an inf in batch simulation")
                return np.nan

            # Compute fit loss
            loss = self.loss(batch_x_sim[:, :, [0]], batch_y).to(self.device)
            #loss = self.loss_MSE(batch_x_sim[:, :, [0]], batch_y).to(self.device)
            loss_temp.append(loss.item())

            if self.curr_epoch < self.batch.epoch:
                LOSS.append(np.mean(loss_temp))
                loss_temp = []

                self.curr_epoch = self.batch.epoch

                # if self.curr_epoch%50==0 and self.curr_epoch>10:
                #    self.scheduler.step()

                with torch.no_grad():
                    (
                        batch_x0_hidden,
                        batch_u_pop,
                        batch_u_ind,
                        batch_y,
                        batch_x_original,
                    ) = self.batch.get_all("All")
                    batch_x_sim = self.nn_solution(
                        batch_x0_hidden, batch_u_pop, batch_u_ind
                    )
                    LOSS_TRAIN.append(
                        torch.sqrt(
                            torch.mean(
                                (
                                    scale_inverse_Q1(
                                        batch_x_sim[:, :, [0]], self.popModelFolder
                                    )
                                    - scale_inverse_Q1(
                                        batch_x_original[:, :, [0]], self.popModelFolder
                                    )
                                )
                                ** 2
                            )
                        ).item()
                    )

                # Early stopping condition

                # if LOSS[-1] < self.best_loss:
                #    self.best_loss = LOSS[-1]
                #    self.best_model = self.nn_solution.ss_ind_model.state_dict()
                #    self.epochs_without_improvement = 0

                # else:
                #    self.epochs_without_improvement += 1

                # if self.epochs_without_improvement >= self.max_epochs_without_improvement:
                #    print("Early stopping after {} epochs without improvement.".format(self.epochs_without_improvement))
                #    break

                if self.curr_epoch == self.n_epochs:
                    break
                else:
                    print(
                        f"Epoch {self.curr_epoch-1} | Loss {LOSS[-1]:.6f}  Simulation Loss {LOSS_TRAIN[-1]:.6f} "
                    )
                    # print('---Epoch {} - lr {}---'.format(self.curr_epoch, self.optimizer.param_groups[0]['lr']))
            # Optimize
            loss.backward()
            self.optimizer.step()

        if self.best_model is None:
            self.best_model = self.nn_solution.ss_ind_model.state_dict()

        if save_model:
            if not os.path.exists(self.pathModel):
                os.mkdir(self.pathModel)
            dump(
                self.scaler_featsRobust,
                open(self.pathModel + "/scaler_robust.pkl", "wb"),
            )
            print("guardando....")
            torch.save(
                self.best_model, os.path.join(self.pathModel + "/individual_model.pt")
            )
            print(self.pathModel + "/individual_model.pt")

        return LOSS_TRAIN[-1]

    def loss(self, y_pred, y_true):
        err_fit = y_pred[1:, :] - y_true[1:, :]
        err_df = torch.diff(y_pred, axis=0) - torch.diff(y_true, axis=0)

        penalty = torch.ones_like(y_true[1:, :])

        penalty[
            torch.logical_and(
                y_true[1:, :] <= self.LIM_INFERIOR, y_pred[1:, :] > y_true[1:, :]
            )
        ] = 6
        penalty[
            torch.logical_and(
                y_true[1:, :] >= self.LIM_SUPERIOR, y_pred[1:, :] < y_true[1:, :]
            )
        ] = 6

        MSE_cgm = torch.mean(((err_fit) ** 2 * penalty))
        MSE_Dcgm = torch.mean((err_df) ** 2)

        return MSE_cgm + 10 * MSE_Dcgm
    
    def loss_MSE(self, y_pred, y_true):
        err_fit = y_pred[1:, :] - y_true[1:, :]
        MSE_cgm = torch.mean(((err_fit) ** 2))

        return MSE_cgm