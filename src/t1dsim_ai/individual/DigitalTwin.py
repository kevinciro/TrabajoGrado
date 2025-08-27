from t1dsim_ai.individual.ForwardEulerSimulator import ForwardEulerSimulator
from t1dsim_ai.individual.Q1_Ind import CGMIndividual
from t1dsim_ai.individual.Q2_Ind import CGMIndividual_Q2
from t1dsim_ai.utils.preprocess import scaler as scaler_pop
from t1dsim_ai.utils.preprocess import scaler_inverse
from t1dsim_ai.population.CGMOHSUSimStateSpaceModel_V2 import CGMOHSUSimStateSpaceModel_V2
from t1dsim_ai.transversal.options import (
    n_neurons_pop,
    hidden_compartments,
    states,
    inputs,
    input_ind,
    idx_robust,
)

import torch
from pathlib import Path
import os
import numpy as np
from pickle import load
import pandas as pd

class DigitalTwin:
    def __init__(self, n_digitalTwin=0, device=torch.device("cpu"), ts=5):
        self.ts = ts
        self.device = device

        #Path(__file__).parent.parent / "models/IndividualModel/"
        #Path(__file__).parent.parent.parent.parent / "RunExamples/example_model"

        print(Path(__file__).parent.parent.parent.parent / "RunExamples/example_model")

        self.n_digitalTwin = n_digitalTwin
        
        digitalTwin_list = [
            f.path
            for f in os.scandir(Path(__file__).parent.parent / "models/IndividualModel/")
        ]
        digitalTwin_list.sort()
        print(digitalTwin_list)
        self.digital_twin_folder = digitalTwin_list[self.n_digitalTwin]

        self.setup_simulator()

    def setup_simulator(self):
        # Population Model
        ss_pop_model = CGMOHSUSimStateSpaceModel_V2(n_feat=n_neurons_pop)
        ss_pop_model.to(self.device)
        ss_pop_model.load_state_dict(
            torch.load(
                Path(__file__).parent.parent
                / "models/PopulationModel/population_model_05022024_epoch_15.pt"
            )
        )

        # Individual Model
        
        ss_individual_model = CGMIndividual(hidden_compartments=hidden_compartments)
        #ss_individual_model = CGMIndividual_Q2(hidden_compartments=hidden_compartments)
        ss_individual_model.to(self.device)
        print(self.digital_twin_folder + "/individual_model.pt")
        ss_individual_model.load_state_dict(
            torch.load(self.digital_twin_folder + "/individual_model.pt")
        )

        for name, param in ss_pop_model.named_parameters():
            param.requires_grad = False
        for name, param in ss_individual_model.named_parameters():
            param.requires_grad = False

        # Simulator
        self.nn_solution = ForwardEulerSimulator(
            ss_pop_model,
            ss_individual_model,
            str(Path(__file__).parent.parent) + "/models/PopulationModel/",
            ts=self.ts,
        )

        self.scaler_featsRobust = load(
            open(self.digital_twin_folder + "/scaler_robust.pkl", "rb")
        )

    def prepare_data(self, df_scenario):
        dfInitStates = pd.read_csv(
            Path(__file__).parent.parent / "models/initSteadyStates.csv"
        ).set_index("initCGM")

        df_scenario[states] = df_scenario[states].astype(float)

        df_scenario.loc[0, states] = dfInitStates.loc[
            df_scenario.loc[0, states[0]].astype("int64"), states
        ]

        sim_time = len(df_scenario)
        batch_start = np.array([0], dtype=np.int)
        batch_idx = batch_start[:, np.newaxis] + np.arange(sim_time)

        x_est = np.array(df_scenario[states].values).astype(np.float32)
        u_pop = np.array(df_scenario[inputs].values).astype(np.float32)
        u_ind = np.array(df_scenario[input_ind].values).astype(np.float32)

        # Scale states and inputs from the population models
        x_est, u_pop = scaler_pop(
            x_est, u_pop, str(Path(__file__).parent.parent) + "/models/PopulationModel/", False
        )

        # Scale new inputs
        u_ind[:, idx_robust] = self.scaler_featsRobust.transform(u_ind[:, idx_robust])

        u_pop = u_pop.reshape(-1, sim_time, len(inputs))[0, batch_idx, :]
        u_ind = u_ind.reshape(-1, sim_time, len(input_ind))[0, batch_idx, :]

        x0_est = torch.tensor(
            df_scenario.loc[0, states].astype(float).values.reshape(1, -1),
            dtype=torch.float32,
        ).to(self.device)

        u_pop = torch.tensor(u_pop[[0], batch_idx.T], dtype=torch.float32).to(
            self.device
        )
        u_ind = torch.tensor(u_ind[[0], batch_idx.T], dtype=torch.float32).to(
            self.device
        )

        return x0_est, u_pop[:, [0], :], u_ind[:, [0], :]

    def simulate(self, df_scenario_original):
        # Prepare data

        df_scenario = df_scenario_original.copy()
        df_scenario = df_scenario.reset_index()
        try:
            df_scenario[states]
        except KeyError:
            df_scenario[states[1:]] = 0

        df_scenario["cgm_Actual"] = df_scenario["output_cgm"]

        x0_est, u_pop, u_ind = self.prepare_data(df_scenario)

        with torch.no_grad():
            x_sim_pop = self.nn_solution(x0_est, u_pop, None, is_pers=False)
            df_scenario[states] = scaler_inverse(
                x_sim_pop[:, 0, :].to("cpu").detach().numpy(),
                str(Path(__file__).parent.parent) + "/models/PopulationModel/",
            )

            x_sim_DT = self.nn_solution(x0_est, u_pop, u_ind, is_pers=True)

            df_scenario[[s + "_DT" for s in states]] = scaler_inverse(
                x_sim_DT[:, 0, :].to("cpu").detach().numpy(),
                str(Path(__file__).parent.parent) + "/models/PopulationModel/",
            )

        df_scenario["cgm_NNPop"] = df_scenario["output_cgm"]
        df_scenario["cgm_NNDT"] = df_scenario["output_cgm_DT"]

        return df_scenario