import torch
import torch.nn as nn
from t1dsim_ai.transversal.WeightClipper import WeightClipper

#from t1dsim_ai.options import input_ind
input_ind = [
    "heart_rate_WRTbaseline",
    "feat_is_weekend",
    "feat_hour_of_day_cos",
    "feat_hour_of_day_sin",
    "sleep_efficiency",
]

#from t1dsim_ai.options import hidden_compartments

n_neurons = 128
hidden_compartments_q1 = {
    "models": [5+ len(input_ind), n_neurons, n_neurons // 2, n_neurons // 4, 1]
}

hidden_compartments_q2 = {
    "models": [5+ len(input_ind), n_neurons, n_neurons // 2, n_neurons // 4, 1]
}

hidden_compartments_s1 = {
    "models": [5+ len(input_ind), n_neurons, n_neurons // 2, n_neurons // 4, 1]
}

hidden_compartments_s2 = {
    "models": [5+ len(input_ind), n_neurons, n_neurons // 2, n_neurons // 4, 1]
}

hidden_compartments_x1 = {
    "models": [5+ len(input_ind), n_neurons, n_neurons // 2, n_neurons // 4, 1]
}

hidden_compartments_x2 = {
    "models": [5+ len(input_ind), n_neurons, n_neurons // 2, n_neurons // 4, 1]
}

hidden_compartments_x3 = {
    "models": [5+ len(input_ind), n_neurons, n_neurons // 2, n_neurons // 4, 1]
}

hidden_compartments_I = {
    "models": [5+ len(input_ind), n_neurons, n_neurons // 2, n_neurons // 4, 1]
}

hidden_compartments_c1 = {
    "models": [5+ len(input_ind), n_neurons, n_neurons // 2, n_neurons // 4, 1]
}

hidden_compartments_c2 = {
    "models": [5+ len(input_ind), n_neurons, n_neurons // 2, n_neurons // 4, 1]
}

class CGMIndividual(nn.Module):
    def __init__(self, hidden_compartments, init_small=True):
        super(CGMIndividual, self).__init__()

        # NN - Model S1
        layers_model_S1 = []
        for i in range(len(hidden_compartments_s1["models"]) - 2):
            layers_model_S1.append(
                nn.Linear(
                    hidden_compartments_s1["models"][i],
                    hidden_compartments_s1["models"][i + 1],
                )
            )
            layers_model_S1.append(nn.ReLU())

        layers_model_S1.append(nn.Linear(hidden_compartments_s1["models"][-2], 1))
        self.net_dS1 = nn.Sequential(*layers_model_S1)

        # NN - Model S2
        layers_model_S2 = []
        for i in range(len(hidden_compartments_s2["models"]) - 2):
            layers_model_S2.append(
                nn.Linear(
                    hidden_compartments_s2["models"][i],
                    hidden_compartments_s2["models"][i + 1],
                )
            )
            layers_model_S2.append(nn.ReLU())

        layers_model_S2.append(nn.Linear(hidden_compartments_s2["models"][-2], 1))
        self.net_dS2 = nn.Sequential(*layers_model_S2)

        # NN - Model I
        layers_model_I = []
        for i in range(len(hidden_compartments_I["models"]) - 2):
            layers_model_X3.append(
                nn.Linear(
                    hidden_compartments_I["models"][i],
                    hidden_compartments_I["models"][i + 1],
                )
            )
            layers_model_I.append(nn.ReLU())

        layers_model_I.append(nn.Linear(hidden_compartments_I["models"][-2], 1))
        self.net_dI = nn.Sequential(*layers_model_I)

        # NN - Model X1
        layers_model_X1 = []
        for i in range(len(hidden_compartments_x1["models"]) - 2):
            layers_model_X1.append(
                nn.Linear(
                    hidden_compartments_x1["models"][i],
                    hidden_compartments_x1["models"][i + 1],
                )
            )
            layers_model_X1.append(nn.ReLU())

        layers_model_X1.append(nn.Linear(hidden_compartments_x1["models"][-2], 1))
        self.net_dX1 = nn.Sequential(*layers_model_X1)

        # NN - Model X2
        layers_model_X2 = []
        for i in range(len(hidden_compartments_x2["models"]) - 2):
            layers_model_X2.append(
                nn.Linear(
                    hidden_compartments_x2["models"][i],
                    hidden_compartments_x2["models"][i + 1],
                )
            )
            layers_model_X2.append(nn.ReLU())

        layers_model_X2.append(nn.Linear(hidden_compartments_x2["models"][-2], 1))
        self.net_dX2 = nn.Sequential(*layers_model_X2)

        # NN - Model X3
        layers_model_X3 = []
        for i in range(len(hidden_compartments_x3["models"]) - 2):
            layers_model_X3.append(
                nn.Linear(
                    hidden_compartments_x3["models"][i],
                    hidden_compartments_x3["models"][i + 1],
                )
            )
            layers_model_X3.append(nn.ReLU())

        layers_model_X3.append(nn.Linear(hidden_compartments_x3["models"][-2], 1))
        self.net_dX3 = nn.Sequential(*layers_model_X3)

        # NN - Model Q1
        layers_model_Q1 = []
        for i in range(len(hidden_compartments_q1["models"]) - 2):
            layers_model_Q1.append(
                nn.Linear(
                    hidden_compartments_q1["models"][i],
                    hidden_compartments_q1["models"][i + 1],
                )
            )
            layers_model_Q1.append(nn.ReLU())

        layers_model_Q1.append(nn.Linear(hidden_compartments_q1["models"][-2], 1))
        self.net_dQ1 = nn.Sequential(*layers_model_Q1)

        # NN - Model Q2
        layers_model_Q2 = []
        for i in range(len(hidden_compartments_q2["models"]) - 2):
            layers_model_Q2.append(
                nn.Linear(
                    hidden_compartments_q2["models"][i],
                    hidden_compartments_q2["models"][i + 1],
                )
            )
            layers_model_Q2.append(nn.ReLU())

        layers_model_Q2.append(nn.Linear(hidden_compartments_q2["models"][-2], 1))
        self.net_dQ2 = nn.Sequential(*layers_model_Q2)

        # NN - Model C1
        layers_model_c1 = []
        for i in range(len(hidden_compartments_c1["models"]) - 2):
            layers_model_c1.append(
                nn.Linear(
                    hidden_compartments_c1["models"][i],
                    hidden_compartments_c1["models"][i + 1],
                )
            )
            layers_model_c1.append(nn.ReLU())

        layers_model_c1.append(nn.Linear(hidden_compartments_c1["models"][-2], 1))
        self.net_dc1 = nn.Sequential(*layers_model_c1)

        # NN - Model C2
        layers_model_c2 = []
        for i in range(len(hidden_compartments_c2["models"]) - 2):
            layers_model_c2.append(
                nn.Linear(
                    hidden_compartments_c2["models"][i],
                    hidden_compartments_c2["models"][i + 1],
                )
            )
            layers_model_c2.append(nn.ReLU())

        layers_model_c2.append(nn.Linear(hidden_compartments_c2["models"][-2], 1))
        self.net_dc2 = nn.Sequential(*layers_model_c2)

        clipper = WeightClipper()
        if init_small:
            networks = {
                "S1": self.net_dS1,
                "S2": self.net_dS2,
                "I": self.net_dI,
                "X1": self.net_dX1,
                "X2": self.net_dX2,
                "X3": self.net_dX3,
                "Q1": self.net_dQ1,
                "Q2": self.net_dQ2,
                "C1": self.net_dc1,
                "C2": self.net_dc2,
                }

            for key in networks.keys():
                net = networks[key]
                for m in net.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, mean=0, std=1e-4)
                        nn.init.constant_(m.bias, val=0)

                net.apply(clipper)

    def forward(self, in_x, u_pop, u_ind, in_u):
        # q1, q2, s1, s2, I, x1, x2, x3, c2, c1
        q1, q2, s1, s2, i, x1, x2, x3, c2, c1 = (
            in_x[..., [0]],
            in_x[..., [1]],
            in_x[..., [2]],
            in_x[..., [3]],
            in_x[..., [4]],
            in_x[..., [5]],
            in_x[..., [6]],
            in_x[..., [7]],
            in_x[..., [8]],
            in_x[..., [9]],
        )

        u_I, u_carbs = in_u[..., [0]], in_u[..., 1:]

        inp_S1 = torch.cat((s1, u_I, u_ind), -1)
        dS1_Ind = self.net_dS1(inp_S1)

        inp_S2 = torch.cat((s1, s2, u_ind), -1)
        dS2_Ind = self.net_dS2(inp_S2)

        inp_I = torch.cat((s2, i, u_ind), -1)
        dI_Ind = self.net_dI(inp_I)

        inp_x1 = torch.cat((x1, i, u_ind), -1)
        dX1_Ind = self.net_dX1(inp_x1)

        inp_x2 = torch.cat((x2, i, u_ind), -1)
        dX2_Ind = self.net_dX1(inp_x2)

        inp_x3 = torch.cat((x3, i, u_ind), -1)
        dX3_Ind = self.net_dX3(inp_x3)

        inp_Q1 = torch.cat((q1, q2, x1, x3, c2, u_ind), -1)
        dQ1_Ind = self.net_dQ1(inp_Q1)

        inp_Q2 = torch.cat((q1, q2, x1, x2, u_ind), -1)
        dQ2_Ind = self.net_dQ2(inp_Q2)

        inp_C1 = torch.cat((c1, u_carbs, u_ind), -1)
        dC1_Ind = self.net_dc1(inp_C1)

        inp_C2 = torch.cat((c1, c2, u_ind), -1)
        dC2_Ind = self.net_dc2(inp_C2)

        dx = torch.cat((dS1_Ind, dS2_Ind, dI_Ind, dX1_Ind, dX2_Ind, dX3_Ind, dQ1_Ind, dQ2_Ind, dC1_Ind, dC2_Ind), -1)

        return dx