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
hidden_compartments_q1 = {
    "models": [5 + len(input_ind), 20, 30, 40, 50, 1]
}

hidden_compartments_I = {
    "models": [2 + len(input_ind), 20, 30, 40, 50, 1]
}


# Se define la clase para Q2
class CGMIndividual_I(nn.Module):
    def __init__(self, hidden_compartments, init_small=True):
        super(CGMIndividual_I, self).__init__()

        # NN - Model Q1
        layers_model_Q1 = []

        # Se generara una arquitectura de red neuronal de modo que:
        # la primera capa tiene 10 neuronas de entrada y 20 de salida
        # la segunda tiene 20 de entrada y 30 de salida
        # la tercera capa tiene 30 de entrada y 40 de salida
        # la cuarta capa tiene 40 de entrada y 50 de salida
        # la quinta capa tiene 50 de entrada y 1 salida
        # Cada capa tiene una funcion de activacion ReLU 
        for i in range(len(hidden_compartments["models"]) - 2):
            layers_model_Q1.append(
                nn.Linear(
                    hidden_compartments_q1["models"][i],
                    hidden_compartments_q1["models"][i + 1],
                )
            )
            layers_model_Q1.append(nn.ReLU())

        layers_model_Q1.append(nn.Linear(hidden_compartments_q1["models"][-2], 1))
        self.net_dQ1 = nn.Sequential(*layers_model_Q1)

        # NN - Model S2
        layers_model_I = []
        for i in range(len(hidden_compartments_I["models"]) - 2):
            layers_model_I.append(
                nn.Linear(
                    hidden_compartments_I["models"][i],
                    hidden_compartments_I["models"][i + 1],
                )
            )
            layers_model_I.append(nn.ReLU())

        layers_model_I.append(nn.Linear(hidden_compartments_I["models"][-2], 1))
        self.net_dI = nn.Sequential(*layers_model_I)

        clipper = WeightClipper()

        if init_small:
            networks = {
                "Q1": self.net_dQ1,
                "I": self.net_dI
                }

            for key in networks.keys():
                net = networks[key]
                for m in net.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, mean=0, std=1e-4)
                        nn.init.constant_(m.bias, val=0)

                net.apply(clipper)

    def forward(self, in_x, u_pop, u_ind):
        # q1, q2, s1, s2, I, x1, x2, x3, c2, c1
        q1, q2, _, s2, i, x1, _, x3, c2, _ = (
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

        inp_I = torch.cat((s2, i, u_ind), -1)
        dI_Ind = self.net_dI(inp_I)

        inp_Q1 = torch.cat((q1, q2, x1, x3, c2, u_ind), -1)
        dQ1_Ind = self.net_dQ1(inp_Q1)

        dx = torch.cat((dI_Ind, dQ1_Ind), -1)

        return dx