from pathlib import Path
import pandas as pd
from t1dsim_ai.transversal.options import states

""" 
    Transversal functions that can be used around all the simulation
"""

def getInitSSFromFile(cgm_target):
    if cgm_target < 40:
        cgm_target = 40
    if cgm_target > 400:
        cgm_target = 400
    dfInitStates = pd.read_csv(
        Path(__file__).parent.parent / "models/initSteadyStates.csv"
    ).set_index("initCGM")

    return dfInitStates.loc[int(cgm_target), states]