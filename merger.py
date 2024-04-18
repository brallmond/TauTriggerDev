import json
import os
import numpy as np
import pandas as pd

#p = "/eos/home-j/jleonhol/www/PNetAtHLT/2024_v1/PNetAtHLT/OptimisationDiTau/result_new/"

p = "/afs/cern.ch/work/b/ballmond/public/TauTriggerDev/Optimisation/result/"

l = []
for f in os.listdir(p):
    with open(p + f) as f:
        l.append(json.load(f))

df = pd.DataFrame(l)

print(max(df["eff"]))

df.to_pickle("results_VBFSingleTau.pickle")
