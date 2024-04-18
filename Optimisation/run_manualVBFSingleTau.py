from helpers import files_from_path, load_cfg_file
import os
from scipy.optimize import minimize
#from HLTClass.DiTauJetDataset import DiTauJetDataset
from HLTClass.VBFSingleTauDataset import VBFSingleTauDataset
import numpy as np
from Optimisation.helpers_opt import loss
import pandas as pd

def f(a):
        a = np.append(a, [0])
        N_den, N_num = dataset_rate.get_Nnum_Nden_VBFSingleTauPNet(a)
        rate = (N_num/N_den)*L1A_physics

        N_den, N_num = dataset_eff.ComputeEffAlgo_VBFSingleTauPNet(a)
        eff_algo = (N_num/N_den)

        print(f'Rate: {rate}')
        print(f'Algo Eff: {eff_algo}') 
        print('---------------------------------------------------------------------------------------------------------')
        return (eff_algo, rate)


if __name__ == '__main__':

    config = load_cfg_file()
    RefRun = int(config['RUNINFO']['ref_run'])
    HLT_name = config['HLT']['HLTname']
    L1A_physics = float(config['RUNINFO']['L1A_physics'])

    Rate_path = os.path.join(config['DATA']['RateDenPath'], f'Run_{RefRun}')
    Eff_path = os.path.join(config['DATA']['EffDenPath'], HLT_name)

    # FileNameList_eff = f"/eos/user/j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/{HLT_name}/ZprimeToTauTau_M-4000.root" 
    FileNameList_eff = [
        "/eos/home-j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60/GluGluHToTauTau_M-125.root",
        # "/eos/home-j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60/GluGlutoHHto2B2Tau_kl-1p00_kt-1p00_c2-0p00.root",
        # "/eos/home-j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60/VBFHHto2B2Tau_CV-1_C2V-1_C3-1.root",
        # "/eos/home-j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60/VBFHToTauTau_M125.root",
    ]
    # FileNameList_eff = f"/afs/cern.ch/work/p/pdebryas/PNetAtHLT/data/EfficiencyDen/{HLT_name}/VBFHToTauTau_M125_ext1.root"
    FileNameList_rate = files_from_path(Rate_path)
    FileNameList_rate = FileNameList_rate[0]
    print(FileNameList_rate)
    #FileNameList_rate = files_from_path(Rate_path)[0] # only one otherwise it's too long (gives good aprosimation for the rate)

    #dataset_eff = DiTauJetDataset(FileNameList_eff)
    #dataset_rate = DiTauJetDataset(FileNameList_rate)
    dataset_eff = VBFSingleTauDataset(FileNameList_eff)
    dataset_rate = VBFSingleTauDataset(FileNameList_rate)

    min_param = 0.2
    max_param = 0.6
    step = 0.01
    effs = []
    rates = []
    params = []


    for i in range(int(round((max_param - min_param) / step)) + 1):
        for j in range(int(round((max_param - min_param) / step)) + 1):
            param = (min_param + i * step, min_param + j * step)
            # print(param)
            params.append(param)
            eff, rate = f(param)
            effs.append(eff)
            rates.append(rate)

    d = {'params': params, 'efficiency': effs, 'rate': rates}
    df = pd.DataFrame(d)
    res = df.to_json()
    with open("results.json", "w+") as f:
        f.write(res)
