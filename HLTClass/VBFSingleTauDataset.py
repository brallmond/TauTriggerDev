import awkward as ak
import numpy as np
import uproot
import math
import numba as nb
import sys
from HLTClass.dataset import Dataset
from HLTClass.dataset import (
    get_L1Taus, get_L1Jets, get_Taus, get_Jets, get_GenTaus, get_GenJets, hGenTau_selection, hGenJet_selection,
    matching_Gentaus, matching_Genjets, matching_L1Taus_obj, matching_L1Jets_obj, compute_PNet_charge_prob
)

# ------------------------------ functions for VBFSingleTau with PNet -----------------------------------------------------------------------------
def compute_PNet_WP_VBFSingleTau(tau_pt, par): # TUNE FOR VBF SINGLE TAU / KEEP FOR VBF+DITAU
    # return PNet WP for DiTauJet (to optimize)
    t1 = par[0]
    t2 = par[1]
    t3 = 0.001
    t4 = 0
    x1 = 45
    x2 = 100
    x3 = 500
    x4 = 1000
    PNet_WP = tau_pt*0.
    ones = tau_pt/tau_pt
    PNet_WP = ak.where((tau_pt <= ones*x1) == False, PNet_WP, ones*t1)
    PNet_WP = ak.where((tau_pt >= ones*x4) == False, PNet_WP, ones*t4)
    PNet_WP = ak.where(((tau_pt < ones*x2) & (tau_pt > ones*x1)) == False, PNet_WP, (t2 - t1) / (x2 - x1) * (tau_pt - ones*x1) + ones*t1)
    PNet_WP = ak.where(((tau_pt >= ones*x2) & (tau_pt < ones*x3))== False, PNet_WP, (t3 - t2) / (x3 - x2) * (tau_pt - ones*x2) + ones*t2)
    PNet_WP = ak.where(((tau_pt >= ones*x3) & (tau_pt < ones*x4))== False, PNet_WP, (t4 - t3) / (x4 - x3) * (tau_pt - ones*x3) + ones*t3)
    return PNet_WP

def Jet_selection_VBFSingleTau(events, par, apply_PNET_WP = True): # TUNE FOR VBF SINGLE TAU / VBF+DITAU (check ORM module)
    # return mask for Jet (Taus) passing selection for DiTauJet path
    Jet_pt_corr = events['Jet_pt'].compute()*events['Jet_PNet_ptcorr'].compute()
    Jets_mask = (events['Jet_pt'].compute() >= 45) & (np.abs(events['Jet_eta'].compute()) <= 2.3) & (Jet_pt_corr >= 45)
    if apply_PNET_WP:
        probTauP = events['Jet_PNet_probtauhp'].compute()
        probTauM = events['Jet_PNet_probtauhm'].compute()
        Jets_mask = Jets_mask & ((probTauP + probTauM) >= compute_PNet_WP_VBFSingleTau(Jet_pt_corr, par)) & (compute_PNet_charge_prob(probTauP, probTauM) >= 0)
    return Jets_mask

@nb.jit(nopython=True)
def phi_mpi_pi(x: float) -> float: 
    # okay
    while (x >= 3.14159):
        x -= (2 * 3.14159)
    while (x < -3.14159):
        x += (2 * 3.14159)
    return x

@nb.jit(nopython=True)
def deltaR(eta1: float, phi1: float, eta2: float, phi2: float) -> float:
    deta = eta1 - eta2
    dphi = phi_mpi_pi(phi1 - phi2)
    return float(math.sqrt(deta * deta + dphi * dphi))

@nb.jit(nopython=True)
def apply_ovrm(builder, tau_eta, tau_phi, jet_pt, jet_eta, jet_phi, jet_pt_th):
    for iev in range(len(tau_eta)):
        builder.begin_list()
        for j_pt, j_eta, j_phi in zip(jet_pt[iev], jet_eta[iev], jet_phi[iev]):
            if j_pt < jet_pt_th:
                builder.append(False) # below threshold
                continue
            else:
              num_matches = 0
              dR = 999
              if (len(tau_eta[iev]) == 0):
                  builder.append(True)
                  continue
              good_jet = True
              for t_eta, t_phi in zip(tau_eta[iev], tau_phi[iev]):
                  # only save on last tau, so set a boolean here and fill it outside of the loop :)
                  dR = deltaR(j_eta, j_phi, t_eta, t_phi)
                  if dR < 0.5:
                      good_jet = False
              builder.append(good_jet)
        builder.end_list()
    return builder

@nb.jit(nopython=True)
def mjj(pt1: float, eta1: float, phi1: float,
        pt2: float, eta2: float, phi2: float) -> float:
    return float(math.sqrt(2*pt1*pt2*(math.cosh(eta1-eta2)-math.cos(phi1-phi2))))

@nb.jit(nopython=True)
def pass_mjj(jet_pt: ak.Array, jet_eta: ak.Array, jet_phi: ak.Array, mjjmax: float) -> ak.Array:
    out_mjj = []
    for iev  in range(len(jet_eta)):
        evt_mjj = 0
        if len(jet_pt[iev])<2:
            out_mjj.append(False)
        else:
            for ij1, (j1_pt, j1_eta, j1_phi) in enumerate(zip(jet_pt[iev], jet_eta[iev], jet_phi[iev])):
                for ij2, (j2_pt, j2_eta, j2_phi) in enumerate(zip(jet_pt[iev], jet_eta[iev], jet_phi[iev])):
                    if ij1 >= ij2:
                        continue
                    tmp_mjj = mjj(j1_pt,j1_eta,j1_phi,j2_pt,j2_eta,j2_phi)
                    if tmp_mjj > evt_mjj:
                        evt_mjj = tmp_mjj
            out_mjj.append(evt_mjj>=mjjmax)
    #print(out_mjj)
    return out_mjj

def Jet_selection_VBFSingleTau_Jets(events, VBFSingleTau_mask, usejets=False) -> ak.Array:
    print("in Jet_selection_VBFSingleTau_Jets")
    # return mask for Jet passing selection for DiTauJet path
    # TODO: mjj selection!!
    if usejets:
        tau_pt  = ak.drop_none(ak.mask(events['Jet_pt'].compute(), VBFSingleTau_mask)) # use with new ovrm
        tau_eta = ak.drop_none(ak.mask(events['Jet_eta'].compute(), VBFSingleTau_mask))
        tau_phi = ak.drop_none(ak.mask(events['Jet_phi'].compute(), VBFSingleTau_mask))
    else:
        tau_pt  = ak.drop_none(ak.mask(events['Tau_pt'].compute(), VBFSingleTau_mask)) # use with new ovrm
        tau_eta = ak.drop_none(ak.mask(events['Tau_eta'].compute(), VBFSingleTau_mask))
        tau_phi = ak.drop_none(ak.mask(events['Tau_phi'].compute(), VBFSingleTau_mask))

    jet_pt = events['Jet_pt'].compute()
    jet_eta = events['Jet_eta'].compute()
    jet_phi = events['Jet_phi'].compute()

    return apply_ovrm(ak.ArrayBuilder(), tau_eta, tau_phi, jet_pt, jet_eta, jet_phi, 45.).snapshot()
    #return apply_ovrm(ak.ArrayBuilder(), tau_pt, tau_eta, tau_phi, jet_eta, jet_phi, 45.).snapshot()

def Jet_selection_VBFSingleTau_mjj(events, Jet_selection_VBFSingleTau_Jet_mask):
    print("in Jet_selection_VBFSingleTau_mjj")
    
    jet_pt  = ak.drop_none(ak.mask(events['Jet_pt'].compute(),  Jet_selection_VBFSingleTau_Jet_mask))
    jet_eta = ak.drop_none(ak.mask(events['Jet_eta'].compute(), Jet_selection_VBFSingleTau_Jet_mask))
    jet_phi = ak.drop_none(ak.mask(events['Jet_phi'].compute(), Jet_selection_VBFSingleTau_Jet_mask))

    mjj_mask = pass_mjj(jet_pt,jet_eta,jet_phi, 650.)#.snapshot() # can be empty
    return mjj_mask

def evt_sel_VBFSingleTau(events, par, n_min=1, is_gen = False):
    print("in evt_sel_VBFSingleTau")
    # Selection of event passing condition of DiTauJet with PNet HLT path + mask of objects passing those conditions

    L1Tau_IsoTau45er2p1_mask = L1Tau_IsoTau45er2p1_selection(events)
    L1Tau_IsoTau45er2p1L2NN_mask = L1Tau_IsoTau45er2p1_mask & L1Tau_L2NN_selection_VBFSingleTau(events)
    L1Jet_Jet45_mask = L1Jet_Jet45_selection(events, L1Tau_IsoTau45er2p1_mask)
    #L1_DoubleJet45_Mass_Min600_mask = L1Jet_DoubleJet45_Mass_Min600_selection(events, L1Jet_Jet45_mask)

    VBFSingleTau_mask     = Jet_selection_VBFSingleTau(events, par, apply_PNET_WP = True)
    VBFSingleTau_Jet_mask = Jet_selection_VBFSingleTau_Jets(events, VBFSingleTau_mask, usejets=True)
    #VBFSingleTau_mjj_mask = Jet_selection_VBFSingleTau_mjj(events, VBFSingleTau_Jet_mask)

    # at least n_min L1tau/ recoJet and 1 L1jet / recoJet should pass
    # applying also the full L1 seed selection to account for the Overlap Removal
   
    # requiring L1 directly, instead of parts, make sure you do these things the same way

    #VBFSingleTau_evt_mask = (
        #(ak.sum(VBFSingleTau_mask, axis=-1) >= n_min) & (ak.sum(L1Tau_IsoTau45er2p1L2NN_mask, axis=-1) >= n_min) & # taus
        #(ak.sum(VBFSingleTau_Jet_mask, axis=-1) >= 2) & (ak.sum(L1Jet_Jet45_mask, axis=-1) >= 2) & # jets
        #(VBFSingleTau_mjj_mask) & L1_DoubleJet45_Mass_Min600_mask # mjj requirement
    #)

    L1_req_mask = (L1_VBFTau_selection(events))
    VBFSingleTau_evt_mask = (
        (L1_req_mask) &                                 # require L1
        (ak.sum(L1Tau_IsoTau45er2p1L2NN_mask, axis=-1) >= n_min) # require L2NN
        #(ak.sum(VBFSingleTau_mask, axis=-1) >= n_min) &  # require 1 offline Tau
        #(ak.sum(VBFSingleTau_Jet_mask, axis=-1) >= 2) &  # require 2 offline Jets, cross-cleaned w Tau
        #(ak.sum(VBFSingleTau_Jet_mask, axis=-1) >= 2)  # require 2 offline Jets, cross-cleaned w Tau
        #(VBFSingleTau_mjj_mask)                          # require dijet mjj
    )

    if is_gen:
        # if MC data, at least n_min GenTau should also pass
        GenTau_mask = hGenTau_selection(events)
        GenJet_mask = hGenJet_selection(events)
        VBFSingleTau_evt_mask = VBFSingleTau_evt_mask & (ak.sum(GenTau_mask, axis=-1) >= n_min)
        VBFSingleTau_evt_mask = VBFSingleTau_evt_mask & (ak.sum(GenJet_mask, axis=-1) >= 2)

    # matching
    L1Taus = get_L1Taus(events)
    L1Jets = get_L1Jets(events)
    Jets = get_Jets(events)

    L1Taus_VBFSingleTau = get_selL1Taus(L1Taus, L1Tau_IsoTau45er2p1L2NN_mask, n_min_taus = n_min)
    L1Jets_VBFSingleTau = get_selL1Jets(L1Jets, L1Jet_Jet45_mask, n_min_jets = 2)
    Jets_VBFSingleTau = Jets[VBFSingleTau_mask]
    Jets_VBFSingleTau_Jet = Jets[VBFSingleTau_Jet_mask]

    if is_gen:
        GenTaus = get_GenTaus(events)
        GenTaus_VBFSingleTau  = GenTaus[GenTau_mask]
        matchingGentaus_mask = matching_Gentaus(L1Taus_VBFSingleTau, Jets_VBFSingleTau, GenTaus_VBFSingleTau)

        GenJets = get_GenJets(events)
        GenJets_VBFSingleTau  = GenJets[GenJet_mask]
        matchingGenjets_mask = matching_Genjets(L1Jets_VBFSingleTau, Jets_VBFSingleTau_Jet, GenJets_VBFSingleTau)

        # at least n_min GenTau should match L1Tau/Taus and 2 GenJet L1Jet/Jets
        evt_mask_matching = (ak.sum(matchingGentaus_mask, axis=-1) >= n_min) & (ak.sum(matchingGenjets_mask, axis=-1) >= 2)
    else:
        matchingJets_mask = matching_L1Taus_obj(L1Taus_VBFSingleTau, Jets_VBFSingleTau)
        # at least n_min Tau should match L1Tau
        evt_mask_matching = (ak.sum(matchingJets_mask, axis=-1) >= n_min)

        matchingJets_Jet_mask = matching_L1Jets_obj(L1Jets_VBFSingleTau, Jets_VBFSingleTau_Jet)
        # at least 2 Jet should match L1Jet
        evt_mask_matching = evt_mask_matching & (ak.sum(matchingJets_Jet_mask, axis=-1) >= 2)

    #VBFSingleTau_evt_mask = VBFSingleTau_evt_mask & evt_mask_matching
    if is_gen: 
        return VBFSingleTau_evt_mask, matchingGentaus_mask, matchingGenjets_mask
    else:
        return VBFSingleTau_evt_mask, matchingJets_mask, matchingJets_Jet_mask


    
# ------------------------------ functions for VBF_DiPFJet45_Mjj650_MediumDeepTauPFTauHPS45_L2NN_eta2p1  ---------------------------------------------------
def compute_DeepTau_WP_DiTau(tau_pt):
    # return DeepTau WP 
    t1 = 0.5701#0.649
    t2 = 0.4610#0.441
    t3 = 0.125#0.05
    x1 = 35
    x2 = 100
    x3 = 300
    Tau_WP = tau_pt*0.
    ones = tau_pt/tau_pt
    Tau_WP = ak.where((tau_pt <= ones*x1) == False, Tau_WP, ones*t1)
    Tau_WP = ak.where((tau_pt >= ones*x3) == False, Tau_WP, ones*t3)
    Tau_WP = ak.where(((tau_pt < ones*x2) & (tau_pt > ones*x1)) == False, Tau_WP, (t2 - t1) / (x2 - x1) * (tau_pt - ones*x1) + ones*t1)
    Tau_WP = ak.where(((tau_pt >= ones*x2) & (tau_pt < ones*x3))== False, Tau_WP, (t3 - t2) / (x3 - x2) * (tau_pt - ones*x2) + ones*t2)
    return Tau_WP

def Tau_selection_VBFSingleTau(events, apply_DeepTau_WP = True):
    # return mask for Tau passing selection
    tau_pt = events['Tau_pt'].compute()
    Tau_mask = (tau_pt >= 45) & (np.abs(events['Tau_eta'].compute()) <= 2.1)
    if apply_DeepTau_WP:
        Tau_mask = Tau_mask & (events['Tau_deepTauVSjet'].compute() >= compute_DeepTau_WP_DiTau(tau_pt))
    return Tau_mask

def evt_sel_VBF_DiPFJet45_Mjj650_MediumDeepTauPFTauHPS45_L2NN_eta2p1(events, n_min = 1, is_gen = False):
    # Selection of event passing condition of VBF_DiPFJet45_Mjj650_MediumDeepTauPFTauHPS45_L2NN_eta2p1 + mask of objects passing those conditions

    # get mask for L1 taus with iso and pt > 45
    # get mask for L1 jets with pt > 45
    # get mask for L1 jets that aren't taus (uses previous two masks)
    # get mask for events where L1 jet pair has mjj > 600 (after ovrm)
    # get mask for L1 taus passing L2NN requirements
    # end L1

    L1Tau_IsoTau45er2p1_mask = L1Tau_IsoTau45er2p1_selection(events)
    L1Jet_Jet45_mask = L1Jet_Jet45_selection(events) # used for debugging only
    L1Jet_Jet45_ovrm_mask = L1Jet_Jet45_ovrm_selection(events, L1Tau_IsoTau45er2p1_mask)

    L1_DoubleJet45_Mass_Min600_mask = L1Jet_DoubleJet45_Mass_Min600_selection(events, L1Jet_Jet45_ovrm_mask)
    # check cutflow?

    #L1_DoubleJet45_Mass_Min600_mask = L1Jet_DoubleJet45_Mass_Min600_selection(events, L1Jet_Jet45_mask) # works fine

    L2NN_mask = L1Tau_L2NN_selection_VBFSingleTau(events)

    VBFSingleTau_mask     = Tau_selection_VBFSingleTau(events)
    VBFSingleTau_Jet_mask = Jet_selection_VBFSingleTau_Jets(events, VBFSingleTau_mask, usejets=False)
    #VBFSingleTau_mjj_mask = Jet_selection_VBFSingleTau_mjj(events, VBFSingleTau_Jet_mask)
   
    #VBFSingleTau_evt_mask = (
    #    (ak.sum(VBFSingleTau_mask, axis=-1) >= n_min) & (ak.sum(L1Tau_IsoTau45er2p1L2NN_mask, axis=-1) >= n_min) & # taus
    #    (ak.sum(VBFSingleTau_Jet_mask, axis=-1) >= 2) & (ak.sum(L1Jet_Jet45_mask, axis=-1) >= 2) & # jets
    #    (VBFSingleTau_mjj_mask) & (L1_DoubleJet45_Mass_Min600_mask) # mjj requirement
    #)

    print("I'm in the DeepTau counting function!")
    L1_req_mask = (L1_VBFTau_selection(events))
    VBFSingleTau_evt_mask = (                            
        (ak.sum(L1Tau_IsoTau45er2p1_mask, axis=-1) >= n_min) &
        (ak.sum(L1Jet_Jet45_ovrm_mask, axis=-1) >= 2) &
        #(ak.sum(L1_DoubleJet45_Mass_Min600_mask, axis=-1) >= 2) &
        (ak.sum(L2NN_mask, axis=-1) >= n_min)
        #(L1_DoubleJet45_Mass_Min600_mask)
        #(ak.sum(L1Tau_IsoTau45er2p1L2NN_mask, axis=-1) >= n_min) # require L2NN
        #(ak.sum(VBFSingleTau_mask, axis=-1) >= n_min) &  # require 1 offline Tau
        #(ak.sum(VBFSingleTau_Jet_mask, axis=-1) >= 2) &  # require 2 offline Jets, cross-cleaned w Tau
        #(ak.sum(VBFSingleTau_Jet_mask, axis=-1) >= 2)  # require 2 offline Jets, cross-cleaned w Tau
        #(VBFSingleTau_mjj_mask)                          # require dijet mjj
    )

    if is_gen:
        # if MC data, at least n_min GenTau should also pass
        GenTau_mask = hGenTau_selection(events)
        GenJet_mask = hGenJet_selection(events)
        #VBFSingleTau_evt_mask = VBFSingleTau_evt_mask & (ak.sum(GenTau_mask, axis=-1) >= n_min)
        #VBFSingleTau_evt_mask = VBFSingleTau_evt_mask & (ak.sum(GenJet_mask, axis=-1) >= 2)

    # matching
    L1Taus = get_L1Taus(events)
    L1Jets = get_L1Jets(events)
    Taus = get_Taus(events)
    Jets = get_Jets(events)

    #L1Taus_VBFSingleTau = get_selL1Taus(L1Taus, L1Tau_IsoTau45er2p1L2NN_mask, n_min_taus = n_min)
    L1Taus_VBFSingleTau = get_selL1Taus(L1Taus, L2NN_mask, n_min_taus = n_min)
    L1Jets_VBFSingleTau = get_selL1Jets(L1Jets, L1Jet_Jet45_mask, n_min_jets = 2)
    Taus_VBFSingleTau = Taus[VBFSingleTau_mask]
    Jets_VBFSingleTau_Jet = Jets[VBFSingleTau_Jet_mask]

    if is_gen:
        GenTaus = get_GenTaus(events)
        GenTaus_VBFSingleTau = GenTaus[GenTau_mask]
        matchingGentaus_mask = matching_Gentaus(L1Taus_VBFSingleTau, Taus_VBFSingleTau, GenTaus_VBFSingleTau)

        GenJets = get_GenJets(events)
        GenJets_VBFSingleTau  = GenJets[GenJet_mask]
        matchingGenjets_mask = matching_Genjets(L1Jets_VBFSingleTau, Jets_VBFSingleTau_Jet, GenJets_VBFSingleTau)

        # at least n_min GenTau should match L1Tau/Taus and 1 GenJet L1Jet/Jets
        evt_mask_matching = (ak.sum(matchingGentaus_mask, axis=-1) >= n_min) & (ak.sum(matchingGenjets_mask, axis=-1) >= 2)
    else:
        matchingTaus_mask = matching_L1Taus_obj(L1Taus_VBFSingleTau, Taus_VBFSingleTau)
        # at least n_min Tau should match L1Tau
        evt_mask_matching = (ak.sum(matchingTaus_mask, axis=-1) >= n_min)

        matchingJets_Jet_mask = matching_L1Jets_obj(L1Jets_VBFSingleTau, Jets_VBFSingleTau_Jet)
        # at least 2 Jet should match L1Jet
        evt_mask_matching = evt_mask_matching & (ak.sum(matchingJets_Jet_mask, axis=-1) >= 2)

    #VBFSingleTau_evt_mask = VBFSingleTau_evt_mask & evt_mask_matching
    if is_gen: 
        return VBFSingleTau_evt_mask, matchingGentaus_mask, matchingGenjets_mask
    else:
        return VBFSingleTau_evt_mask, matchingTaus_mask, matchingJets_Jet_mask



# ------------------------------ Common functions for VBFSingleTau path ---------------------------------------------------------------
def L1_VBFTau_selection(events):
    return (events['L1_DoubleJet45_Mass_Min600_IsoTau45er2p1_RmOvlp_dR0p5'].compute()>0)
    #return L1_DoubleJet45_Mass_Min550_IsoTau45er2p1_Rmovlp_mask

def L1Tau_IsoTau45er2p1_selection(events):
    # return mask for L1tau passing IsoTau45er2p1 selection
    L1_IsoTau45er2p1_mask = (events['L1Tau_pt'].compute() >= 45) & (events['L1Tau_eta'].compute() <= 2.131) & (events['L1Tau_eta'].compute() >= -2.131) & (events['L1Tau_hwIso'].compute() > 0 )
    return L1_IsoTau45er2p1_mask

def L1Jet_Jet45_ovrm_selection(events, VBFSingleTau_mask) -> ak.Array:
    tau_pt  = ak.drop_none(ak.mask(events['L1Tau_pt'].compute(), VBFSingleTau_mask))
    tau_eta = ak.drop_none(ak.mask(events['L1Tau_eta'].compute(), VBFSingleTau_mask))
    tau_phi = ak.drop_none(ak.mask(events['L1Tau_phi'].compute(), VBFSingleTau_mask))

    jet_pt  = events['L1Jet_pt'].compute()
    jet_eta = events['L1Jet_eta'].compute()
    jet_phi = events['L1Jet_phi'].compute()

    return apply_ovrm(ak.ArrayBuilder(), tau_eta, tau_phi, jet_pt, jet_eta, jet_phi, 45.).snapshot()
    #return apply_ovrm(ak.ArrayBuilder(), tau_pt, tau_eta, tau_phi, jet_eta, jet_phi, 45.).snapshot()

def L1Jet_Jet45_selection(events):
    L1_Jet45_mask = (events["L1Jet_pt"].compute() >= 45) & (events['L1Jet_eta'].compute() < 4.7) & (events['L1Jet_eta'].compute() > -4.7)
    return L1_Jet45_mask

#@nb.jit(nopython=True)
def L1_dr(events, L1Tau_mask, L1Jet_mask):
    tau_eta = ak.drop_none(ak.mask(events['L1Tau_eta'].compute(), L1Tau_mask))
    tau_phi = ak.drop_none(ak.mask(events['L1Tau_phi'].compute(), L1Tau_mask))
    jet_eta = ak.drop_none(ak.mask(events['L1Jet_eta'].compute(), L1Jet_mask))
    jet_phi = ak.drop_none(ak.mask(events['L1Jet_phi'].compute(), L1Jet_mask))
    for event_idx in range(len(jet_eta)): # number of Entries in the list is nEvents
        for jet_eta, jet_phi in zip(jet_eta[event_idx], jet_phi[event_idx]):
            for tau_eta, tau_phi in zip(tau_eta[event_idx], tau_phi[event_idx]):
                if tau_eta == None:
                    print('ok')
                    continue
                dR = deltaR(tau_eta, tau_phi, jet_eta, jet_phi)
                print(dR)
    return "happy"


def L1Jet_DoubleJet45_Mass_Min600_selection(events, L1Jet_Jet45_selection_mask):
    jet_pt  = ak.drop_none(ak.mask(events['L1Jet_pt'].compute(),  L1Jet_Jet45_selection_mask))
    jet_eta = ak.drop_none(ak.mask(events['L1Jet_eta'].compute(), L1Jet_Jet45_selection_mask))
    jet_phi = ak.drop_none(ak.mask(events['L1Jet_phi'].compute(), L1Jet_Jet45_selection_mask))
    print(jet_pt, jet_eta, jet_phi)

    L1_DoubleJet45_Mass_Min600_mask = pass_mjj(jet_pt,jet_eta,jet_phi, 600.)
    return L1_DoubleJet45_Mass_Min600_mask

def HLT_VBFTau_selection(events):
    HLT_mask = (events['HLT_VBF_DiPFJet45_Mjj650_MediumDeepTauPFTauHPS45_L2NN_eta2p1'].compute() > 0)
    return HLT_mask

def L1Tau_L2NN_selection_VBFSingleTau(events):
    # return mask for L1tau passing L2NN selection
    #L1_L2NN_mask = ((events['L1Tau_l2Tag'].compute() > 0.386) | (events['L1Tau_pt'].compute() >= 250))
    L1_L2NN_mask = ((events['L1Tau_l2Tag'].compute() > 0.4327) | (events['L1Tau_pt'].compute() >= 250))
    return L1_L2NN_mask

def Denominator_Selection_VBFSingleTau(GenLepton):
    # return mask for event passing minimal GenTau requirements for diTauJet HLT (1 hadronic Taus with min vis. pt and eta)
    mask = (GenLepton['kind'] == 5)
    ev_mask = ak.sum(mask, axis=-1) >= 1  # at least 1 Gen taus should pass this requirements
    return ev_mask

def Denominator_Selection_VBFSingleTau_Jet(events):
    # return mask for event passing minimal GenJet requirements for diTauJet HLT (2 jet)
    mask = (events["GenJet_pt"].compute() > 0)
    ev_mask = ak.sum(mask, axis=-1) >= 2  # at least 2 Gen jet should pass this requirements
    return ev_mask

def get_selL1Taus(L1Taus, L1Tau_IsoTau45er2p1_mask, n_min_taus = 1):
    # return L1tau that pass L1Tau_IsoTau45er2p1
    IsoTau45er2p1 = (ak.sum(L1Tau_IsoTau45er2p1_mask, axis=-1) >= n_min_taus)
    L1Taus_Sel = L1Taus
    L1Taus_Sel = ak.where(IsoTau45er2p1 == False, L1Taus_Sel, L1Taus[L1Tau_IsoTau45er2p1_mask])
    return L1Taus_Sel
# TODO : what is going on here?
def get_selL1Jets(L1Jets, L1Jet_Jet45_mask, n_min_jets = 2):
    # return L1jet that pass L1Jet_Jet35
    Jet35 = (ak.sum(L1Jet_Jet45_mask, axis=-1) >= n_min_jets)
    L1Jets_Sel = L1Jets
    L1Jets_Sel = ak.where(Jet35 == False, L1Jets_Sel, L1Jets[L1Jet_Jet45_mask])
    return L1Jets_Sel



class VBFSingleTauDataset(Dataset):
    def __init__(self, fileName):
        Dataset.__init__(self, fileName)

    # ------------------------------ functions to Compute Rate ---------------------------------------------------------------------

    #def get_Nnum_Nden_HLTDoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60(self):
    def get_Nnum_Nden_HLTVBF_DiPFJet45_Mjj650_MediumDeepTauPFTauHPS45_L2NN_eta2p1(self):
        print(f'Computing rate for VBF_DiPFJet45_Mjj650_MediumDeepTauPFTauHPS45_L2NN_eta2p1:')
        #load all events in the file that belong to the run and lumiSections_range, save the number of events in Denominator
        events = self.get_events()
        N_den = len(events)
        print(f"Number of events in denominator: {len(events)}")

        VBFSingleTau_evt_mask, _, _ = evt_sel_VBF_DiPFJet45_Mjj650_MediumDeepTauPFTauHPS45_L2NN_eta2p1(events, n_min = 1, is_gen = False)
        N_num = len(events[VBFSingleTau_evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num

    def get_Nnum_Nden_VBFSingleTauPNet(self, par):
        print(f'Computing Rate for VBFSingleTau PNet path with param: {par}')
        #load all events in the file that belong to the run and lumiSections_range, save the number of events in Denominator
        events = self.get_events()
        N_den = len(events)
        print(f"Number of events in denominator: {len(events)}")

        VBFSingleTau_evt_mask, _, _ = evt_sel_VBFSingleTau(events, par, n_min=1, is_gen = False)
        print(VBFSingleTau_evt_mask)
        N_num = len(events[VBFSingleTau_evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num

    def save_Event_Nden_eff_VBFSingleTau(self, tmp_file):
        #Save only needed informations for events passing minimal Gen requirements for VBF+tau HLT (passing denominator selection for efficiency)
        events = self.get_events()
        print(f"Number of events in the file: {len(events)}")
        GenLepton = self.get_GenLepton(events)
        evt_mask = Denominator_Selection_VBFSingleTau(GenLepton) & Denominator_Selection_VBFSingleTau_Jet(events)
        print(f"Number of events with at least 1 hadronic Tau and 2 jet: {ak.sum(evt_mask)}")
        self.Save_Event_Nden_Eff(events, GenLepton, evt_mask, tmp_file)
        return

    def save_info(self, events_Den, events_Num, Tau_Den, Tau_Num, Jet_Den, Jet_Num, out_file):
        # saving infos
        lst_evt_Den = {}
        lst_evt_Den['nPFPrimaryVertex'] = np.array(events_Den['nPFPrimaryVertex'].compute())
        lst_evt_Den['nPFSecondaryVertex'] = np.array(events_Den['nPFSecondaryVertex'].compute())

        lst_evt_Num = {}
        lst_evt_Num['nPFPrimaryVertex'] = np.array(events_Num['nPFPrimaryVertex'].compute())
        lst_evt_Num['nPFSecondaryVertex'] = np.array(events_Num['nPFSecondaryVertex'].compute())

        lst_Den = {}
        lst_Den['Tau_pt'] = Tau_Den.pt
        lst_Den['Tau_eta'] = Tau_Den.eta
        lst_Den['Tau_phi'] = Tau_Den.phi
        lst_Den['Tau_nChargedHad'] = Tau_Den.nChargedHad
        lst_Den['Tau_nNeutralHad'] = Tau_Den.nNeutralHad
        lst_Den['Tau_DecayMode'] = Tau_Den.DecayMode
        lst_Den['Tau_charge'] = Tau_Den.charge
        lst_Den['Jet_pt'] = Jet_Den.pt
        lst_Den['Jet_eta'] = Jet_Den.eta
        lst_Den['Jet_phi'] = Jet_Den.phi

        lst_Num = {}
        lst_Num['Tau_pt'] = Tau_Num.pt
        lst_Num['Tau_eta'] = Tau_Num.eta
        lst_Num['Tau_phi'] = Tau_Num.phi
        lst_Num['Tau_nChargedHad'] = Tau_Num.nChargedHad
        lst_Num['Tau_nNeutralHad'] = Tau_Num.nNeutralHad
        lst_Num['Tau_DecayMode'] = Tau_Num.DecayMode
        lst_Num['Tau_charge'] = Tau_Num.charge
        lst_Num['Jet_pt'] = Jet_Num.pt
        lst_Num['Jet_eta'] = Jet_Num.eta
        lst_Num['Jet_phi'] = Jet_Num.phi

        with uproot.create(out_file, compression=uproot.ZLIB(4)) as file:
            file["eventsDen"] = lst_evt_Den
            file["TausDen"] = lst_Den
            file["eventsNum"] = lst_evt_Num
            file["TausNum"] = lst_Num
        return 

    def produceRoot_HLTVBF_DiPFJet45_Mjj650_MediumDeepTauPFTauHPS45_L2NN_eta2p1(self, out_file):
        #load all events that pass denominator Selection
        events = self.get_events()
        # To compute efficiency, we save in denominator GenTau which pass minimal selection
        GenTau_mask = hGenTau_selection(events)
        GenTaus = get_GenTaus(events)
        Tau_Den = GenTaus[GenTau_mask]
        GenJet_mask = hGenJet_selection(events)
        GenJets = get_GenJets(events)
        Jet_Den = GenJets[GenJet_mask]

        mask_den_selection = (ak.num(Tau_Den['pt']) >= 1) & (ak.num(Jet_Den['pt']) >=2)
        Tau_Den = Tau_Den[mask_den_selection]
        Jet_Den = Jet_Den[mask_den_selection]

        events = events[mask_den_selection]

        print(f"Number of GenTaus passing denominator selection: {len(ak.flatten(Tau_Den))}")
        print(f"Number of GenJets passing denominator selection: {len(ak.flatten(Jet_Den))}")
        
        VBFSingleTau_evt_mask, matchingGentaus_mask, matchingGenjets_mask = evt_sel_VBF_DiPFJet45_Mjj650_MediumDeepTauPFTauHPS45_L2NN_eta2p1(events, n_min = 1, is_gen = True)
        Tau_Num = (Tau_Den[matchingGentaus_mask])[VBFSingleTau_evt_mask]
        print(f"Number of GenTaus passing numerator selection: {len(ak.flatten(Tau_Num))}")

        Jet_Num = (Jet_Den[matchingGenjets_mask])[VBFSingleTau_evt_mask]
        print(f"Number of GenJets passing numerator selection: {len(ak.flatten(Jet_Num))}")
        events_Num = events[VBFSingleTau_evt_mask]

        self.save_info(events, events_Num, Tau_Den, Tau_Num, Jet_Den, Jet_Num, out_file)
        return

    def produceRoot_VBFSingleTauPNet(self, out_file, par):
        #load all events that pass denominator Selection
        events = self.get_events()

        # To compute efficiency, we save in denominator GenTau which pass minimal selection
        GenTau_mask = hGenTau_selection(events)
        GenTaus = get_GenTaus(events)
        Tau_Den = GenTaus[GenTau_mask]

        GenJet_mask = hGenJet_selection(events)
        GenJets = get_GenJets(events)
        Jet_Den = GenJets[GenJet_mask]

        mask_den_selection = (ak.num(Tau_Den['pt']) >=1) & (ak.num(Jet_Den['pt']) >=2)
        Tau_Den = Tau_Den[mask_den_selection]
        Jet_Den = Jet_Den[mask_den_selection]
        events = events[mask_den_selection]

        print(f"Number of GenTaus passing denominator selection: {len(ak.flatten(Tau_Den))}")

        GenJet_mask = hGenJet_selection(events)
        GenJets = get_GenJets(events)
        Jet_Den = GenJets[GenJet_mask]

        mask_den_selection = ak.num(Jet_Den['pt']) >=2
        Jet_Den = Jet_Den[mask_den_selection]
        events = events[mask_den_selection]

        print(f"Number of GenJets passing denominator selection: {len(ak.flatten(Jet_Den))}")

        VBFSingleTau_evt_mask, matchingGentaus_mask, matchingGenjets_mask = evt_sel_VBFSingleTau(events, par, n_min=1, is_gen = True)

        Tau_Num = (Tau_Den[matchingGentaus_mask])[VBFSingleTau_evt_mask]
        print(f"Number of GenTaus passing numerator selection: {len(ak.flatten(Tau_Num))}")
        events_Num = events[VBFSingleTau_evt_mask]

        Jet_Num = (Jet_Den[matchingGenjets_mask])[VBFSingleTau_evt_mask]
        print(f"Number of GenJets passing numerator selection: {len(ak.flatten(Jet_Num))}")
        events_Num = events[VBFSingleTau_evt_mask]

        self.save_info(events, events_Num, Tau_Den, Tau_Num, Jet_Den, Jet_Num, out_file)

        return

    # ------------------------------ functions to Compute Efficiency for opt ---------------------------------------------------------------

    def ComputeEffAlgo_VBFSingleTauPNet(self, par):

        #load all events that pass denominator Selection
        events = self.get_events()

        # these are defined in a different place in a similar function, but okay..
        L1Taus = get_L1Taus(events)
        L1Jets = get_L1Jets(events)
        Jets = get_Jets(events) # Taus not used in PNet
        GenTaus = get_GenTaus(events)
        GenJets = get_GenJets(events)
        GenTau_mask = hGenTau_selection(events)
        GenJet_mask = hGenJet_selection(events)
        n_min = 2

        # Selection of L1/Gen and Jets objects without PNET WP
        L1Tau_IsoTau45er2p1_mask = L1Tau_IsoTau45er2p1_selection(events)
        L1Tau_IsoTau45er2p1L2NN_mask = L1Tau_IsoTau45er2p1_mask & L1Tau_L2NN_selection_VBFSingleTau(events)
        L1Jet_Jet45_mask = L1Jet_Jet45_selection(events, L1Tau_IsoTau45er2p1_mask)
        #L1_DoubleJet45_Mass_Min600_mask = (ak.sum(L1Jet_Jet45_mask, axis=-1) >= 2) & L1Jet_DoubleJet45_Mass_Min600_selection(events, L1Jet_Jet45_mask)

        VBFSingleTau_mask = Jet_selection_VBFSingleTau(events, par, apply_PNET_WP = False) # used for denom
        VBFSingleTau_Jet_mask = Jet_selection_VBFSingleTau_Jets(events, VBFSingleTau_mask, usejets=True)
        #VBFSingleTau_mjj_mask = ak.sum(VBFSingleTau_Jet_mask, axis=-1) >= 1  & Jet_selection_VBFSingleTau_mjj(events, VBFSingleTau_Jet_mask)
        #VBFSingleTau_mjj_mask = Jet_selection_VBFSingleTau_mjj(events, VBFSingleTau_Jet_mask)

        L1_req_mask = (L1_VBFTau_selection(events))
        VBFSingleTau_evt_mask = (
            (L1_req_mask) &                                  # require L1
            (ak.sum(L1Tau_IsoTau45er2p1L2NN_mask, axis=-1) >= n_min) # require L2NN
            #(ak.sum(VBFSingleTau_mask, axis=-1) >= n_min) &  # require 1 offline Tau
            #(ak.sum(VBFSingleTau_Jet_mask, axis=-1) >= 2) &  # require 2 offline Jets, cross-cleaned w Tau
            #(ak.sum(VBFSingleTau_Jet_mask, axis=-1) >= 2)   # require 2 offline Jets, cross-cleaned w Tau
            #(VBFSingleTau_mjj_mask)                          # require dijet mjj
        )

        # old selection
        ## at least n_min L1tau/ recoJet and 2 L1jet / recoJet should pass
        ## applying also the full L1 seed selection to account for the Overlap Removal
        ##VBFSingleTau_evt_mask = (
        ##    (ak.sum(VBFSingleTau_mask, axis=-1) >= n_min) & (ak.sum(L1Tau_IsoTau45er2p1L2NN_mask, axis=-1) >= n_min) & # taus
        ##    (ak.sum(VBFSingleTau_Jet_mask, axis=-1) >= 2) & (ak.sum(L1Jet_Jet45_mask, axis=-1) >= 2) & # jets
        ##    (VBFSingleTau_mjj_mask) & L1_DoubleJet45_Mass_Min600_mask # mjj requirement
        ##)

        # matching
        L1Taus_VBFSingleTau = get_selL1Taus(L1Taus, L1Tau_IsoTau45er2p1L2NN_mask, n_min_taus = n_min)
        L1Jets_VBFSingleTau = get_selL1Jets(L1Jets, L1Jet_Jet45_mask, n_min_jets = 2)
        Jets_VBFSingleTau = Jets[VBFSingleTau_mask]
        Jets_VBFSingleTau_Jet = Jets[VBFSingleTau_Jet_mask]
        GenTaus_VBFSingleTau = GenTaus[GenTau_mask]
        GenJets_VBFSingleTau = GenJets[GenJet_mask]

        matchingGentaus_mask = matching_Gentaus(L1Taus_VBFSingleTau, Jets_VBFSingleTau, GenTaus_VBFSingleTau)
        # at least n_min GenTau should match L1Tau/Taus
        evt_mask_matching = (ak.sum(matchingGentaus_mask, axis=-1) >= n_min)

        matchingGenjets_mask = matching_Genjets(L1Jets_VBFSingleTau, Jets_VBFSingleTau_Jet, GenJets_VBFSingleTau)
        # at least 2 GenJet should match L1Jet/Jets
        evt_mask_matching = evt_mask_matching & (ak.sum(matchingGenjets_mask, axis=-1) >= 2)

        VBFSingleTau_evt_mask = VBFSingleTau_evt_mask & evt_mask_matching
        N_den = len(events[VBFSingleTau_evt_mask])
        print(f"Number of events in denominator: {N_den}")

        # Numerator: only need to require PNET WP on the taus, nothing additional on the jets
        # Selection of L1/Gen and Jets objects with PNET WP
        VBFSingleTau_mask = Jet_selection_VBFSingleTau(events, par, apply_PNET_WP = True)
        # at least 1 L1tau/ Jet/ GenTau should pass
        #VBFSingleTau_evt_mask = (
        #    (ak.sum(VBFSingleTau_mask, axis=-1) >= n_min) & (ak.sum(L1Tau_IsoTau45er2p1L2NN_mask, axis=-1) >= n_min) & # taus
        #    (ak.sum(VBFSingleTau_Jet_mask, axis=-1) >= 2) & (ak.sum(L1Jet_Jet45_mask, axis=-1) >= 2) & # jets
        #    (VBFSingleTau_mjj_mask) & L1_DoubleJet45_Mass_Min600_mask # mjj requirement
        #)
        VBFSingleTau_evt_mask = (
            (L1_req_mask) &                                  # require L1
            (ak.sum(L1Tau_IsoTau45er2p1L2NN_mask, axis=-1) >= n_min) # require L2NN
            #(ak.sum(VBFSingleTau_mask, axis=-1) >= n_min) &  # require 1 offline Tau
            #(ak.sum(VBFSingleTau_Jet_mask, axis=-1) >= 2) &  # require 2 offline Jets, cross-cleaned w Tau
            #(ak.sum(VBFSingleTau_Jet_mask, axis=-1) >= 2)  # require 2 offline Jets, cross-cleaned w Tau
            #(VBFSingleTau_mjj_mask)                          # require dijet mjj
        )

        # matching
        # no need to match jets, as they are already included in the denominator
        Jets_VBFSingleTau = Jets[VBFSingleTau_mask]
        matchingGentaus_mask = matching_Gentaus(L1Taus_VBFSingleTau, Jets_VBFSingleTau, GenTaus_VBFSingleTau)
        # at least n_min GenTau should match L1Tau/Taus
        evt_mask_matching = evt_mask_matching & (ak.sum(matchingGentaus_mask, axis=-1) >= n_min)

        #VBFSingleTau_evt_mask = VBFSingleTau_evt_mask & evt_mask_matching
        N_num = len(events[VBFSingleTau_evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num

    def ComputeEffAlgo_HLTVBF_DiPFJet45_Mjj650_MediumDeepTauPFTauHPS45_L2NN_eta2p1(self):
        #load all events that pass denominator Selection
        events = self.get_events()

        L1Taus = get_L1Taus(events)
        L1Jets = get_L1Jets(events)
        Taus = get_Taus(events)
        Jets = get_Jets(events)
        GenTaus = get_GenTaus(events)
        GenJets = get_GenJets(events)
        GenTau_mask = hGenTau_selection(events)
        GenJet_mask = hGenJet_selection(events)
        n_min = 2

        # Selection of L1/Gen and Jets objects without PNET WP
        L1Tau_IsoTau45er2p1_mask = L1Tau_IsoTau45er2p1_selection(events)
        L1Tau_IsoTau45er2p1L2NN_mask = L1Tau_IsoTau45er2p1_mask & L1Tau_L2NN_selection_VBFSingleTau(events)
        L1Jet_Jet45_mask = L1Jet_Jet45_selection(events, L1Tau_IsoTau45er2p1_mask)
        #L1_DoubleJet45_Mass_Min600_mask = ak.sum(L1Jet_Jet45_mask, axis=-1) >= 2 & L1Jet_DoubleJet45_Mass_Min600_selection(events, L1Jet_Jet45_mask)
           
        VBFSingleTau_mask = Tau_selection_VBFSingleTau(events, apply_DeepTau_WP = False)
        VBFSingleTau_Jet_mask = Jet_selection_VBFSingleTau_Jets(events, VBFSingleTau_mask, usejets=False)
        #VBFSingleTau_mjj_mask = ak.sum(VBFSingleTau_Jet_mask, axis=-1) >= 1  & Jet_selection_VBFSingleTau_mjj(events, VBFSingleTau_Jet_mask)
        #VBFSingleTau_mjj_mask = Jet_selection_VBFSingleTau_mjj(events, VBFSingleTau_Jet_mask)

        # at least n_min L1tau/ recoJet and 2 L1jet/jet should pass + L1 trigger
        #VBFSingleTau_evt_mask = (
        #    (ak.sum(VBFSingleTau_mask, axis=-1) >= n_min) & (ak.sum(L1Tau_IsoTau45er2p1L2NN_mask, axis=-1) >= n_min) & # taus
        #    (ak.sum(VBFSingleTau_Jet_mask, axis=-1) >= 2) & (ak.sum(L1Jet_Jet45_mask, axis=-1) >= 2) & # jets
        #    (VBFSingleTau_mjj_mask) & L1_DoubleJet45_Mass_Min600_mask # mjj requirement
        #)

        L1_req_mask = (L1_VBFTau_selection(events))
        VBFSingleTau_evt_mask = (
            (L1_req_mask) &                                  # require L1
            (ak.sum(L1Tau_IsoTau45er2p1L2NN_mask, axis=-1) >= n_min) # require L2NN
            #(ak.sum(L1Tau_IsoTau45er2p1L2NN_mask, axis=-1) >= n_min) & # require L2NN
            #(ak.sum(VBFSingleTau_mask, axis=-1) >= n_min) &  # require 1 offline Tau
            #(ak.sum(VBFSingleTau_Jet_mask, axis=-1) >= 2) &  # require 2 offline Jets, cross-cleaned w Tau
            #(ak.sum(VBFSingleTau_Jet_mask, axis=-1) >= 2)  # require 2 offline Jets, cross-cleaned w Tau
            #(VBFSingleTau_mjj_mask)                          # require dijet mjj
        )

        # matching
        L1Taus_VBFSingleTau = get_selL1Taus(L1Taus, L1Tau_IsoTau45er2p1L2NN_mask, n_min_taus = n_min)
        L1Jets_VBFSingleTau = get_selL1Jets(L1Jets, L1Jet_Jet45_mask, n_min_jets = 2)
        #Jets_VBFSingleTau = Jets[VBFSingleTau_mask]
        Taus_VBFSingleTau = Taus[VBFSingleTau_mask]
        Jets_VBFSingleTau_Jet = Jets[VBFSingleTau_Jet_mask]
        GenTaus_VBFSingleTau = GenTaus[GenTau_mask]
        GenJets_VBFSingleTau = GenJets[GenJet_mask]

        #matchingGentaus_mask = matching_Gentaus(L1Taus_VBFSingleTau, Jets_VBFSingleTau, GenTaus_VBFSingleTau)
        matchingGentaus_mask = matching_Gentaus(L1Taus_VBFSingleTau, Taus_VBFSingleTau, GenTaus_VBFSingleTau)
        # at least n_min GenTau should match L1Tau/Taus
        evt_mask_matching = (ak.sum(matchingGentaus_mask, axis=-1) >= n_min)

        matchingGenjets_mask = matching_Genjets(L1Jets_VBFSingleTau, Jets_VBFSingleTau_Jet, GenJets_VBFSingleTau)
        # at least 1 GenJet should match L1Jet/Jets
        evt_mask_matching = evt_mask_matching & (ak.sum(matchingGenjets_mask, axis=-1) >= 1)

        VBFSingleTau_evt_mask = VBFSingleTau_evt_mask & evt_mask_matching
        N_den = len(events[VBFSingleTau_evt_mask])
        print(f"Number of events in denominator: {N_den}")

        # Selection of L1/Gen and Jets objects with Deeptau WP
        VBFSingleTau_mask = Tau_selection_VBFSingleTau(events, apply_DeepTau_WP = True)
        # at least 1 L1tau/ Jet/ GenTau and 2 L1jet / Jet / GenJet should pass
        #VBFSingleTau_evt_mask = (
        #    (ak.sum(VBFSingleTau_mask, axis=-1) >= n_min) & (ak.sum(L1Tau_IsoTau45er2p1L2NN_mask, axis=-1) >= n_min) & # taus
        #    (ak.sum(VBFSingleTau_Jet_mask, axis=-1) >= 2) & (ak.sum(L1Jet_Jet45_mask, axis=-1) >= 2) & # jets
        #    (VBFSingleTau_mjj_mask) & L1_DoubleJet45_Mass_Min600_mask # mjj requirement
        #)

        VBFSingleTau_evt_mask = (
            (L1_req_mask) &                                  # require L1
            (ak.sum(L1Tau_IsoTau45er2p1L2NN_mask, axis=-1) >= n_min)  # require L2NN
            #(ak.sum(VBFSingleTau_mask, axis=-1) >= n_min) &  # require 1 offline Tau
            #(ak.sum(VBFSingleTau_Jet_mask, axis=-1) >= 2) &  # require 2 offline Jets, cross-cleaned w Tau
            #(ak.sum(VBFSingleTau_Jet_mask, axis=-1) >= 2)  # require 2 offline Jets, cross-cleaned w Tau
            #(VBFSingleTau_mjj_mask)                          # require dijet mjj
        )

        # matching
        # no need to match jets, as they are already included in the denominator
        #Jets_VBFSingleTau = Jets[VBFSingleTau_mask]
        Taus_VBFSingleTau = Taus[VBFSingleTau_mask]
        #matchingGentaus_mask = matching_Gentaus(L1Taus_VBFSingleTau, Jets_VBFSingleTau, GenTaus_VBFSingleTau)
        matchingGentaus_mask = matching_Gentaus(L1Taus_VBFSingleTau, Taus_VBFSingleTau, GenTaus_VBFSingleTau)
        # at least n_min GenTau should match L1Tau/Taus
        evt_mask_matching = evt_mask_matching & (ak.sum(matchingGentaus_mask, axis=-1) >= n_min)

        #VBFSingleTau_evt_mask = VBFSingleTau_evt_mask & evt_mask_matching
        N_num = len(events[VBFSingleTau_evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num

