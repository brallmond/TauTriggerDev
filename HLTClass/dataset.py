import uproot
import collections
import six
import numpy as np
import awkward as ak
from helpers import delta_r

def get_L1Taus(events):
    L1taus_dict = {"pt": events["L1Tau_pt"].compute(), "eta": events["L1Tau_eta"].compute(), "phi": events["L1Tau_phi"].compute(), "Iso": events["L1Tau_hwIso"].compute()}
    L1Taus = ak.zip(L1taus_dict)
    return L1Taus

def get_L1Jets(events):
    L1jets_dict = {"pt": events["L1Jet_pt"].compute(), "eta": events["L1Jet_eta"].compute(), "phi": events["L1Jet_phi"].compute()}
    L1Jets = ak.zip(L1jets_dict)
    return L1Jets

def get_Taus(events):
    taus_dict = {"pt": events["Tau_pt"].compute(), "eta": events["Tau_eta"].compute(), "phi": events["Tau_phi"].compute(), "deepTauVSjet": events["Tau_deepTauVSjet"].compute()}
    Taus = ak.zip(taus_dict)
    return Taus

def get_Jets(events):
    jets_dict = {"pt": events["Jet_PNet_ptcorr"].compute()*events["Jet_pt"].compute(), "eta": events["Jet_eta"].compute(), "phi": events["Jet_phi"].compute(), "probtauhm": events['Jet_PNet_probtauhm'].compute(), "probtauhp": events['Jet_PNet_probtauhp'].compute()}
    Jets = ak.zip(jets_dict)
    return Jets

def get_GenTaus(events):
    gentaus_dict = {"pt": events["GenLepton_pt"].compute(), "eta": events["GenLepton_eta"].compute(), "phi": events["GenLepton_phi"].compute(), "nChargedHad": events["GenLepton_nChargedHad"].compute(), "nNeutralHad": events["GenLepton_nNeutralHad"].compute(), "DecayMode": events["GenLepton_DecayMode"].compute(), "charge": events["GenLepton_charge"].compute()}
    GenTaus = ak.zip(gentaus_dict)
    return GenTaus

def get_GenJets(events):
    genjets_dict = {"pt": events["GenJet_pt"].compute(), "eta": events["GenJet_eta"].compute(), "phi": events["GenJet_phi"].compute()}
    GenJets = ak.zip(genjets_dict)
    return GenJets

def hGenTau_selection(events):
    # return mask for GenLepton for hadronic GenTau passing minimal selection
    hGenTau_mask = (events['GenLepton_pt'].compute() >= 20) & (np.abs(events['GenLepton_eta'].compute()) <= 2.3) & (events['GenLepton_kind'].compute() == 5)
    return hGenTau_mask

def hGenJet_selection(events):
    # return mask for GenJet passing minimal selection
    hGenJet_mask = (events['GenJet_pt'].compute() >= 20)
    return hGenJet_mask

def matching_L1Taus_obj(L1Taus, Obj, dR_matching_min = 0.5):
    obj_inpair, l1taus_inpair = ak.unzip(ak.cartesian([Obj, L1Taus], nested=True))
    dR_obj_l1taus = delta_r(obj_inpair, l1taus_inpair)
    mask_obj_l1taus = (dR_obj_l1taus < dR_matching_min)
    
    mask = ak.any(mask_obj_l1taus, axis=-1)
    return mask

def matching_L1Jets_obj(L1Jets, Obj, dR_matching_min = 0.5):
    obj_inpair, l1jets_inpair = ak.unzip(ak.cartesian([Obj, L1Jets], nested=True))
    dR_obj_l1jets = delta_r(obj_inpair, l1jets_inpair)
    mask_obj_l1jets = (dR_obj_l1jets < dR_matching_min)

    mask = ak.any(mask_obj_l1jets, axis=-1)
    return mask

def matching_Gentaus(L1Taus, Taus, GenTaus, dR_matching_min = 0.5):
    gentaus_inpair, l1taus_inpair = ak.unzip(ak.cartesian([GenTaus, L1Taus], nested=True))
    dR_gentaus_l1taus = delta_r(gentaus_inpair, l1taus_inpair)
    mask_gentaus_l1taus = (dR_gentaus_l1taus < dR_matching_min) 

    gentaus_inpair, taus_inpair = ak.unzip(ak.cartesian([GenTaus, Taus], nested=True))
    dR_gentaus_taus = delta_r(gentaus_inpair, taus_inpair)
    mask_gentaus_taus = (dR_gentaus_taus < dR_matching_min)

    matching_mask = ak.any(mask_gentaus_l1taus, axis=-1) & ak.any(mask_gentaus_taus, axis=-1)  # Gentau should match l1Taus and Taus
    return matching_mask

def matching_Genjets(L1Jets, Jets, GenJets, dR_matching_min = 0.5):
    genjets_inpair, l1jets_inpair = ak.unzip(ak.cartesian([GenJets, L1Jets], nested=True))
    dR_genjets_l1jets = delta_r(genjets_inpair, l1jets_inpair)
    mask_genjets_l1jets = (dR_genjets_l1jets < dR_matching_min)

    genjets_inpair, jets_inpair = ak.unzip(ak.cartesian([GenJets, Jets], nested=True))
    dR_genjets_jets = delta_r(genjets_inpair, jets_inpair)
    mask_genjets_jets = (dR_genjets_jets < dR_matching_min)

    matching_mask = ak.any(mask_genjets_l1jets, axis=-1) & ak.any(mask_genjets_jets, axis=-1)  # Genjet should match l1Jets and Jets
    return matching_mask

def compute_PNet_charge_prob(probTauP, probTauM):
    return np.abs(np.ones(len(probTauP))*0.5 - probTauP/(probTauP + probTauM))
    
def iterable(arg):
    return (
        isinstance(arg, collections.abc.Iterable)
        and not isinstance(arg, six.string_types)
    )



class Dataset:
    def __init__(self, fileName):
        self.fileName = fileName

    @staticmethod
    def compute_decay_mode(nChargedHad, nNeutralHad):
        return (nChargedHad - 1) * 5 + nNeutralHad

    def __define_tree_expression(self):
        treeName = 'Events'
        if iterable(self.fileName):
            tree_path = []
            for file in self.fileName:
                tree_path.append(file + ":" + treeName)
        else:
            tree_path = self.fileName + ":" + treeName
        return tree_path

    def get_events(self):
        tree_path = self.__define_tree_expression()
        events = uproot.dask(tree_path, library='ak')
        return events

    def get_GenLepton(self, events):
        from GenLeptonCode.helpers import get_GenLepton_rdf
        # Add GenLepton information to RdataFrame
        rdf = ak.to_rdataframe({'GenPart_pt': events['GenPart_pt'].compute(),
                                'GenPart_eta': events['GenPart_eta'].compute(),
                                'GenPart_phi': events['GenPart_phi'].compute(),
                                'GenPart_mass': events['GenPart_mass'].compute(),
                                'GenPart_genPartIdxMother': events['GenPart_genPartIdxMother'].compute(),
                                'GenPart_pdgId': events['GenPart_pdgId'].compute(),
                                'GenPart_statusFlags': events['GenPart_statusFlags'].compute()})

        rdf = get_GenLepton_rdf(rdf)
        GenLepton = {}
        GenLepton['pt'] = ak.from_rdataframe(rdf, 'GenLepton_pt')
        GenLepton['eta'] = ak.from_rdataframe(rdf, 'GenLepton_eta')
        GenLepton['phi'] = ak.from_rdataframe(rdf, 'GenLepton_phi')
        GenLepton['kind'] = ak.from_rdataframe(rdf, 'GenLepton_kind')
        GenLepton['nChargedHad'] = ak.from_rdataframe(rdf, 'GenLepton_nChargedHad')
        GenLepton['nNeutralHad'] = ak.from_rdataframe(rdf, 'GenLepton_nNeutralHad')
        GenLepton['DecayMode'] = self.compute_decay_mode(GenLepton['nChargedHad'], GenLepton['nNeutralHad'])
        GenLepton['charge'] = ak.from_rdataframe(rdf, 'GenLepton_charge')
        return GenLepton

# ------------------------------ functions for ComputeRate ---------------------------------------------------------------
    def Save_Event_Nden_Rate(self, tmp_file, run, lumiSections_range):
        ''' 
        Save only needed informations (for numerator cuts) of events passing denominator cuts 
        '''

        conditions = []
        for r, ls in zip(run, lumiSections_range):
            condition = f"((events['run'] == {r}) & "
            ls_conditions = []
            ls = ls.split(", ")
            for ils in range(0, len(ls), 2):
                ls_conditions.append(f"((events['luminosityBlock'] >= {ls[ils]}) & (events['luminosityBlock'] <= {ls[ils + 1]}))")
            condition += f"({' | '.join(ls_conditions)}))"
            conditions.append(condition)

        events = self.get_events()
        N_events = len(events)
        print(f"Number of events: {N_events}")

        events = eval(f"events[{' | '.join(conditions)}]")
        print(f"Number of selected events : {len(events)}")

        events = self.get_events()
        N_events = len(events)
        print(f"Number of events: {N_events}")
        #events = events[events['run'] == run]
        #print(f"Number of events belonging to run {run}: {len(events)}")
        #events = events[(events["luminosityBlock"] >= lumiSections_range[0]) & (events["luminosityBlock"] <= lumiSections_range[1])]
        #print(f"Number of event in LumiSections range {lumiSections_range}: {len(events)}")
        Nevents_L1 = len(events[events['L1_DoubleIsoTau34er2p1'].compute()])
        print(f"   - events passing L1_DoubleIsoTau34er2p1 flag: {Nevents_L1}")
        Nevents_HLT = len(events[events['HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1'].compute()])
        print(f"   - events passing HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1 flag: {Nevents_HLT}")
        Nevents_HLTVBFDeepTau1 = len(events[events['HLT_VBF_DiPFJet45_Mjj650_MediumDeepTauPFTauHPS45_L2NN_eta2p1'].compute()])
        print(f"   - events passing HLT_VBF_DiPFJet45_Mjj650_MediumDeepTauPFTauHPS45_L2NN_eta2p flag: {Nevents_HLTVBFDeepTau1}")
        Nevents_HLTVBFPNet1 = len(events[events['HLT_VBF_DiPFJet45_Mjj650_PNetTauhPFJet45_L2NN_eta2p3'].compute()])
        print(f"   - events passing HLT_VBF_DiPFJet45_Mjj650_PNetTauhPFJet45_L2NN_eta2p3 flag: {Nevents_HLTVBFPNet1}")
        Nevents_HLTVBFDeepTau2 = len(events[events['HLT_VBF_DoubleMediumDeepTauPFTauHPS20_eta2p1'].compute()])
        print(f"   - events passing HLT_VBF_DoubleMediumDeepTauPFTauHPS20_eta2p1 flag: {Nevents_HLTVBFDeepTau2}")
        Nevents_HLTVBFPNet2 = len(events[events['HLT_VBF_DoublePNetTauhPFJet20_eta2p2'].compute()])
        print(f"   - events passing HLT_VBF_DoublePNetTauhPFJet20_eta2p2 flag: {Nevents_HLTVBFPNet2}")
 
        # list of all info you need to save
        saved_info_events = ['event',
                             'luminosityBlock',
                             'run',
                             'nPFPrimaryVertex',
                             'nPFSecondaryVertex',
                             'L1_DoubleIsoTau34er2p1',
                             'L1_DoubleIsoTau26er2p1_Jet55_RmOvlp_dR0p5',
                             'L1_DoubleJet45_Mass_Min600_IsoTau45er2p1_RmOvlp_dR0p5',
                             'L1_DoubleJet_110_35_DoubleJet35_Mass_Min800',
                             'L1_Mu18er2p1_Tau24er2p1',
                             'L1_LooseIsoEG22er2p1_IsoTau26er2p1_dR_Min0p3',
                             'HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1',
                             'HLT_DoublePNetTauhPFJet30_Medium_L2NN_eta2p3',
                             'HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1',
                             'HLT_SinglePNetTauhPFJet130_Loose_L2NN_eta2p3',
                             'HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1',
                             'HLT_IsoMu20_eta2p1_PNetTauhPFJet27_Loose_eta2p3_CrossL1',
                             'HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1',
                             'HLT_Ele24_eta2p1_WPTight_Gsf_PNetTauhPFJet30_Loose_eta2p3_CrossL1',
                             'HLT_VBF_DiPFJet45_Mjj650_MediumDeepTauPFTauHPS45_L2NN_eta2p1',
                             'HLT_VBF_DiPFJet45_Mjj650_PNetTauhPFJet45_L2NN_eta2p3',
                             'HLT_VBF_DoubleMediumDeepTauPFTauHPS20_eta2p1',
                             'HLT_VBF_DoublePNetTauhPFJet20_eta2p2',
                             'Tau_pt',
                             'Tau_eta',
                             'Tau_phi',
                             'Tau_deepTauVSjet',
                             'L1Tau_pt',
                             'L1Tau_eta',
                             'L1Tau_phi',
                             'L1Tau_hwPt',
                             'L1Tau_hwEta',
                             'L1Tau_hwIso',
                             'L1Tau_l2Tag',
                             'Jet_PNet_probtauhm',
                             'Jet_PNet_probtauhp',
                             'Jet_PNet_ptcorr',
                             'Jet_pt',
                             'Jet_eta',
                             'Jet_phi',
                             'L1Jet_pt',
                             'L1Jet_eta',
                             'L1Jet_phi'] 
        
        if len(events)!= 0:
            print('Saving info in tmp file')
            lst = {}
            for element in saved_info_events:
                lst[element] = events[element].compute()
            #saving also initial number of events in the file    
            lst['Nevents_init'] = np.ones(len(events))*N_events

            with uproot.create(tmp_file, compression=uproot.ZLIB(4)) as file:
                file["Events"] = lst
        else:
            print('No events to save')
        return

# ------------------------------ functions for ComputeEff ---------------------------------------------------------------
    def Save_Event_Nden_Eff(self, events, GenLepton, evt_mask, tmp_file):

        saved_info_events = ['event',
                             'nPFPrimaryVertex',
                             'nPFSecondaryVertex',
                             'Tau_pt', 
                             'Tau_eta', 
                             'Tau_phi', 
                             'Tau_deepTauVSjet', 
                             'L1_DoubleIsoTau34er2p1',
                             'L1_DoubleJet45_Mass_Min600_IsoTau45er2p1_RmOvlp_dR0p5',
                             'L1_DoubleJet_110_35_DoubleJet35_Mass_Min800',
                             'HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1',
                             'HLT_DoublePNetTauhPFJet30_Medium_L2NN_eta2p3',
                             'HLT_VBF_DiPFJet45_Mjj650_MediumDeepTauPFTauHPS45_L2NN_eta2p1',
                             'HLT_VBF_DiPFJet45_Mjj650_PNetTauhPFJet45_L2NN_eta2p3',
                             'HLT_VBF_DoubleMediumDeepTauPFTauHPS20_eta2p1',
                             'HLT_VBF_DoublePNetTauhPFJet20_eta2p2',
                             'L1Tau_pt', 
                             'L1Tau_eta', 
                             'L1Tau_phi', 
                             'L1Tau_hwPt', 
                             'L1Tau_hwEta', 
                             'L1Tau_hwIso', 
                             'L1Tau_l2Tag',
                             'Jet_PNet_probtauhm',
                             'Jet_PNet_probtauhp',
                             'Jet_PNet_ptcorr',
                             'Jet_pt',
                             'Jet_eta',
                             'Jet_phi',
                             'L1Jet_pt',
                             'L1Jet_eta',
                             'L1Jet_phi',
                             'GenJet_pt',
                             'GenJet_eta',
                             'GenJet_phi']
        
        saved_info_GenLepton = ['pt', 
                                'eta', 
                                'phi',
                                'nChargedHad',
                                'nNeutralHad',
                                'DecayMode', 
                                'charge',
                                'kind']

        lst = {}
        for element in saved_info_events:
            lst[element] = (events[element].compute())[evt_mask]

        for element in saved_info_GenLepton:
            lst['GenLepton_' + element] = (GenLepton[element])[evt_mask]

        with uproot.create(tmp_file, compression=uproot.ZLIB(4)) as file:
            file["Events"] = lst

        return
    
    def save_info(self, events_Den, events_Num, Tau_Den, Tau_Num, out_file):
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

        lst_Num = {}
        lst_Num['Tau_pt'] = Tau_Num.pt
        lst_Num['Tau_eta'] = Tau_Num.eta
        lst_Num['Tau_phi'] = Tau_Num.phi
        lst_Num['Tau_nChargedHad'] = Tau_Num.nChargedHad
        lst_Num['Tau_nNeutralHad'] = Tau_Num.nNeutralHad
        lst_Num['Tau_DecayMode'] = Tau_Num.DecayMode
        lst_Num['Tau_charge'] = Tau_Num.charge

        with uproot.create(out_file, compression=uproot.ZLIB(4)) as file:
            file["eventsDen"] = lst_evt_Den
            file["TausDen"] = lst_Den
            file["eventsNum"] = lst_evt_Num
            file["TausNum"] = lst_Num
        return 
