#!/usr/bin/env python
import law
import os
import shutil

from law_customizations import Task, HTCondorWorkflow
from helpers import files_from_path, load_cfg_file, hadd_anatuple

class SaveEventsDenRate(Task, HTCondorWorkflow, law.LocalWorkflow):
    '''
    Produce root file where Events passing denominator selection (run and lumiSection) are saved 
    '''
    config = load_cfg_file()
    # RefRun = int(config['RUNINFO']['ref_run'])
    RefRun = list(filter(None, (x.strip() for x in config['RUNINFO']['ref_run'].splitlines())))
    # LumiSectionsRange = [int(config['RUNINFO']['LumiSectionsRange_low']), int(config['RUNINFO']['LumiSectionsRange_up'])]
    LumiSectionsRange =  list(filter(None, (x.strip() for x in config['RUNINFO']['LumiSectionsRange'].splitlines())))
    number_of_ephemeral_folder = int(config['DATA']['number_of_ephemeral_folder'])
    runArea = config['RUNINFO']['Area']
    EphemeralFolder = config['DATA']['SamplesPath']
    #output_path = os.path.join(config['DATA']['RateDenPath'], f'Run_{RefRun}')
    tmp_folder = config['DATA']['tmpPath']
    
    def __init__(self, *args, **kwargs):
        super(SaveEventsDenRate, self).__init__(*args, **kwargs)
        tag = f'Run_{self.RefRun[0]}' if len(self.RefRun) == 1 else f'Run_{"_".join(self.RefRun)}'
        self.output_path = os.path.join(self.config['DATA']['RateDenPath'], tag)

    def create_branch_map(self):
        os.makedirs(self.output_path, exist_ok=True)
        branches = {}
        for i in range(self.number_of_ephemeral_folder):
            branches[i] = os.path.join(self.output_path, f'EphemeralHLTPhysics{i}_Run{self.runArea}.root')
        return branches
    
    def output(self):
        path = os.path.join(self.branch_data)
        return law.LocalFileTarget(path)

    def run(self):
        print(f"-----------------------------------------------Processing folder EphemeralHLTPhysics{self.branch}-----------------------------------------------")
        EphemeralFolderName = os.path.join(self.EphemeralFolder, f"EphemeralHLTPhysics{self.branch}_Run{self.runArea}/")
        output_root_file = os.path.join(self.output_path, f'EphemeralHLTPhysics{self.branch}_Run{self.runArea}.root')
        output_tmp_folder = os.path.join(self.tmp_folder, f"EphemeralHLTPhysics{self.branch}_Run{self.runArea}")
        from HLTClass.dataset import Dataset
        
        if os.path.exists(output_tmp_folder):
            print(f'A tmp folder which store tmp root files exist already: being deleted')
            shutil.rmtree(output_tmp_folder)
        os.makedirs(output_tmp_folder)

        # Produce tmp files
        FileNameList = files_from_path(EphemeralFolderName)
        for FileName in FileNameList:
            # this tmp folder fuck up the code
            if FileName == os.path.join(self.EphemeralFolder, 'EphemeralHLTPhysics1_Run2023C', 'nano.tmp'):
                continue
            if not FileName.endswith(".root"):
                continue
            print(f"Producing tmp file for {os.path.basename(FileName)}:")
            output_tmp_file = os.path.join(output_tmp_folder, os.path.basename(FileName))
            eph_dataset = Dataset(FileName)
            eph_dataset.Save_Event_Nden_Rate(output_tmp_file, self.RefRun, self.LumiSectionsRange)

        # Hadd the tmp files to a single root file
        hadd_anatuple(output_tmp_folder, output_root_file)
        shutil.rmtree(output_tmp_folder)
        return
    
    
class ProduceRateFiles(Task, HTCondorWorkflow, law.LocalWorkflow):
    '''
    Produce json files where number of events which pass denominator and numerator criteria are saved from each ephemerals files
    '''
    config = load_cfg_file()
    #RefRun = int(config['RUNINFO']['ref_run'])
    RefRun = config['RUNINFO']['ref_run']
    number_of_ephemeral_folder = int(config['DATA']['number_of_ephemeral_folder'])
    HLT_name = config['HLT']['HLTname']
    runArea = config['RUNINFO']['Area']
    input_root = os.path.join(config['DATA']['RateDenPath'], f'Run_{RefRun}')
    output_path = os.path.join(config['DATA']['result_rate'], f'result_{RefRun}')

    PNet_mode = config['OPT']['PNet_mode']
    if PNet_mode == 'false':
        PNetMode = False
        output_path = os.path.join(output_path, HLT_name, HLT_name)
    else:
        PNetMode = True
        PNetparam = [float(config['OPT']['PNet_t1']), float(config['OPT']['PNet_t2']), float(config['OPT']['PNet_t3'])]
        output_path = os.path.join(output_path, HLT_name, f'PNetTresh_{config["OPT"]["PNet_t1"]}_{config["OPT"]["PNet_t2"]}_{config["OPT"]["PNet_t3"]}')

    # requires SaveEventsDenRate for Den events
    def workflow_requires(self):
        return { "Counter": SaveEventsDenRate.req(self, branch=self.branch) }

    def requires(self):
        return self.workflow_requires()
    
    def create_branch_map(self):
        os.makedirs(self.output_path, exist_ok=True)
        branches = {}
        for i in range(self.number_of_ephemeral_folder):
            branches[i] = os.path.join(self.output_path, f'EphemeralHLTPhysics{i}_Run{self.runArea}.json')
        return branches
    
    def output(self):
        path = os.path.join(self.branch_data)
        return law.LocalFileTarget(path)

    def run(self):
        print(f"-----------------------------------------------Processing folder EphemeralHLTPhysics{self.branch}_Run{self.runArea}.root-----------------------------------------------")

        input_root_file = os.path.join(self.input_root, f'EphemeralHLTPhysics{self.branch}_Run{self.runArea}.root')

        event_counter = {}
        event_counter[input_root_file] = {}
        print(f"For {os.path.basename(input_root_file)}:")

        HLT_config = ['HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1', 'HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3', 'HLT_DoubleTauOrSingleTau', 'HLT_VBF_DiPFJet45_Mjj650_MediumDeepTauPFTauHPS45_L2NN_eta2p1']
        if self.HLT_name not in HLT_config:
            print(f'HLT name {self.HLT_name} not implemented in the code')
            raise
        
        if self.HLT_name == 'HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1':
            from HLTClass.DiTauDataset import DiTauDataset
            
            eph_dataset = DiTauDataset(input_root_file)
            if self.PNetMode:
                N_den_i, N_num_i = eph_dataset.get_Nnum_Nden_DiTauPNet(self.PNetparam)
            else:
                N_den_i, N_num_i = eph_dataset.get_Nnum_Nden_HLTDoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1()

        if self.HLT_name == 'HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3':
            from HLTClass.SingleTauDataset import SingleTauDataset
            
            eph_dataset = SingleTauDataset(input_root_file)
            if self.PNetMode:
                N_den_i, N_num_i = eph_dataset.get_Nnum_Nden_SingleTauPNet(self.PNetparam)
            else:
                N_den_i, N_num_i = eph_dataset.get_Nnum_Nden_HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3()

        if self.HLT_name == 'HLT_DoubleTauOrSingleTau':
            from HLTClass.DoubleORSingleTauDataset import DoubleORSingleTauDataset
            
            eph_dataset = DoubleORSingleTauDataset(input_root_file)
            if self.PNetMode:
                N_den_i, N_num_i = eph_dataset.get_Nnum_Nden_DoubleORSinglePNet(self.PNetparam)
            else:
                N_den_i, N_num_i = eph_dataset.get_Nnum_Nden_HLT_DoubleORSingleDeepTau()

        if self.HLT_name == 'HLT_VBF_DiPFJet45_Mjj650_MediumDeepTauPFTauHPS45_L2NN_eta2p1':
            from HLTClass.VBFSingleTauDataset import VBFSingleTauDataset

            eph_dataset = VBFSingleTauDataset(input_root_file)
            if self.PNetMode:
                N_den_i, N_num_i = eph_dataset.get_Nnum_Nden_VBFSingleTauPNet(self.PNetparam)
            else:
                N_den_i, N_num_i = eph_dataset.get_Nnum_Nden_HLTVBF_DiPFJet45_Mjj650_MediumDeepTauPFTauHPS45_L2NN_eta2p1()
                
        event_counter[input_root_file]['N_den'] = N_den_i
        event_counter[input_root_file]['N_num'] = N_num_i

        self.output().dump(event_counter)
