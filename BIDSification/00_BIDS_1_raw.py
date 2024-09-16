"""
Organize the directories and BIDSify the data.

More precisely, it generates :
- dataset description file (root)
- MEG raw data (subj/ses/meg/)
- empty-room and resting state (subj/ses/meg/)
- calibration files (subj/ses/meg/sss_config/)
- logs_exp files (subj/ses/logs)  --> legs_exp + eyelogs within the same folder

"""


# %% Imports

import copy
import os
import os.path as op
import shutil
from soundsemtools import utils as sstu
import soundsemtools as sst
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne

from mne_bids import (
    BIDSPath,
    make_report,
    make_dataset_description,
    print_dir_tree,
    write_raw_bids,
)


# %% Paths

username = "marieplgt"   # whoami within a terminal

if username == "marieplgt":
    paths = sst.utils.get_proj_paths(
        "/media/sf_sharedVM/github_repo/phd/code/meg_analysis_code/config.json"
    )


user = paths["user"]
proj_dir = paths["proj_dir"]
code_dir = paths["code_dir"]
data_dir = paths["data_dir"]

bids_dir = f"{proj_dir}/AUD2SEM_BIDS"


def dircreate(dirs_to_create):
    """
    Create directories if they don't exist.

    :param dirs_to_create: Directory, or list of directories to create
    :type dirs_to_create: strings of path
    :return: Nothing, create the directories in the filesystem

    """
    if not op.exists(dirs_to_create):
        os.makedirs(dirs_to_create)


[dircreate(x) for x in [bids_dir]]

# %% BIDSification

# Rename logs_files if needed
logs_folders_name = os.listdir(op.join(proj_dir, "Logs_Exp/"))
for name in logs_folders_name:
    full_path = os.path.join(proj_dir, "Logs_Exp/", name)
    print(full_path)
    if not name.startswith('aud2sem'):
        print(name)
        subj_num = name.split('_', 1)[0]
        print(subj_num)
        new_name = f'aud2sem_s{subj_num}'
        new_full_path = os.path.join(proj_dir, "Logs_Exp/", new_name)
        os.rename(full_path, new_full_path)
        print(f"Renamed '{full_path}' to '{new_full_path}'")
    else :
        print(f'{name} logs folder is already renamed')


# Retrieve raw_data paths
subjs_list = [name for name in os.listdir(data_dir)
                  if os.path.isdir(os.path.join(data_dir, name)) and name.startswith('aud2sem_s')]
# subj_list = sorted(os.listdir(raw_data))
print(subjs_list)
# ['aud2sem_s09', 'aud2sem_s10', 'aud2sem_s11', 'aud2sem_s12', 'aud2sem_s13', 'aud2sem_s14', 'aud2sem_s15', 
# 'aud2sem_s16', 'aud2sem_s17', 'aud2sem_s18', 'aud2sem_s19', 'aud2sem_s20', 'aud2sem_s21', 'aud2sem_s22', 
# 'aud2sem_s01', 'aud2sem_s02', 'aud2sem_s03', 'aud2sem_s04', 'aud2sem_s05', 'aud2sem_s06', 'aud2sem_s07', 'aud2sem_s08']


for subj_index, subj in enumerate(subjs_list, start=1):
    print(f"Processing subject: {subj}")

    tmp_list = ['X']
                        #   'aud2sem_s06',
                #  'aud2sem_s08', 'aud2sem_s01', 'aud2sem_s02', 'aud2sem_s09',  'aud2sem_s10','aud2sem_s11','aud2sem_s12', 'aud2sem_s13','aud2sem_s14','aud2sem_s15', 
                # 'aud2sem_s16', 'aud2sem_s17', 'aud2sem_s18', 'aud2sem_s19', 'aud2sem_s20', 'aud2sem_s21', 'aud2sem_s22','aud2sem_s03', 'aud2sem_s04', 'aud2sem_s05'
    
    if subj in tmp_list: 
        continue
    else:

        # Define the path to the subject's data
        subj_path = os.path.join(data_dir, subj)

        # Session list
        session_list = sorted(os.listdir(subj_path))
        print(session_list)

        # Retrieve logs of the current subject
        all_logs_files = sorted(glob.glob(op.join(proj_dir, "Logs_Exp/", subj, "Logs/", "*.txt")))
        print(all_logs_files)
        # Retrieve logs of the current subject
        all_eyelogs_files = sorted(glob.glob(op.join(proj_dir, "Logs_Exp/", subj, "EyeLink_logs/", "*.edf")))
        print(all_eyelogs_files)

        for sess_index, sess in enumerate(session_list, start=1):
            print(f"  {subj}, session: {sess}")

            # Define the path to the session's data
            sess_path = os.path.join(subj_path, sess)

            # Get list of runs
            tmp_run_files = sorted(glob.glob(os.path.join(sess_path, 'run*.fif')))
            if len(tmp_run_files) < 8 :
                print(f'BE CAREFUL, only {len(tmp_run_files)} runs of the {subj} sess{sess_index}  are available')

            run_files = []
            for run in tmp_run_files:
                if run.endswith('rate.fif') or run.endswith('bad.fif') or run.endswith('abort.fif'):
                    print(f"BE CAREFUL, one run is a failure one : {run}")
                else:
                    run_files.append(run)
            del run
            print(run_files, len(run_files))
                        

            # Extract trigger and soundName from Logs_files for the current session
            if sess_index == 1:  # Session 1
                logs_files = all_logs_files[:8]
                eyelogs_files = all_eyelogs_files[:8]
            if sess_index == 2:  # Session 2
                logs_files = all_logs_files[8:]
                eyelogs_files = all_eyelogs_files[8:]

            print(logs_files, len(logs_files))      
            print(eyelogs_files, len(eyelogs_files))  

            #### Integrate empty room and calibration files
            # define empty room and resting state participant's files
            empty_room_path = op.join(sess_path, "empty_room.fif")
            resting_state_path = op.join(sess_path, "resting_state.fif")

            # define all the Elekta calibration files
            crosstalk_file = op.join(sess_path, "sss_config", "ct_sparse.fif")
            fine_cal_file = op.join(sess_path, "sss_config", "sss_cal_3101_160108.dat")

            for run_index, run in enumerate(run_files, start=1):
                print(f"    Processing run file {run}...")
            
                #### Load MEG runs ------------------------------------------------
                # Load and prepare raw data
                raw = mne.io.read_raw_fif(
                    run,
                    allow_maxshield=True,
                    preload=True,
                )

                raw.crop(1.0, raw.times[-1] - 1)
                raw.drop_channels(raw.copy().pick_types(misc=True).ch_names)
                raw.info['line_freq'] = 50  # specify power line frequency as required by BIDS

                # Retrieve logs of the current run
                run_log_file = logs_files[run_index - 1]   # bc loop starts from 1 and not 0
                print(run_log_file)
                df_log = pd.read_csv(run_log_file)

                # Retrieve eyelogs of the current run
                run_eyelog_file = eyelogs_files[run_index - 1]   # bc loop starts from 1 and not 0
                print(run_eyelog_file)

                #### BIDS MEG runs ------------------------------------------------

                # Set subject, session, and run IDs using incremental indexes
                subject_id = subj.split('_')[1].replace('s','')
                # subject_id = f"{subj_index:02d}"  # Format subject ID as sub-01, sub-02, etc.
                session_id = f"{sess_index:02d}"  # Format session ID as ses-01, ses-02, etc.
                run_id = f"{run_index:02d}"  # Format run ID as run-01, run-02, etc.
                print(f' subj_id : {subject_id}, sess: {session_id}, run : {run_id}')

                task = "aud2sem"
                datatype = 'meg'


                make_dataset_description(
                    path=bids_dir,
                    name="Aud2Sem",
                    # hed_version=None, # to use HED tags ; useful ?
                    dataset_type='raw',
                    data_license=None,
                    authors=None,                   # TO FILL
                    acknowledgements=None,          # TO FILL
                    how_to_acknowledge=None,        # TO FILL
                    funding=None,                   # TO FILL
                    ethics_approvals=None,          # TO FILL
                    references_and_links=None,      # TO FILL
                    # doi=None,
                    # generated_by=None,
                    # source_datasets=None,
                    overwrite=False,
                    )

                # define the bids path for raw
                raw_bids_path = BIDSPath(
                    subject=subject_id,
                    session=session_id,
                    run=run_id,
                    datatype=datatype,
                    root=bids_dir,
                    task=task,
                    # acquisition=None,
                    # processing=None,
                    # recording=None,
                    # space=None,
                    # split=None,
                    # description=None,
                    # suffix=None,
                    # extension=None,
                    # check=True,
                )


                ## Write raw BIDSification
                write_raw_bids(
                    raw=raw,
                    bids_path=raw_bids_path,
                    empty_room=None,   # Define later
                    events=None,    # onset	duration	   trial_type	value	sample
                    event_id=None,   # Don't use the event_id param provided by mne.bids_path ; instead fill bids folder with appropriate logs_exp file
                    format="FIF",
                    allow_preload=True,   # NOT SURE IF WE SHOULD !!!
                    anonymize=None,
                    symlink=False,    # if True, only create symbolic links to preserve storage space.
                    # montage=None,
                    # acpc_aligned=False, # this flag is required to be True when the digitization data is in “mri” for intracranial data to confirm that the T1 is ACPC-aligned.
                    overwrite=True,
                )                   

                base_name = raw_bids_path.basename
                default_path = op.join(bids_dir, f'sub-{subject_id}', f'ses-{session_id}')  
                meg_path = op.join(f"{default_path}/meg")  

                # Save logs in subj/ses/logs/
                log_path = f"{default_path}/logs"
                if not os.path.exists(log_path):
                    os.mkdir(log_path)    
                #print(logs_files)
                df_log.to_csv(f"{log_path}/{base_name}_logs_exp.csv")
                eyelogs_dir_output = op.join(f"{log_path}/{base_name}_eyelogs_exp.edf")
                shutil.copy2(run_eyelog_file, eyelogs_dir_output) 

            # Save empty room, resting state and calibration files in subj/ses/meg
            sss_path = f"{meg_path}/sss_config"
            if not os.path.exists(sss_path):
                os.mkdir(sss_path)  
            
            empty_room_new_path = op.join(meg_path, f"sub-{subject_id}_ses-{session_id}_empty_room.fif")
            resting_state_new_path = op.join(meg_path, f"sub-{subject_id}_ses-{session_id}_resting_state.fif")
            crosstalk_new_path = op.join(sss_path, f"sub-{subject_id}_ses-{session_id}_ct_sparse.fif")
            fine_cal_new_path = op.join(sss_path, f"sub-{subject_id}_ses-{session_id}_sss_cal_3101_160108.dat")
            shutil.copy2(empty_room_path, empty_room_new_path) 
            shutil.copy2(resting_state_path, resting_state_new_path)
            shutil.copy2(crosstalk_file, crosstalk_new_path)
            shutil.copy2(fine_cal_file, fine_cal_new_path)

            #### Summary of methods :
                # - dataset_description.json file
                # - (optional) participants.tsv file
                # - (optional) datatype-agnostic files for (M/I)EEG data, which reads files from the *_scans.tsv file.
            make_report(bids_dir, session=sess)


#### Read the tree
print_dir_tree(bids_dir,
               # max_depth=None,    # 4
               # return_str=False,
               )
make_report(bids_dir)


## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
## At the very end, we can create a last DATA folders with complete anonymization using
## to integrate within the loop if useful
# get_anonymization_daysback(raw)
# anonymize_dataset(bids_root_in=bids_root, bids_root_out=bids_root_anon)


# %%
