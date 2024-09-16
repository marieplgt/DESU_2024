"""
preprocessing of the first Paris pilot.

1) IIR High-pass filtering
2) Low-pass: IIR
3) Notch: FIR

TODO: 1. add figures to the report
      2. better functions for filtering maybe

@author: neurogima, marieplgt
"""

# %% Import modules, packages etc

import glob
import os
import os.path as op
import re

# os.environ['QT_QPA_PLATFORM'] = 'xcb'
import matplotlib
import matplotlib.pyplot as plt
import mne
import mne.filter
import mne.preprocessing
import soundsemtools as sst
from mne.report import Report
from mne.viz import plot_filter


matplotlib.use("Qt5Agg")


from mne_bids import (
    BIDSPath,
    find_matching_paths,
    get_entity_vals,
    make_report,
    print_dir_tree,
    read_raw_bids,
    write_raw_bids,
    mark_channels
)

# os.environ['QT_QPA_PLATFORM'] = 'xcb'


# %% set default paths


def dircreate(dirs_to_create):
    """
    Create directories if they don't exist.

    :param dirs_to_create: Directory, or list of directories to create
    :type dirs_to_create: strings of path
    :return: Nothing, create the directories in the filesystem

    """
    if not op.exists(dirs_to_create):
        os.makedirs(dirs_to_create)

# %% iir creation

def construct_high_and_low_pass_filters(
    sr_target: float, f_pass_h: float, f_pass_l: float
) -> tuple:
    """
    Construct high-pass and low-pass filters using the specified parameters.

    Parameters:
        sr_target (float): The target sampling frequency of the data.
        f_pass_h (float): The high-pass frequency of the filter.
        f_pass_l (float): The low-pass frequency of the filter.

    Returns:
        tuple: The high-pass and low-pass filter parameters.
    """
    # High-pass filter
    iir_params_h = dict(order=4, ftype="butter", output="sos")
    iir_params_h = mne.filter.construct_iir_filter(
        iir_params_h,
        sfreq=sr_target,
        f_pass=f_pass_h,
        btype="highpass",
        phase="zero-double",
    )

    # Low-pass filter
    iir_params_l = dict(order=4, ftype="butter", output="sos")
    iir_params_l = mne.filter.construct_iir_filter(
        iir_params_l,
        sfreq=sr_target,
        f_pass=f_pass_l,
        btype="lowpass",
        phase="zero-double",
    )

    return iir_params_h, iir_params_l

# %% Run the file as script

if __name__ == "__main__":

    username = "marieplgt"   # whoami within a terminal

    if username == "marieplgt":
        paths = sst.utils.get_proj_paths(
            "/media/sf_sharedVM/github_repo/phd/code/meg_analysis_code/config.json"
        )
    # if username == "neurogima":
    #     paths = sst.utils.get_proj_paths(
    #         "/home/neurogima/Code/SoundBrainSem/Aud2Sem_MEG/config.json"
    #     )
    #     bids_dir = "/run/media/neurogima/DATA/Aud2Sem_MEG_BIDS"

    proj_dir = paths["proj_dir"]
    bids_root = f"{proj_dir}/AUD2SEM_BIDS/"

    # define the orig root derivative path
    deriv_proc_dir = op.join(bids_root, "derivatives", "preprocess", "ICA_removal")

    subjs_list = [name for name in os.listdir(deriv_proc_dir)
                    if os.path.isdir(os.path.join(deriv_proc_dir, name)) and name.startswith('sub-')]
    # subj_list = sorted(os.listdir(raw_data))
    print(subjs_list)

    for subj_index, subj in enumerate(subjs_list, start=0):

        # Define the path to the subject's data
        subj_path = os.path.join(deriv_proc_dir, subj)
        print(subj_path)

        # Session list
        session_list = [name for name in os.listdir(subj_path)
                    if os.path.isdir(os.path.join(subj_path, name)) and name.startswith('ses-')]
        print(session_list)
        
        for sess_index, sess in enumerate(session_list, start=1):
            print(f"  {subj}, session: {sess}")

            # Define the path to the session's data
            sess_path = os.path.join(subj_path, sess, "meg")
            print(sess_path)

            # Get list of runs
            run_files = sorted(glob.glob(os.path.join(sess_path, '*ICAremoval_meg.fif')))
            if len(run_files) < 8 :
                print(f'BE CAREFUL, only {len(run_files)} runs of the {subj}_{sess} are available')
            
        
            for run_index, run in enumerate(run_files, start=1):
                print(f"    Processing run file {run}...")

                # Set subject, session, and run IDs using incremental indexes
                subject_id = subj.split('-')[1]
                session_id = sess.split('-')[1]  # Format session ID as ses-01, ses-02, etc.

                run_id = re.search(r'run-\d+', run).group().split('-')[1]  # Format run ID as run-01, run-02, etc.
                print(f' subj_id : {subject_id}, sess: {session_id}, run : {run_id}')

                task = "aud2sem"
                datatype = 'meg'
            
            
                input_bids_path = BIDSPath(
                        root=deriv_proc_dir,
                        subject=subject_id,
                        session=session_id,
                        datatype=datatype,
                        run=run_id,
                        task=task,
                        processing="ICAremoval",
                        suffix='meg',
                        extension=".fif"
                        )
                # print(input_bids_path)


                raw = read_raw_bids(bids_path=input_bids_path, verbose=False,     
                    )                
                
                print(raw.info)

                raw.load_data()
                

                # ===== Construct the high and low pass filters =====
                sr_orig = 1000
                sr_target = 1000
                f_pass_h = 0.05
                f_pass_l = 70

                iir_params_h, iir_params_l = (
                    construct_high_and_low_pass_filters(sr_target, f_pass_h, f_pass_l)
                )

                # ===== filter data high and low =====
                meg_picks = mne.pick_types(raw.info, meg=True)

                # High pass
                raw1 = raw.copy()
                raw1.filter(
                    l_freq=0.05,
                    h_freq=None,
                    picks=meg_picks,
                    n_jobs=None,
                    method="iir",
                    iir_params=iir_params_h,
                )
                print(raw1.info)   # filtering applied but info min seems to be 0.1

                # Low pass
                raw2 = raw1.copy()
                raw2.filter(
                    l_freq=None,
                    h_freq=70,
                    picks=meg_picks,
                    n_jobs=None,
                    method="iir",
                    iir_params=iir_params_l,
                )

                # ===== Notch Filter =====

                raw3 = raw2.copy()
                raw3.notch_filter(
                    freqs=50,
                    picks=meg_picks,
                    notch_widths=None,
                    method="iir",
                    phase="zero-double",
                    )

                # # Notch: FIR (iir bc mem issue) using the raw.method
                # tmp = raw.get_data()  # Noth filter requires array object
                # tmp_filtered = mne.filter.notch_filter(
                #     tmp,
                #     sr_target,
                #     50,
                #     notch_widths=None,  # If None, freqs / 200 is used
                #     n_jobs=None,
                #     method="iir",
                #     phase="zero-double",
                # )

                # raw._data = tmp_filtered


                # ===== Save data filtered =====
                deriv_proc_dir_output = op.join(bids_root, "derivatives", "preprocess", "filtering")

                output_bids_path = BIDSPath(
                        root=deriv_proc_dir_output,
                        subject=subject_id,
                        session=session_id,
                        datatype=datatype,
                        run=run_id,
                        task=task,
                        processing="Filt",
                        suffix='meg',
                        extension=".fif"
                        )

                write_raw_bids(
                    raw3,
                    output_bids_path,
                    overwrite=True,
                    allow_preload=True,
                    format='FIF'
                    )