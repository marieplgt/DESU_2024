"""
preprocessing of the first Paris pilot.

Better code with automatic bidifcatoin will follow

1. load MaxFilter data
2. check ECG and ECG artifacts
3. filtering
4. fit ICA
5. automatic rejection
6. apply ICA
7. save data and report

channels of the Elekta in Paris:
--------------------------------

    Analog outputs of the Eye Tracker:
    ----------------------------------
        MISC007 = X gaze position
        MISC008 = Y gaze position
        MISC009 = pupil size

    Electrodes:
    -----------
        BIO001 = HEOG
        BIO002 = VEOG
        BIO003 = ECG

@author: neurogima, marieplgt
"""


# %% Import modules, packages etc

import glob
import os
import os.path as op
import re

import mne
import mne.filter
import mne.preprocessing
import soundsemtools as sst
from mne.report import Report

import matplotlib
matplotlib.use('Agg')  # ('Qt5Agg') to show in windows

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


# %% Load maxfiltered data


# def load_maxfiltered_data(run_file):
#     """
#     Load maxfiltered data from a specified run number.

#     :param run_number: The number of the run to load
#     :type run_number: int
#     :param runs_files_paths: List of paths to maxfiltered runs
#     :type runs_files_paths: list of strings
#     :return: The raw data from the specified run

#     """
#     raw = mne.io.read_raw_fif(
#         run_file,
#         allow_maxshield=True,
#         preload=True,
#     )

#     return raw


# %% Check EOG artifacts


def check_eog_artifacts(raw):
    """
    Check for EOG artifacts in the raw data.

    :param raw: The raw data
    :type raw: mne.io.Raw
    :return: EOG events and evoked data

    """
    eog_events = mne.preprocessing.find_eog_events(
        raw, ch_name=["BIO001", "BIO002"]
    )
    eog_evoked = mne.preprocessing.create_eog_epochs(
        raw, ch_name=["BIO001", "BIO002"], baseline=(-0.5, -0.2)
    )

    return eog_events, eog_evoked


# %% Check ECG artifacts


def check_ecg_artifacts(raw):
    """
    Check for ECG artifacts in the raw data.

    :param raw: The raw data
    :type raw: mne.io.Raw
    :return: ECG events and evoked data

    """
    ecg_events = mne.preprocessing.find_ecg_events(raw, ch_name="BIO003")
    ecg_evoked = mne.preprocessing.create_ecg_epochs(
        raw
        )

    return ecg_events, ecg_evoked


# %% Filtering to remove slow drift


def filter_data(raw):
    """
    Filter the raw data to remove slow drift.

    :param raw: The raw data
    :type raw: mne.io.Raw
    :return: Filtered raw data

    """
    filt_raw = raw.copy().filter(l_freq=0.05, h_freq=None)  # lowering to 0.05

    return filt_raw


# %% Set up and fit the ICA
# WIP: check the number of components here should be determined by the
# maxfilter output because it change the ranks of the data, but not sure


def fit_ica(filt_raw):
    """
    Set up and fit the ICA.

    :param raw: The filt raw data
    :type raw: mne.io.Raw
    :return: Fitted ICA object

    """
    ica_picard = mne.preprocessing.ICA(
        n_components=25, random_state=97, method="picard", max_iter="auto"
    )

    ica_picard.fit(filt_raw)

    return ica_picard


# %% Look at ICA
# It seems that Maxfilter already cleaned a lot of artifacts,
# for example seems very good with the heartbeat. ICA appear to be not that
# effective or useful like before on continuos data

def check_ica(ica_picard, filt_raw):
    """
    Look at the ICA components and sources.

    :param ica_picard: Fitted ICA object
    :type ica_picard: mne.preprocessing.ICA
    :param raw: The raw data
    :type raw: mne.io.Raw
    :return: List of ICA component figures

    """
    # Variance explained by
    explained_var_ratio = ica_picard.get_explained_variance_ratio(filt_raw)
    for channel_type, ratio in explained_var_ratio.items():
        print(
            f"Fraction of {channel_type} variance explained by all components:"
            f"{ratio}"
        )

    # look at the time-series
    ica_picard.plot_sources(filt_raw, show_scrollbars=False, show=False)

    # look at the components
    comp_ica_figs = []

    ica_picard.plot_components(show=False)
    comp_ica_figs.extend(ica_picard.plot_components(show=False))

    return comp_ica_figs


# %% Automatic ICA components selection


def automatic_ica_components_selection(
    ica_picard, filt_raw, artifacts=["EOG", "ECG"]
):
    """
    Automatically select ICA components based on EOG and ECG patterns.

    :param ica_picard: Fitted ICA object
    :type ica_picard: mne.preprocessing.ICA
    :param raw: The raw data
    :type raw: mne.io.Raw
    :param artifacts: List of automatic steps to run (default: ["EOG", "ECG"])
    :type artifacts: list
    :return: List of excluded component indices and scores

    """
    eog_scores = None
    ecg_scores = None
    ica_picard.exclude = []

    # EOG
    if "EOG" in artifacts:
        # find which ICs match the EOG pattern
        eog_indices, eog_scores = ica_picard.find_bads_eog(
            filt_raw, ch_name=["BIO001", "BIO002"]
        )
    else:
        eog_scores = []
        eog_indices = []

    # ECG
    if "ECG" in artifacts:
        # find which ICs match the ECG pattern
        ecg_indices, ecg_scores = ica_picard.find_bads_ecg(
            filt_raw, ch_name="BIO003"
        )
    else:
        ecg_scores = []
        ecg_indices = []

    # mark components to exclude

    # combine EOG and ECG indices in the final exclude list
    ica_picard.exclude = eog_indices + ecg_indices
    # selelct unique components
    ica_picard.exclude = list(set(ica_picard.exclude))

    return ica_picard, eog_scores, ecg_scores, artifacts

# %% Apply ICA and repair artifacts


def apply_ica(raw, ica_picard):
    """
    Apply ICA and repair artifacts.

    :param raw: The raw data
    :type raw: mne.io.Raw
    :param ica_picard: Fitted ICA object
    :type ica_picard: mne.preprocessing.ICA
    :return: Reconstructed raw data

    """
    # make a copy of the raw data
    reconst_raw = raw.copy()

    # apply ICA and repair artifacts
    ica_picard.apply(reconst_raw)

    return reconst_raw


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
    deriv_proc_dir = op.join(bids_root, "derivatives", "preprocess", "tsss")

    subjs_list = [name for name in os.listdir(deriv_proc_dir)
                    if os.path.isdir(os.path.join(deriv_proc_dir, name)) and name.startswith('sub-')]
    # subj_list = sorted(os.listdir(raw_data))
    print(subjs_list)

    for subj_index, subj in enumerate(subjs_list, start=0):
        if subj in ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-17']:
            continue
        else :
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
                run_files = sorted(glob.glob(os.path.join(sess_path, '*proc-tsss_meg.fif')))
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
                            processing="tsss",
                            suffix='meg',
                            extension=".fif"
                            )
                    print(input_bids_path)

                    # ===== Run ICA preprocessing on each run =====
                    # ===== Report =====
                    # Open html report
                    report = Report("ICA Processing", verbose=False)
                    report = Report(image_format="svg")
                    report = Report(title="ICA_Picard")


                    raw = read_raw_bids(bids_path=input_bids_path, verbose=False,     
                        )                
                    
                    print(raw.info)

                    raw.load_data()

                    # ===== Check EOG =====
                    eog_events, eog_evoked = check_eog_artifacts(raw)

                    # ===== Check ECG =====
                    ecg_events, ecg_evoked = check_ecg_artifacts(raw)

                    # ===== Filtering =====
                    filt_raw = filter_data(raw)

                    # ===== Set up and Fit ICA =====
                    ica_picard = fit_ica(filt_raw)

                    # ===== Check ICA =====
                    comp_ica_figs = check_ica(ica_picard, filt_raw)
                    report.add_figure(comp_ica_figs, title="ICA components")

                    # ===== Automatic ICA components selection =====
                    [ica_picard, eog_scores, ecg_scores, artifacts] = (
                        automatic_ica_components_selection(
                            ica_picard, filt_raw, artifacts=["EOG"]
                        )
                    )

                    # print(ica_picard.exclude)

                    # Generate the ICA Report
                    if ica_picard.exclude:
                        report.add_ica(
                            ica=ica_picard,
                            title="ICA cleaning",
                            picks=ica_picard.exclude,  # plot the excluded EOG and/or ECG components
                            inst=raw,
                            eog_evoked=eog_evoked.average() if "EOG" in artifacts else None,
                            eog_scores=eog_scores if "EOG" in artifacts else None,
                            ecg_evoked=ecg_evoked.average() if "ECG" in artifacts else None,
                            ecg_scores=ecg_scores if "ECG" in artifacts else None,
                            n_jobs=None,
                        )
                    else:
                        print("No components were marked for exclusion. Skipping ICA report addition.")


                    # ===== Apply ICA =====
                    reconst_raw = apply_ica(raw, ica_picard)



                    # ===== Save data ICA preprocessed =====
                    deriv_proc_dir_output = op.join(bids_root, "derivatives", "preprocess", "ICA_removal")

                    output_bids_path = BIDSPath(
                            root=deriv_proc_dir_output,
                            subject=subject_id,
                            session=session_id,
                            datatype=datatype,
                            run=run_id,
                            task=task,
                            processing="ICAremoval",
                            suffix='meg',
                            extension=".fif"
                            )

                    write_raw_bids(
                        reconst_raw,
                        output_bids_path,
                        overwrite=True,
                        allow_preload=True,
                        format='FIF'
                        )


                    
                    # ===== Save report =====

                    # Get the base name and file extension
                    base_run_name, extension = os.path.splitext(output_bids_path.basename)

                    report_name = op.join(
                        deriv_proc_dir_output + "/", f"sub-{subject_id}", f"ses-{session_id}", "meg", base_run_name + "_figures.html"
                    )

                    report.save(report_name, overwrite=True, open_browser=False)

