"""
preprocessing of the first Paris pilot.

Better code with automatic bidificatoin will follow

1. load the MaxFilter data
2. prepare the dictionaries for epoching manipulation
3. find the events in the fif file
4. do the epoching
5. plot a basic average of sounds and silence epochs
5. save the epochs on disk

@author: neurogima, marieplgt
"""

# %% Import modules, packages etc

import glob
import os
import os.path as op
import re
import matplotlib.pyplot as plt

import mne
import mne.filter
import mne.preprocessing
import pandas as pd
from mne.report import Report
import soundsemtools as sst
import numpy as np

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

def replace_consecutive_duplicates(column):
    """
    Change the trigger values for each repeated sound with a fixed value: 1500

    Parameters
    ----------
    column : ARRAY OF INTEGERS
        ARRAY OF TRIGGER VALUES, LAST COLUMN OF THE EVENT STRUCTURE.

    Returns
    -------
    column : ARRAY OF INTEGERS
        MODIFIED REPETITION TRIGGER VALUES COLUMN: NEW VALUE = 1500

    """
    prev_value = None
    for idx, value in enumerate(column):
        if (value == prev_value and value != 601 and value != 2048):
            print(f"Repeated value {value} at positions {idx} and {idx - 1}")
            column[idx] = 1500
        prev_value = value
    return column


# Adjust event onsets for sound delay
def adjust_event_onsets(events, delay, sfreq):
    events[:, 0] = events[:, 0] + int(delay * sfreq)
    return events


# Time onset delay adjustment
SOUND_SPEED = 343  # Speed of sound in air at room temperature in m/s
TUBE_LENGTH = 3.5  # Length of the tube in meters
DELAY = TUBE_LENGTH / SOUND_SPEED  # Delay in seconds
TRIGGER_CHANNEL = 'STI101'  # Identify the trigger channel



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
    deriv_proc_dir = op.join(bids_root, "derivatives", "preprocess", "filtering")

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
            run_files = sorted(glob.glob(os.path.join(sess_path, '*Filt_meg.fif')))
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
                        processing="Filt",
                        suffix='meg',
                        extension=".fif"
                        )
                # print(input_bids_path)
                
                raw = read_raw_bids(bids_path=input_bids_path, verbose=False,     
                    )                
                
                print(raw.info)
                
                # raw.load_data()

                #Check the events
                raw.copy().pick(picks="stim").plot(start=3, duration=6)


                # Prepare the dictionary thanks to the logs
                base_run_name, extension = os.path.splitext(input_bids_path.basename)
                base_name = base_run_name.replace('_proc-Filt_meg', '')
                # store all the runs logs
                run_log_file = sorted(glob.glob(op.join(bids_root, subj, sess,  "logs", f"{base_name}_logs_exp.csv")))
                
                # read the log as pandas dataframe
                df_log = pd.read_csv(run_log_file[0])
                
                # extract the sounds keys and names using the columns as a pandas series
                trial_stim = df_log.loc[:, "trial_stim"]
                sound_name = df_log.loc[:, "soundName"]

                # create the event dictionary
                event_dict = {key: value for key, value in zip(sound_name, trial_stim)}

                # add the starting block, the response, and repetition triggers
                start_and_response_and_repetition = {
                    "start_block": 1000,
                    "response_button": 2048,
                    "repetition": 1500,
                }

                # combine the original dictionary with the additional key-value pairs
                event_dict.update(start_and_response_and_repetition)


                ## simplify the dictionary for quick evoked response and general handling as
                # in MNE-Python way:
                # https://mne.tools/stable/auto_tutorials/raw/20_event_arrays.html#mapping-event-ids-to-trial-descriptors

                # mark the collective sounds events
                event_dict_simple = {
                    re.sub(r"(.*Fsd50k.*)", "sound/\\1", key): value
                    for key, value in event_dict.items()
                }

                # now mark the silence events
                event_dict_simple = {
                    re.sub(r"(.*Silence.*)", "silence/\\1", key): value
                    for key, value in event_dict_simple.items()
                }

                # # drop the start block event
                # if run_number == 6:
                #     del event_dict_simple["start_block"]


                ## Extract samples-trigger events structure
                # Find the events in the fif files and change the event trigger value
                # for the repetitions

                raw.info["ch_names"]

                events = mne.find_events(raw, stim_channel="STI101", min_duration=0.002)

                events_column = replace_consecutive_duplicates(events[:, -1])

                # Create a copy of the original array
                events_copy = events.copy()

                # Replace the old column with the new column
                events[:, -1] = events_column

                # % Plot the events

                # # plot the events
                # fig = mne.viz.plot_events(
                #     events,
                #     event_id=event_dict,
                #     sfreq=raw.info["sfreq"],
                #     first_samp=raw.first_samp,
                # )

                # # plot the events and data
                # raw.plot(
                #     events=events,
                #     start=5,
                #     duration=10,
                #     color="gray",
                # )


                ## Do the Epoching

                tmin, tmax = -0.5, 3.75
                # tmin = -3  # Start of each epoch
                # tmax = 14  # End of each epoch

                # Adjust event onsets for sound delay
                events = adjust_event_onsets(events, DELAY, raw.info['sfreq'])
                epochs = mne.Epochs(
                    raw,
                    events,
                    event_id=event_dict_simple,
                    tmin=tmin,
                    tmax=tmax,
                    baseline=None,
                    picks="meg",
                    preload=True,
                )


                ## Generate epochs Report
                # Open html report
                report = Report('Sound Epochs', verbose=False)
                report = Report(image_format='png')

                report.add_epochs(
                    epochs=epochs, title='Epochs from sound epochs no baseline'
                )
                plt.close()


                ## Visualize the epochs as average

                # use the keys in the dictionary to subselect epochs
                evoked = epochs["sound"].average()
                evoked.silence = epochs["silence"].average()
                evoked_response = epochs["response_button"].average()

                # topoplot
                # times = np.arange(-0.2, 2.5, 0.05)
                # evoked.plot_topomap(times, ch_type="mag", ncols=8, nrows="auto")
                # plt.close()

                # time course
                # evoked.plot()
                # plt.close()


                ## Generate evoked report
                report.add_evokeds(
                    evokeds=evoked,
                    titles='evoked from the sound epochs',  # Manually specify titles
                    # noise_cov=cov_path,
                    n_time_points=10,
                )
                plt.close()



                # ===== Save final raw and epochs =====
                deriv_proc_dir_output = op.join(bids_root, "derivatives", "preprocess", "epoching")

                output_bids_path = BIDSPath(
                        root=deriv_proc_dir_output,
                        subject=subject_id,
                        session=session_id,
                        datatype=datatype,
                        run=run_id,
                        task=task,
                        processing="final",
                        suffix='meg',
                        extension=".fif"
                        )

                write_raw_bids(
                    raw,
                    output_bids_path,
                    events=events,
                    event_id=event_dict_simple,
                    overwrite=True,
                    allow_preload=True,
                    format='FIF'
                    )
                # epochs.plot(n_epochs=10, events=True)


                ## save epochs

                # Get the base name and file extension
                output_path = op.join(
                    deriv_proc_dir_output + "/", f"sub-{subject_id}", f"ses-{session_id}", "meg")
                out_name = f"{tmin}_tmax_{tmax}_no_baseline_new_prep-epo.fif"
                epoch_file_name = op.join(output_path, base_name + out_name)

                epochs.save(epoch_file_name, overwrite=True)


                ## Save the report
                report_name = op.join(
                    output_path, base_name + "_epochs.html"
                )
                report.save(report_name, overwrite=True, open_browser=False)

                del report


