
# %% Import modules, packages etc

"""
preprocessing of the first Paris pilot.

Better code with automatic bidifcation will follow

1. Run the Automatic bad channel rejection for each run
2. Visually insepect the data and mark addionally bad channels
3. Run the spatiotemporal Signal Space Separation (tSSS) of MaxFilter

@author: neurogima, marieplgt
"""

# % matplotlib qt

import glob
import os
import os.path as op

import matplotlib.pyplot as plt
import mne
import mne.filter
import mne.preprocessing
import numpy as np
import pandas as pd
import seaborn as sns

import soundsemtools as sst

from mne.preprocessing import find_bad_channels_maxwell
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


username = "marieplgt"   # whoami within a terminal

if username == "marieplgt":
    paths = sst.utils.get_proj_paths(
        "/media/sf_sharedVM/github_repo/phd/code/meg_analysis_code/config.json"
    )
    bids_root = "/media/marieplgt/T7/docs/PHD/Aud2sem_project/AUD2SEM_BIDS/"

# %% ONLY CHANGE HERE:
# definitions for BIDS reading of participant, run number etc

subject = '08'
session = '02'
run = '08'

datatype = 'meg'
task = 'aud2sem'

# %% BIDS reading


# define all the Elekta calibration files
crosstalk_file = op.join(bids_root, f"sub-{subject}", f"ses-{session}", "meg/", "sss_config", f"sub-{subject}_ses-{session}_ct_sparse.fif")
fine_cal_file = op.join(bids_root, f"sub-{subject}", f"ses-{session}", "meg/", "sss_config", f"sub-{subject}_ses-{session}_sss_cal_3101_160108.dat")

bids_path = BIDSPath(
        root=bids_root,
        subject=subject,
        session=session,
        datatype=datatype,
        run=run,
        task=task,
        extension=".fif"
        )

raw = read_raw_bids(bids_path=bids_path, verbose=False,     
       )
print(raw.preload)                 
# %%


raw_duration_secs = len(raw) / 1000


# %% Automatic Channel rejection (let's test)

raw_autoBad = raw.copy()

# Remeber: to optimize the procedure we need to have the site-specific
# calibration file and crosstalk file
# fine_cal_file = os.path.join(sample_data_folder, "SSS", "sss_cal_mgh.dat")
# crosstalk_file = os.path.join(sample_data_folder, "SSS", "ct_sparse_mgh.fif")

auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
    raw_autoBad,
    cross_talk=crosstalk_file,
    calibration=fine_cal_file,
    return_scores=True,
    verbose=True,
)

print(auto_noisy_chs)
print(auto_flat_chs)

# Now we can update the list of bad channels in the dataset.
bads = raw_autoBad.info["bads"] + auto_noisy_chs + auto_flat_chs
raw_autoBad.info["bads"] = bads


# %% Plot the automatic channels score

# Only select the data for gradiometer channels.
ch_type = "grad"
ch_subset = auto_scores["ch_types"] == ch_type
ch_names = auto_scores["ch_names"][ch_subset]
scores = auto_scores["scores_noisy"][ch_subset]
limits = auto_scores["limits_noisy"][ch_subset]
bins = auto_scores["bins"]  # The the windows that were evaluated.
# We will label each segment by its start and stop time, with up to 3
# digits before and 3 digits after the decimal place (1 ms precision).
bin_labels = [f"{start:3.3f} – {stop:3.3f}" for start, stop in bins]

# We store the data in a Pandas DataFrame. The seaborn heatmap function
# we will call below will then be able to automatically assign the correct
# labels to all axes.
data_to_plot = pd.DataFrame(
    data=scores,
    columns=pd.Index(bin_labels, name="Time (s)"),
    index=pd.Index(ch_names, name="Channel"),
)

# First, plot the "raw" scores.
fig, ax = plt.subplots(1, 2, figsize=(12, 8), layout="constrained")
fig.suptitle(
    f"Automated noisy channel detection: {ch_type}",
    fontsize=16,
    fontweight="bold",
)

sns.heatmap(
    data=data_to_plot, cmap="Reds", cbar_kws=dict(label="Score"), ax=ax[0]
)

[
    ax[0].axvline(x, ls="dashed", lw=0.25, dashes=(25, 15), color="gray")
    for x in range(1, len(bins))
]
ax[0].set_title("All Scores", fontweight="bold")

# Now, adjust the color range to highlight segments that exceeded the limit.
sns.heatmap(
    data=data_to_plot,
    vmin=np.nanmin(limits),  # bads in input data have NaN limits
    cmap="Reds",
    cbar_kws=dict(label="Score"),
    ax=ax[1],
)
[
    ax[1].axvline(x, ls="dashed", lw=0.25, dashes=(25, 15), color="gray")
    for x in range(1, len(bins))
]
ax[1].set_title("Scores > Limit", fontweight="bold")


# %% Detect Bad Channels by visual inspection

# when we close the plot, the channels manually marked as bad are automatically
# saved in the raw.info "bads" dictionary
# raw_autoBad.plot(theme="dark")

# To identify channels responsible of distributed artefacts
# First, remove the ssp from the parameters
# Second, decrease the amplitude of the sensors
raw_visualBad = raw_autoBad.copy()
raw_visualBad.load_data()
raw_visualBad.filter(1, None).plot(theme="dark")


# %% Discard channels marked as bads

visual_bads = [
'MEG0332',
'MEG0413',
'MEG0733',
'MEG0642',
'MEG0643',
'MEG1022',
'MEG1143',
'MEG1332',
'MEG1443',
'MEG1822',
'MEG2122',
'MEG2623',
'MEG0811',
'MEG0521',
'MEG0341',
'MEG2021',
]
# visual_bads = raw_visualBad.info["bads"]

raw_autoBad.info["bads"] = raw_autoBad.info["bads"] + visual_bads

bad_channels_list = raw_autoBad.info["bads"]
print(bad_channels_list)
# %% Compute raw spectrum

spectrum = raw.compute_psd()
spectrum.plot(average=True, picks="data", exclude="bads")


# %% tSSS MaxFilter AKA Spatiotemporal SSS

# Read carefullty:
# https://mne.tools/stable/auto_tutorials/preprocessing/60_maxwell_filtering_sss.html#spatiotemporal-sss-tsss
# In general, larger values of st_duration are better (provided that your
# computer has sufficient memory) because larger values of st_duration will
# have a smaller effect on the signal.

# choose a chunk duration that evenly divides your data length
# (only recommended if analyzing a single subject or run)

# otherwise include at least 2 * st_duration of post-experiment recording time
# at the end of the Raw object, so that the data you intend to further analyze
# is guaranteed not to be in the final or penultimate chunks.

# After discussing with Laurent and Christophe, we will use the exact duration
# of the run.

raw_tsss = mne.preprocessing.maxwell_filter(
    raw_autoBad,
    cross_talk=crosstalk_file,
    calibration=fine_cal_file,
    # st_duration=raw_duration_secs,
    verbose=True,
)
# print(raw_tsss.info['bads'])
# Maxwell filtering reconstrcuts bads channels, it won't have them in .info['bads'] anymore !!

# %% Compute raw_tsss spectrum

spectrum = raw_tsss.compute_psd()
spectrum.plot(average=True, picks="data", exclude="bads")


# %% Check how does the tSSS performed

# and check how did it perform
# raw_autoBad.pick(["meg"]).plot(duration=2, butterfly=True)
# raw_sss.pick(["meg"]).plot(duration=2, butterfly=True)
# raw_tsss.pick(["meg"]).plot(duration=2, butterfly=True)

raw_tsss.plot()
# raw_autoBad.plot()



# %% Save the maxfiltered file for BIDS

# define the new root derivative path for this process
deriv_proc_dir = op.join(bids_root, "derivatives", "preprocess", "tsss")

# define a new BIDSpath
deriv_bids_path = BIDSPath(
        root=deriv_proc_dir,
        subject=subject,
        session=session,
        datatype=datatype,
        run=run,
        task=task,
        processing="tsss",
        suffix='meg',
        extension=".fif"
        )

# define the full path using teh BIDSpath
# maxfilt_file_name = deriv_bids_path.fpath

# Save the data
write_raw_bids(
    raw_tsss,
    deriv_bids_path,
    overwrite=True,
    allow_preload=True,
    format='FIF'
    )


# %% Update the channel file in the raw data, with the list of bad channels
# we mark the bad channels in the channel file at raw level, because after the
# the max filter process "reapair" the channels marking all goods

bad_channels_list = raw_autoBad.info["bads"]

mark_channels(
    bids_path=bids_path,
    ch_names=bad_channels_list,
    status="bad",
    verbose=False
    )


# Get the base name and file extension
base_run_name, extension = os.path.splitext(deriv_bids_path.basename)

bad_chans_name = op.join(
    deriv_proc_dir + "/", f"sub-{subject}", f"ses-{session}", "meg", base_run_name + "_bad_chans.txt"
)

# Write the bad channels list into a csv
# using the savetxt from the numpy module
np.savetxt(bad_chans_name, raw_autoBad.info["bads"], delimiter=" ", fmt="% s")



# %%
