"""
Organize the directories and BIDSify the data.

TODO: [ ] add the dataset description with make description function
"""


# %% Imports

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

import ants



# %% Paths

username = "marieplgt"   # whoami within a terminal

if username == "marieplgt":
    paths = sst.utils.get_proj_paths(
        "/media/sf_sharedVM/github_repo/phd/code/meg_analysis_code/config.json"
    )

user = paths["user"]
proj_dir = paths["proj_dir"]
code_dir = paths["code_dir"]
data_dir = paths["data_dir"]  # MEG !!!


bids_dir = f"{proj_dir}/AUD2SEM_BIDS"
MRIs_dir = op.join(proj_dir, "orig_data/MRIs")


def dircreate(dirs_to_create):
    """
    Create directories if they don't exist.

    :param dirs_to_create: Directory, or list of directories to create
    :type dirs_to_create: strings of path
    :return: Nothing, create the directories in the filesystem

    """
    if not op.exists(dirs_to_create):
        os.makedirs(dirs_to_create)


dircreate(bids_dir)



# convert dicom for sub-X to nifti
# code should be ran in terminal

# dcm2niix -o /home/brungio/Desktop/Projects/AudSemScene/bruno/BIDS/tmp_dicom/sub-02/ /home/brungio/Desktop/Projects/AudSemScene/bruno/BIDS/tmp_dicom/sub-02/DICOM/PRISMA_AUD2SEM_20240620_154551_506000/T1_MEG_0_8ISO_PRISMA_0002/

# also write the anatomical in .nii format (non compressed) for spm segmentation : -z n
# dcm2niix -z n -o /home/brungio/Desktop/Projects/AudSemScene/bruno/BIDS/tmp_dicom/sub-02/ /home/brungio/Desktop/Projects/AudSemScene/bruno/BIDS/tmp_dicom/sub-02/DICOM/PRISMA_AUD2SEM_20240620_154551_506000/T1_MEG_0_8ISO_PRISMA_0002/

# %% BIDSification

#print_dir_tree(bids_dir, max_depth=None)

# Create derivatives directory if it doesn't exist
derivatives_root = os.path.join(bids_dir, 'derivatives', 'acpc')
os.makedirs(derivatives_root, exist_ok=True)
print(f'Derivatives root directory is set up at {derivatives_root}')


# Retrieve raw_data paths
subjs_list = [name for name in os.listdir(bids_dir)
              if os.path.isdir(os.path.join(bids_dir, name)) and name.startswith('sub-')]
print(subjs_list)

sess_list = ["ses-01", "ses-02"]
modality = 'T1w'


for subject in subjs_list:    
    print(subject)
    subj_index = subject.split('-')[1]

    # Create BIDS dir outputs
    bids_anat_dir = os.path.join(bids_dir, subject, 'anat')
    os.makedirs(bids_anat_dir, exist_ok=True)

    # Retrieve orig MRIs data
    Nifti_dir = op.join(MRIs_dir, "NIFTI", f"aud2sem_s{subj_index}")
    Dicom_dir = op.join(MRIs_dir, "DICOM", f"aud2sem_s{subj_index}")    
         

    # 1) Move and rename NIfTI and JSON files into BIDS_anat dir
    if op.exists(Nifti_dir):
        for file in os.listdir(Nifti_dir):
            #print(file)
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                src = os.path.join(Nifti_dir, file)
                dst = os.path.join(bids_anat_dir, f'{subject}_{modality}.nii')
                shutil.copy2(src, dst)
            elif file.endswith('.json'):
                src = os.path.join(Nifti_dir, file)
                dst = os.path.join(bids_anat_dir, f'{subject}_{modality}.json')
                shutil.copy2(src, dst)
        
        print('NIfTI and JSON files organized into BIDS structure.')



    # 2) ACpc alignement computation, saving in Bids_deriv dir
    derivatives_dir = os.path.join(derivatives_root, subject, 'anat')
    os.makedirs(derivatives_dir, exist_ok=True)


    t1_file = os.path.join(bids_anat_dir, f'{subject}_T1w.nii')
    output_file_nii = os.path.join(derivatives_dir, f'{subject}_T1w_acpc.nii')
    output_file_niigz = os.path.join(derivatives_dir, f'{subject}_T1w_acpc.nii.gz')

    if op.exists(t1_file):
        # Load the image
        print(f'Loading T1 image for {subject} from {t1_file}')
        t1 = ants.image_read(t1_file)
        print(f'T1 image for {subject} loaded successfully')

        # Perform AC-PC alignment (rigid registration to MNI template)
        print(f'Loading MNI template for AC-PC alignment')
        mni_template = ants.image_read(ants.get_ants_data('mni'))
        print(f'MNI template loaded successfully')
        
        print(f'Performing rigid registration for {subject}')
        acpc_transform = ants.registration(fixed=mni_template, moving=t1, type_of_transform='Rigid')
        print(f'Rigid registration completed for {subject}')

        # Apply the transformation
        print(f'Applying transformation to align T1 image for {subject}')
        t1_acpc = ants.apply_transforms(fixed=mni_template, moving=t1, transformlist=acpc_transform['fwdtransforms'])
        print(f'Transformation applied successfully for {subject}')

        # Save the aligned image in NIfTI format
        print(f'Saving AC-PC aligned T1 image for {subject} in .nii format')
        ants.image_write(t1_acpc, output_file_nii)
        print(f'Saved AC-PC aligned T1 image for {subject} in .nii format at {output_file_nii}')

        print(f'Saving AC-PC aligned T1 image for {subject} in .nii.gz format')
        ants.image_write(t1_acpc, output_file_niigz)
        print(f'Saved AC-PC aligned T1 image for {subject} in .nii.gz format at {output_file_niigz}')

        print(f'AC-PC aligned T1 image processing completed for {subject}')
    else:
        print(f"{subject} doesn't have MRI files")

print('AC-PC alignment and saving completed for all subjects.')

