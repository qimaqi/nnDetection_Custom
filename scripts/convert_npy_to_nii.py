import pickle 
import numpy as np
import json
import nibabel as nib
import os 


data_file = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnDet_raw/Task016_Luna/preprocessed/D3V001_3d/imagesTr/1_3_6_1_4_1_14519_5_2_1_6279_6001_997611074084993415992563148335.npy'

data = np.load(data_file)[0]
# data_file = '/usr/bmicnas02/data-biwi-01/lung_detection/nnDet_data/Task017_Luna_liu_prep_int_16/preprocessed/D3V001_3d/imagesTr/1_3_6_1_4_1_14519_5_2_1_6279_6001_980362852713685276785310240144.npy'
# seg_file = '/usr/bmicnas02/data-biwi-01/lung_detection/nnDet_data/Task017_Luna_liu_prep_int_16/preprocessed/D3V001_3d/imagesTr/1_3_6_1_4_1_14519_5_2_1_6279_6001_980362852713685276785310240144_seg.npy'

# data_file = '/usr/bmicnas02/data-biwi-01/lung_detection/nnDet_data/Task017_Luna_liu_prep_int_16/raw_cropped/imagesTr/1_3_6_1_4_1_14519_5_2_1_6279_6001_997611074084993415992563148335.npz'
# seg_file = '/usr/bmicnas02/data-biwi-01/lung_detection/nnDet_data/Task017_Luna_liu_prep_int_16/preprocessed/D3V001_3d/imagesTr/1_3_6_1_4_1_14519_5_2_1_6279_6001_997611074084993415992563148335_seg.npy'


# raw_data = np.load(data_file)['data']
# data = raw_data[0]
# seg = raw_data[1]
print("Data shape: ", data.shape, data.min(), data.max())
# print("Seg shape: ", seg.shape, seg.min(), seg.max())
# seg = seg.astype(np.int16)

# # Save as nifti
nii_data = nib.Nifti1Image(data, np.eye(4))
# nii_seg = nib.Nifti1Image(seg, np.eye(4))


nii_data.to_filename('./demo_data.nii.gz')
# nii_seg.to_filename('./demo_data_seg.nii.gz')

# # find the slice while label [1] exist
# slice_idx = np.where(seg == 0)
# print("Slice index: ",  np.sum(slice_idx), "out of", seg.shape[1]*seg.shape[2]*seg.shape[0])

# # 6518394
# # 51963765

# # finx the x,y,z coordinate of the slice
# x, y, z = slice_idx[0][0], slice_idx[0][1], slice_idx[0][2]

# slices_16_data_start_index = [x, np.max(y-112,0), np.max(z-112)]
# slices_16_data_end_index =  [x+16, np.max(y-112,0) + 224, np.max(z-112) + 224]


# slices_64_data_start_index = [x, np.max(y-64,0), np.max(z-64)]
# slices_64_data_end_index = [x+64, np.max(y-64,0) + 128, np.max(z-64)+128]

# slice_16_data = data[slices_16_data_start_index[0]:slices_16_data_end_index[0], slices_16_data_start_index[1]:slices_16_data_end_index[1], slices_16_data_start_index[2]:slices_16_data_end_index[2]]
# slice_64_data = data[slices_64_data_start_index[0]:slices_64_data_end_index[0], slices_64_data_start_index[1]:slices_64_data_end_index[1], slices_64_data_start_index[2]:slices_64_data_end_index[2]]

# # nii_data.to_filename('./demo_data.nii.gz')
# # nii_seg.to_filename('./demo_seg.nii.gz')

# nii_data_16 = nib.Nifti1Image(slice_16_data, np.eye(4))
# nii_data_64 = nib.Nifti1Image(slice_64_data, np.eye(4))

# nii_data_16.to_filename('./demo_data_16.nii.gz')
# nii_data_64.to_filename('./demo_data_64.nii.gz')