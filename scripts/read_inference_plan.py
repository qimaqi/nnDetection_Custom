import pickle 
import numpy as np
import json

# file = '/usr/bmicnas02/data-biwi-01/lung_detection/nnDet_data/Task019_Luna_liu_prep_int_96/preprocessed/D3V001_3d.pkl'
# '/usr/bmicnas02/data-biwi-01/lung_detection/nnDet_models/Task017_Luna_crop/RetinaUNetV001_D3V001_3d/16_224_224/fold1/plan_inference.pkl'
# '/usr/bmicnas02/data-biwi-01/lung_detection/nnDet_models/Task017_Luna_crop/RetinaUNetV001_D3V001_3d/16_224_224/fold0/plan_inference.pkl'
# '/usr/bmicnas02/data-biwi-01/lung_detection/nnDet_models/Task020_Luna_16_pre/VideoMAEUNetV001_D3V001_3d/16_224_224/fold1/plan_inference.pkl'

# file ='/usr/bmicnas02/data-biwi-01/lung_detection/nnDet_models/Task023_Luna_liu_prep_int_16_pretrain_freeze/VideoMAEUNetV001_D3V001_3d/16_224_224/fold0/plan_inference.pkl'
#file = '/usr/bmicnas02/data-biwi-01/lung_detection/nnDet_models/Task017_Luna_liu_prep_int_16/RetinaUNetV001_D3V001_3d/16_224_224/fold0/plan_inference.pkl'
#file = '/usr/bmicnas02/data-biwi-01/lung_detection/nnDet_models/Task017_Luna_liu_prep_int_16/final_official_VideoMAEUNetV001_D3V001_3d/16_224_224/fold0/plan_inference.pkl' #facebook 19
#file = '/usr/bmicnas02/data-biwi-01/lung_detection/nnDet_models/Task020_Luna_liu_prep_int_16_pretrain/cropped_at_epoch19_VideoMAEUNetV001_D3V001_3d/16_224_224/fold0/plan_inference.pkl' #pretrain 19
# file = '/usr/bmicnas02/data-biwi-01/lung_detection/nnDet_models/Task017_Luna_liu_prep_int_16/VideoMAEUNetV001_D3V001_3d/16_224_224/fold0/plan_inference.pkl' #facebook 19

file = "/usr/bmicnas02/data-biwi-01/lung_detection/nnDet_models/Task020_Luna_liu_prep_int_16_pretrain/official_finished_version1_VideoMAEUNetV001_D3V001_3d/16_224_224/fold0/plan_inference.pkl"
with open(file, 'rb') as f:
    data = pickle.load(f)

for key, value in data.items():

    print(key, value)
    

# nndet:
# planner_id D3V001
# network_dim 3
# dataloader_kwargs {}
# data_identifier D3V001_3d
# postprocessing {}
# patch_size [ 16 224 224]
# batch_size 4
# architecture {'arch_name': 'RetinaUNetV001', 'max_channels': 320, 'start_channels': 32, 'fpn_channels': 128, 'head_channels': 128, 'classifier_classes': 1, 'seg_classes': 1, 'in_channels': 1, 'dim': 3, 'class_weight': [0.5, 0.0], 'conv_kernels': [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]], 'strides': [[2, 2, 2], [2, 2, 2], [1, 2, 2], [1, 2, 2]], 'decoder_levels': (1, 2, 3, 4)}
# anchors {'width': [(4.0, 6.0, 8.0), (8.0, 12.0, 16.0), (8.0, 12.0, 16.0), (8.0, 12.0, 16.0)], 'height': [(7.0, 10.0, 8.0), (14.0, 20.0, 16.0), (28.0, 40.0, 32.0), (56.0, 80.0, 64.0)], 'depth': [(7.0, 9.0, 11.0), (14.0, 18.0, 22.0), (28.0, 36.0, 44.0), (56.0, 72.0, 88.0)], 'stride': 1}
# target_spacing_transposed [1.25     0.703125 0.703125]
# median_shape_transposed [254. 512. 512.]
# do_dummy_2D_data_aug True
# trigger_lr1 True
# inference_plan {'model_iou': 1e-05, 'model_nms_fn': <function batched_weighted_nms_model at 0x7f634f668c10>, 'model_score_thresh': 0.6, 'model_topk': 1000, 'model_detections_per_image': 100, 'ensemble_iou': 0.2, 'ensemble_nms_fn': <function batched_wbc_ensemble at 0x7f634b5ef670>, 'ensemble_topk': 1000, 'remove_small_boxes': 3.0, 'ensemble_score_thresh': 0.0}


# planner_id D3V001
# network_dim 3
# dataloader_kwargs {}
# data_identifier D3V001_3d
# postprocessing {}
# patch_size [ 16 224 224]
# batch_size 4
# architecture {'arch_name': 'RetinaUNetV001', 'max_channels': 320, 'start_channels': 32, 'fpn_channels': 128, 'head_channels': 128, 'classifier_classes': 1, 'seg_classes': 1, 'in_channels': 1, 'dim': 3, 'class_weight': [0.5, 0.0], 'conv_kernels': [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]], 'strides': [[2, 2, 2], [2, 2, 2], [1, 2, 2], [1, 2, 2]], 'decoder_levels': (1, 2, 3, 4)}
# anchors {'width': [(4.0, 6.0, 8.0), (8.0, 12.0, 16.0), (8.0, 12.0, 16.0), (8.0, 12.0, 16.0)], 'height': [(7.0, 10.0, 8.0), (14.0, 20.0, 16.0), (28.0, 40.0, 32.0), (56.0, 80.0, 64.0)], 'depth': [(7.0, 9.0, 11.0), (14.0, 18.0, 22.0), (28.0, 36.0, 44.0), (56.0, 72.0, 88.0)], 'stride': 1}
# target_spacing_transposed [1.25     0.703125 0.703125]
# median_shape_transposed [254. 512. 512.]
# do_dummy_2D_data_aug True
# trigger_lr1 True
# inference_plan {'model_iou': 1e-05, 'model_nms_fn': <function batched_weighted_nms_model at 0x7f824cfb7c10>, 'model_score_thresh': 0.0, 'model_topk': 1000, 'model_detections_per_image': 100, 'ensemble_iou': 1e-05, 'ensemble_nms_fn': <function batched_wbc_ensemble at 0x7f8248f3f670>, 'ensemble_topk': 1000, 'remove_small_boxes': 0.01, 'ensemble_score_thresh': 0.0}







# save_pkl = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnDet_raw/Task018_Luna64x128/preprocessed/D3V001_3d.pkl'
# with open(save_pkl, 'wb') as f:
#     pickle.dump(data, f)


# map pkl to json, save json, include numpy arrays, int64, float64 in json
# def map_pkl_to_json(data):
#     for key, value in data.items():
#         if isinstance(value, (np.ndarray, np.int64, np.float64)):
#             data[key] = (value).astype(np.float32).tolist()
#         if isinstance(value, dict):
#             data[key] = map_pkl_to_json(value)
#     return data

# data = map_pkl_to_json(data)
# with open(save_json, 'w') as f:
#     json.dump(data, f, indent=4)

# Dict([('original_size_of_raw_data', array([256, 256, 256])), ('original_spacing', array([1., 1., 1.])), ('list_of_data_files', [PosixPath('/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnDet_raw/Task000D3_Example/raw_splitted/imagesTr/case_0_0000.nii.gz')]), ('seg_file', PosixPath('/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnDet_raw/Task000D3_Example/raw_splitted/labelsTr/case_0.nii.gz')), ('itk_origin', (0.0, 0.0, 0.0)), ('itk_spacing', (1.0, 1.0, 1.0)), ('itk_direction', (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)), ('instances', {'1': 1}), ('crop_bbox', [(0, 256), (0, 256), (0, 256)]), ('classes', array([0., 1.], dtype=float32)), ('size_after_cropping', (256, 256, 256)), ('num_instances', {0: 0, 1: 1}), ('has_classes', [1]), ('volume_per_class', OrderedDict([(0, 0), (1, 10752.0)])), ('region_volume_per_class', OrderedDict([(1, [10752.0])])), ('boxes', array([[ 12, 216,  41, 245,  75, 104]])), ('all_ious', array([], shape=(1, 0), dtype=float64)), ('class_ious', OrderedDict([(1, array([], shape=(1, 0), dtype=float64))]))])


# mode 3d
# target_spacing [1. 1. 1.]
# normalization_schemes OrderedDict([(0, 'nonCT')])
# use_mask_for_norm OrderedDict([(0, False)])
# anisotropy_threshold 3
# resample_anisotropy_threshold 3
# target_spacing_percentile 50
# dim 3
# num_modalities 1
# all_classes [0 1]
# num_classes 2
# transpose_forward [0, 1, 2]
# transpose_backward [0, 1, 2]
# dataset_properties {'dim': 3, 'all_sizes': [(256, 256, 256), (256, 256, 256), (256, 256, 256), (256, 256, 256), (256, 256, 256), (256, 256, 256), (256, 256, 256), (256, 256, 256), (256, 256, 256), (256, 256, 256)], 'all_spacings': [array([1., 1., 1.]), array([1., 1., 1.]), array([1., 1., 1.]), array([1., 1., 1.]), array([1., 1., 1.]), array([1., 1., 1.]), array([1., 1., 1.]), array([1., 1., 1.]), array([1., 1., 1.]), array([1., 1., 1.])], 'size_reductions': OrderedDict([('case_0', 1.0), ('case_1', 1.0), ('case_2', 1.0), ('case_3', 1.0), ('case_4', 1.0), ('case_5', 1.0), ('case_6', 1.0), ('case_7', 1.0), ('case_8', 1.0), ('case_9', 1.0)]), 'modalities': {0: 'MRI'}, 'class_dct': {'0': 'Square', '1': 'SquareHole'}, 'all_classes': array([0, 1]), 'instance_props_per_patient': OrderedDict([('case_0', OrderedDict([('original_size_of_raw_data', array([256, 256, 256])), ('original_spacing', array([1., 1., 1.])), ('list_of_data_files', [PosixPath('/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnDet_raw/Task000D3_Example/raw_splitted/imagesTr/case_0_0000.nii.gz')]), ('seg_file', PosixPath('/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnDet_raw/Task000D3_Example/raw_splitted/labelsTr/case_0.nii.gz')), ('itk_origin', (0.0, 0.0, 0.0)), ('itk_spacing', (1.0, 1.0, 1.0)), ('itk_direction', (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)), ('instances', {'1': 1}), ('crop_bbox', [(0, 256), (0, 256), (0, 256)]), ('classes', array([0., 1.], dtype=float32)), ('size_after_cropping', (256, 256, 256)), ('num_instances', {0: 0, 1: 1}), ('has_classes', [1]), ('volume_per_class', OrderedDict([(0, 0), (1, 10752.0)])), ('region_volume_per_class', OrderedDict([(1, [10752.0])])), ('boxes', array([[ 12, 216,  41, 245,  75, 104]])), ('all_ious', array([], shape=(1, 0), dtype=float64)), ('class_ious', OrderedDict([(1, array([], shape=(1, 0), dtype=float64))]))])), ('case_1', OrderedDict([('original_size_of_raw_data', array([256, 256, 256])), ('original_spacing', array([1., 1., 1.])), ('list_of_data_files', [PosixPath('/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnDet_raw/Task000D3_Example/raw_splitted/imagesTr/case_1_0000.nii.gz')]), ('seg_file', PosixPath('/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnDet_raw/Task000D3_Example/raw_splitted/labelsTr/case_1.nii.gz')), ('itk_origin', (0.0, 0.0, 0.0)), ('itk_spacing', (1.0, 1.0, 1.0)), ('itk_direction', (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)), ('instances', {'1': 1}), ('crop_bbox', [(0, 256), (0, 256), (0, 256)]), ('classes', array([0., 1.], dtype=float32)), ('size_after_cropping', (256, 256, 256)), ('num_instances', {0: 0, 1: 1}), ('has_classes', [1]), ('volume_per_class', OrderedDict([(0, 0), (1, 5712.0)])), ('region_volume_per_class', OrderedDict([(1, [5712.0])])), ('boxes', array([[112,  43, 134,  65, 230, 252]])), ('all_ious', array([], shape=(1, 0), dtype=float64)), ('class_ious', OrderedDict([(1, array([], shape=(1, 0), dtype=float64))]))])), ('case_2', OrderedDict([('original_size_of_raw_data', array([256, 256, 256])), ('original_spacing', array([1., 1., 1.])), ('list_of_data_files', [PosixPath('/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnDet_raw/Task000D3_Example/raw_splitted/imagesTr/case_2_0000.nii.gz')]), ('seg_file', PosixPath('/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnDet_raw/Task000D3_Example/raw_splitted/labelsTr/case_2.nii.gz')), ('itk_origin', (0.0, 0.0, 0.0)), ('itk_spacing', (1.0, 1.0, 1.0)), ('itk_direction', (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)), ('instances', {'1': 1}), ('crop_bbox', [(0, 256), (0, 256), (0, 256)]), ('classes', array([0., 1.], dtype=float32)), ('size_after_cropping', (256, 256, 256)), ('num_instances', {0: 0, 1: 1}), ('has_classes', [1]), ('volume_per_class', OrderedDict([(0, 0), (1, 7680.0)])), ('region_volume_per_class', OrderedDict([(1, [7680.0])])), ('boxes', array([[164, 119, 189, 144, 142, 167]])), ('all_ious', array([], shape=(1, 0), dtype=float64)), ('class_ious', OrderedDict([(1, array([], shape=(1, 0), dtype=float64))]))])), ('case_3', OrderedDict([('original_size_of_raw_data', array([256, 256, 256])), ('original_spacing', array([1., 1., 1.])), ('list_of_data_files', [PosixPath('/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnDet_raw/Task000D3_Example/raw_splitted/imagesTr/case_3_0000.nii.gz')]), ('seg_file', PosixPath('/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnDet_raw/Task000D3_Example/raw_splitted/labelsTr/case_3.nii.gz')), ('itk_origin', (0.0, 0.0, 0.0)), ('itk_spacing', (1.0, 1.0, 1.0)), ('itk_direction', (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)), ('instances', {'1': 0}), ('crop_bbox', [(0, 256), (0, 256), (0, 256)]), ('classes', array([0., 1.], dtype=float32)), ('size_after_cropping', (256, 256, 256)), ('num_instances', {0: 1, 1: 0}), ('has_classes', [0]), ('volume_per_class', OrderedDict([(0, 17576.0), (1, 0)])), ('region_volume_per_class', OrderedDict([(0, [17576.0])])), ('boxes', array([[132, 186, 159, 213, 164, 191]])), ('all_ious', array([], shape=(1, 0), dtype=float64)), ('class_ious', OrderedDict([(0, array([], shape=(1, 0), dtype=float64))]))])), ('case_4', OrderedDict([('original_size_of_raw_data', array([256, 256, 256])), ('original_spacing', array([1., 1., 1.])), ('list_of_data_files', [PosixPath('/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnDet_raw/Task000D3_Example/raw_splitted/imagesTr/case_4_0000.nii.gz')]), ('seg_file', PosixPath('/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnDet_raw/Task000D3_Example/raw_splitted/labelsTr/case_4.nii.gz')), ('itk_origin', (0.0, 0.0, 0.0)), ('itk_spacing', (1.0, 1.0, 1.0)), ('itk_direction', (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)), ('instances', {'1': 0}), ('crop_bbox', [(0, 256), (0, 256), (0, 256)]), ('classes', array([0., 1.], dtype=float32)), ('size_after_cropping', (256, 256, 256)), ('num_instances', {0: 1, 1: 0}), ('has_classes', [0]), ('volume_per_class', OrderedDict([(0, 17576.0), (1, 0)])), ('region_volume_per_class', OrderedDict([(0, [17576.0])])), ('boxes', array([[139, 165, 166, 192,  81, 108]])), ('all_ious', array([], shape=(1, 0), dtype=float64)), ('class_ious', OrderedDict([(0, array([], shape=(1, 0), dtype=float64))]))])), ('case_5', OrderedDict([('original_size_of_raw_data', array([256, 256, 256])), ('original_spacing', array([1., 1., 1.])), ('list_of_data_files', [PosixPath('/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnDet_raw/Task000D3_Example/raw_splitted/imagesTr/case_5_0000.nii.gz')]), ('seg_file', PosixPath('/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnDet_raw/Task000D3_Example/raw_splitted/labelsTr/case_5.nii.gz')), ('itk_origin', (0.0, 0.0, 0.0)), ('itk_spacing', (1.0, 1.0, 1.0)), ('itk_direction', (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)), ('instances', {'1': 0}), ('crop_bbox', [(0, 256), (0, 256), (0, 256)]), ('classes', array([0., 1.], dtype=float32)), ('size_after_cropping', (256, 256, 256)), ('num_instances', {0: 1, 1: 0}), ('has_classes', [0]), ('volume_per_class', OrderedDict([(0, 6859.0), (1, 0)])), ('region_volume_per_class', OrderedDict([(0, [6859.0])])), ('boxes', array([[30, 73, 50, 93, 52, 72]])), ('all_ious', array([], shape=(1, 0), dtype=float64)), ('class_ious', OrderedDict([(0, array([], shape=(1, 0), dtype=float64))]))])), ('case_6', OrderedDict([('original_size_of_raw_data', array([256, 256, 256])), ('original_spacing', array([1., 1., 1.])), ('list_of_data_files', [PosixPath('/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnDet_raw/Task000D3_Example/raw_splitted/imagesTr/case_6_0000.nii.gz')]), ('seg_file', PosixPath('/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnDet_raw/Task000D3_Example/raw_splitted/labelsTr/case_6.nii.gz')), ('itk_origin', (0.0, 0.0, 0.0)), ('itk_spacing', (1.0, 1.0, 1.0)), ('itk_direction', (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)), ('instances', {'1': 1}), ('crop_bbox', [(0, 256), (0, 256), (0, 256)]), ('classes', array([0., 1.], dtype=float32)), ('size_after_cropping', (256, 256, 256)), ('num_instances', {0: 0, 1: 1}), ('has_classes', [1]), ('volume_per_class', OrderedDict([(0, 0), (1, 9152.0)])), ('region_volume_per_class', OrderedDict([(1, [9152.0])])), ('boxes', array([[159,  16, 186,  43,  12,  39]])), ('all_ious', array([], shape=(1, 0), dtype=float64)), ('class_ious', OrderedDict([(1, array([], shape=(1, 0), dtype=float64))]))])), ('case_7', OrderedDict([('original_size_of_raw_data', array([256, 256, 256])), ('original_spacing', array([1., 1., 1.])), ('list_of_data_files', [PosixPath('/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnDet_raw/Task000D3_Example/raw_splitted/imagesTr/case_7_0000.nii.gz')]), ('seg_file', PosixPath('/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnDet_raw/Task000D3_Example/raw_splitted/labelsTr/case_7.nii.gz')), ('itk_origin', (0.0, 0.0, 0.0)), ('itk_spacing', (1.0, 1.0, 1.0)), ('itk_direction', (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)), ('instances', {'1': 0}), ('crop_bbox', [(0, 256), (0, 256), (0, 256)]), ('classes', array([0., 1.], dtype=float32)), ('size_after_cropping', (256, 256, 256)), ('num_instances', {0: 1, 1: 0}), ('has_classes', [0]), ('volume_per_class', OrderedDict([(0, 29791.0), (1, 0)])), ('region_volume_per_class', OrderedDict([(0, [29791.0])])), ('boxes', array([[192,  31, 224,  63,  82, 114]])), ('all_ious', array([], shape=(1, 0), dtype=float64)), ('class_ious', OrderedDict([(0, array([], shape=(1, 0), dtype=float64))]))])), ('case_8', OrderedDict([('original_size_of_raw_data', array([256, 256, 256])), ('original_spacing', array([1., 1., 1.])), ('list_of_data_files', [PosixPath('/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnDet_raw/Task000D3_Example/raw_splitted/imagesTr/case_8_0000.nii.gz')]), ('seg_file', PosixPath('/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnDet_raw/Task000D3_Example/raw_splitted/labelsTr/case_8.nii.gz')), ('itk_origin', (0.0, 0.0, 0.0)), ('itk_spacing', (1.0, 1.0, 1.0)), ('itk_direction', (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)), ('instances', {'1': 0}), ('crop_bbox', [(0, 256), (0, 256), (0, 256)]), ('classes', array([0., 1.], dtype=float32)), ('size_after_cropping', (256, 256, 256)), ('num_instances', {0: 1, 1: 0}), ('has_classes', [0]), ('volume_per_class', OrderedDict([(0, 6859.0), (1, 0)])), ('region_volume_per_class', OrderedDict([(0, [6859.0])])), ('boxes', array([[ 50,  63,  70,  83, 162, 182]])), ('all_ious', array([], shape=(1, 0), dtype=float64)), ('class_ious', OrderedDict([(0, array([], shape=(1, 0), dtype=float64))]))])), ('case_9', OrderedDict([('original_size_of_raw_data', array([256, 256, 256])), ('original_spacing', array([1., 1., 1.])), ('list_of_data_files', [PosixPath('/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnDet_raw/Task000D3_Example/raw_splitted/imagesTr/case_9_0000.nii.gz')]), ('seg_file', PosixPath('/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnDet_raw/Task000D3_Example/raw_splitted/labelsTr/case_9.nii.gz')), ('itk_origin', (0.0, 0.0, 0.0)), ('itk_spacing', (1.0, 1.0, 1.0)), ('itk_direction', (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)), ('instances', {'1': 0}), ('crop_bbox', [(0, 256), (0, 256), (0, 256)]), ('classes', array([0., 1.], dtype=float32)), ('size_after_cropping', (256, 256, 256)), ('num_instances', {0: 1, 1: 0}), ('has_classes', [0]), ('volume_per_class', OrderedDict([(0, 27000.0), (1, 0)])), ('region_volume_per_class', OrderedDict([(0, [27000.0])])), ('boxes', array([[ 21,  81,  52, 112, 141, 172]])), ('all_ious', array([], shape=(1, 0), dtype=float64)), ('class_ious', OrderedDict([(0, array([], shape=(1, 0), dtype=float64))]))]))]), 'num_instances': defaultdict(<class 'int'>, {0: 6, 1: 4}), 'class_ious': defaultdict(<class 'list'>, {1: array([], dtype=float64), 0: array([], dtype=float64)}), 'all_ious': array([], dtype=float64), 'intensity_properties': OrderedDict([(0, OrderedDict([('local_props', OrderedDict([('case_0', {'median': 0.89304155, 'mean': 0.8151397, 'std': 0.20067374, 'min': 0.4000893, 'max': 1.0, 'percentile_99_5': 1.0, 'percentile_00_5': 0.4038710966706276}), ('case_1', {'median': 0.88741374, 'mean': 0.82317114, 'std': 0.19485347, 'min': 0.4008933, 'max': 1.0, 'percentile_99_5': 1.0, 'percentile_00_5': 0.40345924109220505}), ('case_2', {'median': 0.8987428, 'mean': 0.8210632, 'std': 0.197915, 'min': 0.40059903, 'max': 1.0, 'percentile_99_5': 1.0, 'percentile_00_5': 0.4026064185798168}), ('case_3', {'median': 0.92768216, 'mean': 0.8296321, 'std': 0.19717818, 'min': 0.4014379, 'max': 1.0, 'percentile_99_5': 1.0, 'percentile_00_5': 0.41033535107970237}), ('case_4', {'median': 0.9036, 'mean': 0.8236405, 'std': 0.19637193, 'min': 0.40038034, 'max': 1.0, 'percentile_99_5': 1.0, 'percentile_00_5': 0.4043998111784458}), ('case_5', {'median': 0.92180187, 'mean': 0.824384, 'std': 0.20285589, 'min': 0.40010166, 'max': 1.0, 'percentile_99_5': 1.0, 'percentile_00_5': 0.4027950368821621}), ('case_6', {'median': 0.90362275, 'mean': 0.8194111, 'std': 0.20116888, 'min': 0.40165293, 'max': 1.0, 'percentile_99_5': 1.0, 'percentile_00_5': 0.40553907603025435}), ('case_7', {'median': 0.904618, 'mean': 0.82284516, 'std': 0.19817714, 'min': 0.40014488, 'max': 1.0, 'percentile_99_5': 1.0, 'percentile_00_5': 0.4029020240902901}), ('case_8', {'median': 0.9246541, 'mean': 0.8393138, 'std': 0.18853667, 'min': 0.40055338, 'max': 1.0, 'percentile_99_5': 1.0, 'percentile_00_5': 0.4067072279751301}), ('case_9', {'median': 0.9019394, 'mean': 0.820961, 'std': 0.19949464, 'min': 0.4001987, 'max': 1.0, 'percentile_99_5': 1.0, 'percentile_00_5': 0.40474733993411066})])), ('median', 0.90862525), ('mean', 0.82341903), ('std', 0.19815694), ('min', 0.4000893), ('max', 1.0), ('percentile_99_5', 1.0), ('percentile_00_5', 0.4042625856399536)]))])}
# planner_id D3V001
# network_dim 3
# dataloader_kwargs {}
# data_identifier D3V001_3d
# postprocessing {}
# patch_size [128 128 128]
# batch_size 4
# architecture {'arch_name': 'RetinaUNetV001', 'max_channels': 320, 'start_channels': 32, 'fpn_channels': 128, 'head_channels': 128, 'classifier_classes': 2, 'seg_classes': 2, 'in_channels': 1, 'dim': 3, 'class_weight': [0.3333333333333333, 0.2666666666666667, 0.4], 'conv_kernels': [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]], 'strides': [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]], 'decoder_levels': (2, 3, 4, 5)}
# anchors {'width': [(6.0, 7.0, 15.0), (12.0, 14.0, 30.0), (24.0, 28.0, 60.0), (48.0, 56.0, 120.0)], 'height': [(6.0, 7.0, 16.0), (12.0, 14.0, 32.0), (24.0, 28.0, 64.0), (48.0, 56.0, 128.0)], 'depth': [(6.0, 7.0, 16.0), (12.0, 14.0, 32.0), (24.0, 28.0, 64.0), (48.0, 56.0, 128.0)], 'stride': 1}
# target_spacing_transposed [1. 1. 1.]
# median_shape_transposed [256. 256. 256.]
# do_dummy_2D_data_aug False
# trigger_lr1 False


# patch_size [ 80 192 192]
# batch_size 4
# architecture {'arch_name': 'RetinaUNetV001', 'max_channels': 320, 'start_channels': 32, 'fpn_channels': 128, 'head_channels': 128, 'classifier_classes': 1, 'seg_classes': 1, 'in_channels': 1, 'dim': 3, 'class_weight': [0.5, 0.0], 'conv_kernels': [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]], 'strides': [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]], 'decoder_levels': (2, 3, 4, 5)}
# anchors {'width': [(4.0, 6.0, 5.0), (8.0, 12.0, 10.0), (16.0, 24.0, 20.0), (16.0, 24.0, 20.0)], 'height': [(6.0, 8.0, 10.0), (12.0, 16.0, 20.0), (24.0, 32.0, 40.0), (48.0, 64.0, 80.0)], 'depth': [(8.0, 6.0, 10.0), (16.0, 12.0, 20.0), (32.0, 24.0, 40.0), (64.0, 48.0, 80.0)], 'stride': 1}