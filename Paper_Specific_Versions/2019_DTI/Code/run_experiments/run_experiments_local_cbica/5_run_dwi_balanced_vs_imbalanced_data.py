from run_experiments import run_dwi_voxel_with_feature_rescaling, run_dwi_roi_with_feature_rescaling


caps_directory= '/run/user/1000/gvfs/sftp:host=cbica-cluster,user=wenju/cbica/home/wenju/Dataset/Neuroinformatics_CAPS/CAPS_DWI'
output_dir = '/run/user/1000/gvfs/sftp:host=cbica-cluster,user=wenju/cbica/home/wenju/Dataset/Neuroinformatics_CAPS/classification_output'
n_threads = 8
n_iterations = 250
test_size = 0.2
grid_search_folds = 10
caps_reg_method = 'single_modal'
balanced_down_sample = True
atlas= ['JHUDTI81', 'JHUTracts25']

# ########################
# ### Balanced classification voxel-wise
# ########################


#####  CN vs pMCI
task='CN_vs_pMCI_VB'
subjects_visits_tsv  = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_pMCI.tsv'
diagnoses_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_pMCI_diagnosis.tsv'
run_dwi_voxel_with_feature_rescaling(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, task, n_threads, n_iterations, test_size, grid_search_folds, caps_reg_method=caps_reg_method, balanced_down_sample=balanced_down_sample)

## TODO run on cluster, memory issue
# ######  CN vs MCI
# task='CN_vs_MCI_VB'
# subjects_visits_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_MCI.tsv'
# diagnoses_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_MCI_diagnosis.tsv'
# run_dwi_voxel_with_feature_rescaling(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, task, n_threads, n_iterations, test_size, grid_search_folds, caps_reg_method=caps_reg_method, balanced_down_sample=balanced_down_sample)

######  sMCI vs pMCI
task='sMCI_vs_pMCI_VB'
diagnoses_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/sMCI_vs_pMCI_diagnosis.tsv'
subjects_visits_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/sMCI_vs_pMCI.tsv'
run_dwi_voxel_with_feature_rescaling(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, task, n_threads, n_iterations, test_size, grid_search_folds, caps_reg_method=caps_reg_method, balanced_down_sample=balanced_down_sample)


# ########################
# ### Balanced classification ROI-wise
# ########################

#####  CN vs pMCI
task='CN_vs_pMCI_RB'
subjects_visits_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_pMCI.tsv'
diagnoses_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_pMCI_diagnosis.tsv'
run_dwi_roi_with_feature_rescaling(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, atlas,
                                task, n_threads, n_iterations, test_size, grid_search_folds, balanced_down_sample=balanced_down_sample)

######  CN vs MCI
task='CN_vs_MCI_RB'
subjects_visits_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_MCI.tsv'
diagnoses_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_MCI_diagnosis.tsv'
run_dwi_roi_with_feature_rescaling(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, atlas,
                                task, n_threads, n_iterations, test_size, grid_search_folds, balanced_down_sample=balanced_down_sample)

######  sMCI vs pMCI
task='sMCI_vs_pMCI_RB'
diagnoses_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/sMCI_vs_pMCI_diagnosis.tsv'
subjects_visits_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/sMCI_vs_pMCI.tsv'
run_dwi_roi_with_feature_rescaling(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, atlas,
                                task, n_threads, n_iterations, test_size, grid_search_folds, balanced_down_sample=balanced_down_sample)


# ########################
# ### Original classification voxel-wise without balanced_sklearn
# ########################
#####  CN vs pMCI
task='CN_vs_pMCI_VB'
subjects_visits_tsv  = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_pMCI.tsv'
diagnoses_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_pMCI_diagnosis.tsv'
run_dwi_voxel_with_feature_rescaling(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, task, n_threads, n_iterations, test_size, grid_search_folds, caps_reg_method=caps_reg_method, balance_sklearn=False)

## TODO run on cluster, memory issue
# ######  CN vs MCI
# task='CN_vs_MCI_VB'
# subjects_visits_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_MCI.tsv'
# diagnoses_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_MCI_diagnosis.tsv'
# run_dwi_voxel_with_feature_rescaling(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, task, n_threads, n_iterations, test_size, grid_search_folds, caps_reg_method=caps_reg_method, balance_sklearn=False)

######  sMCI vs pMCI
task='sMCI_vs_pMCI_VB'
diagnoses_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/sMCI_vs_pMCI_diagnosis.tsv'
subjects_visits_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/sMCI_vs_pMCI.tsv'
run_dwi_voxel_with_feature_rescaling(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, task, n_threads, n_iterations, test_size, grid_search_folds, caps_reg_method=caps_reg_method, balance_sklearn=False)

# ########################
# ### Original classification ROI-wise without balanced_sklearn
# ########################

#####  CN vs pMCI
task='CN_vs_pMCI_RB'
subjects_visits_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_pMCI.tsv'
diagnoses_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_pMCI_diagnosis.tsv'
run_dwi_roi_with_feature_rescaling(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, atlas,
                                task, n_threads, n_iterations, test_size, grid_search_folds, balance_sklearn=False)

######  CN vs MCI
task='CN_vs_MCI_RB'
subjects_visits_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_MCI.tsv'
diagnoses_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_MCI_diagnosis.tsv'
run_dwi_roi_with_feature_rescaling(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, atlas,
                                task, n_threads, n_iterations, test_size, grid_search_folds, balance_sklearn=False)

######  sMCI vs pMCI
task='sMCI_vs_pMCI_RB'
diagnoses_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/sMCI_vs_pMCI_diagnosis.tsv'
subjects_visits_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/sMCI_vs_pMCI.tsv'
run_dwi_roi_with_feature_rescaling(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, atlas,
                                task, n_threads, n_iterations, test_size, grid_search_folds, balance_sklearn=False)