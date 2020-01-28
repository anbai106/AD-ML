from run_experiments import run_t1_voxel_without_feature_rescaling, run_t1_roi_without_feature_rescaling

# ########################
# ### Original classification
# ########################

caps_directory= '/run/user/1000/gvfs/sftp:host=cbica-cluster,user=wenju/cbica/home/wenju/Dataset/Neuroinformatics_CAPS/CAPS_T1'
output_dir = '/run/user/1000/gvfs/sftp:host=cbica-cluster,user=wenju/cbica/home/wenju/Dataset/Neuroinformatics_CAPS/classification_output'
n_threads = 8
n_iterations = 250
test_size = 0.2
grid_search_folds = 10

# ########################
# ### T1 Voxel
# ########################
######  CN vs AD
task='AD_vs_CN_VB'
diagnoses_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_AD_diagnosis.tsv'
subjects_visits_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_AD.tsv'
run_t1_voxel_without_feature_rescaling(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, task, n_threads, n_iterations, test_size, grid_search_folds)

#####  CN vs pMCI
task='CN_vs_pMCI_VB'
subjects_visits_tsv  = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_pMCI.tsv'
diagnoses_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_pMCI_diagnosis.tsv'
run_t1_voxel_without_feature_rescaling(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, task, n_threads, n_iterations, test_size, grid_search_folds)

######  CN vs MCI
task='CN_vs_MCI_VB'
subjects_visits_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_MCI.tsv'
diagnoses_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_MCI_diagnosis.tsv'
run_t1_voxel_without_feature_rescaling(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, task, n_threads, n_iterations, test_size, grid_search_folds)

######  sMCI vs pMCI
task='sMCI_vs_pMCI_VB'
diagnoses_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/sMCI_vs_pMCI_diagnosis.tsv'
subjects_visits_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/sMCI_vs_pMCI.tsv'
run_t1_voxel_without_feature_rescaling(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, task, n_threads, n_iterations, test_size, grid_search_folds)


# ########################
# ### T1 ROI
# ########################
######  CN vs AD
task='AD_vs_CN_RB'
atlas= ['AAL2']
diagnoses_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_AD_diagnosis.tsv'
subjects_visits_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_AD.tsv'
run_t1_roi_without_feature_rescaling(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, atlas,
                                task, n_threads, n_iterations, test_size, grid_search_folds, group_id='ADNIbl')
#####  CN vs pMCI
task='CN_vs_pMCI_RB'
subjects_visits_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_pMCI.tsv'
diagnoses_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_pMCI_diagnosis.tsv'
run_t1_roi_without_feature_rescaling(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, atlas,
                                task, n_threads, n_iterations, test_size, grid_search_folds, group_id='ADNIbl')

######  CN vs MCI
task='CN_vs_MCI_RB'
subjects_visits_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_MCI.tsv'
diagnoses_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_MCI_diagnosis.tsv'
run_t1_roi_without_feature_rescaling(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, atlas,
                                task, n_threads, n_iterations, test_size, grid_search_folds, group_id='ADNIbl')

######  sMCI vs pMCI
task='sMCI_vs_pMCI_RB'
diagnoses_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/sMCI_vs_pMCI_diagnosis.tsv'
subjects_visits_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/sMCI_vs_pMCI.tsv'
run_t1_roi_without_feature_rescaling(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, atlas,
                                task, n_threads, n_iterations, test_size, grid_search_folds, group_id='ADNIbl')
