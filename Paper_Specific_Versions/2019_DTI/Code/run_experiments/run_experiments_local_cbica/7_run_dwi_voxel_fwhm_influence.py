from run_experiments import run_dwi_voxel_with_feature_rescaling

# ########################
# ### Classification with feature rescaling, with diff number of fwhm
# ########################

caps_directory= '/run/user/1000/gvfs/sftp:host=cbica-cluster,user=wenju/cbica/home/wenju/Dataset/Neuroinformatics_CAPS/CAPS_DWI'
output_dir = '/run/user/1000/gvfs/sftp:host=cbica-cluster,user=wenju/cbica/home/wenju/Dataset/Neuroinformatics_CAPS/classification_output'
n_threads = 8
n_iterations = 250
test_size = 0.2
grid_search_folds = 10
caps_reg_method = 'single_modal'
fwhm = [0, 4, 8, 12]


# ########################
# ### DWI Voxel
# ########################
######  CN vs AD
task='AD_vs_CN_VB'
diagnoses_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_AD_diagnosis.tsv'
subjects_visits_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_AD.tsv'
run_dwi_voxel_with_feature_rescaling(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, task, n_threads, n_iterations, test_size, grid_search_folds, caps_reg_method=caps_reg_method, fwhm=fwhm)