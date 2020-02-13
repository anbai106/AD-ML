from run_experiments import run_dwi_voxel_with_feature_selection

# ########################
# ### Classification with nested feature selection and non-nested one.
# ########################

caps_directory= '/run/user/1000/gvfs/sftp:host=cbica-cluster,user=wenju/cbica/home/wenju/Dataset/Neuroinformatics_CAPS/CAPS_DWI'
output_dir = '/run/user/1000/gvfs/sftp:host=cbica-cluster,user=wenju/cbica/home/wenju/Dataset/Neuroinformatics_CAPS/classification_output'
n_threads = 8
n_iterations = 250
test_size = 0.2
grid_search_folds = 10
top_k = [0.1, 0.001, 0.01, 1] + list(range(10, 110, 10))
tissue_type = ['GM_WM']

# ########################
# ### Consider first Feautre rescaling.
# ########################

# ########################
# ### ANOVA nested
# ########################
######  CN vs AD
task='AD_vs_CN_VB'
diagnoses_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_AD_diagnosis.tsv'
subjects_visits_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_AD.tsv'
feature_selection_nested = True
feature_selection_method = 'ANOVA'
run_dwi_voxel_with_feature_selection(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, task, n_threads, n_iterations, test_size, grid_search_folds, top_k=top_k,
                                     feature_selection_nested=feature_selection_nested, feature_selection_method=feature_selection_method, tissue_type=tissue_type)

# ########################
# ### RFE nested
# ########################
feature_selection_nested = True
feature_selection_method = 'RFE'
run_dwi_voxel_with_feature_selection(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, task, n_threads, n_iterations, test_size, grid_search_folds, top_k=top_k,
                                     feature_selection_nested=feature_selection_nested, feature_selection_method=feature_selection_method, tissue_type=tissue_type)

# ########################
# ### ANOVA non-nested
# ########################
######  CN vs AD
feature_selection_nested = False
feature_selection_method = 'ANOVA'
run_dwi_voxel_with_feature_selection(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, task, n_threads, n_iterations, test_size, grid_search_folds, top_k=top_k,
                                     feature_selection_nested=feature_selection_nested, feature_selection_method=feature_selection_method, tissue_type=tissue_type)

# ########################
# ### RFE non-nested
# ########################
feature_selection_nested = False
feature_selection_method = 'RFE'
run_dwi_voxel_with_feature_selection(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, task, n_threads, n_iterations, test_size, grid_search_folds, top_k=top_k,
                                     feature_selection_nested=feature_selection_nested, feature_selection_method=feature_selection_method, tissue_type=tissue_type)

# ########################
# ### Consider no Feautre rescaling.
# ########################

# ########################
# ### ANOVA nested
# ########################
######  CN vs AD
task='AD_vs_CN_VB'
diagnoses_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_AD_diagnosis.tsv'
subjects_visits_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_AD.tsv'
feature_selection_nested = True
feature_selection_method = 'ANOVA'
feature_rescaling_method = None
run_dwi_voxel_with_feature_selection(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, task, n_threads, n_iterations, test_size, grid_search_folds, top_k=top_k,
                                     feature_selection_nested=feature_selection_nested, feature_selection_method=feature_selection_method, tissue_type=tissue_type, feature_rescaling_method=feature_rescaling_method)

# ########################
# ### RFE nested
# ########################
feature_selection_nested = True
feature_selection_method = 'RFE'
run_dwi_voxel_with_feature_selection(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, task, n_threads, n_iterations, test_size, grid_search_folds, top_k=top_k,
                                     feature_selection_nested=feature_selection_nested, feature_selection_method=feature_selection_method, tissue_type=tissue_type, feature_rescaling_method=feature_rescaling_method)

# ########################
# ### ANOVA non-nested
# ########################
######  CN vs AD
feature_selection_nested = False
feature_selection_method = 'ANOVA'
run_dwi_voxel_with_feature_selection(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, task, n_threads, n_iterations, test_size, grid_search_folds, top_k=top_k,
                                     feature_selection_nested=feature_selection_nested, feature_selection_method=feature_selection_method, tissue_type=tissue_type, feature_rescaling_method=feature_rescaling_method)

# ########################
# ### RFE non-nested
# ########################
feature_selection_nested = False
feature_selection_method = 'RFE'
run_dwi_voxel_with_feature_selection(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, task, n_threads, n_iterations, test_size, grid_search_folds, top_k=top_k,
                                     feature_selection_nested=feature_selection_nested, feature_selection_method=feature_selection_method, tissue_type=tissue_type, feature_rescaling_method=feature_rescaling_method)