from run_experiments import run_t1_voxel_with_feature_selection

# ########################
# ### Classification with nested feature selection for T1 GM density maps
# ########################

caps_directory= '/run/user/1000/gvfs/sftp:host=cbica-cluster,user=wenju/cbica/home/wenju/Dataset/Neuroinformatics_CAPS/CAPS_T1'
output_dir = '/run/user/1000/gvfs/sftp:host=cbica-cluster,user=wenju/cbica/home/wenju/Dataset/Neuroinformatics_CAPS/classification_output'
n_threads = 8
n_iterations = 250
test_size = 0.2
grid_search_folds = 10
top_k = [1] + list(range(10, 110, 10))

# ########################
# ### ANOVA nested
# ########################
######  CN vs AD
task='AD_vs_CN_VB'
diagnoses_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_AD_diagnosis.tsv'
subjects_visits_tsv = '/home/hao/Project/aramis/AD-ML-anbai106/AD-ML/Paper_Specific_Versions/2019_DTI/Code/subjects_lists/CN_vs_AD.tsv'
feature_selection_nested = True
feature_selection_method = 'ANOVA'
# run_t1_voxel_with_feature_selection(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, task, n_threads, n_iterations, test_size, grid_search_folds, top_k=top_k,
#                                      feature_selection_nested=feature_selection_nested, feature_selection_method=feature_selection_method)

# ########################
# ### RFE nested
# ########################
feature_selection_nested = True
feature_selection_method = 'RFE'
run_t1_voxel_with_feature_selection(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, task, n_threads, n_iterations, test_size, grid_search_folds, top_k=top_k,
                                     feature_selection_nested=feature_selection_nested, feature_selection_method=feature_selection_method)
