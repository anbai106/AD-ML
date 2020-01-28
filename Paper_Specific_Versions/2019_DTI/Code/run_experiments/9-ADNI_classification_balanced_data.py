from mlworkflow_dwi_utils import run_voxel_based_classification

# ########################
# ### Original classification
# ########################

caps_directory= 'PATH/TO/CAPS'
output_dir = 'PATH/TO/CLASSIFICATION_OUTPUT'
n_threads = 72
n_iterations = 250
test_size = 0.2
grid_search_folds = 10

# ########################
# ### Balanced classification
# ########################
balanced_down_sample = True

#####  CN vs pMCI
task='CN_vs_pMCI_VB'
diagnoses_tsv = 'PATH/TO/DIAGONISIS_TSV'
subjects_visits_tsv = 'PATH/TO/SUBJECTS_VISITS_TSV'
run_voxel_based_classification(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir,
                                task, n_threads, n_iterations, test_size, grid_search_folds, balanced_down_sample=balanced_down_sample)

######  CN vs MCI
task='CN_vs_MCI_VB'
diagnoses_tsv = 'PATH/TO/DIAGONISIS_TSV'
subjects_visits_tsv = 'PATH/TO/SUBJECTS_VISITS_TSV'
run_voxel_based_classification(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir,
                                task, n_threads, n_iterations, test_size, grid_search_folds, balanced_down_sample=balanced_down_sample)

######  sMCI vs pMCI
task='sMCI_vs_pMCI_VB'
diagnoses_tsv = 'PATH/TO/DIAGONISIS_TSV'
subjects_visits_tsv = 'PATH/TO/SUBJECTS_VISITS_TSV'
run_voxel_based_classification(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir,
                                task, n_threads, n_iterations, test_size, grid_search_folds, balanced_down_sample=balanced_down_sample)