from run_experiments import classification_performances_violin_plot_neuroinformatics, classification_performances_scatter_plot_feature_selection_neuroinformatics

# ########################
# ### Figure imaging modality ## TODO may need change at the end, NOT USED IN THE PAPER
# ########################
classification_output_dir = '/run/user/1000/gvfs/sftp:host=cbica-cluster,user=wenju/cbica/home/wenju/Dataset/Neuroinformatics_CAPS/classification_output'
task_name = 'T1_vs_DTI'
# classification_performances_violin_plot_neuroinformatics(classification_output_dir, task_name)

# ########################
# ### Figure  imbalanded data
# ########################
classification_output_dir = '/run/user/1000/gvfs/sftp:host=cbica-cluster,user=wenju/cbica/home/wenju/Dataset/Neuroinformatics_CAPS/classification_output'
task_name = 'Balanced_vs_imbalanced'
# classification_performances_violin_plot_neuroinformatics(classification_output_dir, task_name)

# ########################
# ### Figure influence of smoothing
# ########################
classification_output_dir = '/run/user/1000/gvfs/sftp:host=cbica-cluster,user=wenju/cbica/home/wenju/Dataset/Neuroinformatics_CAPS/classification_output'
task_name = 'Influence_of_fwhm'
# classification_performances_violin_plot_neuroinformatics(classification_output_dir, task_name)

# ########################
# ### Figure influence of smoothing
# ########################
classification_output_dir = '/run/user/1000/gvfs/sftp:host=cbica-cluster,user=wenju/cbica/home/wenju/Dataset/Neuroinformatics_CAPS/classification_output'
task_name = 'Influence_of_reg'
# classification_performances_violin_plot_neuroinformatics(classification_output_dir, task_name)

# ########################
# ### Figure influence of feature selection bias
# ########################
classification_result_path = '/run/user/1000/gvfs/sftp:host=cbica-cluster,user=wenju/cbica/home/wenju/Dataset/Neuroinformatics_CAPS/classification_output'
metric = 'fa'
task_name = 'Influence_of_feature_selection'
classification_performances_scatter_plot_feature_selection_neuroinformatics(classification_result_path, task_name, metric=metric, fs_technique='ANOVA+RFE')
metric = 'md'
classification_performances_scatter_plot_feature_selection_neuroinformatics(classification_result_path, task_name, metric=metric, fs_technique='ANOVA+RFE')

# ########################
# ### Figure influence of modality
# ########################
classification_result_path = '/run/user/1000/gvfs/sftp:host=cbica-cluster,user=wenju/cbica/home/wenju/Dataset/Neuroinformatics_CAPS/classification_output'
task_name = 'Influence_of_modality'
classification_performances_scatter_plot_feature_selection_neuroinformatics(classification_result_path, task_name, metric=metric, fs_technique='RFE')
