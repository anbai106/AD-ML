from run_experiments import classification_performances_violin_plot_neuroinformatics

# ########################
# ### Figure imaging modality ## TODO may need change at the end
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
classification_performances_violin_plot_neuroinformatics(classification_output_dir, task_name)