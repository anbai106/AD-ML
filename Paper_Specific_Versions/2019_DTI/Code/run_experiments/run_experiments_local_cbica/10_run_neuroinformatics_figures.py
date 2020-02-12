from run_experiments import classification_performances_violin_plot_neuroinformatics

# ########################
# ### Figure 1 imaging modality
# ########################
classification_output_dir = '/run/user/1000/gvfs/sftp:host=cbica-cluster,user=wenju/cbica/home/wenju/Dataset/Neuroinformatics_CAPS/classification_output'
task_name = 'T1_vs_DTI'
classification_performances_violin_plot_neuroinformatics(classification_output_dir, task_name)