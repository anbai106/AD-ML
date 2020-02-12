# -*- coding: utf-8 -*-
__author__ = ["Junhao Wen", "Jorge Samper-Gonzalez"]
__copyright__ = "Copyright 2016-2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__status__ = "Development"

import os
from pandas.io import parsers
import numpy as np
import pandas as pd
from scipy import stats

def random_donsample_subjects(diagnoses_tsv, n=None):
    """
    This function is to randomly downsample the subjects.
    :param diagnoses_tsv:
    :return:
    """
    from collections import Counter
    import os

    diagnoses = pd.io.parsers.read_csv(diagnoses_tsv, sep='\t')
    print 'Do random subsampling the majority group to number of subjects of minority group:'
    counts = Counter(list(diagnoses.diagnosis))
    label1 = counts.keys()[0]
    label2 = counts.keys()[1]
    count_label1 = counts[label1]
    count_label2 = counts[label2]
    if count_label1 < count_label2:
        print '%s is the majority group and will be randomly downsampled.' % label2
        majority_df_index = diagnoses.index[diagnoses['diagnosis'] == label2]
        drop_index = np.random.choice(majority_df_index, count_label2 - count_label1, replace= False)
        diagnoses_balanced = diagnoses.drop(drop_index)

    elif count_label1 > count_label2:
        print '%s is the majority group and will be randomly downsampled.' % label1
        majority_df_index = diagnoses.index[diagnoses['diagnosis'] == label1]
        drop_index = np.random.choice(majority_df_index, count_label1 - count_label2, replace= False)
        diagnoses_balanced = diagnoses.drop(drop_index)
    else:
        raise Exception("""The data is balanced already, please deactivate the balanced_down_sample flag""")
    # save the balanced tsv
    if n == None:
        if os.path.isfile(os.path.join(os.path.dirname(diagnoses_tsv), os.path.basename(diagnoses_tsv).split('.')[0] + '_balanced.tsv')):
            pass
        else:
            diagnoses_balanced.to_csv(os.path.join(os.path.dirname(diagnoses_tsv), os.path.basename(diagnoses_tsv).split('.')[0] + '_balanced.tsv'), sep='\t', index=False)
    else:
        if os.path.isfile(os.path.join(os.path.dirname(diagnoses_tsv), os.path.basename(diagnoses_tsv).split('.')[0] + '_' + str(n) + '_balanced.tsv')):
            pass
        else:
            diagnoses_balanced.to_csv(os.path.join(os.path.dirname(diagnoses_tsv), os.path.basename(diagnoses_tsv).split('.')[0] + '_' + str(n) + '_balanced.tsv'), sep='\t', index=False)

    list_diagnoses = list(diagnoses_balanced.diagnosis)
    list_subjects = list(diagnoses_balanced.participant_id)
    list_sessions = list(diagnoses_balanced.session_id)

    return list_subjects, list_sessions, list_diagnoses

def split_subjects_to_pickle(diagnoses_tsv, n_iterations=250, test_size=0.2, balanced=False):

    from os import path
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit
    import pickle
    from collections import Counter

    diagnoses = pd.io.parsers.read_csv(diagnoses_tsv, sep='\t')
    if 'diagnosis' not in list(diagnoses.columns.values):
        raise Exception('Diagnoses file is not in the correct format.')
    diagnoses_list = list(diagnoses.diagnosis)
    unique = list(set(diagnoses_list))
    y = np.array([unique.index(x) for x in diagnoses_list])

    if balanced == False:
        splits_indices_pickle = path.join(path.dirname(diagnoses_tsv), path.basename(diagnoses_tsv).split('.')[0] + '.pkl')
    else:
        splits_indices_pickle = path.join(path.dirname(diagnoses_tsv), path.basename(diagnoses_tsv).split('.')[0] + '_balanced.pkl')

    ## try to see if the shuffle has been done
    if os.path.isfile(splits_indices_pickle):
        splits_indices = pickle.load(open(splits_indices_pickle, 'rb'))
    else:
        splits = StratifiedShuffleSplit(n_splits=n_iterations, test_size=test_size)
        if balanced == False:
            splits_indices = list(splits.split(np.zeros(len(y)), y))
        else:
            print 'Do random subsampling the majority group to number of subjects of minority group:'
            splits_indices = []
            n_iteration = 0
            for train_index, test_index in splits.split(np.zeros(len(y)), y):

                # for training
                train_label1 = []
                train_label2 = []
                counts = Counter(diagnoses.diagnosis[train_index])
                label1 = counts.keys()[0]
                label2 = counts.keys()[1]
                count_label1 = counts[label1]
                count_label2 = counts[label2]
                for i  in train_index:
                    if diagnoses.diagnosis[i] == label2:
                        train_label2.append(i)
                    else:
                        train_label1.append(i)
                if count_label1 < count_label2:
                    print 'In training data for iteration %d, %s is the majority group and will be randomly downsampled.' % (n_iteration, label2)
                    drop_index_train = np.random.choice(train_label2, count_label2 - count_label1, replace=False)
                    train_index_balanced = np.asarray([item for item in train_index if item not in drop_index_train])
                elif count_label1 > count_label2:
                    print 'In training data for iteration %d, %s is the majority group and will be randomly downsampled.' % (n_iteration, label1)
                    drop_index_train = np.random.choice(train_label1, count_label1 - count_label2, replace=False)
                    train_index_balanced = np.asarray([item for item in train_index if item not in drop_index_train])
                else:
                    raise Exception("""The data is balanced already, please deactivate the balanced_down_sample flag""")

                # for test
                test_label1 = []
                test_label2 = []
                counts = Counter(diagnoses.diagnosis[test_index])
                label1 = counts.keys()[0]
                label2 = counts.keys()[1]
                count_label1 = counts[label1]
                count_label2 = counts[label2]
                for i  in test_index:
                    if diagnoses.diagnosis[i] == label2:
                        test_label2.append(i)
                    else:
                        test_label1.append(i)
                if count_label1 < count_label2:
                    print 'In test data for iteration %d, %s is the majority group and will be randomly downsampled.' % (n_iteration, label2)
                    drop_index_test= np.random.choice(test_label2, count_label2 - count_label1, replace=False)
                    test_index_balanced = np.asarray([item for item in test_index if item not in drop_index_test])
                elif count_label1 > count_label2:
                    print 'In test data for iteration %d, %s is the majority group and will be randomly downsampled.' % (n_iteration, label1)
                    drop_index_test = np.random.choice(test_label1, count_label1 - count_label2, replace=False)
                    test_index_balanced = np.asarray([item for item in test_index if item not in drop_index_test])
                else:
                    raise Exception("""The data is balanced already, please deactivate the balanced_down_sample flag""")
                ##
                n_iteration += 1
                splits_indices.append((train_index_balanced, test_index_balanced))
                ## save each iteration as tsv files
                diagnoses_balanced_tsv = diagnoses.drop(np.append(drop_index_train, drop_index_test))
                diagnoses_balanced_tsv.to_csv(os.path.join(os.path.dirname(diagnoses_tsv),
                                                       os.path.basename(diagnoses_tsv).split('.')[0] + '_' + str(
                                                           n_iteration) + '_balanced.tsv'), sep='\t', index=False)
    with open(splits_indices_pickle, 'wb') as s:
        pickle.dump(splits_indices, s)

    return splits_indices, splits_indices_pickle

def compute_t(subjects_1_tsv, subjects_2_tsv, test_size=0.2):
    """
    This is a function to compute the corrected resampled paired t-test based on the paper of Nadeau and Bengio 2003.
    Also please refer this post to understand the different metrics used to compare two classifiers (https://stats.stackexchange.com/questions/217466/for-model-selection-comparison-what-kind-of-test-should-i-use)

    Also, please refer the package mlxtend.evaluate, but they did not include the corrected resampled paired t-test

    :param subjects_1_tsv:
    :param subjects_2_tsv:
    :param test_size:
    :return:
    """

    subjects_1 = pd.io.parsers.read_csv(subjects_1_tsv, sep='\t')
    subjects_2 = pd.io.parsers.read_csv(subjects_2_tsv, sep='\t')

    num_split = len(subjects_1.iteration.unique())
    n_subj = subjects_1.shape[0] / num_split

    test_error_split = np.zeros((num_split, 1))  # this list will contain the list of mu_j hat for j = 1 to J

    q1 = (subjects_1.y == subjects_1.y_hat) * 1.0
    q2 = (subjects_2.y == subjects_2.y_hat) * 1.0
    l = q1 - q2

    for i in range(num_split):
        test_error_split[i] = np.mean(l[(i * n_subj):((i + 1) * n_subj)])

    # compute mu_{n_1}^{n_2}
    average_test_error = np.mean(test_error_split)

    # compute S2_{mu_J}
    approx_variance = np.sum((test_error_split - average_test_error) ** 2)

    resampled_t = average_test_error * np.sqrt(num_split) / np.sqrt(approx_variance / (num_split - 1))

    resampled_p_value = stats.t.sf(np.abs(resampled_t), num_split - 1) * 2.

    corrected_resampled_t = average_test_error * np.sqrt(num_split) / np.sqrt((test_size / (1 - test_size) + 1/(num_split - 1)) * approx_variance)

    corrected_resampled_p_value = stats.t.sf(np.abs(corrected_resampled_t), num_split - 1) * 2.

    return resampled_t, resampled_p_value, corrected_resampled_t, corrected_resampled_p_value

def classification_performances_dot_plot_feature_selection(classification_result_path, fs_methods, tasks, metric = 'fa', fs_technique='ANOVA+RFE', tissue='GM_WM'):
    """
    :return:
    """
    import os, errno
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt

    results_balanced_acc_nested_anova = []
    results_balanced_acc_nonnested_anova = []
    results_balanced_acc_nested_rfe = []
    results_balanced_acc_nonnested_rfe = []

    x_persent_voxels = ['0.001', '0.01', '0.1', '0.5', '1', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']

    if fs_technique == 'ANOVA+RFE':

        for task in tasks:
            for threshold in x_persent_voxels:
                result_tsv = os.path.join(classification_result_path, 'feature_selection_results', fs_methods[0], task + '_VB_' + tissue + '_0.3_8_fs_' + str(threshold), metric, 'mean_results.tsv')
                balanced_accuracy = []

                if os.path.isfile(result_tsv):
                    balanced_accuracy.append((pd.io.parsers.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                else:
                    raise OSError(
                        errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                results_balanced_acc_nested_anova.append(balanced_accuracy)

        for task in tasks:
            for threshold in x_persent_voxels:
                result_tsv = os.path.join(classification_result_path, 'feature_selection_results', fs_methods[1], task + '_VB_' + tissue + '_0.3_8_fs_' + str(threshold), metric, 'mean_results.tsv')
                balanced_accuracy = []

                if os.path.isfile(result_tsv):
                    balanced_accuracy.append((pd.io.parsers.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                else:
                    raise OSError(
                        errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                results_balanced_acc_nonnested_anova.append(balanced_accuracy)

        for task in tasks:
            for threshold in x_persent_voxels:
                result_tsv = os.path.join(classification_result_path, 'feature_selection_results', fs_methods[2], task + '_VB_' + tissue + '_0.3_8_fs_' + str(threshold), metric, 'mean_results.tsv')
                balanced_accuracy = []

                if os.path.isfile(result_tsv):
                    balanced_accuracy.append((pd.io.parsers.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                else:
                    raise OSError(
                        errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                results_balanced_acc_nested_rfe.append(balanced_accuracy)

        for task in tasks:
            for threshold in x_persent_voxels:
                result_tsv = os.path.join(classification_result_path, 'feature_selection_results', fs_methods[3], task + '_VB_' + tissue + '_0.3_8_fs_' + str(threshold), metric, 'mean_results.tsv')
                balanced_accuracy = []

                if os.path.isfile(result_tsv):
                    balanced_accuracy.append((pd.io.parsers.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                else:
                    raise OSError(
                        errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                results_balanced_acc_nonnested_rfe.append(balanced_accuracy)


        num_rois = len(x_persent_voxels)
        # x = np.asarray(x_persent_voxels, dtype=float)
        x = np.arange(1, num_rois+1, 1)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Remove the plot frame lines. They are unnecessary chartjunk.
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.scatter(x, results_balanced_acc_nested_anova, s=100, c='g', marker="^", label='nested ANOVA')
        ax.scatter(x, results_balanced_acc_nonnested_anova, s=100, c='r', marker="^", label='non-nested ANOVA')
        ax.scatter(x, results_balanced_acc_nested_rfe, s=100, c='b', marker="o", label='nested SVM-RFE')
        ax.scatter(x, results_balanced_acc_nonnested_rfe, s=100, c='y', marker="o", label='non-nested SVM-RFE')

        plt.plot(x, results_balanced_acc_nested_anova, c='g', linestyle='-')
        plt.plot(x, results_balanced_acc_nonnested_anova, c='r', linestyle='-')
        plt.plot(x, results_balanced_acc_nested_rfe, c='b', linestyle='--')
        plt.plot(x, results_balanced_acc_nonnested_rfe, c='y', linestyle='--')

        plt.legend(loc='upper right',fontsize=25, frameon=False)
        # plt.xticks(x, x_persent_voxels, rotation='vertical')
        plt.xticks(x, x_persent_voxels)
        plt.tick_params(axis='x', labelsize=20)
        plt.tick_params(axis='y', labelsize=25)
        # plt.xlabel('ROIs', fontsize=5)
        # plt.ylabel('f2', fontsize=5)
        plt.ylim(0.4, 1.1)
        plt.show()

    else:
        print 'Not implemented yet'
    print 'finish'