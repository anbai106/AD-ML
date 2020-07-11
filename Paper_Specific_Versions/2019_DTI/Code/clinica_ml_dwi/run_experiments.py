# -*- coding: utf-8 -*-
__author__ = ["Junhao Wen", "Jorge Samper-Gonzalez"]
__copyright__ = "Copyright 2016-2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__status__ = "Development"

from mlworkflow_dwi import DWI_VB_RepHoldOut_DualSVM, DWI_RB_RepHoldOut_DualSVM, DWI_RB_RepHoldOut_DualSVM_FeatureSelectionNested, \
    DWI_VB_RepHoldOut_DualSVM_FeatureSelectionNested, DWI_VB_RepHoldOut_DualSVM_FeatureSelectionNonNested, T1_RB_RepHoldOut_DualSVM_FeatureSelectionNested, \
    T1_VB_RepHoldOut_DualSVM_FeatureSelectionNested
from clinica.pipelines.machine_learning.ml_workflows import RB_RepHoldOut_DualSVM, VB_RepHoldOut_DualSVM
from os import path
from mlworkflow_dwi_utils import *
import matplotlib.pyplot as plt
import errno

#############################
# DWI voxel-wise without feature rescaling
#############################

def run_dwi_voxel_without_feature_rescaling(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, task, n_threads, n_iterations, test_size, grid_search_folds, dwi_maps=['fa', 'md'], fwhm=[8],
                                   tissue_type=['WM', 'GM', 'GM_WM'], threshold=[0.3], caps_reg_method='single_modal'):
    """
    This is to run voxel-wise classification for DTI without feature rescaling.
    Args:
        caps_directory:
        diagnoses_tsv:
        subjects_visits_tsv:
        output_dir:
        task:
        n_threads:
        n_iterations:
        test_size:
        grid_search_folds:
        dwi_maps:
        fwhm:
        tissue_type:
        threshold:
        group_id:
        caps_reg_method: single_modal or multi_modal

    Returns:

    """

    splits_indices, splits_indices_pickle = split_subjects_to_pickle(diagnoses_tsv, n_iterations=n_iterations, test_size=test_size)

    print("Original experiments voxel-wise DWI experiments without feature rescaling and any other tricks")
    for dwi_map in dwi_maps:
        for i in tissue_type:
            for j in threshold:
                for k in fwhm:
                    classification_dir = path.join(output_dir, 'DWIOriginalWithoutFeatureRescalingVoxel',
                                                   task + '_' + i + '_' + str(j) + '_' + str(
                                                       k) + '_' + dwi_map)
                    if not path.exists(classification_dir):
                        os.makedirs(classification_dir)

                        print("\nRunning %s" % classification_dir)
                        wf = DWI_VB_RepHoldOut_DualSVM(caps_directory, subjects_visits_tsv, diagnoses_tsv, dwi_map,
                                                       i, j, classification_dir, fwhm=k,
                                                       n_threads=n_threads, n_iterations=n_iterations,
                                                       test_size=test_size,
                                                       grid_search_folds=grid_search_folds, splits_indices=splits_indices, caps_reg_method=caps_reg_method)
                        wf.run()
                    else:
                        print("This combination has been classified, just skip: %s " % classification_dir)

#############################
# DWI ROI-wise without feature rescaling
#############################
def run_dwi_roi_without_feature_rescaling(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, atlas,
                                task, n_threads, n_iterations, test_size, grid_search_folds,
                                 dwi_maps=['md']):
    """
        This is to run ROI-wise classification for DTI without feature rescaling.
    Args:
        caps_directory:
        diagnoses_tsv:
        subjects_visits_tsv:
        output_dir:
        atlas: ['JHUDTI81', 'JHUTracts0', 'JHUTracts25']
        task:
        n_threads:
        n_iterations:
        test_size:
        grid_search_folds:
        balanced_down_sample:
        dwi_maps:

    Returns:

    """
    splits_indices, splits_indices_pickle = split_subjects_to_pickle(diagnoses_tsv, n_iterations=n_iterations,
                                                                     test_size=test_size)

    print("Original experiments ROI-wise DWI experiments without feature rescaling and any other tricks")

    for dwi_map in dwi_maps:
        for i in atlas:
                    classification_dir = path.join(output_dir, 'DWIOriginalWithoutFeatureRescalingROI',
                                                   task + '_' + i,
                                                   dwi_map)
                    if not path.exists(classification_dir):
                        os.makedirs(classification_dir)

                        print("\nRunning %s" % classification_dir)

                        wf = DWI_RB_RepHoldOut_DualSVM(caps_directory, subjects_visits_tsv, diagnoses_tsv, i, dwi_map,
                                                   classification_dir,
                                                   n_threads=n_threads, n_iterations=n_iterations,
                                                   test_size=test_size,
                                                   grid_search_folds=grid_search_folds,
                                                       splits_indices=splits_indices)
                        wf.run()
                    else:
                        print("This combination has been classified, just skip: %s " % classification_dir)

#############################
# T1 voxel-wise without feature rescaling
#############################
def run_t1_voxel_without_feature_rescaling(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, task, n_threads, n_iterations, test_size, grid_search_folds, fwhm=[8], group_id='ADNIbl'):
    """
    This is to run voxel-wise classification for T1 or DTI
    Args:
        caps_directory:
        diagnoses_tsv:
        subjects_visits_tsv:
        output_dir:
        task:
        n_threads:
        n_iterations:
        test_size:
        grid_search_folds:
        balanced_down_sample:
        modality:
        dwi_maps:
        fwhm:
        tissue_type:
        threshold:
        group_id:
        feature_rescaling: if feature rescaling is performed, by default no.
        feature_selection_nested: if feature selection should be done. None means no feature selection is done. False
        for non-nested feature selection; True for nested feature selection
        top_k:
        feature_selection_method: By default 'ANOVA'. Other options are RF, PCA, RFE

    Returns:

    """

    splits_indices, splits_indices_pickle = split_subjects_to_pickle(diagnoses_tsv, n_iterations=n_iterations,
                                                                     test_size=test_size)

    print("Original experiments for voxel-wise T1 experiments without feature rescaling!!!")

    for k in fwhm:
        classification_dir = path.join(output_dir, 'T1OriginalWithoutFeatureRescalingVoxel',
                                       task + '_fwhm_' + str(k))
        if not path.exists(classification_dir):
            os.makedirs(classification_dir)

            print("\nRunning %s" % classification_dir)
            wf = VB_RepHoldOut_DualSVM(caps_directory, subjects_visits_tsv, diagnoses_tsv, group_id, 'T1',
                                       classification_dir, fwhm=k, n_threads=n_threads,
                                       n_iterations=n_iterations,
                                       test_size=test_size, splits_indices=splits_indices, grid_search_folds=grid_search_folds)
            wf.run()
        else:
            print("This combination has been classified, just skip: %s " % classification_dir)


#############################
# T1 ROI-wise without feature rescaling
#############################
def run_t1_roi_without_feature_rescaling(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, atlas,
                                task, n_threads, n_iterations, test_size, grid_search_folds, group_id='ADNIbl'):
    """
    This is a function to run the Voxel-based calssification tasks_imbalanced after the imaging processing pipeline of ADNI
    :param caps_directory: caps directory for Clinica outputs
    :param diagnoses_tsv:
    :param subjects_visits_tsv:
    :param output_dir: the path to store the classification outputs
    :param atlas: one of these: ['JHUDTI81', 'JHUTracts0', 'JHUTracts25']
    :param dwi_maps: the maps based on DWI, currently, we just have maps from DTI model.
    :param balanced_down_sample: int, how many times to repeat for the downsampling procedures to creat the balanced data, default is 0, which means we do not force the data to be balanced
    :param task: the name of the task to store the classification results
    :param n_threads: number of cores to use for this classification
    :param n_iterations: number of runs for the RepHoldOut
    :param test_size: propotion for the testing dataset
    :param grid_search_folds: number of runs to search the hyperparameters

    :return:
    """
    splits_indices, splits_indices_pickle = split_subjects_to_pickle(diagnoses_tsv, n_iterations=n_iterations,
                                                                     test_size=test_size)
    print("Original experiments for ROI-wise T1 experiments without feature rescaling!!!")
    for i in atlas:
        classification_dir = path.join(output_dir, 'T1OriginalWithoutFeatureRescalingROI',
                                       task + '_' + i)
        if not path.exists(classification_dir):
            os.makedirs(classification_dir)

            print("\nRunning %s" % classification_dir)

            wf = RB_RepHoldOut_DualSVM(caps_directory, subjects_visits_tsv, diagnoses_tsv, group_id,
                                       'T1', i,
                                       classification_dir, n_threads=n_threads, n_iterations=n_iterations,
                                       test_size=test_size, splits_indices=splits_indices, grid_search_folds=grid_search_folds)
            wf.run()
        else:
            print("This combination has been classified, just skip: %s " % classification_dir)

#############################
# DWI voxel-wise with feature rescaling
#############################

def run_dwi_voxel_with_feature_rescaling(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, task, n_threads, n_iterations, test_size, grid_search_folds, dwi_maps=['fa', 'md'], fwhm=[8],
                                   tissue_type=['WM', 'GM', 'GM_WM'], threshold=[0.3], caps_reg_method='single_modal', balanced_down_sample=False, balance_sklearn=True):
    """
    This is to run voxel-wise classification for DTI with z score feature rescaling.
    Args:
        caps_directory:
        diagnoses_tsv:
        subjects_visits_tsv:
        output_dir:
        task:
        n_threads:
        n_iterations:
        test_size:
        grid_search_folds:
        dwi_maps:
        fwhm:
        tissue_type:
        threshold:
        group_id:
        caps_reg_method: single_modal or multi_modal
        balanced_down_sample: if random balance the sample size, by default: False

    Returns:

    """

    if balanced_down_sample:
        print("Random downsample the bigger samples to balance the groups\n")
        splits_indices, splits_indices_pickle = split_subjects_to_pickle(diagnoses_tsv, n_iterations=n_iterations,
                                                                         test_size=test_size, balanced=True)
        for dwi_map in dwi_maps:
            for i in tissue_type:
                for j in threshold:
                    for k in fwhm:
                        classification_dir = path.join(output_dir, 'DWIWithFeatureRescalingVoxelRandomBalancedData',
                                                       task + '_' + i + '_' + str(j) + '_' + str(
                                                           k) + '_' + dwi_map)
                        if not path.exists(classification_dir):
                            os.makedirs(classification_dir)

                            print("\nRunning %s" % classification_dir)

                            wf = DWI_VB_RepHoldOut_DualSVM_FeatureSelectionNested(caps_directory, subjects_visits_tsv,
                                                                                  diagnoses_tsv, dwi_map,
                                                                                  i, j, classification_dir, fwhm=k,
                                                                                  n_threads=n_threads,
                                                                                  n_iterations=n_iterations,
                                                                                  test_size=test_size,
                                                                                  grid_search_folds=grid_search_folds,
                                                                                  splits_indices=splits_indices, caps_reg_method=caps_reg_method,
                                                                                  feature_rescaling_method='zscore')

                            wf.run()
                        else:
                            print("This combination has been classified, just skip: %s " % classification_dir)

    else:
        print("Experiments voxel-wise DWI experiments with z-score feature rescaling\n")
        splits_indices, splits_indices_pickle = split_subjects_to_pickle(diagnoses_tsv, n_iterations=n_iterations, test_size=test_size)
        for dwi_map in dwi_maps:
            for i in tissue_type:
                for j in threshold:
                    for k in fwhm:
                        if balance_sklearn and caps_reg_method == 'single_modal': ## original setting
                            classification_dir = path.join(output_dir, 'DWIWithFeatureRescalingVoxel',
                                                           task + '_' + i + '_' + str(j) + '_' + str(
                                                               k) + '_' + dwi_map)
                        elif balance_sklearn == False and caps_reg_method == 'single_modal': ## without any trick for balancing data
                            classification_dir = path.join(output_dir, 'DWIWithFeatureRescalingVoxelBalancedSklearnOffWithoutAnyTrick',
                                                           task + '_' + i + '_' + str(j) + '_' + str(
                                                               k) + '_' + dwi_map)
                        elif caps_reg_method == 'multi_modal' and balance_sklearn: ## multicontrast reg
                            classification_dir = path.join(output_dir, 'DWIWithFeatureRescalingVoxelMultimodalReg',
                                                           task + '_' + i + '_' + str(j) + '_' + str(
                                                               k) + '_' + dwi_map)
                        if not path.exists(classification_dir):
                            os.makedirs(classification_dir)

                            print("\nRunning %s" % classification_dir)

                            wf = DWI_VB_RepHoldOut_DualSVM_FeatureSelectionNested(caps_directory, subjects_visits_tsv,
                                                                                  diagnoses_tsv, dwi_map,
                                                                                  i, j, classification_dir, fwhm=k,
                                                                                  n_threads=n_threads,
                                                                                  n_iterations=n_iterations,
                                                                                  test_size=test_size,
                                                                                  grid_search_folds=grid_search_folds,
                                                                                  splits_indices=splits_indices, caps_reg_method=caps_reg_method,
                                                                                  feature_rescaling_method='zscore', balanced=balance_sklearn)

                            wf.run()
                        else:
                            print("This combination has been classified, just skip: %s " % classification_dir)

#############################
# DWI ROI-wise with feature rescaling
#############################
def run_dwi_roi_with_feature_rescaling(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, atlas,
                                task, n_threads, n_iterations, test_size, grid_search_folds,
                                 dwi_maps=['fa', 'md'], balanced_down_sample=False, balance_sklearn=True):
    """
        This is to run ROI-wise classification for DTI with feature rescaling.
    Args:
        caps_directory:
        diagnoses_tsv:
        subjects_visits_tsv:
        output_dir:
        atlas: ['JHUDTI81', 'JHUTracts0', 'JHUTracts25']
        task:
        n_threads:
        n_iterations:
        test_size:
        grid_search_folds:
        balanced_down_sample:
        dwi_maps:

    Returns:

    """
    if balanced_down_sample:
        print("Random downsample the bigger samples to balance the groups\n")
        splits_indices, splits_indices_pickle = split_subjects_to_pickle(diagnoses_tsv, n_iterations=n_iterations,
                                                                         test_size=test_size, balanced=True)
        for dwi_map in dwi_maps:
            for i in atlas:
                classification_dir = path.join(output_dir, 'DWIWithFeatureRescalingROIRandomBalancedData',
                                               task + '_' + i,
                                               dwi_map)
                if not path.exists(classification_dir):
                    os.makedirs(classification_dir)

                    print("\nRunning %s" % classification_dir)


                    wf = DWI_RB_RepHoldOut_DualSVM_FeatureSelectionNested(caps_directory, subjects_visits_tsv,
                                                                          diagnoses_tsv, i,
                                                                          dwi_map,
                                                                          classification_dir,
                                                                          n_threads=n_threads,
                                                                          n_iterations=n_iterations,
                                                                          test_size=test_size,
                                                                          grid_search_folds=grid_search_folds,
                                                                          splits_indices=splits_indices,
                                                                          feature_rescaling_method='zscore')
                    wf.run()
                else:
                    print("This combination has been classified, just skip: %s " % classification_dir)

    else:
        splits_indices, splits_indices_pickle = split_subjects_to_pickle(diagnoses_tsv, n_iterations=n_iterations, test_size=test_size)
        print("ROI-wise DWI experiments with feature rescaling")
        for dwi_map in dwi_maps:
            for i in atlas:
                if balance_sklearn:
                    classification_dir = path.join(output_dir, 'DWIWithFeatureRescalingROI',
                                                   task + '_' + i,
                                                   dwi_map)
                else:
                    classification_dir = path.join(output_dir, 'DWIWithFeatureRescalingROIBalancedSklearnOffWithoutAnyTrick',
                                                   task + '_' + i,
                                                   dwi_map)
                if not path.exists(classification_dir):
                    os.makedirs(classification_dir)

                    print("\nRunning %s" % classification_dir)


                    wf = DWI_RB_RepHoldOut_DualSVM_FeatureSelectionNested(caps_directory, subjects_visits_tsv,
                                                                          diagnoses_tsv, i,
                                                                          dwi_map,
                                                                          classification_dir,
                                                                          n_threads=n_threads,
                                                                          n_iterations=n_iterations,
                                                                          test_size=test_size,
                                                                          grid_search_folds=grid_search_folds,
                                                                          splits_indices=splits_indices,
                                                                          feature_rescaling_method='zscore', balanced=balance_sklearn)
                    wf.run()
                else:
                    print("This combination has been classified, just skip: %s " % classification_dir)

#############################
# T1 voxel-wise with feature rescaling
#############################
def run_t1_voxel_with_feature_rescaling(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, task, n_threads, n_iterations, test_size, grid_search_folds, fwhm=[8], group_id='ADNIbl', feature_rescaling_method='zscore'):
    """
    This is to run voxel-wise classification for T1 or DTI
    Args:
        caps_directory:
        diagnoses_tsv:
        subjects_visits_tsv:
        output_dir:
        task:
        n_threads:
        n_iterations:
        test_size:
        grid_search_folds:
        balanced_down_sample:
        modality:
        dwi_maps:
        fwhm:
        tissue_type:
        threshold:
        group_id:
        feature_rescaling: if feature rescaling is performed, by default no.
        feature_selection_nested: if feature selection should be done. None means no feature selection is done. False
        for non-nested feature selection; True for nested feature selection
        top_k:
        feature_rescaling_method: zscore or minmax

    Returns:

    """

    splits_indices, splits_indices_pickle = split_subjects_to_pickle(diagnoses_tsv, n_iterations=n_iterations,
                                                                     test_size=test_size)

    print("Experiments for voxel-wise T1 experiments with feature rescaling!!!")

    for k in fwhm:
        classification_dir = path.join(output_dir, 'T1WithFeatureRescalingVoxel',
                                       task + '_fwhm_' + str(k))
        if not path.exists(classification_dir):
            os.makedirs(classification_dir)

            print("\nRunning %s" % classification_dir)
            wf = T1_VB_RepHoldOut_DualSVM_FeatureSelectionNested(caps_directory, subjects_visits_tsv, diagnoses_tsv,
                                                                 group_id,
                                                                 'T1',
                                                                 classification_dir, fwhm=k, n_threads=n_threads,
                                                                 n_iterations=n_iterations,
                                                                 test_size=test_size, splits_indices=splits_indices,
                                                                 grid_search_folds=grid_search_folds,
                                                                 feature_rescaling_method=feature_rescaling_method)

            wf.run()
        else:
            print("This combination has been classified, just skip: %s " % classification_dir)


#############################
# T1 ROI-wise with feature rescaling
#############################
def run_t1_roi_with_feature_rescaling(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, atlas,
                                task, n_threads, n_iterations, test_size, grid_search_folds, group_id='ADNIbl'):
    """
    This is a function to run the Voxel-based calssification tasks_imbalanced after the imaging processing pipeline of ADNI
    :param caps_directory: caps directory for Clinica outputs
    :param diagnoses_tsv:
    :param subjects_visits_tsv:
    :param output_dir: the path to store the classification outputs
    :param atlas: one of these: ['JHUDTI81', 'JHUTracts0', 'JHUTracts25']
    :param dwi_maps: the maps based on DWI, currently, we just have maps from DTI model.
    :param balanced_down_sample: int, how many times to repeat for the downsampling procedures to creat the balanced data, default is 0, which means we do not force the data to be balanced
    :param task: the name of the task to store the classification results
    :param n_threads: number of cores to use for this classification
    :param n_iterations: number of runs for the RepHoldOut
    :param test_size: propotion for the testing dataset
    :param grid_search_folds: number of runs to search the hyperparameters

    :return:
    """
    splits_indices, splits_indices_pickle = split_subjects_to_pickle(diagnoses_tsv, n_iterations=n_iterations,
                                                                     test_size=test_size)
    print("Original experiments for ROI-wise T1 experiments with feature rescaling!!!")
    for i in atlas:
        classification_dir = path.join(output_dir, 'T1WithFeatureRescalingROI',
                                       task + '_' + i)
        if not path.exists(classification_dir):
            os.makedirs(classification_dir)

            print("\nRunning %s" % classification_dir)
            wf = T1_RB_RepHoldOut_DualSVM_FeatureSelectionNested(caps_directory, subjects_visits_tsv, diagnoses_tsv,
                                                            group_id,
                                                            'T1', i,
                                                            classification_dir, n_threads=n_threads,
                                                            n_iterations=n_iterations,
                                                            test_size=test_size, splits_indices=splits_indices,
                                                            grid_search_folds=grid_search_folds,
                                                            feature_rescaling_method='zscore')

            wf.run()
        else:
            print("This combination has been classified, just skip: %s " % classification_dir)

#############################
# DWI voxel-wise with feature selection
#############################

def run_dwi_voxel_with_feature_selection(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, task, n_threads, n_iterations, test_size, grid_search_folds, dwi_maps=['fa', 'md'], fwhm=[8],
                                   tissue_type=['WM', 'GM', 'GM_WM'], threshold=[0.3], caps_reg_method='single_modal', feature_rescaling_method='zscore',
                                   feature_selection_nested=True, feature_selection_method='RFE', top_k=[50]):
    """
    This is to run voxel-wise classification for DTI with different feature selection procedures:
    1) nested feature selection
    2) non-nested feature selection
    Args:
        caps_directory:
        diagnoses_tsv:
        subjects_visits_tsv:
        output_dir:
        task:
        n_threads:
        n_iterations:
        test_size:
        grid_search_folds:
        dwi_maps:
        fwhm:
        tissue_type:
        threshold:
        group_id:
        caps_reg_method: single_modal or multi_modal
        balanced_down_sample: if random balance the sample size, by default: False
        feature_selection_nested: nested feature selection, by default True
        feature_selection_method: one of 'ANOVA' and 'RFE'
        feature_rescaling_method: default is zscore, if None, not perform feature rescaling

    Returns:

    """

    if feature_selection_nested and feature_rescaling_method != None:
        print("Classification with nested feature resclaling and feature selection!\n")
        splits_indices, splits_indices_pickle = split_subjects_to_pickle(diagnoses_tsv, n_iterations=n_iterations, test_size=test_size)
        for dwi_map in dwi_maps:
            for i in tissue_type:
                for j in threshold:
                    for k in fwhm:
                        for l in top_k:
                            classification_dir = path.join(output_dir, 'DWIWithFeatureRescalingFeatureSelectionVoxelNestedFS' + feature_selection_method,
                                                           task + '_' + i + '_' + str(j) + '_' + str(
                                                               k) + '_fs_' + str(l) + '_' + dwi_map)
                            if not path.exists(classification_dir):
                                os.makedirs(classification_dir)

                                print("\nRunning %s" % classification_dir)

                                wf = DWI_VB_RepHoldOut_DualSVM_FeatureSelectionNested(caps_directory, subjects_visits_tsv,
                                                                                      diagnoses_tsv, dwi_map,
                                                                                      i, j, classification_dir, fwhm=k,
                                                                                      n_threads=n_threads,
                                                                                      n_iterations=n_iterations,
                                                                                      test_size=test_size,
                                                                                      grid_search_folds=grid_search_folds,
                                                                                      splits_indices=splits_indices, caps_reg_method=caps_reg_method,
                                                                                      feature_selection_method=feature_selection_method,
                                                                                      feature_rescaling_method=feature_rescaling_method,
                                                                                      top_k=l)

                                wf.run()
                            else:
                                print("This combination has been classified, just skip: %s " % classification_dir)

    elif feature_selection_nested == False and feature_rescaling_method != None:
        print("Classification with non-nested feature rescaling and feature selection\n")
        splits_indices, splits_indices_pickle = split_subjects_to_pickle(diagnoses_tsv, n_iterations=n_iterations, test_size=test_size)
        for dwi_map in dwi_maps:
            for i in tissue_type:
                for j in threshold:
                    for k in fwhm:
                        for l in top_k:
                            classification_dir = path.join(output_dir, 'DWIWithFeatureRescalingFeatureSelectionVoxelNonNestedFS' + feature_selection_method,
                                                           task + '_' + i + '_' + str(j) + '_' + str(
                                                               k) + '_fs_' + str(l) + '_' + dwi_map)

                            if not path.exists(classification_dir):
                                os.makedirs(classification_dir)

                                print("\nRunning %s" % classification_dir)

                                wf = DWI_VB_RepHoldOut_DualSVM_FeatureSelectionNonNested(caps_directory, subjects_visits_tsv,
                                                                                      diagnoses_tsv, dwi_map,
                                                                                      i, j, classification_dir, fwhm=k,
                                                                                      n_threads=n_threads,
                                                                                      n_iterations=n_iterations,
                                                                                      test_size=test_size,
                                                                                      grid_search_folds=grid_search_folds,
                                                                                      splits_indices=splits_indices,
                                                                                      feature_selection_method=feature_selection_method,
                                                                                      feature_rescaling_method=feature_rescaling_method,
                                                                                      top_k=l)
                                wf.run()
                            else:
                                print("This combination has been classified, just skip: %s " % classification_dir)
    elif feature_selection_nested and feature_rescaling_method == None:
        print("Classification with nested feature selection without feature rescaling!\n")
        splits_indices, splits_indices_pickle = split_subjects_to_pickle(diagnoses_tsv, n_iterations=n_iterations, test_size=test_size)
        for dwi_map in dwi_maps:
            for i in tissue_type:
                for j in threshold:
                    for k in fwhm:
                        for l in top_k:
                            classification_dir = path.join(output_dir, 'DWIWithoutFeatureRescalingFeatureSelectionVoxelNestedFS' + feature_selection_method,
                                                           task + '_' + i + '_' + str(j) + '_' + str(
                                                               k) + '_fs_' + str(l) + '_' + dwi_map)
                            if not path.exists(classification_dir):
                                os.makedirs(classification_dir)

                                print("\nRunning %s" % classification_dir)

                                wf = DWI_VB_RepHoldOut_DualSVM_FeatureSelectionNested(caps_directory, subjects_visits_tsv,
                                                                                      diagnoses_tsv, dwi_map,
                                                                                      i, j, classification_dir, fwhm=k,
                                                                                      n_threads=n_threads,
                                                                                      n_iterations=n_iterations,
                                                                                      test_size=test_size,
                                                                                      grid_search_folds=grid_search_folds,
                                                                                      splits_indices=splits_indices, caps_reg_method=caps_reg_method,
                                                                                      feature_selection_method=feature_selection_method,
                                                                                      feature_rescaling_method=None,
                                                                                      top_k=l)

                                wf.run()
                            else:
                                print("This combination has been classified, just skip: %s " % classification_dir)

    elif feature_selection_nested == False and feature_rescaling_method == None:
        print("Classification with non-nested feature selection but wihtout feature rescaling\n")
        splits_indices, splits_indices_pickle = split_subjects_to_pickle(diagnoses_tsv, n_iterations=n_iterations, test_size=test_size)
        for dwi_map in dwi_maps:
            for i in tissue_type:
                for j in threshold:
                    for k in fwhm:
                        for l in top_k:
                            classification_dir = path.join(output_dir, 'DWIWithoutFeatureRescalingFeatureSelectionVoxelNonNestedFS' + feature_selection_method,
                                                           task + '_' + i + '_' + str(j) + '_' + str(
                                                               k) + '_fs_' + str(l) + '_' + dwi_map)

                            if not path.exists(classification_dir):
                                os.makedirs(classification_dir)

                                print("\nRunning %s" % classification_dir)

                                wf = DWI_VB_RepHoldOut_DualSVM_FeatureSelectionNonNested(caps_directory, subjects_visits_tsv,
                                                                                      diagnoses_tsv, dwi_map,
                                                                                      i, j, classification_dir, fwhm=k,
                                                                                      n_threads=n_threads,
                                                                                      n_iterations=n_iterations,
                                                                                      test_size=test_size,
                                                                                      grid_search_folds=grid_search_folds,
                                                                                      splits_indices=splits_indices,
                                                                                      feature_selection_method=feature_selection_method,
                                                                                      feature_rescaling_method=None,
                                                                                      top_k=l)
                                wf.run()
                            else:
                                print("This combination has been classified, just skip: %s " % classification_dir)

#############################
# T1 voxel-wise with feature selection
#############################
def run_t1_voxel_with_feature_selection(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, task, n_threads, n_iterations, test_size, grid_search_folds, fwhm = [8],
                                        group_id='ADNIbl', feature_rescaling_method='zscore', feature_selection_nested=True, feature_selection_method='RFE', top_k=[50]):

    """
    This is to run voxel-wise classification for T1 with different feature selection procedures:
    1) nested feature selection
    2) non-nested feature selection
    Args:
        caps_directory:
        diagnoses_tsv:
        subjects_visits_tsv:
        output_dir:
        task:
        n_threads:
        n_iterations:
        test_size:
        grid_search_folds:
        dwi_maps:
        fwhm:
        tissue_type:
        threshold:
        group_id:
        caps_reg_method: single_modal or multi_modal
        balanced_down_sample: if random balance the sample size, by default: False
        feature_selection_nested: nested feature selection, by default True
        feature_selection_method: one of 'ANOVA' and 'RFE'

    Returns:

    """

    if feature_selection_nested:
        print("T1 Classification with nested feature selection!\n")
        splits_indices, splits_indices_pickle = split_subjects_to_pickle(diagnoses_tsv, n_iterations=n_iterations, test_size=test_size)

        for k in fwhm:
            for l in top_k:
                classification_dir = path.join(output_dir, 'T1WithFeatureRescalingFeatureSelectionVoxelNestedFS' + feature_selection_method,
                                               task + '_fwhm_' + str(k) + '_fs_' + str(l))
                if not path.exists(classification_dir):
                    os.makedirs(classification_dir)

                    print("\nRunning %s" % classification_dir)

                    wf = T1_VB_RepHoldOut_DualSVM_FeatureSelectionNested(caps_directory, subjects_visits_tsv,
                                                                        diagnoses_tsv, group_id, 'T1', classification_dir,
                                                                        fwhm=k, n_threads=n_threads,
                                                                        n_iterations=n_iterations,
                                                                        test_size=test_size,
                                                                        splits_indices=splits_indices,
                                                                        grid_search_folds=grid_search_folds,
                                                                        feature_selection_method=feature_selection_method,
                                                                        feature_rescaling_method=feature_rescaling_method,
                                                                        top_k=l)

                    wf.run()
                else:
                    print("This combination has been classified, just skip: %s " % classification_dir)

    else:
        print("Classification with non-nested feature selection.\n")
        print("No need to run.\n")

def classification_performances_violin_plot_neuroinformatics(classification_output_dir, task_name):
    """
    This is a function to plot the classification performances in Neuroinformatic paper
    :param classification_output_dir: str, should be absolute path to the classification results.
    :param task_name: str, task name in the paper
    :return:
    """
    n_iterations = 250

    if task_name == 'T1_vs_DTI': ## compare the T1 with DTI results
        ## NOTE: NOT USED IN THE PAPER
        ## results list to contain both DTI and T1
        results_voxel = []
        results_roi = []

        tissue_combinations = ['GM']
        tasks_list = ['AD_vs_CN', 'CN_vs_MCI', 'CN_vs_pMCI', 'sMCI_vs_pMCI']

        ## get T1
        for task in tasks_list:
            tsvs_path = os.path.join(classification_output_dir, 'T1WithFeatureRescalingVoxel',
                                     task + '_VB_fwhm_8')
            balanced_accuracy = []
            for i in xrange(n_iterations):
                result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                if os.path.isfile(result_tsv):
                    balanced_accuracy.append(
                        (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                else:
                    raise OSError(
                        errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
            results_voxel.append(balanced_accuracy)

        ## get GM-FA
        for task in tasks_list:
            for tissue in tissue_combinations:
                tsvs_path = os.path.join(classification_output_dir, 'DWIWithFeatureRescalingVoxel',
                                         task + '_VB_' + tissue + '_0.3_8_fa')
                balanced_accuracy = []
                for i in xrange(n_iterations):
                    result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                    if os.path.isfile(result_tsv):
                        balanced_accuracy.append(
                            (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                    else:
                        raise OSError(
                            errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                results_voxel.append(balanced_accuracy)

        ## get GM-MD
        for task in tasks_list:
            for tissue in tissue_combinations:
                tsvs_path = os.path.join(classification_output_dir, 'DWIWithFeatureRescalingVoxel',
                                         task + '_VB_' + tissue + '_0.3_8_md')
                balanced_accuracy = []
                for i in xrange(n_iterations):
                    result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                    if os.path.isfile(result_tsv):
                        balanced_accuracy.append(
                            (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                    else:
                        raise OSError(
                            errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                results_voxel.append(balanced_accuracy)

        ## get T1 AAL
        for task in tasks_list:
            tsvs_path = os.path.join(classification_output_dir, 'T1WithFeatureRescalingROI',
                                     task + '_RB_AAL2')
            balanced_accuracy = []
            for i in xrange(n_iterations):
                result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                if os.path.isfile(result_tsv):
                    balanced_accuracy.append(
                        (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                else:
                    raise OSError(
                        errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
            results_roi.append(balanced_accuracy)

        ## get T1 JHULabel-FA
        for task in tasks_list:
            tsvs_path = os.path.join(classification_output_dir, 'DWIWithFeatureRescalingROI',
                                     task + '_RB_JHUDTI81', 'fa')
            balanced_accuracy = []
            for i in xrange(n_iterations):
                result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                if os.path.isfile(result_tsv):
                    balanced_accuracy.append(
                        (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                else:
                    raise OSError(
                        errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
            results_roi.append(balanced_accuracy)

        ## get T1 JHULabel-MD
        for task in tasks_list:
            tsvs_path = os.path.join(classification_output_dir, 'DWIWithFeatureRescalingROI',
                                     task + '_RB_JHUDTI81', 'md')
            balanced_accuracy = []
            for i in xrange(n_iterations):
                result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                if os.path.isfile(result_tsv):
                    balanced_accuracy.append(
                        (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                else:
                    raise OSError(
                        errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
            results_roi.append(balanced_accuracy)

        ## voxel plot
        ### transfer the list into an array with this shape: n_iterations * num_tasks
        metric = np.array(results_voxel).transpose()
        ## rearranage the metric in order to have the right position for each task
        index_order = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
        metric = metric[:, index_order]
        ## define the violin's postions
        pos = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]
        color = ['#FF0000', '#87CEFA', '#90EE90'] * len(tasks_list)  # red, blue and green
        legendA = ['GM-density', 'GM-FA', 'GM-MD']

        ## define the size of th image
        fig, ax = plt.subplots(2, figsize=[15, 10])
        line_coll = ax[0].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
        for cc, ln in enumerate(line_coll['bodies']):
            ln.set_facecolor(color[cc])

        ax[0].legend(legendA, loc='upper right', fontsize=10, frameon=True)
        ax[0].grid(axis='y', which='major', linestyle='dotted')
        ax[0].set_xticks([2, 6, 10, 14])
        ax[0].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax[0].set_xticklabels(tasks_list, rotation=0, fontsize=15)  # 'vertical'
        ax[0].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
        mean = np.mean(metric, 0)
        std = np.std(metric, 0)
        inds = np.array(pos)
        ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
        ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
        ax[0].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
        ax[0].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
        ax[0].set_ylim(0.1, 1)
        ax[0].set_title('A: Voxel-based features for T1w and diffusion MRI', fontsize=15)

        ##### ROI
        ### transfer the list into an array with this shape: n_iterations*n_tasks_imbalanced
        metric = np.array(results_roi).transpose()
        ## rearranage the metric in order to have the right position for each task
        index_order = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
        metric = metric[:, index_order]
        ## define the size of th image
        line_coll = ax[1].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
        for cc, ln in enumerate(line_coll['bodies']):
            ln.set_facecolor(color[cc])
        legendB = ['WM-MD', 'GM-MD', 'GM+WM-MD']

        ax[1].legend(legendB, loc='upper right', fontsize=10, frameon=True)
        ax[1].grid(axis='y', which='major', linestyle='dotted')
        ax[1].set_xticks([2, 6, 10, 14])
        ax[1].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax[1].set_xticklabels(tasks_list, rotation=0, fontsize=15)  # 'vertical'
        ax[1].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
        mean = np.mean(metric, 0)
        std = np.std(metric, 0)
        inds = np.array(pos)
        ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
        ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
        ax[1].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
        ax[1].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
        ax[1].set_ylim(0.1, 1)
        ax[1].set_title('B: Region-based features for T1w and diffusion MRI', fontsize=15)

        plt.savefig(os.path.join(classification_output_dir, 'figures',
                                 'figure_t1w_vs_dti.png'), additional_artists=plt.legend,
                    bbox_inches="tight")

    elif task_name == 'Balanced_vs_imbalanced':
        ## results list to contain both DTI and T1
        results_voxel_fa = []
        results_voxel_md = []
        results_roi_fa = []
        results_roi_md = []

        tissue_combinations = ['GM_WM']
        # atlases = ['JHUTracts25']
        tasks_list = ['CN_vs_MCI', 'CN_vs_pMCI', 'sMCI_vs_pMCI']

        ## get original without any trick FA
        for task in tasks_list:
            for tissue in tissue_combinations:
                tsvs_path = os.path.join(classification_output_dir, 'DWIWithFeatureRescalingVoxelBalancedSklearnOffWithoutAnyTrick',
                                         task + '_VB_' + tissue + '_0.3_8_fa')
                balanced_accuracy = []
                for i in xrange(n_iterations):
                    result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                    if os.path.isfile(result_tsv):
                        balanced_accuracy.append(
                            (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                    else:
                        raise OSError(
                            errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                results_voxel_fa.append(balanced_accuracy)

        ## get balanced downsample results for FA
        for task in tasks_list:
            for tissue in tissue_combinations:
                tsvs_path = os.path.join(classification_output_dir, 'DWIWithFeatureRescalingVoxelRandomBalancedData',
                                         task + '_VB_' + tissue + '_0.3_8_fa')
                balanced_accuracy = []
                for i in xrange(n_iterations):
                    result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                    if os.path.isfile(result_tsv):
                        balanced_accuracy.append(
                            (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                    else:
                        raise OSError(
                            errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                results_voxel_fa.append(balanced_accuracy)

        ## get original without any trick MD
        for task in tasks_list:
            for tissue in tissue_combinations:
                tsvs_path = os.path.join(classification_output_dir, 'DWIWithFeatureRescalingVoxelBalancedSklearnOffWithoutAnyTrick',
                                         task + '_VB_' + tissue + '_0.3_8_md')
                balanced_accuracy = []
                for i in xrange(n_iterations):
                    result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                    if os.path.isfile(result_tsv):
                        balanced_accuracy.append(
                            (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                    else:
                        raise OSError(
                            errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                results_voxel_md.append(balanced_accuracy)

        ## get balanced downsample results for MD
        for task in tasks_list:
            for tissue in tissue_combinations:
                tsvs_path = os.path.join(classification_output_dir, 'DWIWithFeatureRescalingVoxelRandomBalancedData',
                                         task + '_VB_' + tissue + '_0.3_8_md')
                balanced_accuracy = []
                for i in xrange(n_iterations):
                    result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                    if os.path.isfile(result_tsv):
                        balanced_accuracy.append(
                            (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                    else:
                        raise OSError(
                            errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                results_voxel_md.append(balanced_accuracy)

        ## get T1 JHUTract-FA without any trick for imbalance
        for task in tasks_list:
            tsvs_path = os.path.join(classification_output_dir, 'DWIWithFeatureRescalingROIBalancedSklearnOffWithoutAnyTrick',
                                     task + '_RB_JHUTracts25', 'fa')
            balanced_accuracy = []
            for i in xrange(n_iterations):
                result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                if os.path.isfile(result_tsv):
                    balanced_accuracy.append(
                        (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                else:
                    raise OSError(
                        errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
            results_roi_fa.append(balanced_accuracy)

        ## get T1 JHUTract-FA with random downsampling
        for task in tasks_list:
            tsvs_path = os.path.join(classification_output_dir, 'DWIWithFeatureRescalingROIRandomBalancedData',
                                     task + '_RB_JHUTracts25', 'fa')
            balanced_accuracy = []
            for i in xrange(n_iterations):
                result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                if os.path.isfile(result_tsv):
                    balanced_accuracy.append(
                        (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                else:
                    raise OSError(
                        errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
            results_roi_fa.append(balanced_accuracy)

        ## get T1 JHUTract-MD without any trick for imbalance
        for task in tasks_list:
            tsvs_path = os.path.join(classification_output_dir, 'DWIWithFeatureRescalingROIBalancedSklearnOffWithoutAnyTrick',
                                     task + '_RB_JHUTracts25', 'md')
            balanced_accuracy = []
            for i in xrange(n_iterations):
                result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                if os.path.isfile(result_tsv):
                    balanced_accuracy.append(
                        (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                else:
                    raise OSError(
                        errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
            results_roi_md.append(balanced_accuracy)

        ## get T1 JHUTract-FA with random downsampling
        for task in tasks_list:
            tsvs_path = os.path.join(classification_output_dir, 'DWIWithFeatureRescalingROIRandomBalancedData',
                                     task + '_RB_JHUTracts25', 'md')
            balanced_accuracy = []
            for i in xrange(n_iterations):
                result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                if os.path.isfile(result_tsv):
                    balanced_accuracy.append(
                        (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                else:
                    raise OSError(
                        errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
            results_roi_md.append(balanced_accuracy)

        ## voxel FA
        ### transfer the list into an array with this shape: n_iterations * num_tasks
        metric = np.array(results_voxel_fa).transpose()
        ## rearranage the metric in order to have the right position for each task
        index_order = [0, 3, 1, 4, 2, 5]
        metric = metric[:, index_order]
        ## define the violin's postions
        pos = [1, 2, 4, 5, 7, 8]
        color = ['#FF0000', '#87CEFA'] * len(tasks_list)  # red, blue and green
        legendA = ['GM+WM-original', 'GM+WM-balanced']

        ## define the size of th image
        fig, ax = plt.subplots(2, figsize=[15, 10])
        line_coll = ax[0].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
        for cc, ln in enumerate(line_coll['bodies']):
            ln.set_facecolor(color[cc])

        ax[0].legend(legendA, loc='upper right', fontsize=10, frameon=True)
        ax[0].grid(axis='y', which='major', linestyle='dotted')
        ax[0].set_xticks([1.5 , 4.5, 7.5])
        ax[0].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax[0].set_xticklabels(tasks_list, rotation=0, fontsize=15)  # 'vertical'
        ax[0].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
        mean = np.mean(metric, 0)
        std = np.std(metric, 0)
        inds = np.array(pos)
        ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
        ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
        ax[0].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
        ax[0].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
        ax[0].set_ylim(0.1, 1)
        ax[0].set_title('A: FA voxel-based features', fontsize=15)

        ##### voxel MD
        ### transfer the list into an array with this shape: n_iterations*n_tasks_imbalanced
        metric = np.array(results_voxel_md).transpose()
        ## rearranage the metric in order to have the right position for each task
        metric = metric[:, index_order]
        ## define the size of th image
        line_coll = ax[1].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
        for cc, ln in enumerate(line_coll['bodies']):
            ln.set_facecolor(color[cc])

        ax[1].legend(legendA, loc='upper right', fontsize=10, frameon=True)
        ax[1].grid(axis='y', which='major', linestyle='dotted')
        ax[1].set_xticks([1.5 , 4.5, 7.5])
        ax[1].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax[1].set_xticklabels(tasks_list, rotation=0, fontsize=15)  # 'vertical'
        ax[1].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
        mean = np.mean(metric, 0)
        std = np.std(metric, 0)
        inds = np.array(pos)
        ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
        ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
        ax[1].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
        ax[1].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
        ax[1].set_ylim(0.1, 1)
        ax[1].set_title('B: MD voxel-based features', fontsize=15)

        plt.savefig(os.path.join(classification_output_dir, 'figures',
                                 'figure_influence_of_imbalance_VOXEL.png'), additional_artists=plt.legend,
                    bbox_inches="tight")

        ## ROI FA
        ### transfer the list into an array with this shape: n_iterations * num_tasks
        metric = np.array(results_roi_fa).transpose()
        ## rearranage the metric in order to have the right position for each task
        index_order = [0, 3, 1, 4, 2, 5]
        metric = metric[:, index_order]
        ## define the violin's postions
        pos = [1, 2, 4, 5, 7, 8]
        color = ['#FF0000', '#87CEFA'] * len(tasks_list)  # red, blue and green
        legendB = ['JHUTract25-original', 'JHUTract25-balanced']

        ## define the size of th image
        fig, ax = plt.subplots(2, figsize=[15, 10])
        line_coll = ax[0].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
        for cc, ln in enumerate(line_coll['bodies']):
            ln.set_facecolor(color[cc])

        ax[0].legend(legendB, loc='upper right', fontsize=10, frameon=True)
        ax[0].grid(axis='y', which='major', linestyle='dotted')
        ax[0].set_xticks([1.5 , 4.5, 7.5])
        ax[0].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax[0].set_xticklabels(tasks_list, rotation=0, fontsize=15)  # 'vertical'
        ax[0].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
        mean = np.mean(metric, 0)
        std = np.std(metric, 0)
        inds = np.array(pos)
        ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
        ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
        ax[0].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
        ax[0].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
        ax[0].set_ylim(0.1, 1)
        ax[0].set_title('A: FA region-based features', fontsize=15)

        ##### voxel MD
        ### transfer the list into an array with this shape: n_iterations*n_tasks_imbalanced
        metric = np.array(results_roi_md).transpose()
        ## rearranage the metric in order to have the right position for each task
        metric = metric[:, index_order]
        ## define the size of th image
        line_coll = ax[1].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
        for cc, ln in enumerate(line_coll['bodies']):
            ln.set_facecolor(color[cc])

        ax[1].legend(legendB, loc='upper right', fontsize=10, frameon=True)
        ax[1].grid(axis='y', which='major', linestyle='dotted')
        ax[1].set_xticks([1.5 , 4.5, 7.5])
        ax[1].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax[1].set_xticklabels(tasks_list, rotation=0, fontsize=15)  # 'vertical'
        ax[1].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
        mean = np.mean(metric, 0)
        std = np.std(metric, 0)
        inds = np.array(pos)
        ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
        ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
        ax[1].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
        ax[1].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
        ax[1].set_ylim(0.1, 1)
        ax[1].set_title('B: MD region-based features', fontsize=15)

        plt.savefig(os.path.join(classification_output_dir, 'figures',
                                 'figure_influence_of_imbalance_REGION.png'), additional_artists=plt.legend,
                    bbox_inches="tight")

    elif task_name == 'Influence_of_fwhm':
        ## results list to contain results with diff fwhm
        results_fwhm = []
        tissue_combinations = ['GM_WM']
        tasks_list = ['AD_vs_CN']
        fwhm_list = [0, 4, 8, 12]
        metric_list = ['fa', 'md']

        ## get DWI voxel with diff smoothing fwhm
        for metric in metric_list:
            for task in tasks_list:
                for tissue in tissue_combinations:
                    for f in fwhm_list:
                        tsvs_path = os.path.join(classification_output_dir, 'DWIWithFeatureRescalingVoxel',
                                                 task + '_VB_' + tissue + '_0.3_' + str(f) + '_' + metric)
                        balanced_accuracy = []
                        for i in xrange(n_iterations):
                            result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                            if os.path.isfile(result_tsv):
                                balanced_accuracy.append(
                                    (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                            else:
                                raise OSError(
                                    errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                        results_fwhm.append(balanced_accuracy)



        ## voxel plot
        ### transfer the list into an array with this shape: n_iterations * num_tasks
        metric = np.array(results_fwhm).transpose()

        ## define the violin's postions
        pos = [1, 2, 3, 4, 6, 7, 8, 9]
        color = ['#FF0000', '#87CEFA', '#90EE90', 'y'] * len(metric_list)  # red, blue and green
        legendA = ['No smoothing', 'fwhm=4mm', 'fwhm=8mm', 'fwhm=12mm']

        ## define the size of th image
        fig, ax = plt.subplots(1, figsize=[15, 10])
        line_coll = ax.violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
        for cc, ln in enumerate(line_coll['bodies']):
            ln.set_facecolor(color[cc])

        ax.legend(legendA, loc='lower right', fontsize=10, frameon=True)
        ax.grid(axis='y', which='major', linestyle='dotted')
        ax.set_xticks([2.5, 7.5])
        ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_xticklabels(tasks_list, rotation=0, fontsize=15)  # 'vertical'
        ax.set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
        mean = np.mean(metric, 0)
        std = np.std(metric, 0)
        inds = np.array(pos)
        ax.vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
        ax.vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
        ax.hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
        ax.hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
        ax.set_ylim(0.4, 1)
        ax.set_title('Influence of the smoothing on classification performance', fontsize=15)

        plt.savefig(os.path.join(classification_output_dir, 'figures',
                                 'figure_influence_of_fwhm.png'), additional_artists=plt.legend,
                    bbox_inches="tight")

    elif task_name == 'Influence_of_reg':
        ## results for registration
        results_fa = []
        results_md = []

        tissue_combinations = ['WM', 'GM', 'GM_WM']
        tasks_list = ['AD_vs_CN']

        ## get FA singlemodal
        for task in tasks_list:
            for tissue in tissue_combinations:
                tsvs_path = os.path.join(classification_output_dir, 'DWIWithFeatureRescalingVoxel',
                                         task + '_VB_' + tissue + '_0.3_8_fa')
                balanced_accuracy = []
                for i in xrange(n_iterations):
                    result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                    if os.path.isfile(result_tsv):
                        balanced_accuracy.append(
                            (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                    else:
                        raise OSError(
                            errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                results_fa.append(balanced_accuracy)

        ## get FA multimodal
        for task in tasks_list:
            for tissue in tissue_combinations:
                tsvs_path = os.path.join(classification_output_dir, 'DWIWithFeatureRescalingVoxelMultimodalReg',
                                         task + '_VB_' + tissue + '_0.3_8_fa')
                balanced_accuracy = []
                for i in xrange(n_iterations):
                    result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                    if os.path.isfile(result_tsv):
                        balanced_accuracy.append(
                            (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                    else:
                        raise OSError(
                            errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                results_fa.append(balanced_accuracy)

        ## get MD singlemodal
        for task in tasks_list:
            for tissue in tissue_combinations:
                tsvs_path = os.path.join(classification_output_dir, 'DWIWithFeatureRescalingVoxel',
                                         task + '_VB_' + tissue + '_0.3_8_md')
                balanced_accuracy = []
                for i in xrange(n_iterations):
                    result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                    if os.path.isfile(result_tsv):
                        balanced_accuracy.append(
                            (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                    else:
                        raise OSError(
                            errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                results_md.append(balanced_accuracy)

        ## get md multimodal
        for task in tasks_list:
            for tissue in tissue_combinations:
                tsvs_path = os.path.join(classification_output_dir, 'DWIWithFeatureRescalingVoxelMultimodalReg',
                                         task + '_VB_' + tissue + '_0.3_8_md')
                balanced_accuracy = []
                for i in xrange(n_iterations):
                    result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                    if os.path.isfile(result_tsv):
                        balanced_accuracy.append(
                            (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                    else:
                        raise OSError(
                            errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                results_md.append(balanced_accuracy)

        ## voxel plot fa
        ### transfer the list into an array with this shape: n_iterations * num_tasks
        metric = np.array(results_fa).transpose()
        ## rearranage the metric in order to have the right position for each task
        index_order = [0, 3, 2, 4, 3, 5]
        metric = metric[:, index_order]
        ## define the violin's postions
        pos = [1, 2, 4, 5, 7, 8]
        color = ['#FF0000', '#87CEFA', '#90EE90'] * len(tissue_combinations)  # red, blue and green
        legendA = ['Single-modal', 'Multimodal']

        ## define the size of th image
        fig, ax = plt.subplots(2, figsize=[15, 10])
        line_coll = ax[0].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
        for cc, ln in enumerate(line_coll['bodies']):
            ln.set_facecolor(color[cc])

        ax[0].legend(legendA, loc='lower right', fontsize=10, frameon=True)
        ax[0].grid(axis='y', which='major', linestyle='dotted')
        ax[0].set_xticks([1.5, 4.5, 7.5])
        ax[0].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax[0].set_xticklabels(tissue_combinations, rotation=0, fontsize=15)  # 'vertical'
        ax[0].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
        mean = np.mean(metric, 0)
        std = np.std(metric, 0)
        inds = np.array(pos)
        ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
        ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
        ax[0].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
        ax[0].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
        ax[0].set_ylim(0.4, 1)
        ax[0].set_title('A: FA feature for different registration methods', fontsize=15)

        ##### ROI
        ### transfer the list into an array with this shape: n_iterations*n_tasks_imbalanced
        metric = np.array(results_md).transpose()
        ## rearranage the metric in order to have the right position for each task
        metric = metric[:, index_order]
        ## define the size of th image
        line_coll = ax[1].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
        for cc, ln in enumerate(line_coll['bodies']):
            ln.set_facecolor(color[cc])

        ax[1].legend(legendA, loc='lower right', fontsize=10, frameon=True)
        ax[1].grid(axis='y', which='major', linestyle='dotted')
        ax[1].set_xticks([1.5, 4.5, 7.5])
        ax[1].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax[1].set_xticklabels(tissue_combinations, rotation=0, fontsize=15)  # 'vertical'
        ax[1].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
        mean = np.mean(metric, 0)
        std = np.std(metric, 0)
        inds = np.array(pos)
        ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
        ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
        ax[1].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
        ax[1].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
        ax[1].set_ylim(0.4, 1)
        ax[1].set_title('B: MD feature for different registration methods', fontsize=15)

        plt.savefig(os.path.join(classification_output_dir, 'figures',
                                 'figure_influence_of_reg.png'), additional_artists=plt.legend,
                    bbox_inches="tight")

def classification_performances_scatter_plot_feature_selection_neuroinformatics(classification_result_path, task_name, metric = 'fa', fs_technique='ANOVA+RFE'):
    """
    Figure for Neuroinformatics paper for feature selection bias!
    :return:
    """

    if task_name == 'Influence_of_feature_selection':
        results_balanced_acc_nested_anova = []
        results_balanced_acc_nonnested_anova = []
        results_balanced_acc_nested_rfe = []
        results_balanced_acc_nonnested_rfe = []

        x_persent_voxels = ['1', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']

        if fs_technique == 'ANOVA+RFE':

            for k in x_persent_voxels:
                result_tsv = os.path.join(classification_result_path, 'DWIWithoutFeatureRescalingFeatureSelectionVoxelNestedFSANOVA', 'AD_vs_CN_VB_GM_WM_0.3_8_fs_' + str(k) + '_' + metric, 'mean_results.tsv')
                balanced_accuracy = []

                if os.path.isfile(result_tsv):
                    balanced_accuracy.append((pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                else:
                    raise OSError(
                        errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                results_balanced_acc_nested_anova.append(balanced_accuracy)


            for k in x_persent_voxels:
                result_tsv = os.path.join(classification_result_path, 'DWIWithoutFeatureRescalingFeatureSelectionVoxelNonNestedFSANOVA', 'AD_vs_CN_VB_GM_WM_0.3_8_fs_' + str(k) + '_' + metric, 'mean_results.tsv')
                balanced_accuracy = []

                if os.path.isfile(result_tsv):
                    balanced_accuracy.append((pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                else:
                    raise OSError(
                        errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                results_balanced_acc_nonnested_anova.append(balanced_accuracy)

            for k in x_persent_voxels:
                result_tsv = os.path.join(classification_result_path, 'DWIWithoutFeatureRescalingFeatureSelectionVoxelNestedFSRFE', 'AD_vs_CN_VB_GM_WM_0.3_8_fs_' + str(k) + '_' + metric, 'mean_results.tsv')
                balanced_accuracy = []

                if os.path.isfile(result_tsv):
                    balanced_accuracy.append((pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                else:
                    raise OSError(
                        errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                results_balanced_acc_nested_rfe.append(balanced_accuracy)

            for k in x_persent_voxels:
                result_tsv = os.path.join(classification_result_path, 'DWIWithoutFeatureRescalingFeatureSelectionVoxelNonNestedFSRFE', 'AD_vs_CN_VB_GM_WM_0.3_8_fs_' + str(k) + '_' + metric, 'mean_results.tsv')
                balanced_accuracy = []

                if os.path.isfile(result_tsv):
                    balanced_accuracy.append((pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                else:
                    raise OSError(
                        errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                results_balanced_acc_nonnested_rfe.append(balanced_accuracy)


            num_rois = len(x_persent_voxels)
            # x = np.asarray(x_persent_voxels, dtype=float)
            x = np.arange(1, num_rois+1, 1)

            fig = plt.figure(figsize=[15, 10])
            ax = fig.add_subplot(111)

            # Remove the plot frame lines. They are unnecessary chartjunk.
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.scatter(x, results_balanced_acc_nested_anova, s=200, c='g', marker="^", label='nested ANOVA')
            ax.scatter(x, results_balanced_acc_nonnested_anova, s=200, c='r', marker="^", label='non-nested ANOVA')
            ax.scatter(x, results_balanced_acc_nested_rfe, s=200, c='b', marker="o", label='nested SVM-RFE')
            ax.scatter(x, results_balanced_acc_nonnested_rfe, s=200, c='y', marker="o", label='non-nested SVM-RFE')

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
            plt.ylim(0.5, 1.0)
            # plt.show()

            plt.savefig(os.path.join(classification_result_path, 'figures',
                                     'figure_influence_of_feature_selection_' + metric + '.png'), additional_artists=plt.legend,
                        bbox_inches="tight")

        else:
            print 'Not implemented yet'
        print 'finish'

    elif task_name == 'Influence_of_modality':
        results_balanced_acc_nested_rfe_t1w = []
        results_balanced_acc_nested_rfe_dti_fa_GM = []
        results_balanced_acc_nested_rfe_dti_md_GM = []
        results_balanced_acc_nested_rfe_dti_fa_GM_WM = []
        results_balanced_acc_nested_rfe_dti_md_GM_WM = []
        results_balanced_acc_nested_rfe_dti_fa_WM = []
        results_balanced_acc_nested_rfe_dti_md_WM = []


        x_persent_voxels = ['1', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']

        if fs_technique == 'RFE':
            ## for T1w
            for k in x_persent_voxels:
                result_tsv = os.path.join(classification_result_path,
                                          'T1WithFeatureRescalingFeatureSelectionVoxelNestedFSRFE',
                                          'AD_vs_CN_VB_fwhm_8_fs_' + str(k), 'mean_results.tsv')
                balanced_accuracy = []

                if os.path.isfile(result_tsv):
                    balanced_accuracy.append((pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                else:
                    raise OSError(
                        errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                results_balanced_acc_nested_rfe_t1w.append(balanced_accuracy)

            ### GM
            for k in x_persent_voxels:
                result_tsv = os.path.join(classification_result_path,
                                          'DWIWithFeatureRescalingFeatureSelectionVoxelNestedFSRFE',
                                          'AD_vs_CN_VB_GM_0.3_8_fs_' + str(k) + '_fa', 'mean_results.tsv')
                balanced_accuracy = []

                if os.path.isfile(result_tsv):
                    balanced_accuracy.append((pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                else:
                    raise OSError(
                        errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                results_balanced_acc_nested_rfe_dti_fa_GM.append(balanced_accuracy)

            for k in x_persent_voxels:
                result_tsv = os.path.join(classification_result_path,
                                          'DWIWithFeatureRescalingFeatureSelectionVoxelNestedFSRFE',
                                          'AD_vs_CN_VB_GM_0.3_8_fs_' + str(k) + '_md', 'mean_results.tsv')
                balanced_accuracy = []

                if os.path.isfile(result_tsv):
                    balanced_accuracy.append((pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                else:
                    raise OSError(
                        errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                results_balanced_acc_nested_rfe_dti_md_GM.append(balanced_accuracy)

            ## WM
            for k in x_persent_voxels:
                result_tsv = os.path.join(classification_result_path,
                                          'DWIWithFeatureRescalingFeatureSelectionVoxelNestedFSRFE',
                                          'AD_vs_CN_VB_WM_0.3_8_fs_' + str(k) + '_fa', 'mean_results.tsv')
                balanced_accuracy = []

                if os.path.isfile(result_tsv):
                    balanced_accuracy.append((pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                else:
                    raise OSError(
                        errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                results_balanced_acc_nested_rfe_dti_fa_WM.append(balanced_accuracy)

            for k in x_persent_voxels:
                result_tsv = os.path.join(classification_result_path,
                                          'DWIWithFeatureRescalingFeatureSelectionVoxelNestedFSRFE',
                                          'AD_vs_CN_VB_WM_0.3_8_fs_' + str(k) + '_md', 'mean_results.tsv')
                balanced_accuracy = []

                if os.path.isfile(result_tsv):
                    balanced_accuracy.append((pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                else:
                    raise OSError(
                        errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                results_balanced_acc_nested_rfe_dti_md_WM.append(balanced_accuracy)

            ## GM_WM
            for k in x_persent_voxels:
                result_tsv = os.path.join(classification_result_path,
                                          'DWIWithFeatureRescalingFeatureSelectionVoxelNestedFSRFE',
                                          'AD_vs_CN_VB_GM_WM_0.3_8_fs_' + str(k) + '_fa', 'mean_results.tsv')
                balanced_accuracy = []

                if os.path.isfile(result_tsv):
                    balanced_accuracy.append((pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                else:
                    raise OSError(
                        errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                results_balanced_acc_nested_rfe_dti_fa_GM_WM.append(balanced_accuracy)

            for k in x_persent_voxels:
                result_tsv = os.path.join(classification_result_path,
                                          'DWIWithFeatureRescalingFeatureSelectionVoxelNestedFSRFE',
                                          'AD_vs_CN_VB_GM_WM_0.3_8_fs_' + str(k) + '_md', 'mean_results.tsv')
                balanced_accuracy = []

                if os.path.isfile(result_tsv):
                    balanced_accuracy.append((pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                else:
                    raise OSError(
                        errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                results_balanced_acc_nested_rfe_dti_md_GM_WM.append(balanced_accuracy)

            num_rois = len(x_persent_voxels)
            # x = np.asarray(x_persent_voxels, dtype=float)
            x = np.arange(1, num_rois + 1, 1)

            fig = plt.figure(figsize=[15, 10])
            ax = fig.add_subplot(111)

            # Remove the plot frame lines. They are unnecessary chartjunk.
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.scatter(x, results_balanced_acc_nested_rfe_t1w, s=100, c='g', marker="^", label='GM-density')
            ax.scatter(x, results_balanced_acc_nested_rfe_dti_fa_GM, s=100, c='r', marker="o", label='GM-FA')
            ax.scatter(x, results_balanced_acc_nested_rfe_dti_md_GM, s=100, c='b', marker="o", label='GM-MD')
            ax.scatter(x, results_balanced_acc_nested_rfe_dti_fa_WM, s=100, c='y', marker="o", label='WM-FA')
            ax.scatter(x, results_balanced_acc_nested_rfe_dti_md_WM, s=100, c='m', marker="o", label='WM-MD')
            ax.scatter(x, results_balanced_acc_nested_rfe_dti_fa_GM_WM, s=100, c='c', marker="o", label='GM+WM-FA')
            ax.scatter(x, results_balanced_acc_nested_rfe_dti_md_GM_WM, s=100, c='k', marker="o", label='GM+WM-MD')

            plt.plot(x, results_balanced_acc_nested_rfe_t1w, c='g', linestyle='-')
            plt.plot(x, results_balanced_acc_nested_rfe_dti_fa_GM, c='r', linestyle='-.')
            plt.plot(x, results_balanced_acc_nested_rfe_dti_md_GM, c='b', linestyle='--')
            plt.plot(x, results_balanced_acc_nested_rfe_dti_fa_WM, c='y', linestyle='--')
            plt.plot(x, results_balanced_acc_nested_rfe_dti_md_WM, c='m', linestyle='--')
            plt.plot(x, results_balanced_acc_nested_rfe_dti_fa_GM_WM, c='c', linestyle='--')
            plt.plot(x, results_balanced_acc_nested_rfe_dti_md_GM_WM, c='k', linestyle='--')

            plt.legend(loc='upper right', fontsize=15, frameon=False)
            # plt.xticks(x, x_persent_voxels, rotation='vertical')
            plt.xticks(x, x_persent_voxels)
            plt.tick_params(axis='x', labelsize=20)
            plt.tick_params(axis='y', labelsize=25)
            # plt.xlabel('ROIs', fontsize=5)
            # plt.ylabel('f2', fontsize=5)
            plt.ylim(0.7, 1.0)
            # plt.show()

            plt.savefig(os.path.join(classification_result_path, 'figures',
                                     'figure_influence_of_modality_.png'),
                        additional_artists=plt.legend,
                        bbox_inches="tight")

        else:
            print 'Not implemented yet'
        print 'finish'