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
                                 dwi_maps=['fa', 'md']):
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

    Returns:

    """

    if feature_selection_nested:
        print("Classification with nested feature selection!\n")
        splits_indices, splits_indices_pickle = split_subjects_to_pickle(diagnoses_tsv, n_iterations=n_iterations, test_size=test_size)
        for dwi_map in dwi_maps:
            for i in tissue_type:
                for j in threshold:
                    for k in fwhm:
                        for l in top_k:
                            classification_dir = path.join(output_dir, 'DWIWithFeatureRescalingVoxelNestedFS' + feature_selection_method,
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

    else:
        print("Classification with non-nested feature selection\n")
        splits_indices, splits_indices_pickle = split_subjects_to_pickle(diagnoses_tsv, n_iterations=n_iterations, test_size=test_size)
        for dwi_map in dwi_maps:
            for i in tissue_type:
                for j in threshold:
                    for k in fwhm:
                        for l in top_k:
                            classification_dir = path.join(output_dir, 'DWIWithFeatureRescalingVoxelNonNestedFS' + feature_selection_method,
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
                classification_dir = path.join(output_dir, 'T1WithFeatureRescalingVoxelNestedFS' + feature_selection_method,
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



















    #
    #
    #
    # if modality == 'dwi' and figure_number == 0:
    #     results_balanced_acc_fa_imbalanced = []
    #     results_balanced_acc_fa_balanced = []
    #     results_balanced_acc_md_imbalanced = []
    #     results_balanced_acc_md_balanced = []
    #     if feature_type == 'voxel':
    #         tissue_combinations = ['WM', 'GM', 'GM_WM']
    #         ticklabels_imbalanced = [i.replace('_', ' ') for i in tasks_imbalanced]
    #         # ticklabels_imbalanced = ['CN vs AD', 'CN_vs_MCI', 'CN_vs_pMCI', 'sMCI_vs_pMCI']
    #         ticklabels_balanced = [i.replace('_', ' ') for i in tasks_balanced]
    #         # ticklabels_balanced = ['CN_vs_MCI', 'CN_vs_pMCI', 'sMCI_vs_pMCI']
    #
    #         if raw_classification == True:
    #             print "Plot for original classifications"
    #             ## get FA
    #             for task in tasks_imbalanced:
    #                 for tissue in tissue_combinations:
    #                     tsvs_path = os.path.join(classification_output_dir, 'original_results', task + '_VB_' + tissue + '_0.3_8_fa')
    #                     balanced_accuracy = []
    #                     for i in xrange(n_iterations):
    #                         result_tsv = os.path.join(tsvs_path, 'iteration-' +str(i), 'results.tsv')
    #                         if os.path.isfile(result_tsv):
    #                             balanced_accuracy.append((pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
    #                         else:
    #                             raise OSError(
    #         errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
    #                     results_balanced_acc_fa_imbalanced.append(balanced_accuracy)
    #
    #             ## get MD
    #             for task in tasks_imbalanced:
    #                 for tissue in tissue_combinations:
    #                     tsvs_path = os.path.join(classification_output_dir, 'original_results', task + '_VB_' + tissue + '_0.3_8_md')
    #                     balanced_accuracy = []
    #                     for i in xrange(n_iterations):
    #                         result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
    #                         if os.path.isfile(result_tsv):
    #                             balanced_accuracy.append((pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
    #                         else:
    #                             raise OSError(
    #         errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
    #                     results_balanced_acc_md_imbalanced.append(balanced_accuracy)
    #
    #             ##### FAs
    #             ### transfer the list into an array with this shape: n_iterations*n_tasks_imbalanced
    #             metric = np.array(results_balanced_acc_fa_imbalanced).transpose()
    #             ## define the violin's postions
    #             pos = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]
    #             color = ['#FF0000', '#87CEFA', '#90EE90'] *len(tasks_imbalanced)# red, blue and green
    #             legendA = ['WM-FA', 'GM-FA', 'GM+WM-FA']
    #
    #             ## define the size of th image
    #             fig, ax = plt.subplots(2,figsize=[15, 10])
    #             line_coll = ax[0].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
    #             for cc, ln in enumerate(line_coll['bodies']):
    #                 ln.set_facecolor(color[cc])
    #
    #             ax[0].legend(legendA, loc='upper right', fontsize=10, frameon=True)
    #             ax[0].grid(axis='y', which='major', linestyle='dotted')
    #             ax[0].set_xticks([2, 6, 10, 14])
    #             ax[0].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    #             ax[0].set_xticklabels(ticklabels_imbalanced, rotation=0, fontsize=15)  # 'vertical'
    #             ax[0].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
    #             mean = np.mean(metric, 0)
    #             std = np.std(metric, 0)
    #             inds = np.array(pos)
    #             ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #             ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #             ax[0].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #             ax[0].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #             ax[0].set_ylim(0.1, 1)
    #             ax[0].set_title('A: FA Voxel-based classifications', fontsize=15)
    #
    #             ##### MD
    #             ### transfer the list into an array with this shape: n_iterations*n_tasks_imbalanced
    #             metric = np.array(results_balanced_acc_md_imbalanced).transpose()
    #             ## define the size of th image
    #             line_coll = ax[1].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
    #             for cc, ln in enumerate(line_coll['bodies']):
    #                 ln.set_facecolor(color[cc])
    #             legendB = ['WM-MD', 'GM-MD', 'GM+WM-MD']
    #
    #             ax[1].legend(legendB, loc='upper right', fontsize=10, frameon=True)
    #             ax[1].grid(axis='y', which='major', linestyle='dotted')
    #             ax[1].set_xticks([2, 6, 10, 14])
    #             ax[1].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    #             ax[1].set_xticklabels(ticklabels_imbalanced, rotation=0, fontsize=15)  # 'vertical'
    #             ax[1].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
    #             mean = np.mean(metric, 0)
    #             std = np.std(metric, 0)
    #             inds = np.array(pos)
    #             ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #             ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #             ax[1].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #             ax[1].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #             ax[1].set_ylim(0.1, 1)
    #             ax[1].set_title('B: MD Voxel-based classifications', fontsize=15)
    #
    #             plt.savefig(os.path.join(classification_output_dir,
    #                                      'voxel_violin_imbalanced.png'), additional_artists=plt.legend, bbox_inches="tight")
    #
    #         else:
    #             print "Plot for balanced classifications"
    #             ## get FA
    #             for task in tasks_balanced:
    #                 for tissue in tissue_combinations:
    #                     balanced_accuracy = []
    #                     tsvs_path = os.path.join(classification_output_dir, 'balanced_results', 'RandomBalanced', task + '_VB_' + tissue + '_0.3_8', 'fa')
    #                     for k in xrange(n_iterations):
    #                         result_tsv = os.path.join(tsvs_path, 'iteration-' + str(k), 'results.tsv')
    #                         if os.path.isfile(result_tsv):
    #                             balanced_accuracy.append((pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
    #                         else:
    #                             raise OSError(
    #     errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
    #                     results_balanced_acc_fa_balanced.append(balanced_accuracy)
    #
    #             ## get MD
    #             for task in tasks_balanced:
    #                 for tissue in tissue_combinations:
    #                     balanced_accuracy = []
    #                     tsvs_path = os.path.join(classification_output_dir, 'balanced_results', 'RandomBalanced', task + '_VB_' + tissue + '_0.3_8', 'md')
    #                     for k in xrange(n_iterations):
    #                         result_tsv = os.path.join(tsvs_path, 'iteration-' + str(k), 'results.tsv')
    #                         if os.path.isfile(result_tsv):
    #                             balanced_accuracy.append((pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
    #                         else:
    #                             raise OSError(
    #     errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
    #                     results_balanced_acc_md_balanced.append(balanced_accuracy)
    #
    #             ##### FA
    #             ### transfer the list into an array with this shape: n_iterations*n_tasks_balanced
    #             metric = np.array(results_balanced_acc_fa_balanced).transpose()
    #             ## define the violin's postions
    #             pos = [1, 2, 3, 5, 6, 7, 9, 10, 11]
    #             color = ['#FF0000', '#87CEFA', '#90EE90'] *len(tasks_imbalanced)# red, blue and green
    #             legend = ['WM', 'GM', 'GM+WM']
    #
    #             ## define the size of th image
    #             fig, ax = plt.subplots(2, figsize=[15, 10])
    #             line_coll = ax[0].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
    #             for cc, ln in enumerate(line_coll['bodies']):
    #                 ln.set_facecolor(color[cc])
    #
    #             ax[0].legend(legend, loc='upper right', fontsize=10, frameon=True)
    #             ax[0].grid(axis='y', which='major', linestyle='dotted')
    #             ax[0].set_xticks([2, 6, 10, 14])
    #             ax[0].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    #             ax[0].set_xticklabels(ticklabels_balanced, rotation=0, fontsize=15)  # 'vertical'
    #             ax[0].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
    #             mean = np.mean(metric, 0)
    #             std = np.std(metric, 0)
    #             inds = np.array(pos)
    #             ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #             ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #             ax[0].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #             ax[0].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #             ax[0].set_ylim(0.1, 1)
    #             ax[0].set_title('A: FA Voxel-based classification with balanced data', fontsize=15)
    #
    #             ##### MD
    #             ### transfer the list into an array with this shape: n_iterations*n_tasks_balanced
    #             metric = np.array(results_balanced_acc_md_balanced).transpose()
    #             ## define the size of th image
    #             line_coll = ax[1].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
    #             for cc, ln in enumerate(line_coll['bodies']):
    #                 ln.set_facecolor(color[cc])
    #
    #             ax[1].legend(legend, loc='upper right', fontsize=10, frameon=True)
    #             ax[1].grid(axis='y', which='major', linestyle='dotted')
    #             ax[1].set_xticks([2, 6, 10, 14])
    #             ax[1].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    #             ax[1].set_xticklabels(ticklabels_balanced, rotation=0, fontsize=15)  # 'vertical'
    #             ax[1].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
    #             mean = np.mean(metric, 0)
    #             std = np.std(metric, 0)
    #             inds = np.array(pos)
    #             ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #             ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #             ax[1].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #             ax[1].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #             ax[1].set_ylim(0.1, 1)
    #             ax[1].set_title('B: MD Voxel-based classification with balanced data', fontsize=15)
    #
    #             plt.savefig(os.path.join(classification_output_dir,
    #                                      'voxel_violin_balanced.png'), additional_artists=plt.legend, bbox_inches="tight")
    #     else:  ##### for DWI regions
    #         atlases = ['JHUDTI81', 'JHUTracts25']
    #         ticklabels_imbalanced = [i.replace('_', ' ') for i in tasks_imbalanced]
    #         # ticklabels_imbalanced = ['CN vs AD', 'CN_vs_MCI', 'CN_vs_pMCI', 'sMCI_vs_pMCI']
    #         ticklabels_balanced = [i.replace('_', ' ') for i in tasks_balanced]
    #         # ticklabels_balanced = ['CN_vs_MCI', 'CN_vs_pMCI', 'sMCI_vs_pMCI']
    #
    #         if raw_classification == True:
    #             print "Plot for original classifications"
    #             ## get FA
    #             for task in tasks_imbalanced:
    #                 for atlas in atlases:
    #                     tsvs_path = os.path.join(classification_output_dir, 'original_results', task + '_RB_' + atlas, 'fa')
    #                     balanced_accuracy = []
    #                     for i in xrange(n_iterations):
    #                         result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
    #                         if os.path.isfile(result_tsv):
    #                             balanced_accuracy.append(
    #                                 (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
    #                         else:
    #                             raise OSError(
    #                                 errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
    #                     results_balanced_acc_fa_imbalanced.append(balanced_accuracy)
    #
    #             ## get MD
    #             for task in tasks_imbalanced:
    #                 for atlas in atlases:
    #                     tsvs_path = os.path.join(classification_output_dir, 'original_results', task + '_RB_' + atlas, 'md')
    #                     balanced_accuracy = []
    #                     for i in xrange(n_iterations):
    #                         result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
    #                         if os.path.isfile(result_tsv):
    #                             balanced_accuracy.append(
    #                                 (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
    #                         else:
    #                             raise OSError(
    #                                 errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
    #                     results_balanced_acc_md_imbalanced.append(balanced_accuracy)
    #
    #             ##### FAs
    #             ### transfer the list into an array with this shape: n_iterations*n_tasks_imbalanced
    #             metric = np.array(results_balanced_acc_fa_imbalanced).transpose()
    #             ## define the violin's postions
    #             pos = [1, 2, 4, 5, 7, 8, 10, 11]
    #             # color = ['#FF0000', '#87CEFA', '#90EE90'] * len(tasks_imbalanced)  # red, blue and green
    #             color = ['#FF0000', '#87CEFA'] * len(tasks_imbalanced)  # red, blue
    #             legendC = ['JHULabel', 'JHUTract25']
    #
    #             ## define the size of th image
    #             fig, ax = plt.subplots(2, figsize=[15, 10])
    #             line_coll = ax[0].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
    #             for cc, ln in enumerate(line_coll['bodies']):
    #                 ln.set_facecolor(color[cc])
    #
    #             ax[0].legend(legendC, loc='upper right', fontsize=10, frameon=True)
    #             ax[0].grid(axis='y', which='major', linestyle='dotted')
    #             ax[0].set_xticks([1.5, 4.5, 7.5, 10.5])
    #             ax[0].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    #             ax[0].set_xticklabels(ticklabels_imbalanced, rotation=0, fontsize=15)  # 'vertical'
    #             ax[0].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
    #             mean = np.mean(metric, 0)
    #             std = np.std(metric, 0)
    #             inds = np.array(pos)
    #             ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #             ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #             ax[0].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #             ax[0].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #             ax[0].set_ylim(0.1, 1)
    #             ax[0].set_title('C: FA Region-based classifications', fontsize=15)
    #
    #             ##### MD
    #             ### transfer the list into an array with this shape: n_iterations*n_tasks_imbalanced
    #             metric = np.array(results_balanced_acc_md_imbalanced).transpose()
    #             ## define the size of th image
    #             line_coll = ax[1].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
    #             for cc, ln in enumerate(line_coll['bodies']):
    #                 ln.set_facecolor(color[cc])
    #
    #             ax[1].legend(legendC, loc='upper right', fontsize=10)
    #             ax[1].grid(axis='y', which='major', linestyle='dotted')
    #             ax[1].set_xticks([1.5, 4.5, 7.5, 10.5])
    #             ax[1].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    #             ax[1].set_xticklabels(ticklabels_imbalanced, rotation=0, fontsize=15)  # 'vertical'
    #             ax[1].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
    #             mean = np.mean(metric, 0)
    #             std = np.std(metric, 0)
    #             inds = np.array(pos)
    #             ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #             ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #             ax[1].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #             ax[1].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #             ax[1].set_ylim(0.1, 1)
    #             ax[1].set_title('D: MD Region-based classifications', fontsize=15)
    #
    #             plt.savefig(os.path.join(classification_output_dir,
    #                                      'region_violin_imbalanced.png'), additional_artists=plt.legend,
    #                         bbox_inches="tight")
    #
    #         else:
    #             print "Plot for balanced classifications"
    #             ## get FA
    #             for task in tasks_balanced:
    #                 for atlas in atlases:
    #                     balanced_accuracy = []
    #                     tsvs_path = os.path.join(classification_output_dir, 'RandomBalanced',
    #                                              task + '_RB_' + atlas, 'fa')
    #                     for k in xrange(n_iterations):
    #                         result_tsv = os.path.join(tsvs_path, 'iteration-' + str(k), 'results.tsv')
    #                         if os.path.isfile(result_tsv):
    #                             balanced_accuracy.append(
    #                                 (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
    #                         else:
    #                             raise OSError(
    #                                 errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
    #                     results_balanced_acc_fa_balanced.append(balanced_accuracy)
    #
    #             ## get MD
    #             for task in tasks_balanced:
    #                 for atlas in atlases:
    #                     balanced_accuracy = []
    #                     tsvs_path = os.path.join(classification_output_dir, 'RandomBalanced',
    #                                              task + '_RB_' + atlas, 'md')
    #                     for k in xrange(n_iterations):
    #                         result_tsv = os.path.join(tsvs_path, 'iteration-' + str(k), 'results.tsv')
    #                         if os.path.isfile(result_tsv):
    #                             balanced_accuracy.append(
    #                                 (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
    #                         else:
    #                             raise OSError(
    #                                 errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
    #                     results_balanced_acc_md_balanced.append(balanced_accuracy)
    #
    #             ##### FA
    #             ### transfer the list into an array with this shape: n_iterations*n_tasks_balanced
    #             metric = np.array(results_balanced_acc_fa_balanced).transpose()
    #             ## define the violin's postions
    #             pos = [1, 2, 4, 5, 7, 8]
    #             color = ['#FF0000', '#87CEFA'] * len(tasks_imbalanced)  # red, blue and green
    #             legend = ['JHULabel', 'JHUTract']
    #
    #             ## define the size of th image
    #             fig, ax = plt.subplots(2, figsize=[15, 10])
    #             line_coll = ax[0].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
    #             for cc, ln in enumerate(line_coll['bodies']):
    #                 ln.set_facecolor(color[cc])
    #
    #             ax[0].legend(legend, loc='upper right', fontsize=10, frameon=True)
    #             ax[0].grid(axis='y', which='major', linestyle='dotted')
    #             ax[0].set_xticks([1.5, 4.5, 7.5, 10.5])
    #             ax[0].set_xticklabels(ticklabels_balanced, rotation=0, fontsize=15)  # 'vertical'
    #             ax[0].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
    #             mean = np.mean(metric, 0)
    #             std = np.std(metric, 0)
    #             inds = np.array(pos)
    #             ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #             ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #             ax[0].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #             ax[0].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #             ax[0].set_ylim(0.1, 1)
    #             ax[0].set_title('C: FA Region-based classification with balanced data', fontsize=15)
    #
    #             ##### MD
    #             ### transfer the list into an array with this shape: n_iterations*n_tasks_balanced
    #             metric = np.array(results_balanced_acc_md_balanced).transpose()
    #             ## define the size of th image
    #             line_coll = ax[1].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
    #             for cc, ln in enumerate(line_coll['bodies']):
    #                 ln.set_facecolor(color[cc])
    #
    #             ax[1].legend(legend, loc='upper right', fontsize=10, frameon=True)
    #             ax[1].grid(axis='y', which='major', linestyle='dotted')
    #             ax[1].set_xticks([1.5, 4.5, 7.5, 10.5])
    #             ax[1].set_xticklabels(ticklabels_balanced, rotation=0, fontsize=15)  # 'vertical'
    #             ax[1].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
    #             mean = np.mean(metric, 0)
    #             std = np.std(metric, 0)
    #             inds = np.array(pos)
    #             ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #             ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #             ax[1].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #             ax[1].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #             ax[1].set_ylim(0.1, 1)
    #             ax[1].set_title('D: MD Region-based classification with balanced data', fontsize=15)
    #
    #             plt.savefig(os.path.join(classification_output_dir,
    #                                      'region_violin_balanced.png'), additional_artists=plt.legend,
    #                         bbox_inches="tight")
    #
    #     print 'finish DWI'
    # elif modality == 'T1':
    #     results_balanced_acc_voxel_imbalanced = []
    #     results_balanced_acc_regional_imbalanced = []
    #     tissue_combinations = ['GM_WM']
    #     ticklabels_imbalanced = [i.replace('_', ' ') for i in tasks_imbalanced]
    #
    #     if raw_classification == True:
    #         print "Plot for original classification, to compare T1 with DWI, we use GM+WM"
    #         ## T1
    #         for task in tasks_imbalanced:
    #             tsvs_path = os.path.join(classification_output_dir, 'original_results', task + '_VB_T1_fwhm_8')
    #             balanced_accuracy = []
    #             for i in xrange(n_iterations):
    #                 result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
    #                 if os.path.isfile(result_tsv):
    #                     balanced_accuracy.append(
    #                         (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
    #                 else:
    #                     raise OSError(
    #                         errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
    #             results_balanced_acc_voxel_imbalanced.append(balanced_accuracy)
    #         ## GM+WM FA
    #         for task in tasks_imbalanced:
    #             for tissue in tissue_combinations:
    #                 tsvs_path = os.path.join(classification_output_dir, 'original_results',
    #                                          task + '_VB_' + tissue + '_0.3_8_fa')
    #                 balanced_accuracy = []
    #                 for i in xrange(n_iterations):
    #                     result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
    #                     if os.path.isfile(result_tsv):
    #                         balanced_accuracy.append(
    #                             (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
    #                     else:
    #                         raise OSError(
    #                             errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
    #                 results_balanced_acc_voxel_imbalanced.append(balanced_accuracy)
    #
    #         ## GM+WM MD
    #         for task in tasks_imbalanced:
    #             for tissue in tissue_combinations:
    #                 tsvs_path = os.path.join(classification_output_dir, 'original_results',
    #                                          task + '_VB_' + tissue + '_0.3_8_md')
    #                 balanced_accuracy = []
    #                 for i in xrange(n_iterations):
    #                     result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
    #                     if os.path.isfile(result_tsv):
    #                         balanced_accuracy.append(
    #                             (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
    #                     else:
    #                         raise OSError(
    #                             errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
    #                 results_balanced_acc_voxel_imbalanced.append(balanced_accuracy)
    #
    #         ### transfer the list into an array with this shape: n_iterations*n_tasks_imbalanced
    #         metric = np.array(results_balanced_acc_voxel_imbalanced).transpose()
    #         ## reorder the order of the column to make sure the right order in the image
    #         metric_new = metric[:, [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]]
    #
    #         ## define the violin's postions
    #         pos = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]
    #         color = ['#FF0000', '#87CEFA', '#90EE90'] * len(tasks_imbalanced)  # red, blue and green
    #         legendA = ['GM-Density', 'GM-FA', 'GM-MD']
    #
    #         ## define the size of th image
    #         fig, ax = plt.subplots(2, figsize=[15, 10])
    #         line_coll = ax[0].violinplot(metric_new, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
    #         for cc, ln in enumerate(line_coll['bodies']):
    #             ln.set_facecolor(color[cc])
    #
    #         ax[0].legend(legendA, loc='upper right', fontsize=10, frameon=True)
    #         ax[0].grid(axis='y', which='major', linestyle='dotted')
    #         ax[0].set_xticks([2, 6, 10, 14])
    #         ax[0].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    #         ax[0].set_xticklabels(ticklabels_imbalanced, rotation=0, fontsize=15)  # 'vertical'
    #         ax[0].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
    #         mean = np.mean(metric_new, 0)
    #         std = np.std(metric_new, 0)
    #         inds = np.array(pos)
    #         ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #         ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #         ax[0].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #         ax[0].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #         ax[0].set_ylim(0.1, 1)
    #         ax[0].set_title('A: Voxel-based classifications for T1w and diffusion MRI', fontsize=15)
    #
    #
    #         ### T1 atlaes
    #         atlases_T1 = ['AAL2']
    #         for task in tasks_imbalanced:
    #             for atlas in atlases_T1:
    #                 tsvs_path = os.path.join(classification_output_dir, 'original_results', task + '_RB_T1_' + atlas)
    #                 balanced_accuracy = []
    #                 for i in xrange(n_iterations):
    #                     result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
    #                     if os.path.isfile(result_tsv):
    #                         balanced_accuracy.append(
    #                             (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
    #                     else:
    #                         raise OSError(
    #                             errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
    #                 results_balanced_acc_regional_imbalanced.append(balanced_accuracy)
    #
    #         atlases_DTI = ['JHUDTI81']
    #         ## get DTI atlases FA
    #         for task in tasks_imbalanced:
    #             for atlas in atlases_DTI:
    #                 tsvs_path = os.path.join(classification_output_dir, 'original_results', task + '_RB_' + atlas, 'fa')
    #                 balanced_accuracy = []
    #                 for i in xrange(n_iterations):
    #                     result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
    #                     if os.path.isfile(result_tsv):
    #                         balanced_accuracy.append(
    #                             (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
    #                     else:
    #                         raise OSError(
    #                             errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
    #                 results_balanced_acc_regional_imbalanced.append(balanced_accuracy)
    #         #
    #         # ## get DTI atlases MD
    #         # for task in tasks_imbalanced:
    #         #     for atlas in atlases_DTI:
    #         #         tsvs_path = os.path.join(classification_output_dir, task + '_RB_' + atlas, 'md')
    #         #         balanced_accuracy = []
    #         #         for i in xrange(n_iterations):
    #         #             result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
    #         #             if os.path.isfile(result_tsv):
    #         #                 balanced_accuracy.append(
    #         #                     (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
    #         #             else:
    #         #                 raise OSError(
    #         #                     errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
    #         #         results_balanced_acc_regional_imbalanced.append(balanced_accuracy)
    #
    #         ##### FAs
    #         ### transfer the list into an array with this shape: n_iterations*n_tasks_imbalanced
    #         metric = np.array(results_balanced_acc_regional_imbalanced).transpose()
    #         ## reorder the order of the column to make sure the right order in the image
    #         metric_new = metric[:, [0, 4, 1, 5, 2, 6, 3, 7]]
    #
    #         ## define the violin's postions
    #         pos = [1, 2, 4, 5, 7, 8, 10, 11]
    #         color = ['#FF0000', '#87CEFA'] * len(tasks_imbalanced)  # red, blue and green
    #         legendB = ['AAL2', 'JHULabel-FA']
    #
    #         ## define the size of th image
    #         line_coll = ax[1].violinplot(metric_new, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
    #         for cc, ln in enumerate(line_coll['bodies']):
    #             ln.set_facecolor(color[cc])
    #
    #         ax[1].legend(legendB, loc='upper right', fontsize=10, frameon=True)
    #         ax[1].grid(axis='y', which='major', linestyle='dotted')
    #         ax[1].set_xticks([1.5, 4.5, 7.5, 10.5])
    #         ax[1].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    #         ax[1].set_xticklabels(ticklabels_imbalanced, rotation=0, fontsize=15)  # 'vertical'
    #         ax[1].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
    #         mean = np.mean(metric_new, 0)
    #         std = np.std(metric_new, 0)
    #         inds = np.array(pos)
    #         ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #         ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #         ax[1].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #         ax[1].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #         ax[1].set_ylim(0.1, 1)
    #         ax[1].set_title('B: Region-based classifications for T1w and diffusion MRI', fontsize=15)
    #         plt.savefig(os.path.join(classification_output_dir,
    #                                  'violin_T1_compare_dwi.png'), additional_artists=plt.legend,
    #                     bbox_inches="tight")
    #     #
    #     print 'finish T1'
    #
    # else:
    #     pass
    #
    # if figure_number == 3:
    #     results_acc_fa_voxel = []
    #     results_acc_md_voxel = []
    #     results_acc_fa_region = []
    #     results_acc_md_region = []
    #
    #     if feature_type == 'voxel':
    #         tissue_combinations = ['GM_WM']
    #         ticklabels_imbalanced = [i.replace('_', ' ') for i in tasks_imbalanced]
    #         # ticklabels_imbalanced = ['CN vs AD', 'CN_vs_MCI', 'CN_vs_pMCI', 'sMCI_vs_pMCI']
    #         ticklabels_balanced = [i.replace('_', ' ') for i in tasks_balanced]
    #         # ticklabels_balanced = ['CN_vs_MCI', 'CN_vs_pMCI', 'sMCI_vs_pMCI']
    #         atlases = ['JHUTracts25']
    #
    #         print "Plot for figure to compare the balanced and imbalanced results"
    #
    #         ## for region
    #         ## get FA region original classification
    #         for task in tasks_imbalanced:
    #             for atlas in atlases:
    #                 tsvs_path = os.path.join(classification_output_dir, 'original_results', task + '_RB_' + atlas, 'fa')
    #                 balanced_accuracy = []
    #                 for i in xrange(n_iterations):
    #                     result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
    #                     if os.path.isfile(result_tsv):
    #                         balanced_accuracy.append(
    #                             (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
    #                     else:
    #                         raise OSError(
    #                             errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
    #                 results_acc_fa_region.append(balanced_accuracy)
    #
    #         ## get MD region original classification
    #         for task in tasks_imbalanced:
    #             for atlas in atlases:
    #                 tsvs_path = os.path.join(classification_output_dir, 'original_results', task + '_RB_' + atlas,
    #                                          'md')
    #                 balanced_accuracy = []
    #                 for i in xrange(n_iterations):
    #                     result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
    #                     if os.path.isfile(result_tsv):
    #                         balanced_accuracy.append(
    #                             (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
    #                     else:
    #                         raise OSError(
    #                             errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
    #                 results_acc_md_region.append(balanced_accuracy)
    #
    #         ## get FA region balanced
    #         for task in tasks_balanced:
    #             for atlas in atlases:
    #                 balanced_accuracy = []
    #                 tsvs_path = os.path.join(classification_output_dir, 'balanced_results', 'RandomBalanced',
    #                                          task + '_RB_' + atlas, 'fa')
    #                 for k in xrange(n_iterations):
    #                     result_tsv = os.path.join(tsvs_path, 'iteration-' + str(k), 'results.tsv')
    #                     if os.path.isfile(result_tsv):
    #                         balanced_accuracy.append(
    #                             (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
    #                     else:
    #                         raise OSError(
    #                             errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
    #                 results_acc_fa_region.append(balanced_accuracy)
    #
    #         ## get MD region balanced
    #         for task in tasks_balanced:
    #             for atlas in atlases:
    #                 balanced_accuracy = []
    #                 tsvs_path = os.path.join(classification_output_dir, 'balanced_results', 'RandomBalanced',
    #                                          task + '_RB_' + atlas, 'md')
    #                 for k in xrange(n_iterations):
    #                     result_tsv = os.path.join(tsvs_path, 'iteration-' + str(k), 'results.tsv')
    #                     if os.path.isfile(result_tsv):
    #                         balanced_accuracy.append(
    #                             (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
    #                     else:
    #                         raise OSError(
    #                             errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
    #                 results_acc_md_region.append(balanced_accuracy)
    #
    #         ##### FAs
    #         ### transfer the list into an array with this shape: n_iterations*n_tasks_imbalanced
    #         metric = np.array(results_acc_fa_region).transpose()
    #         metric = metric[:, [0, 3, 1, 4, 2, 5]]
    #
    #         ## define the violin's postions
    #         pos = [1, 2, 4, 5, 7, 8]
    #         # color = ['#FF0000', '#87CEFA', '#90EE90'] * len(tasks_imbalanced)  # red, blue and green
    #         color = ['#FF0000', '#87CEFA'] * len(tasks_imbalanced)  # red, blue
    #         legendC = ['JHUTract25-original', 'JHUTract25-balanced']
    #
    #         ## define the size of th image
    #         fig, ax = plt.subplots(2, figsize=[15, 10])
    #         line_coll = ax[0].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
    #         for cc, ln in enumerate(line_coll['bodies']):
    #             ln.set_facecolor(color[cc])
    #
    #         ax[0].legend(legendC, loc='upper right', fontsize=10, frameon=True)
    #         ax[0].grid(axis='y', which='major', linestyle='dotted')
    #         ax[0].set_xticks([1.5, 4.5, 7.5])
    #         ax[0].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    #         ax[0].set_xticklabels(ticklabels_balanced, rotation=0, fontsize=15)  # 'vertical'
    #         ax[0].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
    #         mean = np.mean(metric, 0)
    #         std = np.std(metric, 0)
    #         inds = np.array(pos)
    #         ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #         ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #         ax[0].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #         ax[0].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #         ax[0].set_ylim(0.1, 1)
    #         ax[0].set_title('C: FA Region-based classifications', fontsize=15)
    #
    #         ##### MDs
    #         ### transfer the list into an array with this shape: n_iterations*n_tasks_imbalanced
    #         metric = np.array(results_acc_md_region).transpose()
    #         metric = metric[:, [0, 3, 1, 4, 2, 5]]
    #
    #         ## define the violin's postions
    #         pos = [1, 2, 4, 5, 7, 8]
    #         # color = ['#FF0000', '#87CEFA', '#90EE90'] * len(tasks_imbalanced)  # red, blue and green
    #         color = ['#FF0000', '#87CEFA'] * len(tasks_imbalanced)  # red, blue
    #
    #         ## define the size of th image
    #         line_coll = ax[1].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
    #         for cc, ln in enumerate(line_coll['bodies']):
    #             ln.set_facecolor(color[cc])
    #
    #         ax[1].legend(legendC, loc='upper right', fontsize=10, frameon=True)
    #         ax[1].grid(axis='y', which='major', linestyle='dotted')
    #         ax[1].set_xticks([1.5, 4.5, 7.5])
    #         ax[1].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    #         ax[1].set_xticklabels(ticklabels_balanced, rotation=0, fontsize=15)  # 'vertical'
    #         ax[1].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
    #         mean = np.mean(metric, 0)
    #         std = np.std(metric, 0)
    #         inds = np.array(pos)
    #         ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #         ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #         ax[1].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #         ax[1].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #         ax[1].set_ylim(0.1, 1)
    #         ax[1].set_title('D: MD Region-based classifications', fontsize=15)
    #         plt.savefig(os.path.join(classification_output_dir,
    #                                  'figure3_CD.png'), additional_artists=plt.legend,
    #                     bbox_inches="tight")
    #
    #         ### for voxel
    #         ## get FA raw
    #         for task in tasks_imbalanced:
    #             for tissue in tissue_combinations:
    #                 tsvs_path = os.path.join(classification_output_dir, 'original_results', task + '_VB_' + tissue + '_0.3_8_fa')
    #                 balanced_accuracy = []
    #                 for i in xrange(n_iterations):
    #                     result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
    #                     if os.path.isfile(result_tsv):
    #                         balanced_accuracy.append(
    #                             (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
    #                     else:
    #                         raise OSError(
    #                             errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
    #                 results_acc_fa_voxel.append(balanced_accuracy)
    #
    #         ## get MD raw
    #         for task in tasks_imbalanced:
    #             for tissue in tissue_combinations:
    #                 tsvs_path = os.path.join(classification_output_dir, 'original_results', task + '_VB_' + tissue + '_0.3_8_md')
    #                 balanced_accuracy = []
    #                 for i in xrange(n_iterations):
    #                     result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
    #                     if os.path.isfile(result_tsv):
    #                         balanced_accuracy.append(
    #                             (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
    #                     else:
    #                         raise OSError(
    #                             errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
    #                 results_acc_md_voxel.append(balanced_accuracy)
    #
    #         ## get FA balanced
    #         for task in tasks_balanced:
    #             for tissue in tissue_combinations:
    #                 balanced_accuracy = []
    #                 tsvs_path = os.path.join(classification_output_dir, 'balanced_results', 'RandomBalanced',
    #                                          task + '_VB_' + tissue + '_0.3_8', 'fa')
    #                 for k in xrange(n_iterations):
    #                     result_tsv = os.path.join(tsvs_path, 'iteration-' + str(k), 'results.tsv')
    #                     if os.path.isfile(result_tsv):
    #                         balanced_accuracy.append(
    #                             (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
    #                     else:
    #                         raise OSError(
    #                             errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
    #                 results_acc_fa_voxel.append(balanced_accuracy)
    #
    #         ## get MD balanced
    #         for task in tasks_balanced:
    #             for tissue in tissue_combinations:
    #                 balanced_accuracy = []
    #                 tsvs_path = os.path.join(classification_output_dir, 'balanced_results', 'RandomBalanced',
    #                                          task + '_VB_' + tissue + '_0.3_8', 'md')
    #                 for k in xrange(n_iterations):
    #                     result_tsv = os.path.join(tsvs_path, 'iteration-' + str(k), 'results.tsv')
    #                     if os.path.isfile(result_tsv):
    #                         balanced_accuracy.append(
    #                             (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
    #                     else:
    #                         raise OSError(
    #                             errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
    #                 results_acc_md_voxel.append(balanced_accuracy)
    #
    #         ##### FAs
    #         ### transfer the list into an array with this shape: n_iterations*n_tasks_imbalanced
    #         metric = np.array(results_acc_fa_voxel).transpose()
    #         metric = metric[:, [0, 3, 1, 4, 2, 5]]
    #
    #         ## define the violin's postions
    #         # pos = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]
    #         pos = [1, 2, 4, 5, 7, 8]
    #         # color = ['#FF0000', '#87CEFA', '#90EE90'] * len(tasks_imbalanced)  # red, blue and green
    #         color = ['#FF0000', '#87CEFA'] * len(tasks_imbalanced)  # red, blue and green
    #         legendA = ['GM+WM-original', 'GM+WM-balanced']
    #
    #         ## define the size of th image
    #         fig, ax = plt.subplots(2, figsize=[15, 10])
    #         line_coll = ax[0].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
    #         for cc, ln in enumerate(line_coll['bodies']):
    #             ln.set_facecolor(color[cc])
    #
    #         ax[0].legend(legendA, loc='upper right', fontsize=10, frameon=True)
    #         ax[0].grid(axis='y', which='major', linestyle='dotted')
    #         ax[0].set_xticks([1.5, 4.5, 7.5])
    #         ax[0].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    #         ax[0].set_xticklabels(ticklabels_balanced, rotation=0, fontsize=15)  # 'vertical'
    #         ax[0].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
    #         mean = np.mean(metric, 0)
    #         std = np.std(metric, 0)
    #         inds = np.array(pos)
    #         ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #         ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #         ax[0].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #         ax[0].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #         ax[0].set_ylim(0.1, 1)
    #         ax[0].set_title('A: FA Voxel-based classifications', fontsize=15)
    #
    #         ##### MD
    #         ### transfer the list into an array with this shape: n_iterations*n_tasks_imbalanced
    #         metric = np.array(results_acc_md_voxel).transpose()
    #         metric = metric[:, [0, 3, 1, 4, 2, 5]]
    #         ## define the size of th image
    #         line_coll = ax[1].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
    #         for cc, ln in enumerate(line_coll['bodies']):
    #             ln.set_facecolor(color[cc])
    #
    #         ax[1].legend(legendA, loc='upper right', fontsize=10, frameon=True)
    #         ax[1].grid(axis='y', which='major', linestyle='dotted')
    #         ax[1].set_xticks([1.5, 4.5, 7.5])
    #         ax[1].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    #         ax[1].set_xticklabels(ticklabels_balanced, rotation=0, fontsize=15)  # 'vertical'
    #         ax[1].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
    #         mean = np.mean(metric, 0)
    #         std = np.std(metric, 0)
    #         inds = np.array(pos)
    #         ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #         ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #         ax[1].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #         ax[1].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #         ax[1].set_ylim(0.1, 1)
    #         ax[1].set_title('B: MD Voxel-based classifications', fontsize=15)
    #
    #         plt.savefig(os.path.join(classification_output_dir,
    #                                  'figure3_AB.png'), additional_artists=plt.legend,
    #                     bbox_inches="tight")
    #     print 'finish Figure 3'
    #
    #
    # elif figure_number == 4:
    #     results_balanced_acc_voxel_imbalanced = []
    #     results_balanced_acc_regional_imbalanced = []
    #     tissue_combinations = ['GM_WM']
    #     ticklabels_imbalanced = [i.replace('_', ' ') for i in tasks_imbalanced]
    #
    #     if raw_classification == True:
    #         print "Plot for original classification, to compare T1 with DWI, we use GM+WM"
    #         ## T1
    #         for task in tasks_imbalanced:
    #             tsvs_path = os.path.join(classification_output_dir, task + '_VB_T1_fwhm_8')
    #             balanced_accuracy = []
    #             for i in xrange(n_iterations):
    #                 result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
    #                 if os.path.isfile(result_tsv):
    #                     balanced_accuracy.append(
    #                         (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
    #                 else:
    #                     raise OSError(
    #                         errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
    #             results_balanced_acc_voxel_imbalanced.append(balanced_accuracy)
    #         ## GM+WM FA
    #         for task in tasks_imbalanced:
    #             for tissue in tissue_combinations:
    #                 tsvs_path = os.path.join(classification_output_dir,
    #                                          task + '_VB_' + tissue + '_0.3_8', 'fa')
    #                 balanced_accuracy = []
    #                 for i in xrange(n_iterations):
    #                     result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
    #                     if os.path.isfile(result_tsv):
    #                         balanced_accuracy.append(
    #                             (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
    #                     else:
    #                         raise OSError(
    #                             errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
    #                 results_balanced_acc_voxel_imbalanced.append(balanced_accuracy)
    #
    #         ## GM+WM MD
    #         for task in tasks_imbalanced:
    #             for tissue in tissue_combinations:
    #                 tsvs_path = os.path.join(classification_output_dir,
    #                                          task + '_VB_' + tissue + '_0.3_8', 'md')
    #                 balanced_accuracy = []
    #                 for i in xrange(n_iterations):
    #                     result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
    #                     if os.path.isfile(result_tsv):
    #                         balanced_accuracy.append(
    #                             (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
    #                     else:
    #                         raise OSError(
    #                             errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
    #                 results_balanced_acc_voxel_imbalanced.append(balanced_accuracy)
    #
    #         ### transfer the list into an array with this shape: n_iterations*n_tasks_imbalanced
    #         metric = np.array(results_balanced_acc_voxel_imbalanced).transpose()
    #         ## reorder the order of the column to make sure the right order in the image
    #         metric_new = metric[:, [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]]
    #
    #         ## define the violin's postions
    #         pos = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]
    #         color = ['#FF0000', '#87CEFA', '#90EE90'] * len(tasks_imbalanced)  # red, blue and green
    #         legend = ['GM-T1w', 'GM+WM-FA', 'GM+WM-MD']
    #
    #         ## define the size of th image
    #         fig, ax = plt.subplots(2, figsize=[15, 10])
    #         line_coll = ax[0].violinplot(metric_new, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
    #         for cc, ln in enumerate(line_coll['bodies']):
    #             ln.set_facecolor(color[cc])
    #
    #         ax[0].legend(legend, loc='upper right', fontsize=10, frameon=True)
    #         ax[0].grid(axis='y', which='major', linestyle='dotted')
    #         ax[0].set_xticks([2, 6, 10, 14])
    #         ax[0].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    #         ax[0].set_xticklabels(ticklabels_imbalanced, rotation=0, fontsize=15)  # 'vertical'
    #         ax[0].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
    #         mean = np.mean(metric_new, 0)
    #         std = np.std(metric_new, 0)
    #         inds = np.array(pos)
    #         ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #         ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #         ax[0].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #         ax[0].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #         ax[0].set_ylim(0.1, 1)
    #         ax[0].set_title('A: Voxel-based classifications for T1w and diffusion MRI', fontsize=15)
    #
    #
    #         ### T1 atlaes
    #         atlases_T1 = ['AAL2']
    #         for task in tasks_imbalanced:
    #             for atlas in atlases_T1:
    #                 tsvs_path = os.path.join(classification_output_dir, task + '_RB_T1_' + atlas)
    #                 balanced_accuracy = []
    #                 for i in xrange(n_iterations):
    #                     result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
    #                     if os.path.isfile(result_tsv):
    #                         balanced_accuracy.append(
    #                             (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
    #                     else:
    #                         raise OSError(
    #                             errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
    #                 results_balanced_acc_regional_imbalanced.append(balanced_accuracy)
    #
    #         atlases_DTI = ['JHUDTI81']
    #         ## get DTI atlases FA
    #         for task in tasks_imbalanced:
    #             for atlas in atlases_DTI:
    #                 tsvs_path = os.path.join(classification_output_dir, task + '_RB_' + atlas, 'fa')
    #                 balanced_accuracy = []
    #                 for i in xrange(n_iterations):
    #                     result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
    #                     if os.path.isfile(result_tsv):
    #                         balanced_accuracy.append(
    #                             (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
    #                     else:
    #                         raise OSError(
    #                             errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
    #                 results_balanced_acc_regional_imbalanced.append(balanced_accuracy)
    #         #
    #         # ## get DTI atlases MD
    #         # for task in tasks_imbalanced:
    #         #     for atlas in atlases_DTI:
    #         #         tsvs_path = os.path.join(classification_output_dir, task + '_RB_' + atlas, 'md')
    #         #         balanced_accuracy = []
    #         #         for i in xrange(n_iterations):
    #         #             result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
    #         #             if os.path.isfile(result_tsv):
    #         #                 balanced_accuracy.append(
    #         #                     (pd.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
    #         #             else:
    #         #                 raise OSError(
    #         #                     errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
    #         #         results_balanced_acc_regional_imbalanced.append(balanced_accuracy)
    #
    #         ##### FAs
    #         ### transfer the list into an array with this shape: n_iterations*n_tasks_imbalanced
    #         metric = np.array(results_balanced_acc_regional_imbalanced).transpose()
    #         ## reorder the order of the column to make sure the right order in the image
    #         metric_new = metric[:, [0, 4, 1, 5, 2, 6, 3, 7]]
    #
    #         ## define the violin's postions
    #         pos = [1, 2, 4, 5, 7, 8, 10, 11]
    #         color = ['#FF0000', '#87CEFA'] * len(tasks_imbalanced)  # red, blue and green
    #         legend = ['AAL2-T1w', 'JHULabel-FA']
    #
    #         ## define the size of th image
    #         line_coll = ax[1].violinplot(metric_new, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
    #         for cc, ln in enumerate(line_coll['bodies']):
    #             ln.set_facecolor(color[cc])
    #
    #         ax[1].legend(legend, loc='upper right', fontsize=10, frameon=True)
    #         ax[1].grid(axis='y', which='major', linestyle='dotted')
    #         ax[1].set_xticks([1.5, 4.5, 7.5, 10.5])
    #         ax[1].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    #         ax[1].set_xticklabels(ticklabels_imbalanced, rotation=0, fontsize=15)  # 'vertical'
    #         ax[1].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
    #         mean = np.mean(metric_new, 0)
    #         std = np.std(metric_new, 0)
    #         inds = np.array(pos)
    #         ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #         ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
    #         ax[1].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #         ax[1].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
    #         ax[1].set_ylim(0.1, 1)
    #         ax[1].set_title('B: Region-based classifications for T1w and diffusion MRI', fontsize=15)
    #         plt.savefig(os.path.join(classification_output_dir,
    #                                  'violin_T1_compare_dwi.png'), additional_artists=plt.legend,
    #                     bbox_inches="tight")
    #     #
    #     print 'finish T1'