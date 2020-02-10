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