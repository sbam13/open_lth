# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from datasets import base, cifar10, mnist, imagenet
from foundations.hparams import DatasetHparams
from platforms.platform import get_platform

registered_datasets = {'cifar10': cifar10, 'mnist': mnist, 'imagenet': imagenet}


def get(dataset_hparams: DatasetHparams, train: bool = True):
    """Get the train or test set corresponding to the hyperparameters."""

    seed = dataset_hparams.transformation_seed or 0

    # Get the dataset itself.
    if dataset_hparams.dataset_name in registered_datasets:
        if train:
            use_augmentation = not dataset_hparams.do_not_augment
            dataset = registered_datasets[dataset_hparams.dataset_name].Dataset.get_train_set(use_augmentation)
        else:
            dataset = registered_datasets[dataset_hparams.dataset_name].Dataset.get_test_set()
    else:
        raise ValueError('No such dataset: {}'.format(dataset_hparams.dataset_name))

    # Transform the dataset.
    randomize = False
    if dataset_hparams.label_randomization_targets is not None:
        rand_targets = dataset_hparams.label_randomization_targets.split(',')
        rand_targets = [elem.strip() for elem in rand_targets]
        if train:
            randomize = 'train' in rand_targets
        else:
            randomize = 'test' in rand_targets
    
    if randomize and dataset_hparams.label_randomization_type is not None:
        _type = dataset_hparams.label_randomization_type
        if _type == 'shuffle':
            dataset.shuffle_labels(seed=seed)
        elif _type == 'corrupt':
            dataset.corrupt_labels(seed=seed, corrupt_prob=dataset_hparams.corruption_probability)
        elif _type == 'fraction':
            dataset.randomize_labels(seed=seed, fraction=dataset_hparams.random_labels_fraction)
        else:
            raise ValueError(f"'{dataset_hparams.label_randomization_type}' is not a valid randomization type.")

    if train and dataset_hparams.subsample_fraction is not None:
        dataset.subsample(seed=seed, fraction=dataset_hparams.subsample_fraction)

    if train and dataset_hparams.blur_factor is not None:
        if not isinstance(dataset, base.ImageDataset):
            raise ValueError('Can blur images.')
        else:
            dataset.blur(seed=seed, blur_factor=dataset_hparams.blur_factor)

    if dataset_hparams.unsupervised_labels is not None:
        if dataset_hparams.unsupervised_labels != 'rotation':
            raise ValueError('Unknown unsupervised labels: {}'.format(dataset_hparams.unsupervised_labels))
        elif not isinstance(dataset, base.ImageDataset):
            raise ValueError('Can only do unsupervised rotation to images.')
        else:
            dataset.unsupervised_rotation(seed=seed)

    # Create the loader.
    return registered_datasets[dataset_hparams.dataset_name].DataLoader(
        dataset, batch_size=dataset_hparams.batch_size, num_workers=get_platform().num_workers)


def iterations_per_epoch(dataset_hparams: DatasetHparams):
    """Get the number of iterations per training epoch."""

    if dataset_hparams.dataset_name in registered_datasets:
        num_train_examples = registered_datasets[dataset_hparams.dataset_name].Dataset.num_train_examples()
    else:
        raise ValueError('No such dataset: {}'.format(dataset_hparams.dataset_name))

    if dataset_hparams.subsample_fraction is not None:
        num_train_examples *= dataset_hparams.subsample_fraction

    return np.ceil(num_train_examples / dataset_hparams.batch_size).astype(int)


def num_classes(dataset_hparams: DatasetHparams):
    """Get the number of classes."""

    if dataset_hparams.dataset_name in registered_datasets:
        num_classes = registered_datasets[dataset_hparams.dataset_name].Dataset.num_classes()
    else:
        raise ValueError('No such dataset: {}'.format(dataset_hparams.dataset_name))

    if dataset_hparams.unsupervised_labels is not None:
        if dataset_hparams.unsupervised_labels != 'rotation':
            raise ValueError('Unknown unsupervised labels: {}'.format(dataset_hparams.unsupervised_labels))
        else:
            return 4

    return num_classes
