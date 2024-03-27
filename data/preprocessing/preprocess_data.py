import os
from os.path import join
from shutil import rmtree

from data_splitter import split_data
from data_cropper import crop_scanpaths
from data_cropper import crop_heatmaps


def create_datasets(project_path, create_heatmaps=True, create_scanpaths=True):
    """
    Creates the preprocessed datasets.
    :param project_path: path to the project
    :param create_heatmaps: flag to create the directory with heatmaps
    :param create_scanpaths: flag to create the directory with scanpaths
    :return: None
    """
    assert project_path is not None, "Please provide the project path"
    assert (os.path.exists(project_path))
    if create_scanpaths:
        crop_scanpaths(project_path)
    if create_heatmaps:
        crop_heatmaps(project_path)
    split_data(project_path, create_heatmaps, create_scanpaths)
    rmtree(join(project_path, "data/cropped/"))


if __name__ == '__main__':
    project_path = None
    create_datasets(project_path, create_heatmaps=True, create_scanpaths=True)
