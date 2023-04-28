import os.path as osp
import os
from typing import NamedTuple, Optional

import logging
import glob
from collections import defaultdict
from sacred import Ingredient
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from torchvision import transforms
import numpy as np
import pickle

from .image_dataset import ImageDataset
from .utils import RandomReplacedIdentitySampler, TripletSampler
from sklearn.neighbors import NearestNeighbors

data_ingredient = Ingredient('dataset')


@data_ingredient.config
def config():
    name = 'sop'
    data_path = 'data/Stanford_Online_Products'
    train_file = 'train.txt'
    test_file = 'test.txt'

    batch_size = 128
    sample_per_id = 2
    assert (batch_size % sample_per_id == 0)
    test_batch_size = 256
    sampler = 'random'

    num_workers = 8  
    pin_memory = True

    crop_size = 224
    recalls = [1, 5, 10, 20]

    num_identities = batch_size // sample_per_id 
    num_iterations = 59551 // batch_size

    train_cache_nn_inds  = None
    test_cache_nn_inds   = None


@data_ingredient.named_config
def sop_global():
    name = 'sop_global'
    batch_size = 800
    test_batch_size = 800
    sampler = 'random_id'


@data_ingredient.named_config
def sop_rerank():
    name = 'sop_rerank'
    batch_size = 300
    test_batch_size = 600
    sampler = 'triplet'
    # Recall 1, 5, 10, 20
    recalls = [1, 10, 100]

    train_cache_nn_inds  = 'rrt_sop_caches/rrt_r50_sop_nn_inds_train.pkl'
    test_cache_nn_inds   = 'rrt_sop_caches/rrt_r50_sop_nn_inds_test.pkl'


@data_ingredient.capture
def get_transforms(crop_size):
    train_transform, test_transform = [], []
    train_transform.extend([
        transforms.RandomResizedCrop(size=crop_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()])
    test_transform.append(transforms.Resize((256, 256)))
    test_transform.append(transforms.CenterCrop(size=224))
    test_transform.append(transforms.ToTensor())
    return transforms.Compose(train_transform), transforms.Compose(test_transform)


def read_file(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines


@data_ingredient.capture
def get_sets(name, data_path, train_file, test_file, num_workers, M=10, alpha=30, N=5, L=2,
                 current_group=0, min_images_per_class=10, queries_folder_name = "queries",
                 positive_dist_threshold=25):

    # Open training folder
    logging.debug(f"Searching training images in {train_file}")
        
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Folder {train_file} does not exist")
        
    images_paths = sorted(glob(f"{train_file}/**/*.jpg", recursive=True))
    logging.debug(f"Found {len(images_paths)} images")
        
    logging.debug("For each image, get its UTM east, UTM north and heading from its path")
    images_metadatas = [p.split("@") for p in images_paths]
    # field 1 is UTM east, field 2 is UTM north, field 9 is heading
    utmeast_utmnorth_heading = [(m[1], m[2], m[9]) for m in images_metadatas]
    utmeast_utmnorth_heading = np.array(utmeast_utmnorth_heading).astype(np.float)
    
    logging.debug("For each image, get class and group to which it belongs")
    class_id = [get__class_id(*m, M, alpha, N, L)
                            for m in utmeast_utmnorth_heading]
    
    logging.debug("Group together images belonging to the same class")
    images_per_class = defaultdict(list)
    for image_path, class_id in zip(images_paths, class_id):
        images_per_class[class_id].append(image_path)
    
    # Images_per_class is a dict where the key is class_id, and the value
    # is a list with the paths of images within that class.
    images_per_class = [(v, k) for k, v in images_per_class.items() if len(v) >= min_images_per_class]
    
    samples = [item for sublist in images_per_class for item in sublist]

    train_set = ImageDataset(samples=samples)

    # Open test/val folder
    database_folder = os.path.join(test_file, "database")
    queries_folder = os.path.join(test_file, queries_folder_name)

    base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    database_paths = sorted(glob(os.path.join(database_folder, "**", "*.jpg"), recursive=True))
    queries_paths = sorted(glob(os.path.join(queries_folder, "**", "*.jpg"),  recursive=True))
        
    # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
    database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in database_paths]).astype(float)
    queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in queries_paths]).astype(float)
     
    # img, class_id
    class_id_database = [get__class_id(*m, M, alpha, N, L)
                            for m in database_utms]
    class_id_queries = [get__class_id(*m, M, alpha, N, L)
                            for m in queries_utms]
    
    images_per_class_database = defaultdict(list)
    for image_path, class_id in zip(database_paths, class_id_database):
        images_per_class_database[class_id].append(image_path)

    images_per_class_database = [(v, k) for k, v in images_per_class_database.items() if len(v) >= min_images_per_class]

    images_per_class_queries = defaultdict(list)
    for image_path, class_id in zip(queries_paths, class_id_queries):
        images_per_class_queries[class_id].append(image_path)
    
    images_per_class_queries = [(v, k) for k, v in images_per_class_queries.items() if len(v) >= min_images_per_class]

    samples_database = [item for sublist in images_per_class_database for item in sublist]
    samples_queries = [item for sublist in images_per_class_queries for item in sublist]

    knn = NearestNeighbors(n_jobs=-1)
    knn.fit(database_utms)
    positives_per_query = knn.radius_neighbors(queries_utms,
                                                    radius=positive_dist_threshold,
                                                    return_distance=False)
        
    with open("rrt_sop_caches/rrt_r50_sop_nn_inds_test.pkl", "wb") as f:
        pickle.dump(positives_per_query, f)
    
    # queries_v1 folder
    query_set = ImageDataset(samples=samples_queries, transform=base_transform)
    # database folder
    gallery_set =  ImageDataset(samples=samples_database, transform=base_transform)

    # query = queriesv1, gallery = database
    return train_set, (query_set, gallery_set)


class MetricLoaders(NamedTuple):
    train: DataLoader
    num_classes: int
    query: DataLoader
    gallery: Optional[DataLoader] = None


@data_ingredient.capture
def get_loaders(batch_size, test_batch_size, 
        num_workers, pin_memory, 
        sampler, recalls,
        num_iterations=None, 
        num_identities=None,
        train_cache_nn_inds=None,
        test_cache_nn_inds=None):

    # TODO Add train_file and test_file in the get_sets() function
    train_set, (query_set, gallery_set) = get_sets()

    if sampler == 'random':
        train_sampler = BatchSampler(RandomSampler(train_set), batch_size=batch_size, drop_last=True)
    elif sampler == 'triplet':
        if train_cache_nn_inds and osp.exists(train_cache_nn_inds):
            train_sampler = TripletSampler(train_set.targets, batch_size, train_cache_nn_inds)
        else:
            # For evaluation only
            train_sampler = None
    elif sampler == 'random_id':
        train_sampler = RandomReplacedIdentitySampler(train_set.targets, batch_size, 
            num_identities=num_identities, num_iterations=num_iterations)
    else:
        raise ValueError('Invalid choice of sampler ({}).'.format(sampler))
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
    query_loader = DataLoader(query_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
    gallery_loader = None
    if gallery_set is not None:
        gallery_loader = DataLoader(gallery_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return MetricLoaders(train=train_loader, query=query_loader, gallery=gallery_loader, num_classes=max(train_set.targets) + 1), recalls


def get__class_id(utm_east, utm_north, heading, M, alpha):
    """Return class_id and group_id for a given point.
        The class_id is a triplet (tuple) of UTM_east, UTM_north and
        heading (e.g. (396520, 4983800,120)).
        The group_id represents the group to which the class belongs
        (e.g. (0, 1, 0)), and it is between (0, 0, 0) and (N, N, L).
    """
    rounded_utm_east = int(utm_east // M * M)  # Rounded to nearest lower multiple of M
    rounded_utm_north = int(utm_north // M * M)
    rounded_heading = int(heading // alpha * alpha)
    
    class_id = (rounded_utm_east, rounded_utm_north, rounded_heading)

    return class_id