import h5py
import os
import numpy as np
import random
import multiprocessing
from sklearn.model_selection import train_test_split
from load_raw_data import *

def generate_h5py(path_raw_data,path_h5py,index_to_use,point_no=8000,split_ratio={'tr':0.6,'va':0.2,'te':0.2}):
    assert sum([split_ratio[k] for k in split_ratio.keys()]) == 1

    files = os.listdir(path_raw_data)

    dataset = {
        'points':[],
        'is_focal_plant':[],
        'leaf_index':[],
        'leaf_part_index':[],
        'leaf_part_full_index': [], 
        'plant_index':[],
        'ground_index':[],
        'names': []
        }

    for i,f in enumerate(files):
        pcd = load_pcd_plyfile(os.path.join(path_raw_data,f),down_sample_n=point_no)
        if pcd is None:
            print(f">>> Error occured. Ignoring {f}.")
            continue
        
        dataset['points'].append(pcd['points'])
        dataset['is_focal_plant'].append(pcd['is_focal_plant'])
        dataset['leaf_index'].append(pcd['leaf_index'])
        dataset['leaf_part_index'].append(pcd['leaf_part_index'])
        dataset['leaf_part_full_index'].append(pcd['leaf_part_full_index'])
        dataset['plant_index'].append(pcd['plant_index'])
        dataset['ground_index'].append(pcd['ground_index'])
        dataset['names'].append(f)
        print(f":: {i}/{len(files)}",end='\r')

    
    dataset['points'] = np.stack(dataset['points'],0)
    dataset['points'] = np.array(dataset['points'])
    dataset['is_focal_plant'] = np.array(dataset['is_focal_plant'])
    dataset['leaf_index'] = np.array(dataset['leaf_index'])
    dataset['leaf_part_index'] = np.array(dataset['leaf_part_index'])
    dataset['leaf_part_full_index'] = np.array(dataset['leaf_part_full_index'])
    dataset['plant_index'] = np.array(dataset['plant_index'])
    dataset['ground_index'] = np.array(dataset['ground_index'])

    # print(dataset['points'])
    train_dataset = {k:[] for k in dataset.keys()}
    validation_dataset = {k:[] for k in dataset.keys()}
    test_dataset = {k:[] for k in dataset.keys()}

    train_dataset['points'], test_dataset['points'] ,train_dataset['leaf_index'], test_dataset['leaf_index'], \
    train_dataset['leaf_part_index'], test_dataset['leaf_part_index'], train_dataset['leaf_part_full_index'], test_dataset['leaf_part_full_index'], \
    train_dataset['plant_index'], test_dataset['plant_index'], train_dataset['ground_index'], test_dataset['ground_index'], \
    train_dataset['is_focal_plant'], test_dataset['is_focal_plant'], train_dataset['names'], test_dataset['names'], \
        = train_test_split(
            dataset['points'], 
            dataset['leaf_index'], 
            dataset['leaf_part_index'], 
            dataset['leaf_part_full_index'],
            dataset['plant_index'],
            dataset['ground_index'],
            dataset['is_focal_plant'],
            dataset['names'], 
            test_size=split_ratio['te'], random_state=42)
    
    train_dataset['points'], validation_dataset['points'] ,train_dataset['leaf_index'], validation_dataset['leaf_index'], \
    train_dataset['leaf_part_index'], validation_dataset['leaf_part_index'], train_dataset['leaf_part_full_index'], validation_dataset['leaf_part_full_index'], \
    train_dataset['plant_index'], validation_dataset['plant_index'], train_dataset['ground_index'], validation_dataset['ground_index'], \
    train_dataset['is_focal_plant'], validation_dataset['is_focal_plant'], train_dataset['names'], validation_dataset['names'], \
        = train_test_split(
            train_dataset['points'], 
            train_dataset['leaf_index'], 
            train_dataset['leaf_part_index'], 
            train_dataset['leaf_part_full_index'],
            train_dataset['plant_index'],
            train_dataset['ground_index'],
            train_dataset['is_focal_plant'],
            train_dataset['names'], 
            test_size=split_ratio['va']/(1-split_ratio['te']), random_state=42)

    for k in train_dataset.keys():
        if k == 'names':
            continue
        train_dataset[k] = np.array(train_dataset[k])

    for k in validation_dataset.keys():
        if k == 'names':
            continue
        validation_dataset[k] = np.array(validation_dataset[k])
        
    for k in test_dataset.keys():
        if k == 'names':
            continue
        test_dataset[k] = np.array(test_dataset[k])
        
    print(train_dataset['points'].shape)
    print(validation_dataset['points'].shape)
    print(test_dataset['points'].shape)

    with h5py.File(os.path.join(path_h5py,'sorghum__labeled_train.hdf5'),'w') as f:
        for k in train_dataset.keys():
            f.create_dataset(k,data=train_dataset[k])
            
    with h5py.File(os.path.join(path_h5py,'sorghum__labeled_validation.hdf5'),'w') as f:
        for k in validation_dataset.keys():
            f.create_dataset(k,data=validation_dataset[k])

    with h5py.File(os.path.join(path_h5py,'sorghum__labeled_test.hdf5'),'w') as f:
        for k in test_dataset.keys():
            f.create_dataset(k,data=test_dataset[k])

# generate_h5py("/space/ariyanzarei/sorghum_segmentation/dataset/2022-03-10/PointCloud",\
#    "/space/ariyanzarei/sorghum_segmentation/dataset/2022-03-10","leaf_index")
# generate_h5py("/space/ariyanzarei/sorghum_segmentation/PointCloud","/space/ariyanzarei/sorghum_segmentation/SorghumData","leaf_part_index")