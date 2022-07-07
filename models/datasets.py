import torch
import numpy as np
import torch.utils.data as data
import h5py

class SorghumDataset(data.Dataset):
    ''''
    Semantic label guide:
        * 0 --> ground
        * 1 --> focal plant
        * 2 --> surrounding plants

    Ground label guide:
        * 0 --> not ground
        * 1 --> ground

    '''
    def __init__(self, h5_filename):
        super().__init__()
        self.h5_filename=h5_filename
        self.length = -1

    def __getitem__(self, index):
        f = h5py.File(self.h5_filename,'r')
        points = f['points'][index]
        is_focal_plant = f['is_focal_plant'][index]
        ground_index = f['ground_index'][index]
        plant_index = f['plant_index'][index]
        leaf_index = f['leaf_index'][index]
        
        # Converting arbitrary and non-contigiouse plant IDs to contigiouse list of indices
        plant_ind = list(set(list(plant_index)))
        ind = list(range(0,len(plant_ind)))
        mapping = dict(zip(ind,plant_ind))
        new_plant = np.zeros(plant_index.shape)
        for key in mapping:
            new_plant[plant_index==mapping[key]] = key
        plant_index = new_plant

        # creating semantic labeling using the guide above
        semantic_label = is_focal_plant.copy()
        semantic_label[np.where((is_focal_plant==0) & (ground_index==1))] = 0
        semantic_label[np.where((is_focal_plant==0) & (ground_index==0))] = 2

        f.close()

        return torch.from_numpy(points).float(),torch.from_numpy(ground_index).float(),\
            torch.from_numpy(semantic_label).type(torch.LongTensor),\
            torch.from_numpy(plant_index).type(torch.LongTensor),\
            torch.from_numpy(leaf_index).type(torch.LongTensor),\
            

    def __len__(self):
        if self.length != -1:
            return self.length
        else:
            f = h5py.File(self.h5_filename,'r')
            self.length=len(f['names'])
            f.close()
            return self.length

    def get_name(self,index):
        f = h5py.File(self.h5_filename,'r')
        name = f['names'][index].decode("utf-8")
        return name


class LeafDataset(data.Dataset):
    ''''
    Leaf label guide:
        * 0 --> single leaf
        * 1 --> double leaf
    '''
    def __init__(self, h5_filename):
        super().__init__()
        self.h5_filename=h5_filename
        self.length = -1

    def __getitem__(self, index):
        f = h5py.File(self.h5_filename,'r')
        points = f['points'][index]

        # Note leaf counts are expected to be either 1 or 2
        is_single_leaf = 2 - f['leaf_count'][index]
        
        f.close()

        return (
            torch.from_numpy(points).float(),
            torch.from_numpy(np.array(is_single_leaf)).type(torch.LongTensor)
        )
            

    def __len__(self):
        if self.length != -1:
            return self.length
        else:
            f = h5py.File(self.h5_filename,'r')
            self.length=len(f['points'])
            f.close()
            return self.length

    def get_name(self,index):
        f = h5py.File(self.h5_filename,'r')
        name = f['names'][index].decode("utf-8")
        return name