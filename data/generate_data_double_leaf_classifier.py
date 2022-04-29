from lib2to3.pytree import LeafPattern
import sys
sys.path.append("/work/ariyanzarei/SorghumSegmentation/TreePartNet/utils")
sys.path.append("/work/ariyanzarei/SorghumSegmentation/TreePartNet/pointnet2_ops_lib")
from TreeDataset import SorghumDataset
from load_raw_data import *
import matplotlib.pyplot as plt
import torch

'''
algorithm:

for each point cloud in dataset:
    remove ground 
    for each plant in the point cloud:
        for each pair of leaves in the plant point cloud: (or randomly sample pairs of leaves)
            find min and mix (boundary) of the pcd
            randomly select a subspace from the boundary
            pick full space too
            this gives example to the double leaf class (Not OK)

        for each leaf in the plant point cloud: (or sample randomly)
            find boundary
            randomly sample a subspace from the boundary
            pick full space too
            this gives example to the single leaf class (OK)

visualize and sanity check
train on these point clouds using a PointNet++ module

'''

def generate_data_single_pcd(points, ground, plant, leaf, leaf_pair_sample_no = 10, space_sample_no = 10):

    single_leaf_data_points = []
    double_leaf_data_points = []

    points = points[ground==0]
    leaf = leaf[ground==0]
    plant = plant[ground==0]

    min_plant_index = torch.min(plant)
    max_plant_index = torch.max(plant)
    min_leaf_index = torch.min(leaf)
    max_leaf_index = torch.max(leaf)

    for pl in range(min_plant_index,max_plant_index+1):
        for i in range(leaf_pair_sample_no):
            leaves = random.sample(range(min_leaf_index,max_leaf_index+1),2)
            leaf_1 = leaves[0]
            leaf_2 = leaves[1]
                
            criteria = ((leaf==leaf_1) | (leaf==leaf_2))
            points_leaf = points[(criteria) & (plant == pl),:]
            
            if points_leaf.shape[0] < 50:
                continue

            mins_leafs,_ = torch.min(points_leaf,axis=0)
            maxs_leafs,_ = torch.max(points_leaf,axis=0)
            
            for j in range(space_sample_no):
                sampled_min_x = random.uniform(mins_leafs[0],mins_leafs[0]+(maxs_leafs[0]-mins_leafs[0])/6)
                sampled_max_x = random.uniform(maxs_leafs[0]-(maxs_leafs[0]-mins_leafs[0])/6,maxs_leafs[0])

                sampled_min_y = random.uniform(mins_leafs[1],mins_leafs[1]+(maxs_leafs[1]-mins_leafs[1])/6)
                sampled_max_y = random.uniform(maxs_leafs[1]-(maxs_leafs[1]-mins_leafs[1])/6,maxs_leafs[1])

                sampled_min_z = random.uniform(mins_leafs[2],mins_leafs[2]+(maxs_leafs[2]-mins_leafs[2])/6)
                sampled_max_z = random.uniform(maxs_leafs[2]-(maxs_leafs[2]-mins_leafs[2])/6,maxs_leafs[2])

                points_subleaf = points_leaf[(points_leaf[:,0]>=sampled_min_x)&(points_leaf[:,0]<=sampled_max_x)&
                                            (points_leaf[:,1]>=sampled_min_y)&(points_leaf[:,1]<=sampled_max_y)&
                                            (points_leaf[:,2]>=sampled_min_z)&(points_leaf[:,2]<=sampled_max_z),:]

                double_leaf_data_points.append(points_subleaf.cpu().numpy())
        
        for lf in range(min_leaf_index+1,max_leaf_index+1):
            criteria = (leaf==lf)
            points_leaf = points[(criteria) & (plant == pl),:]
            single_leaf_data_points.append(points_leaf.cpu().numpy())
    
    return single_leaf_data_points,double_leaf_data_points
            
def generate_data(dataset_path):
    ds = SorghumDataset(dataset_path)
    points, ground, semantic, plant, leaf = ds[4]
    single,double = generate_data_single_pcd(points,ground,plant,leaf)
    print(len(single),len(double))

generate_data("/space/ariyanzarei/sorghum_segmentation/dataset/2022-03-10/sorghum__labeled_train.hdf5")