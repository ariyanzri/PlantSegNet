from turtle import position
from unittest import result
import matplotlib.pyplot as plt
import torch
import os
import pandas as pd
import random
import numpy as np
from scipy.spatial.distance import cdist
import sys
sys.path.append("..")
from data.load_raw_data import load_pcd_plyfile
import open3d as o3d

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

def calculate_leaf_tip_distances(positions):
    distances = cdist(positions,positions)
    return distances

def get_leaves_with_close_distance(distances,pcd_path,result_path,pcd_name):
    x,y = np.where((distances>0)&(distances<0.1))
    list_leaf_indices = list(zip(x,y))
    unique_list_leaf_indices = []
    for l1,l2 in list_leaf_indices:
        if (l2,l1) not in unique_list_leaf_indices and (l1,l2) not in unique_list_leaf_indices and l1!=0 and l2!=0:
            unique_list_leaf_indices.append((l1,l2))
    
    pcd = load_pcd_plyfile(os.path.join(pcd_path,pcd_name+".ply"))
    leaf_ind = pcd['leaf_index']
    points = pcd['points']
    plant_ind = pcd['is_focal_plant']

    for l1,l2 in unique_list_leaf_indices:
        new_points = points[(leaf_ind==l1)|(leaf_ind==l2)&(plant_ind==1)]
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(new_points)    
        o3d.io.write_point_cloud(os.path.join(result_path,f"test_{l1}_{l2}_{pcd_name}.ply"),new_pcd)

def generate_data(dataset_path):
    pcd_path = os.path.join(dataset_path,"PointCloud")
    csv_path = os.path.join(dataset_path,"CSV")

    pcd_files = os.listdir(pcd_path)
    for pcd_f in pcd_files:
        csv = pd.read_csv(os.path.join(csv_path,pcd_f.replace("ply","csv")))
        x = csv["tip_pos_x"].to_numpy()
        y = csv["tip_pos_y"].to_numpy()
        z = csv["tip_pos_z"].to_numpy()

        positions = np.stack((x,y,z),axis=-1)
        distances = calculate_leaf_tip_distances(positions)
        
        get_leaves_with_close_distance(distances,pcd_path,"/space/ariyanzarei/sorghum_segmentation/results",pcd_f.replace(".ply",""))

    # points, ground, semantic, plant, leaf = ds[4]
    # single,double = generate_data_single_pcd(points,ground,plant,leaf)
    # print(len(single),len(double))

generate_data("/space/ariyanzarei/sorghum_segmentation/dataset/2022-05-06")