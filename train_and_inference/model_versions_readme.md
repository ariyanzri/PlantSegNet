# Model Versions Information

Here we summerize the details of the different trained models and their different versions.

- **SorghumPartNetDGCNN**: Basic model trained on the initial syntehtic data.
  - **version_0**: NOT IMPORTANT.
- **SorghumPartNetGroundDGCNN**: Model trained on data with ground labeled separately.
  - **version_0**: NOT IMPORTANT.
  - **version_1**: NOT IMPORTANT.
  - **version_2**: NOT IMPORTANT.
- **SorghumPartNetInstance**: The main class of models for instance segmentation. Using SGPN network loss function.
  - **version_0**: Trained on `2022-03-10` dataset (about 12000 point clouds). This dataset had a couple of issues including incorrect model of the ground, inaccurate model of the plants, incorrect scale and ratio of the points and so on. These are fixed in the next version.
- **SorghumPartNetSemantic**: The main class of models for semantic segmentation. We use DGCNN as backend to do the semantic segmentation.
  - **version_0**: Trained on `2022-03-10` dataset (about 12000 point clouds). This dataset had a couple of issues including incorrect model of the ground, inaccurate model of the plants, incorrect scale and ratio of the points and so on. These are fixed in the next version.
- **SorghumPartNetInstanceWithLeafBranch**: Instance segmentation with the leaf classification branch to address the multiple leaves issue. For now, this is not the main focus.
  - **version_0**: NOT IMPORTANT.
  - **version_1**: NOT IMPORTANT.
