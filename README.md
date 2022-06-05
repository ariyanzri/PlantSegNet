# SorghumPartNet

## Development

### Environment

This project depends on
[`TreePartNet`](https://github.com/marktube/TreePartNet) which should be
accessable in the `PYTHONPATH` when working with the `data` directory of
`SorghumPartNet`.

Note: A suitable version of CUDA-toolkit is needed on the host to build the CUDA
extensions for the TreePartNet repository when it is being installed. However,
only the dataset generation scripts depend TreePartNet currently, so it should
be possible to run training and inference on existing preprocessed datasets
without worrying about this dependency.

It is recommended to use a conda environment with python 3.7 or 3.9 for
development and usage.

## Usage

Currently, the directory for the inputs, and model checkpoints are hard-coded.
This means that in each of the three scripts described below it will be
necessary to modify the filepaths inline before attempting to run locally.

### How to run training

There are two training scripts; one for whole plant instance segmentation, and
one for leaf semantic segmentation.

``` 
python3 ./train_and_inference/train_instance_segmentor.py 
python3 ./train_and_inference/train_semantic_segmentor.py 
```

Once the input and model checkpoint paths have been updated to be valid on your
system it will be possible to run each command to train the respective models.

The inputs directory should contain three dataset files intended to be used for
training.

The output will be in the form of model checkpoints in the specified model
directory. A path to a particular checkpoint can be specified in the inference
script.

### How to run inference

Similar to the training scripts, to run inference locally it will be necessary
to modify the file paths which ar currently hard-coded in the file
`train_and_inference/predict_and_visualize.py`.

With the paths correctly specified, it will be possible to execute the command.

``` 
python3 ./train_and_inference/predict_and_visualize.py -i=23 
```

Where the `-i` flag will specify to run inference on the file `23.ply` in your
inputs directory.

The this will execute and produce 4 output files 

``` 
23_instance_full.ply 23_instance.ply 23_semantic_full.ply 23_semantic.ply
```

Each result ".ply" files contains a 2d array where each row represents
`(x,y,z,r,g,b)` where points segmented together have the same `r,g,b` value. 






