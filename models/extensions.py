from torch.autograd import Function
import numpy as np
import torch

def multi_random_choice(samples, s_size, max):

  out = np.zeros((samples, s_size))

  for i in range(samples):
    out[i,:] = np.random.choice(max, s_size, replace=False)

  return out

"""
# TODO Implement the following pseudo-code

# We want downsample a tensor from size (B, N, D) to (B, S, D) where S < N but pass the gradients 
# to all of the original vectors.

class Cluster(Function):

    @staticmethod
    def forward(ctx, input):
        # (B, N, 256 )

        # return (B, N, 1)
    
    def backwards(ctx, grad_output):
        # input (B, N, 1)
        # Todo do we duplicate the grad value or split?
        # return (B, N, 256)



class LeafBranch(Function): 

    @staticmethod
    def forward(ctx):
        
        # todo save cluster assignments
        cluster_assignments = cluster(feature_vector)

        clusters = unique(cluster_assignments)

        # we know the number of clusters
        # result_tensor (B, max, clusters)
        for batch in batches:
            for cluster in clusters:

                cluster_points_ds = downsample(cluster_points)

                pred_leaf = leaf_model(cluster_points_ds)
                # assign to result_tensor
            
        return leaf_score_vector # (B, N, 1) or (B, 1)

    @staticmethod
    def backward(ctx, grad_output):
        # input size (B, 1) 

        # B, N, 256

"""


class Downsample(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, result_size):

        rng = np.random.default_rng()

        ds_dim_size = input.shape[1]
        batch_size = input.shape[0]

        ds_idx = multi_random_choice(batch_size, result_size, ds_dim_size) 

        ds_idx = np.expand_dims(ds_idx, axis=2)
        gather_idx = torch.tensor(np.repeat(ds_idx, 3, axis=2), dtype=torch.int64)

        ctx.save_for_backward(input, gather_idx)

        output = torch.gather(input, 1, gather_idx)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):

        input, gather_idx = ctx.saved_tensors

        grad_input = torch.zeros(input.shape)
        """
        example gather_idx:
        
        tensor([[[0, 0, 0],
         [3, 3, 3]],

        [[1, 1, 1],
         [3, 3, 3]]])

        example grad_output:
         tensor([[[1., 1., 1.],
         [1., 1., 1.]],

        [[1., 1., 1.],
         [1., 1., 1.]]])

        Assuming input N = 4 output should be
         tensor([[[1., 1., 1.],
         [0, 0, 0],
         [1., 1., 1.]
         [0, 0, 0]
         ],

        [[0, 0, 0],
         [1., 1., 1.],
         [0, 0, 0]
         [1., 1., 1.]]])
        """

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.

        # we need to upsample grad_output, gradient will be zero for points which where
        # not included in the downsample. 

        grad_input = grad_input.scatter(1, gather_idx, grad_output)

        return grad_input, None