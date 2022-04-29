import torch.nn as nn
import torch.optim.lr_scheduler as lr_sched
import torch

def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn

class BNMomentumScheduler(lr_sched.LambdaLR):
    def __init__(self, model, bn_lambda, last_epoch=-1, setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(type(model)._name_)
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def state_dict(self):
        return dict(last_epoch=self.last_epoch)

    def load_state_dict(self, state):
        self.last_epoch = state["last_epoch"]
        self.step(self.last_epoch)

class SpaceSimilarityLossV2(nn.Module):

    def __init__(self,M1=1,M2=10):
        super().__init__()
        self.M1 = M1
        self.M2 = M2
    
    def forward(self,input,target):
        
        distance_pred = torch.cdist(input,input)
        
        target = torch.unsqueeze(target.float(),dim=-1)
        distances_gt = torch.cdist(target,target)
        
        normalized_distance_gt = torch.where(distances_gt==0,torch.clamp(distance_pred-self.M1,min=0),torch.clamp(self.M2-distance_pred,min=0))
        
        return torch.mean(normalized_distance_gt)

class LeafMetrics(nn.Module):

    def __init__(self,dist):
        super().__init__()
        self.threshold = dist
        
    def forward(self,input,target):
        
        cluster_pred = torch.cdist(input,input)

        target = torch.unsqueeze(target.float(),dim=-1)
        cluster_gt = torch.cdist(target,target)
        ones = torch.ones(cluster_pred.shape).cuda()
        zeros = torch.zeros(cluster_pred.shape).cuda()
        
        
        TP = torch.sum(torch.where((cluster_gt == 0) & (cluster_pred < 5), ones, zeros))
        TN = torch.sum(torch.where((cluster_gt > 0) & (cluster_pred >= 5), ones, zeros))
        FP = torch.sum(torch.where((cluster_gt > 0) & (cluster_pred < 5), ones, zeros))
        FN = torch.sum(torch.where((cluster_gt == 0) & (cluster_pred >= 5), ones, zeros))
        
        Acc = (TP+TN)/(TP+FP+TN+FN)
        Precision = TP/(TP+FP)
        Recall = TP/(TP+FN) 
        F = 2*(Precision*Recall)/(Precision+Recall)

        return Acc, Precision, Recall, F
