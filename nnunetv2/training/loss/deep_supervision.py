import torch
from torch import nn

class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        Wraps a loss function so that it can be applied to multiple outputs. Forward accepts an arbitrary number of
        inputs. Each input is expected to be a tuple/list. Each tuple/list must have the same length. The loss is then
        applied to each entry like this:
        l = w0 * loss(input0[0], input1[0], ...) +  w1 * loss(input0[1], input1[1], ...) + ...
        If weights are None, all w will be 1.
        """
        super(DeepSupervisionWrapper, self).__init__()
        assert any([x != 0 for x in weight_factors]), "At least one weight factor should be != 0.0"
        self.weight_factors = tuple(weight_factors)
        self.loss = loss

    def forward(self, *args):
        assert all([isinstance(i, (tuple, list)) for i in args]), \
            f"all args must be either tuple or list, got {[type(i) for i in args]}"
        # we could check for equal lengths here as well, but we really shouldn't overdo it with checks because
        # this code is executed a lot of times!

        if self.weight_factors is None:
            weights = (1, ) * len(args[0])
        else:
            weights = self.weight_factors

        return sum([weights[i] * self.loss(*inputs) for i, inputs in enumerate(zip(*args)) if weights[i] != 0.0])


class DeepSupervisionWrapperV2(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        Wraps a loss function so that it can be applied to multiple outputs. Forward accepts an arbitrary number of
        inputs. Each input is expected to be a tuple/list. Each tuple/list must have the same length. The loss is then
        applied to each entry like this:
        l = w0 * loss(input0[0], input1[0], ...) +  w1 * loss(input0[1], input1[1], ...) + ...
        If weights are None, all w will be 1.
        
        Since the bottleneck layer is not used for computing the loss values, why not use the segmentation output
        to check if it does classification well. Multi-label classification to be exact. 
        
        I will use torch.nn.MultiLabelSoftMarginLoss. This requires target as one-hot encoded.
        
        """
        super(DeepSupervisionWrapperV2, self).__init__()
        assert any([x != 0 for x in weight_factors]), "At least one weight factor should be != 0.0"
        self.weight_factors = tuple(weight_factors)
        self.loss = loss
        
    def multilabel_classification_loss(self, mlc_output, targets):
    
        # There is probably an easier way to do this.
        # We need to convert target array to target labels, as mentioned here: 
        # https://discuss.pytorch.org/t/what-kind-of-loss-is-better-to-use-in-multilabel-classification/32203/4
        # For that, first we need to get the classes from the target array with the highest resolution.
        target_batch_classes_onehot = torch.zeros(mlc_output.shape, device=mlc_output.device, dtype=torch.float32)
        
        # Index 0 has the highest resolution
        for idx, target_item in enumerate(targets[0]):
            target_item_classes = torch.unique(target_item).unsqueeze(0).to(torch.int64) # The output of torch.unique is int16 and scatter needs int64.
            target_item_classes_onehot = torch.zeros((1, mlc_output.shape[-1]), device=target_item_classes.device)
            target_item_classes_onehot.scatter_(1, target_item_classes, 1)
            target_batch_classes_onehot[idx] = target_item_classes_onehot
        
        loss_fn = nn.MultiLabelSoftMarginLoss()
        return loss_fn(mlc_output, target_batch_classes_onehot)
        
    def forward(self, seg_outputs, targets, mlc_output=None):
        args = [seg_outputs, targets]
        assert all([isinstance(i, (tuple, list)) for i in args]), \
            f"all args must be either tuple or list, got {[type(i) for i in args]}"
        # we could check for equal lengths here as well, but we really shouldn't overdo it with checks because
        # this code is executed a lot of times!

        if self.weight_factors is None:
            weights = (1, ) * len(args[0])
        else:
            weights = self.weight_factors

        segmentationLoss = sum([weights[i] * self.loss(*inputs) for i, inputs in enumerate(zip(*args)) if weights[i] != 0.0])
        
        if mlc_output is not None:
            classificationLoss = self.multilabel_classification_loss(mlc_output, targets)
        else:
            classificationLoss = 0.0
        
        return segmentationLoss + classificationLoss
