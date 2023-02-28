from torch import nn

class LinearClassifier(nn.Module):
    """
    Linear Evaluation with a linear classifier defined as 1-layer:
        input size: feature dimension (Ex: 512 for ResNet18 )
        output size: number of class (Ex: 10 for CIFAR-10)
    Args:
        model_name (string): Backbone model name. Default as 'resnet-18'.
        dataset (string): Dataset used for linear evaluation. Default as 'cifar-10'
        x (torch.tensor): input with size (batchsize)x(features_dim)
    Returns:
        x (torch.tensor): logits with size (batchsize)x(num_classes). 
    """
    def __init__(self,features_dim=512,num_classes = 10):
        super().__init__()
        self.features_dim = features_dim
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.features_dim, self.num_classes)
    
    def forward(self, x):
        x = self.classifier(x)
        return x