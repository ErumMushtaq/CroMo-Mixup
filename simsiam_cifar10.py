import torch
from byol_pytorch import BYOL
from torchvision import models
import sys
import os
import torch.nn as nn


# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

# from BYOL.byol_pytorch import BYOL

resnet = models.resnet50(pretrained=True)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

#Turning off omentum encoder makes BYOL SimSiam
learner = BYOL(
    resnet,
    image_size = 256,
    hidden_layer = 'avgpool',
    use_momentum = False       # turn off momentum in the target encoder
)

optimizer = torch.optim.Adam(learner.parameters(), lr=3e-4)

#TODO: @Yavuz: make cifar10 Class incremetal dataloader
def sample_unlabelled_images():
    return torch.randn(20, 3, 256, 256)


# Training loop
epoch = 1
for _ in range(epoch):
    # print('running')
    images = sample_unlabelled_images()
    loss = learner(images)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# save your encoder network
save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'resnet18',
                'state_dict': resnet.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))


# Downstream Task
model = models.resnet50(pretrained=True)
path_file = 'checkpoint_{:04d}.pth.tar'.format(epoch)
checkpoint = torch.load(path_file, map_location="cpu")

# rename moco pre-trained keys
state_dict = checkpoint['state_dict']
for k in list(state_dict.keys()):
    # retain only encoder up to before the embedding layer
    if k == 'fc.weight' or k == 'fc.bias':
        del state_dict[k]
    
print(state_dict.keys())
msg = model.load_state_dict(state_dict, strict=False)
print(msg.missing_keys)
assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

# freeze all layers but the last fc
for name, param in model.named_parameters():
    if name not in ['fc.weight', 'fc.bias']:
        param.requires_grad = False

 # init the fc layer
model.fc.weight.data.normal_(mean=0.0, std=0.01)
model.fc.bias.data.zero_()


# optimize only the linear classifier
parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
assert len(parameters) == 2  # fc.weight, fc.bias

optimizer = torch.optim.SGD(parameters, 3e-4,
                            momentum=0.9,
                            weight_decay=0.001)
criterion = nn.CrossEntropyLoss()

# To be tested once we have dataloaders
# for _ in range(epoch):
#     print('running')
#     images = sample_unlabelled_images() #it will be labelled data and will have target class as well.
#     outputs = model(images)
#     loss = criterion(output, target)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()


