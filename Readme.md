## Testing SimCLR for Class Incremental (CI) Scenario

1. Added a basic code of simsiam from BYOL repository for resnet50 model + downstream task code.
2. Go to CL_SSL_BYOL ad run the followig command
    ```python3 simsiam_cifar10.py```
2. TODO: @Yavuz Dataloader part
3. TODO: @erum Matching the optimizers, and hyper-parameters (lr, batch sizes) from the original paper



## Experiments Discused in previous meeting:
1. Self Supervised Experiment with all classes (no CI)
2. CI:
    a) Easy CI (5 v 5)
    b) Medium-complexity (5 v 3 v 2)
    d) Hard (one class based increments)

3. May be (include one type of class in one batch and compare to 1)