## Testing SimCLR for Class Incremental (CI) Scenario

1. To run SimSiam:
```python3 main.py -cs [10] --cuda_device 0```
2. To run post processing for t-SNE plots or correlation:
```python3 post_processing.py --pretrained_dir './checkpoints/checkpoint_0.030000_cs_[5, 5]_bs_512.pth.tar' -cs 5,5``` 

## Command to Run Continual SSL:
```$ python3 main_cont.py -cs 5,5 -e 500,500 --cuda_device 0```

```cs``` corresponds to class split, ```e``` is for the number of epochs for the corresponding task (length of cs and e should be the same)


## Experiments Discused in previous meeting:
1. Self Supervised Experiment with all classes (no CI)
2. CI:
    a) Easy CI (5 v 5)
    b) Medium-complexity (5 v 3 v 2)
    d) Hard (one class based increments)

3. May be (include one type of class in one batch and compare to 1)
4. We need to tune Linear Classifier, it seems like not performing optimal
