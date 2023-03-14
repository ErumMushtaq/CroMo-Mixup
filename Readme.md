## Testing SimSiam for Class Incremental (CI) Scenario

1. To run SimSiam:
```python3 main.py -cs [10] --cuda_device 0 --algo simsiam```
2. To run Infomax:
```python3 main.py -cs [10] --cuda_device 0 --algo infomax```
3. To run post processing for t-SNE plots or correlation:
```python3 post_processing.py --pretrained_dir './checkpoints/checkpoint_0.030000_cs_[5, 5]_bs_512.pth.tar' -cs 5,5``` 

## Command to Run Continual SSL:
```$ python3 main_cont.py -cs 5,5 -e 500,500 --cuda_device 0```

```cs``` corresponds to class split, ```e``` is for the number of epochs for the corresponding task (length of cs and e should be the same)


