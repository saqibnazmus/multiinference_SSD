# Title: Multi-stage Inference Strategy for Self-supervised Denoising Problem - _PyTorch implementation_

**Master Thesis**

## Abstract:
Self-supervised image denoising is a challenging problem that aims at signal reconstruction on a 
sparse set of noise measurements without any supervision of clean ground truths. Conventional supervised methods
consider the noise recovery process as an ill-posed optimization problem with the availability of ground truth 
which is challenging in numerous domains. Self-supervised techniques alleviate the ground truth-unavailability issue 
by incorporating several complicated objective functions for proper noise removal and reconstruction. 
However, the diverse noise distribution of images is crucial for noise recovery. Moreover, to form a complex loss function,
the methods need to rely on additional hyperparameters. However,  optimal hyperparameter estimation is complicated, 
and any mistuning of the parameters results in over-smoothing and inconsistent structure recovery that is responsible for performance degradation.
This paper proposes a self-regularization technique without using any hyperparameter to alleviate the aforementioned issues.
Our multiple predictions acquired from a multi-inference self-supervised strategy are exploited as the regularization parameters 
and produce a compact loss function. Moreover, the proposed self-regularized method achieves satisfactory performance using multiple predictions 
and follows a simple training strategy without any complexity. Our experimental results represent that our compact loss function can achieve satisfactory performances
in comparison to other existing methods for both synthetic and real noise domains. We also implement our algorithm on practical applications to represent how such low-level
vision task is effective in high-level vision applications. We represent a comparison scenario with weakly and un-supervised denoising methods to highlight our improved performance in the above applications. 

## Implementations

The below sections details what python requirements are required to set up for the project. 
dataset.

### Dependencies
- PyTorch, NumPy, OpenCV

### Dataset
We have worked on ImageNet dataset for the training procedure. For the testing procedure on synthetic noisy images, we have used BSD68, KODAK24 and Set14 datasets. For real noisy images, we have used the SIDD benchmark, CC and PolyU datasets.


##Dataloader
For loading the data, run the following: 
```
python dataloader.py 
```

### Train
We have provided the training code only for the synthetic noisy images. For real noisy images, we will provide the code later.  
```
python train_synthetic.py 
```

### Test
To get the results of the testing procedure, write the following on your command prompt and run. 

```
python test_synthetic.py"
```
