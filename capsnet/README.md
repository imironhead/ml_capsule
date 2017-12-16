# Replicate "Dynamic Routing Between Capsules"

[arXiv: Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)

## Requirements

- numpy
- pillow
- scipy
- tensorflow

## Usage

Train a model

- **--mnist-root-path** : path to mnist of fashion mnist datasets. 4 files are expected in the directory (t10k-images-idx3-ubyte.gz, t10k-labels-idx1-ubyte.gz, train-images-idx3-ubyte.gz , train-labels-idx1-ubyte.gz).
- **--batch-size** : batch size of each training iteration.
- **--routing-frequency** : routing frequency (1 or 3 on the paper).
- **--reconstruction-loss** : True for considering reconstruction loss.
- **--ckpt-dir** : path to save checkpoints.
- **--logs-dir** : path to save summaries.

```
python train.py \
--mnist-root-path=/path/to/your/datasets/mnist/ \
--batch-size=128 \
--routing-frequency=3 \
--reconstruction-loss=True \
--ckpt-dir=./ckpt/routing_3/ \
--logs-dir=./ckpt/routing_3/
```

Reconstruct

- **--mnist-root-path** : path to mnist of fashion mnist datasets. 4 files are expected in the directory (t10k-images-idx3-ubyte.gz, t10k-labels-idx1-ubyte.gz, train-images-idx3-ubyte.gz , train-labels-idx1-ubyte.gz).
- **--ckpt-path** : path to a saved checkpoint.
- **--meta-path** : path to a saved meta.
- **--result-path** : path to save the reconstructed result.

```
python reconstruct.py \
--mnist-root-path=/path/to/your/datasets/mnist/ \
--ckpt-path=./ckpt/routing_3/model.ckpt-93801 \
--meta-path=./ckpt/routing_3/model.ckpt-93801.meta \
--result-path=./qooo.png
```

Dimension Perturbations

- **--mnist-root-path** : path to mnist of fashion mnist datasets. 4 files are expected in the directory (t10k-images-idx3-ubyte.gz, t10k-labels-idx1-ubyte.gz, train-images-idx3-ubyte.gz , train-labels-idx1-ubyte.gz).
- **--ckpt-path** : path to a saved checkpoint.
- **--meta-path** : path to a saved meta.
- **--result-path** : path to save the perturbed result.

```
python perturb.py \
--mnist-root-path=/path/to/your/datasets/mnist/ \
--ckpt-path=./ckpt/routing_3/model.ckpt-93801 \
--meta-path=./ckpt/routing_3/model.ckpt-93801.meta \
--result-path=./zooo.png
```

## MNIST

The best testing accuracy is 0.9974 (3 routing with reconstruction loss) with exponentially decayed learning rate.

``` python
learning_rate = 0.001 * (0.95 ** epoch)
```

### Dataset

[The MNIST Database](http://yann.lecun.com/exdb/mnist/)

### Accuracy

![](/assets/capsnet_accuracies_mnist.png)Accuracies on testing set. The relative time of each run is different due to they were trained on 2 machines with different GPUs (GTX 1070/GTX 1080).

![](/assets/capsnet_accuracies_mnist_routing_3_v1.png)Accuracies of routing frequency 3. The red line split the trials into 2 groups w/ or w/o reconstruction.

![](/assets/capsnet_accuracies_mnist_routing_1_v0.png)Accuracies of routing frequency 1. Note that the run name **routing_1_True_1** is the lowest even with reconstruction loss.

### Loss

![](/assets/capsnet_loss_mnist.png)The gap is from reconstruction loss.

Lower loss is possible (with reconstruction) by tuning learning rate. However, the accuracy would also be lower (overfit).

### Reconstruction

![](/assets/capsnet_reconstruct_mnist.png)

- Top 2 rows: reconstructed from correctly predicted images.
- Bottom 2 rows: reconstructed from incorrectly predicted images.

### Dimension Perturbations

![](/assets/capsnet_dimension_perturbations_mnist.png)

* 10 randomly picked digits from 10 classes.
* the middle column of each digits are reconstructed images of testing set (10000 images).
* each digit has 16 rows for 16D digit capsules.
* each digit has 10 columns (-0.25 ~ 0.0 ~ +0.25, as the descriptions on the paper)

## Fashion-MNIST

### Dataset

Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. Han Xiao, Kashif Rasul, Roland Vollgraf. [arXiv:1708.07747](http://arxiv.org/abs/1708.07747)

[GitHub](https://github.com/zalandoresearch/fashion-mnist)

### Accuracy w/o Tuning

![](/assets/capsnet_accuracies_fashion_routing_3.png)

### Loss (Include Reconstruction Loos)

![](/assets/capsnet_loss_fashion.png)

### Reconstruction

![](/assets/capsnet_reconstruct_fashion.png)

- Top 2 rows: reconstructed from correctly predicted images.
- Bottom 2 rows: reconstructed from incorrectly predicted images.

### Dimension Perturbations

![](/assets/capsnet_dimension_perturbations_fashion.png)

* 10 randomly picked digits from 10 classes.
* the middle column of each digits are reconstructed images of testing set (10000 images).
* each digit has 16 rows for 16D digit capsules.
* each digit has 10 columns (-0.25 ~ 0.0 ~ +0.25, as the descriptions on the paper)
