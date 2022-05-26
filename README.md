# Knowledge Graph Completion

## Prerequisite

* Python 3.6

* Pytorch 1.7

* [Geoopt](https://github.com/geoopt/geoopt)

* tqdm

## Experiments

### 1. Init repo

To uncompress the data and create the output directories, run the following command

```
sh init.sh
```

The dataset is included in the source code from this [link](https://github.com/fedelopez77/gyrospd), 
which is taken from this [link](https://github.com/villmow/datasets_knowledge_embedding).  

### 2. Data Preparation

To preprocess the dataset, run the following command

```
python preprocess.py
```

### 3. Training & Testing

```
python -m torch.distributed.launch --nproc_per_node=N_CPUS --master_port=2055 train.py \\
            --n_procs=N_CPUS \\
            --data=PREP \\
            --run_id=RUN_ID \\
            --results_file=out/results.csv \\
            --model=MODEL \\
            --metric=riem \\
            --dims=10 \\
            --pdim=10 \\
            --kdim=10 \\
            --learning_rate=1e-3 \\
            --val_every=5 \\
            --patience=500 \\
            --batch_size=4096 \\
            --epochs=5000 \\
            --train_bias
```

To train and test the GyroLE-KGCNet model, run the following command

```
python -m torch.distributed.launch --nproc_per_node=N_CPUS --master_port=2055 train.py \\
            --n_procs=N_CPUS \\
            --data=WN18RR \\
            --run_id=test \\
            --results_file=out/results.csv \\
            --model=tgspdle \\
            --metric=riem \\
            --dims=6 \\
            --learning_rate=1e-3 \\
            --val_every=5 \\
            --patience=500 \\
            --batch_size=4096 \\
            --epochs=5000 \\
            --train_bias
```

To train and test the GyroGR-KGCNet model, run the following command

```
python -m torch.distributed.launch --nproc_per_node=N_CPUS --master_port=2055 train.py \\
            --n_procs=N_CPUS \\
            --data=WN18RR \\
            --run_id=test \\
            --results_file=out/results.csv \\
            --model=tggr \\
            --metric=riem \\
            --dims=5 \\
            --pdim=2 \\
            --learning_rate=1e-3 \\
            --val_every=5 \\
            --patience=500 \\
            --batch_size=4096 \\
            --epochs=5000 \\
            --train_bias
```

To train and test the GyroGRLE-KGCNet model, run the following command

```
python -m torch.distributed.launch --nproc_per_node=N_CPUS --master_port=2055 train.py \\
            --n_procs=N_CPUS \\
            --data=WN18RR \\
            --run_id=test \\
            --results_file=out/results.csv \\
            --model=tgspdlegr \\
            --metric=riem \\
            --dims=5 \\
            --pdim=5 \\
            --kdim=2 \\
            --learning_rate=1e-3 \\
            --val_every=5 \\
            --patience=500 \\
            --batch_size=4096 \\
            --epochs=5000 \\
            --train_bias
```


### 4. Models and Metrics

The available options for the parameter `--model` are:

* `tgspdle`: Applies a scaling on the head embedding. Embeddings are SPD matrices with a Log-Euclidean geometry. 

* `tggr`: Applies a scaling on the head embedding. Embedding are p-dimensional subspaces of R^n. 

* `tgspdlegr`: Applies a scaling on the head embedding with the mixture model. 

In the current version, the only option for the parameter `--metric` is 

* `riem`: Riemannian metric

## Acknowledgement

We thank the authors of [Vector-valued Distance and Gyrocalculus on the Space of Symmetric Positive Definite Matrices](https://arxiv.org/pdf/2110.13475.pdf) for releasing their code. The code in this repository is based on their source code release ([link](https://github.com/fedelopez77/gyrospd)). If you find this code useful, please consider citing their work.







----------------------------------------------------------------------------------------------------------------------------------
# Knowledge Graph Completion

## Prerequisite



## Experiments

### 1. Data Preparation



### 2. Training & Testing

![\operatorname{SPD}_6^{le}](https://latex.codecogs.com/svg.image?\operatorname{SPD}_6^{le})

![\operatorname{Gr}_{20,10}](https://latex.codecogs.com/svg.image?\operatorname{Gr}_{20,10})

![\operatorname{SPD}_5^{le} \times \operatorname{Gr}_{5,2}](https://latex.codecogs.com/svg.image?\operatorname{SPD}_5^{le}&space;\times&space;\operatorname{Gr}_{5,2})

![\sum_{\forall i}{x_i^{2}}](https://latex.codecogs.com/svg.latex?%5Csum_%7B%5Cforall+i%7D%7Bx_i%5E%7B2%7D%7D)

### 3. Models and Metrics



## Acknowledgement
