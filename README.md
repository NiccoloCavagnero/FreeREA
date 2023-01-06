# FreeREA
Code release for **[FreeREA: Training-Free Evolution-Based Architecture Search](https://openaccess.thecvf.com/content/WACV2023/papers/Cavagnero_FreeREA_Training-Free_Evolution-Based_Architecture_Search_WACV_2023_paper.pdf)**

If you use this code or the attached files for research purposes, please cite
```
@InProceedings{cavagnero2022freerea,
    author    = {Cavagnero, Niccol\`o and Robbiano, Luca and Caputo, Barbara and Averta, Giuseppe},
    title     = {FreeREA: Training-Free Evolution-Based Architecture Search},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {1493-1502}
}
```

## Software requirements
* Python 3.9 or newer
* PyTorch 1.9 or newer
* Other Python libraries listed in `requirements.txt`

## Hardware requirements
A CUDA-capable GPU is required to compute the metrics.
However, precomputed metrics for the benchmarks NASBench101 and NATS-Bench are available in the directory `cached_metrics`.

## Run experiments
Results can be reproduced with:
```
export NATS_PATH=/data/path/to/NATS-tss-v1_0-3ffb9-simple
./run_search.py --space nats --dataset cifar10
./run_search.py --space nats --dataset cifar100
./run_search.py --space nats --dataset ImageNet16-120

export NASBENCH101_PATH=/data/path/to/NASBench-101/nasbench_full.pkl
./run_search.py --space nasbench101 --dataset cifar10 --max-time 12
```

## License
This code and the attached files are distributed under the MIT license.

Code within the directory `nas_spaces/_nasbench101` is derived from [this repository](https://github.com/google-research/nasbench) and is released under the Apache 2.0 license.

### Contributors
* Niccolò Cavagnero <niccolo.cavagnero@polito.it>
* Luca Robbiano <luca.robbiano@polito.it>
