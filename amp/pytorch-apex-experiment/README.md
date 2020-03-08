# pytorch-apex-experiment
Simple experiment of [Apex: PyTorch Extension with Tools to Realize the Power of Tensor Cores](https://github.com/NVIDIA/apex) 

## Usage
### 1.Install Apex package
[Apex: A PyTorch Extension](https://github.com/NVIDIA/apex)
### 2.Train
```
python CIFAR.py --GPU gpu_name --mode 'FP16' --batch_size 128 --iteration 100
```
### 3.plot (optional)
```
python make_plot.py --GPU 'gpu_name1' 'gpu_name2' 'gpu_name3' --method 'FP32' 'FP16' 'amp' --batch 128 256 512 1024 2048
```
### Folder structure
The following shows basic folder structure.
```
├── cifar
├── CIFAR.py  # training code
├── utils.py
├── make_plot.py
└── results
    └── gpu_name  # results to be saved here
```
## Experiment settings
 * Network: vgg16
 * Dataset: CIFAR10
 * Method: FP32 (float32), FP16 (float16; half tensor), AMP (Automatic Mixed Precision)
 * GPU: GTX 1080 Ti, GTX TITAN X, Tesla V100
 * Batch size: 128, 256, 512, 1024, 2048
 * All random seeds are fixed
 * Result: The mean and std of 5 times (each 100 iterations)
 * Ubuntu 16.04
 * Python 3
 * Cuda 9.0
 * PyTorch 0.4.1
 * torchvision 0.2.1
 
## Resutls
<table align='center'>
<tr align='center'>
  <td rowspan="2"> GPU - Method </td>
  <td rowspan="2"> Metric </td>
  <td colspan="5"> Batch size </td>
</tr>
<tr align='center'>
  <td> 128 </td>
  <td> 256 </td>
  <td> 512 </td>
  <td> 1024 </td>
  <td> 2048 </td>
</tr>
<tr align='center'>
  <td rowspan="3"> 1080 Ti - FP32 </td>
  <td> Accuracy (%) </td>
  <td> 40.92 ± 2.08 </td>
  <td> 50.74 ± 3.64 </td>
  <td> 61.32 ± 2.43 </td>
  <td> 64.79 ± 1.56 </td>
  <td> 63.44 ± 1.76 </td>
</tr>
<tr align='center'>
  <td> Time (sec) </td>
  <td> 5.16 ± 0.73 </td>
  <td> 9.12 ± 1.20 </td>
  <td> 16.75 ± 2.05 </td>
  <td> 32.23 ± 3.23 </td>
  <td> 63.42 ± 4.89 </td>
</tr>
<tr align='center'>
  <td> Memory (Mb) </td>
  <td> 1557.00 ± 0.00 </td>
  <td> 2053.00 ± 0.00 </td>
  <td> 2999.00 ± 0.00 </td>
  <td> 4995.00 ± 0.00 </td>
  <td> 8763.00 ± 0.00 </td>
</tr>
<tr align='center'>
  <td rowspan="3"> 1080 Ti - FP16 </td>
  <td> Accuracy (%) </td>
  <td> 43.35 ± 2.04 </td>
  <td> 51.00 ± 3.75 </td>
  <td> 57.70 ± 1.58 </td>
  <td> 63.79 ± 3.95 </td>
  <td> 62.64 ± 1.91 </td>
</tr>
<tr align='center'>
  <td> Time (sec) </td>
  <td> 5.42 ± 0.71 </td>
  <td> 9.11 ± 1.14 </td>
  <td> 16.54 ± 1.78 </td>
  <td> 31.49 ± 3.01 </td>
  <td> 61.79 ± 5.15 </td>
</tr>
<tr align='center'>
  <td> Memory (Mb) </td>
  <td> 1405.00 ± 0.00 </td>
  <td> 1745.00 ± 0.00 </td>
  <td> 2661.00 ± 0.00 </td>
  <td> 4013.00 ± 0.00 </td>
  <td> 6931.00 ± 0.00 </td>
</tr>
<tr align='center'>
  <td rowspan="3"> 1080 Ti - AMP </td>
  <td> Accuracy (%) </td>
  <td> 41.11 ± 1.19 </td>
  <td> 47.59 ± 1.79 </td>
  <td> 60.37 ± 2.48 </td>
  <td> 63.31 ± 1.92 </td>
  <td> 63.41 ± 3.75 </td>
</tr>
<tr align='center'>
  <td> Time (sec) </td>
  <td> 6.32 ± 0.70 </td>
  <td> 10.70 ± 1.11 </td>
  <td> 18.95 ± 1.80 </td>
  <td> 36.15 ± 3.01 </td>
  <td> 72.64 ± 5.11 </td>
</tr>
<tr align='center'>
  <td> Memory (Mb) </td>
  <td> 1941.00 ± 317.97 </td>
  <td> 1907.00 ± 179.63 </td>
  <td> 2371.00 ± 0.00 </td>
  <td> 4073.00 ± 0.00 </td>
  <td> 7087.00 ± 0.00 </td>
</tr>
<tr align='center'>
  <td rowspan="3"> TITAN X - FP32 </td>
  <td> Accuracy (%) </td>
  <td> 42.90 ± 2.42 </td>
  <td> 45.78 ± 1.22 </td>
  <td> 60.88 ± 1.78 </td>
  <td> 64.22 ± 2.62 </td>
  <td> 63.79 ± 1.62 </td>
</tr>
<tr align='center'>
  <td> Time (sec) </td>
  <td> 5.86 ± 0.80 </td>
  <td> 9.59 ± 1.29 </td>
  <td> 18.19 ± 1.84 </td>
  <td> 35.62 ± 4.07 </td>
  <td> 66.56 ± 4.62 </td>
</tr>
<tr align='center'>
  <td> Memory (Mb) </td>
  <td> 1445.00 ± 0.00 </td>
  <td> 1879.00 ± 0.00 </td>
  <td> 2683.00 ± 0.00 </td>
  <td> 4439.00 ± 0.00 </td>
  <td> 7695.00 ± 0.00 </td>
</tr>
<tr align='center'>
  <td rowspan="3"> TITAN X - FP16 </td>
  <td> Accuracy (%) </td>
  <td> 39.13 ± 3.56 </td>
  <td> 49.87 ± 2.42 </td>
  <td> 59.77 ± 1.77 </td>
  <td> 65.57 ± 2.82 </td>
  <td> 64.08 ± 1.80 </td>
</tr>
<tr align='center'>
  <td> Time (sec) </td>
  <td> 5.66 ± 0.97 </td>
  <td> 9.72 ± 1.23 </td>
  <td> 17.14 ± 1.82 </td>
  <td> 33.23 ± 3.50 </td>
  <td> 65.86 ± 4.94 </td>
</tr>
<tr align='center'>
  <td> Memory (Mb) </td>
  <td> 1361.00 ± 0.00 </td>
  <td> 1807.00 ± 0.00 </td>
  <td> 2233.00 ± 0.00 </td>
  <td> 3171.00 ± 0.00 </td>
  <td> 5535.00 ± 0.00 </td>
</tr>
<tr align='center'>
  <td rowspan="3"> TITAN X - AMP </td>
  <td> Accuracy (%) </td>
  <td> 42.57 ± 1.25 </td>
  <td> 49.59 ± 2.14 </td>
  <td> 59.76 ± 1.60 </td>
  <td> 63.76 ± 4.24 </td>
  <td> 65.14 ± 2.93 </td>
</tr>
<tr align='center'>
  <td> Time (sec) </td>
  <td> 7.55 ± 1.03 </td>
  <td> 11.82 ± 1.07 </td>
  <td> 20.96 ± 1.83 </td>
  <td> 38.82 ± 3.17 </td>
  <td> 76.54 ± 6.60 </td>
</tr>
<tr align='center'>
  <td> Memory (Mb) </td>
  <td> 1729.00 ± 219.51 </td>
  <td> 1999.00 ± 146.97 </td>
  <td> 2327.00 ± 0.00 </td>
  <td> 3453.00 ± 0.00 </td>
  <td> 5917.00 ± 0.00 </td>
</tr>
<tr align='center'>
  <td rowspan="3"> V100 - FP32 </td>
  <td> Accuracy (%) </td>
  <td> 42.56 ± 1.37 </td>
  <td> 49.50 ± 1.81 </td>
  <td> 60.91 ± 0.88 </td>
  <td> 65.26 ± 1.76 </td>
  <td> 63.93 ± 3.69 </td>
</tr>
<tr align='center'>
  <td> Time (sec) </td>
  <td> 3.93 ± 0.54 </td>
  <td> 6.90 ± 0.82 </td>
  <td> 12.97 ± 1.27 </td>
  <td> 25.11 ± 1.83 </td>
  <td> 49.43 ± 3.46 </td>
</tr>
<tr align='center'>
  <td> Memory (Mb) </td>
  <td> 1834.00 ± 0.00 </td>
  <td> 2214.00 ± 0.00 </td>
  <td> 2983.60 ± 116.80 </td>
  <td> 4674.00 ± 304.00 </td>
  <td> 8534.80 ± 826.40 </td>
</tr>
<tr align='center'>
  <td rowspan="3"> V100 - FP16 </td>
  <td> Accuracy (%) </td>
  <td> 43.37 ± 2.13 </td>
  <td> 51.78 ± 2.48 </td>
  <td> 58.46 ± 1.81 </td>
  <td> 64.72 ± 2.37 </td>
  <td> 63.21 ± 1.60 </td>
</tr>
<tr align='center'>
  <td> Time (sec) </td>
  <td> 3.28 ± 0.52 </td>
  <td> 5.95 ± 1.03 </td>
  <td> 10.50 ± 1.27 </td>
  <td> 19.65 ± 1.95 </td>
  <td> 37.32 ± 3.73 </td>
</tr>
<tr align='center'>
  <td> Memory (Mb) </td>
  <td> 1777.20 ± 25.60 </td>
  <td> 2040.00 ± 0.00 </td>
  <td> 2464.00 ± 0.00 </td>
  <td> 3394.00 ± 0.00 </td>
  <td> 4748.00 ± 0.00 </td>
</tr>
<tr align='center'>
  <td rowspan="3"> V100 - AMP </td>
  <td> Accracy (%) </td>
  <td> 42.39 ± 2.35 </td>
  <td> 51.33 ± 1.84 </td>
  <td> 61.41 ± 2.10 </td>
  <td> 65.05 ± 3.29 </td>
  <td> 61.67 ± 3.13 </td>
</tr>
<tr align='center'>
  <td> Time (sec) </td>
  <td> 4.27 ± 0.54 </td>
  <td> 7.18 ± 0.90 </td>
  <td> 13.31 ± 1.26 </td>
  <td> 23.99 ± 2.29 </td>
  <td> 45.68 ± 3.77 </td>
</tr>
<tr align='center'>
  <td> Memory (Mb) </td>
  <td> 2174.80 ± 211.74 </td>
  <td> 2274.00 ± 172.15 </td>
  <td> 2775.20 ± 77.60 </td>
  <td> 3790.80 ± 154.40 </td>
  <td> 5424.00 ± 0.00 </td>
</tr>
</table>

### Visualization
<table align='center'>
<tr align='center'>
  <td> Time </td>
  <td> Memory </td>
</tr>
<tr align='center'>
  <td> <img src = 'assets/CIFAR - Time.png'> </td>
  <td> <img src = 'assets/CIFAR - Memory.png'> </td>
</tr>
<tr align='center'>
  <td> Time with std </td>
  <td> Memory with std </td>
</tr>
<tr align='center'>
  <td> <img src = 'assets/CIFAR - Time (std).png'> </td>
  <td> <img src = 'assets/CIFAR - Memory (std).png'> </td>
</tr>
</table>

