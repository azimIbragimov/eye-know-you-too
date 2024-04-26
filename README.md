
<div align="center">
    <img src="./assets/logo.png" height="70%" width="70%">
</div>

<div align="center">
  <a href="https://faculty.eng.ufl.edu/jain/"><img src="https://img.shields.io/badge/JainLab-blue?style=for-the-badge"  alt="JainLab"></a>
  <a href="https://www.linkedin.com/in/azim-ibragimov/"><img src="https://img.shields.io/badge/Azim%20Ibragimov-green?style=for-the-badge"  alt="Azim Ibragimov"></a>
  <img src="https://img.shields.io/github/stars/azimIbragimov/eye-know-you-too?style=for-the-badge&logo=Trustpilot&logoColor=white&labelColor=d0ab23&color=b0901e" alt="GitHub Repo stars">
</div>

<br>

---

# Eye Know You Too
This repository is an unofficial PyTorch implementation of the paper "Eye Know You Too: Toward Viable End-to-End Eye Movement Biometrics for User Authentication." The official implementation is available [here](https://dataverse.tdl.org/dataset.xhtml?persistentId=doi:10.18738/T8/61ZGZN).

While the official implementation utilizes PyTorch Lightning, this repository offers a simpler alternative using standard PyTorch, which is more widely recognized and commonly used among researchers. This adaptation makes it easier for those familiar with PyTorch to understand and modify the code without needing to learn an additional framework.

## 📔 Table of Contents
- [Comparison with the official repository](#-comparison-with-the-official-repository)
  - [Task Group](#task-group)
  - [Test-retest Interval](#test-retest-interval)
  - [Duration](#duration)
  - [Sampling Rate](#sampling-rate)
- [Environment](#-environment)
  - [Conda](#conda)
  - [Docker](#docker)
- [Dataset](#-dataset)
- [Training](#-training)
  - [Instructions](#instructions)
  - [Pre-trained weights](#pre-trained-weights)
- [Testing](#%EF%B8%8F-training)
- [Evaluation](#-evaluation)
- [Acknowledgements](#-acknowledgements)
- [Citations](#%EF%B8%8F-citations)


## 📊 Comparison with the official repository
To ensure a fair evaluation of both implementations, we have retrained EKYT models using both the official and this repository. We then assessed their performance in terms of Equal Error Rate (EER %), with the results presented in the tables below. The difference in EER (%) is minimal, indicating a correct implementation in this repository. For some tasks, the original implementation performs slightly better; for others, the EER is the same across both approaches, and in some cases, our model slightly outperforms the original. These variations are mainly due to the stochastic nature of deep learning and are entirely random.

*Note: To verify the results, you can download the re-trained original codebase weights [here](https://www.dropbox.com/scl/fo/ziqke9npi1qfpvwb50cgj/AN7Mzr8tsfn0LyggwgREvb0?rlkey=261gnvsxwgoqult6kpyguhc3l&st=vmpzirwn&dl=0) and weights from this codebase in section [Pre-trained weights](#pre-trained-weights)*

### Task Group
| Task | Official Implementation (EER %) | Our Implementation (EER %) | 
| -- | -- | -- |
| TEX | 3.95 | 4.39 | 
| HSS |  5.08  | 5.87  |
| RAN |  5.08 | 5.08 |
| FXS | 11.25  | 11.86  |
| VD1 | 7.27  | 6.41 | 
| VD2 | 4.96  | 5.08 |
| BLG | 7.97  | 6.25 |
 
### Test-retest Interval
| Round | Official Implementation (EER %) | Our Implementation (EER %) | 
| -- | -- | -- |
| R2 | 9.21 | 6.89 |
| R3 | 6.89 | 8.26  |
| R4 | 9.21 | 8.62  |
| R5 | 9.73 | 6.89  |
| R6 | 6.89 | 6.15  |
| R7 | 8.82 | 8.82  |
| R8 | 10.00 | 8.18  |
| R9 | 7.69  | 7.69  |

### Duration
| Duration (s) | Official Implementation (EER %) | Our Implementation (EER %) | 
| -- | -- | -- |
| 5 x 2 | 1.72 | 3.44  | 
| 5 x 3 | 1.69 | 1.69  | 
| 5 x 4 | 1.11 | 0.40  | 
| 5 x 5 | 1.02 | 0.49   | 
| 5 x 6 | 0.81 | 0.49   | 
| 5 x 7 | 0.61 | 0.61  | 
| 5 x 8 | 0.49 | 0.61   | 
| 5 x 9 | 0.49 | 0.61  | 
| 5 x 10 | 0.37 | 0.61   | 
| 5 x 11 | 0.55 | 0.54  | 
| 5 x 12 | 0.46 | 0.75  | 

### Sampling Rate
| Sampling Rate (Hz) | Official Implementation (EER %) | Our Implementation (EER %) | 
| -- | -- | -- |
| 500 Hz | 5.66 | 5.66  |
| 250 Hz | 6.13 | 5.19  |
| 125 Hz | 8.77 | 8.74 |
| 50 Hz | 13.79 | 13.79  |
| 31.25 Hz | 25.72 | 22.41 |


## 🌳 Environment
### Conda
To install a Conda environment for this codebase, ensure that you have Conda installed on your system. Once this is done, run the following command:
```bash
conda env create -n eye-know-you-too python=3.11
conda activate eye-know-you-too
pip install -r requirements.txt
```
### Docker
First ensure that you have Docker installed, as well as any other NVIDIA drivers and toolkits. To build the Docker image from the Dockerfile, use the following command in the terminal. Make sure to execute this command in the directory where your Dockerfile is located.
```bash
docker build -t eye-know-you-image .
```
Once the image is built, you can run a container based on this image. Use the following command to start the container:
```bash
docker run --gpus all -it eye-know-you-image
```

## 💿 Dataset
We utilize the GazeBase dataset, the same one used in the original implementation. This dataset contains eye-tracking data recorded at 1000 Hz while participants engaged in various tasks such as watching videos, reading, etc. Upon initiating the training of the model, the script automatically downloads the dataset and processes it into .pkl files. The processing technique adheres to the descriptions in the referenced paper and the original implementation. It includes converting raw gaze coordinates into smoothed first derivative points using a Savitzky-Golay filter, followed by downsampling the recordings to the desired frequency. 

If you wish to bypass the processing step to save time, you can download the pre-processed .pkl files from the link below:

| Name | Link | 
| -- | -- | 
| GazeBase - 1000 HZ | https://www.dropbox.com/scl/fi/q7rn48pudd7cyp9t8950v/gazebase_savgol_ds1_normal.pkl?rlkey=o7o57856y6ozz7lh2ernwikoz&st=1nbowchk&dl=0 | 
| GazeBase - 500 HZ | https://www.dropbox.com/scl/fi/y2c10di7313c5lb5kvhpu/gazebase_savgol_ds2_normal.pkl?rlkey=d5yoxma548kfm5iyxv5zoxc4o&st=eerp8r59&dl=0 | 
| GazeBase - 250 HZ | https://www.dropbox.com/scl/fi/2cku2vdf3qhrnigclp0z8/gazebase_savgol_ds4_normal.pkl?rlkey=71nt2ybh4rpemmwrwsxcseysf&st=xk7ct4fm&dl=0 | 
| GazeBase - 125 HZ | https://www.dropbox.com/scl/fi/3axxu21idvhpvtajeiwai/gazebase_savgol_ds8_normal.pkl?rlkey=r8mwa7qf1exht911gba7obyfg&st=41u9ffey&dl=0 |
| GazeBase - 50 HZ | https://www.dropbox.com/scl/fi/iy8iqxwtlqrqqs3x0y4ih/gazebase_savgol_ds20_normal.pkl?rlkey=cbm72r8hdbwulm0z9meqcgdcn&st=sl4eyd2i&dl=0 | 
| GazeBase - 31.25 HZ | https://www.dropbox.com/scl/fi/hvz770g58g50cl7dzwnue/gazebase_savgol_ds32_normal.pkl?rlkey=3a0ggn3sb6jcwwaowxgivmljb&st=hhcm754e&dl=0 |

Then make sure to place them within `data/processed/` directory. The directory tree must look like shown below: 

```
data/
├── processed
│   ├── gazebase_savgol_ds1_normal.pkl
│   ├── gazebase_savgol_ds2_normal.pkl
│   ├── gazebase_savgol_ds4_normal.pkl
│   ├── gazebase_savgol_ds8_normal.pkl
│   ├── gazebase_savgol_ds20_normal.pkl
│   └── gazebase_savgol_ds32_normal.pkl
```

Alternitavely, if you wish to download all pickle files at once, you can run the following command:
```bash
bash misc/download_dataset_pkl.sh
```

## 🏋️ Training
### Instructions
Once the dataset is placed in the correct directory, you can begin training the model. If you wish to train models across all frequencies (1000 Hz, 500 Hz, 250 Hz, 125 Hz, 50 Hz, and 31.25 Hz), you can run the following command:
```bash
bash train.sh
```
However, if you want to train a model at a specific frequency, please run the following command:
```bash
python src/train.py --ds=<ds_value> --fold=0
python src/train.py --ds=<ds_value> --fold=1
python src/train.py --ds=<ds_value> --fold=2
python src/train.py --ds=<ds_value> --fold=3

#Example when running the model on 125Hz:
#python src/train.py --ds=8 --fold=0
#python src/train.py --ds=8 --fold=1
#python src/train.py --ds=8 --fold=2
#python src/train.py --ds=8 --fold=3
```

### Pre-trained weights
If you wish to skip the training process, you can use pre-trained weights. The table below provides the links from where you can obtain the weights.

| Name | Link |
| -- | -- |
| EyeKnowYouToo - 1000HZ | https://www.dropbox.com/scl/fo/sn0qcmind2ln5gkmrvvwe/AFjySWnObz3nkjt0lXsOCIw?rlkey=uus626eexxonx5p1735jq34x0&st=vz9ux8es&dl=0 |
| EyeKnowYouToo - 500HZ | https://www.dropbox.com/scl/fo/xp2zkr5xvyakcoudrus6i/ANA4DJfYEFxPpIYnrJE0MTE?rlkey=3v75adr7n3txtmov71pci9lqr&st=13ui29tf&dl=0  |
| EyeKnowYouToo - 250 HZ | https://www.dropbox.com/scl/fo/rwfrgflkjveslq9shzsri/ALIbUo__eDNyx1e3mkC5klI?rlkey=04i7furdytgby266oqxjkowmp&st=jq3vdebz&dl=0 | 
| EyeKnowYouToo - 125 HZ | https://www.dropbox.com/scl/fo/to2ad1bpzscc639a59a9l/AKBTpm8tR9RKSVCB-EW06L8?rlkey=vqjtnu0ckv92g5wz6r6mdpod8&st=glu1uvne&dl=0 |
| EyeKnowYouToo - 50 HZ | https://www.dropbox.com/scl/fo/k5wg2fqjtji1psipzfx2h/ABe4uapHBuKz309iRzPX2Q0?rlkey=z5fsiic0ou20lbwx39dkxbsl3&st=21ley2ct&dl=0 |
| EyeKnowYouToo - 31.25 HZ | https://www.dropbox.com/scl/fo/zasfxht5df7i7i2e9huzj/ADHEpNrWKaWiN0El3VUpQGA?rlkey=50dduktd5qdwfzj7lbzddmqu2&st=5tdoooh9&dl=0 |

*Note: Training was conducted on an NVIDIA A100 GPU with 80GB VRAM*

Then place the weights in the following manner: 
```bash
output/
├── ekyt_t5000_ds1_bc16_bs16_wms10_wce01_normal_f0_epoch=99.ckpt
├── ekyt_t5000_ds1_bc16_bs16_wms10_wce01_normal_f1_epoch=99.ckpt
├── ekyt_t5000_ds1_bc16_bs16_wms10_wce01_normal_f2_epoch=99.ckpt
├── ekyt_t5000_ds1_bc16_bs16_wms10_wce01_normal_f3_epoch=99.ckpt
├── ...
├── ...
├── ekyt_t5000_ds32_bc16_bs16_wms10_wce01_normal_f0_epoch=99.ckpt
├── ekyt_t5000_ds32_bc16_bs16_wms10_wce01_normal_f1_epoch=99.ckpt
├── ekyt_t5000_ds32_bc16_bs16_wms10_wce01_normal_f2_epoch=99.ckpt
└── ekyt_t5000_ds32_bc16_bs16_wms10_wce01_normal_f3_epoch=99.ckpt
```

## 🧪 Testing
If you already have trained models across all frequencies (or have downloaded weights for each frequency), then you can run the following command to test all the models:
```bash
bash test.sh
```
However, if you want to test a model at a specific frequency, please run the following command:
```bash
python src/test.py --ds=<ds_value> --fold=0
python src/test.py --ds=<ds_value> --fold=1
python src/test.py --ds=<ds_value> --fold=2
python src/test.py --ds=<ds_value> --fold=3

#Example when running the model on 125Hz:
#python src/test.py --ds=8 --fold=0
#python src/test.py --ds=8 --fold=1
#python src/test.py --ds=8 --fold=2
#python src/test.py --ds=8 --fold=3
```

## 🤔 Evaluation
Once the model has been tested, you can evaluate the results running the following command: 
```bash
python src/evaluate.py --model=<model_name>

#Example when running 125 Hz model
#python src/evaluate.py --model=ekyt_t5000_ds2_bc16_bs16_wms10_wce01_normal
```

## 🙏 Acknowledgements
We thank Dillon Lohr and Oleg Komogortsev for making their original implementation open-source to the public. Without this, implementation of this repository would not be possible. When using this repository, make sure to cite their paper listed in [citations](#citations) 

## 🖊️ Citations
```
@ARTICLE{9865991,
  author={Lohr, Dillon and Komogortsev, Oleg V.},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Eye Know You Too: Toward Viable End-to-End Eye Movement Biometrics for User Authentication}, 
  year={2022},
  volume={17},
  number={},
  pages={3151-3164},
  keywords={Authentication;Biometrics (access control);Convolution;Transformers;Performance evaluation;Behavioral sciences;Task analysis;Eye tracking;user authentication;metric learning;template aging;permanence;signal quality},
  doi={10.1109/TIFS.2022.3201369}}
```



