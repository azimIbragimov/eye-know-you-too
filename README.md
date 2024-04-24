# Eye Know You Too

This repository is an unofficial PyTorch implementation of the paper "Eye Know You Too: Toward Viable End-to-End Eye Movement Biometrics for User Authentication." The official implementation is available [here](https://dataverse.tdl.org/dataset.xhtml?persistentId=doi:10.18738/T8/61ZGZN).

While the official implementation utilizes PyTorch Lightning, this repository offers a simpler alternative using standard PyTorch, which is more widely recognized and commonly used among researchers. This adaptation makes it easier for those familiar with PyTorch to understand and modify the code without needing to learn an additional framework.

## Dataset
We utilize the GazeBase dataset, the same one used in the original implementation. This dataset contains eye-tracking data recorded at 1000 Hz while participants engaged in various tasks such as watching videos, reading, etc. Upon initiating the training of the model, the script automatically downloads the dataset and processes it into .pkl files. The processing technique adheres to the descriptions in the referenced paper and the original implementation. It includes converting raw gaze coordinates into smoothed first derivative points using a Savitzky-Golay filter, followed by downsampling the recordings to the desired frequency. If you wish to bypass the processing step to save time, you can download the pre-processed .pkl files from the link below:

| -- | -- | 
| Name | Link | 
| GazeBase - 1000 HZ | https://www.dropbox.com/scl/fi/q7rn48pudd7cyp9t8950v/gazebase_savgol_ds1_normal.pkl?rlkey=o7o57856y6ozz7lh2ernwikoz&st=1nbowchk&dl=0 | 
| GazeBase - 500 HZ | https://www.dropbox.com/scl/fi/y2c10di7313c5lb5kvhpu/gazebase_savgol_ds2_normal.pkl?rlkey=d5yoxma548kfm5iyxv5zoxc4o&st=eerp8r59&dl=0 | 
| GazeBase - 250 HZ | https://www.dropbox.com/scl/fi/2cku2vdf3qhrnigclp0z8/gazebase_savgol_ds4_normal.pkl?rlkey=71nt2ybh4rpemmwrwsxcseysf&st=xk7ct4fm&dl=0 | 
| GazeBase - 125 HZ | https://www.dropbox.com/scl/fi/3axxu21idvhpvtajeiwai/gazebase_savgol_ds8_normal.pkl?rlkey=r8mwa7qf1exht911gba7obyfg&st=41u9ffey&dl=0 |
| GazeBase - 50 HZ | https://www.dropbox.com/scl/fi/iy8iqxwtlqrqqs3x0y4ih/gazebase_savgol_ds20_normal.pkl?rlkey=cbm72r8hdbwulm0z9meqcgdcn&st=sl4eyd2i&dl=0 | 
| GazeBase - 31.25 HZ | https://www.dropbox.com/scl/fi/hvz770g58g50cl7dzwnue/gazebase_savgol_ds32_normal.pkl?rlkey=3a0ggn3sb6jcwwaowxgivmljb&st=hhcm754e&dl=0 |
