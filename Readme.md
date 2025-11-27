# Citation
If you have used this code, please consider citing the following paper:
"HULFSynth : An INR based Super-Resolution and Ultra Low-Field MRI Synthesis via Contrast factor estimation", Pranav Indrakanti and Ivor Simpson. 
https://arxiv.org/abs/2511.14897

#Default hyperparameters :
| Hyperparam |      Description      |    Default <br>value     |      Range       |                                           Notes                                            |
| :--------: | :-------------------: | :----------------------: | :--------------: | :----------------------------------------------------------------------------------------: |
|   $l_1$    |       MAE Recon       |           1e3            |    1e2 to 1e5    |                                Tuned in parallel with $l_2$                                |
|   $l_2$    |     Segmentation      |           1e2            |   0.5e2 to 5e3   |                                Tuned in parallel with $l_1$                                |
|   $l_3$    |        TV seg         | [0.95, 0.95, 0.95, 0.01] |   0.1 to 10.0    |                tuned in parallel with fff; weight CSF low if fff $\geq$ 64                 |
|   $l_4$    |     TV intensity      |   [1.5, 1.5, 1.5, 0.1]   |   0.1 to 10.0    |                tuned in parallel with fff; weight CSF low if fff $\geq$ 64                 |
|    fff     | Fourier <br>Frequency |            72            |    32 to 156     |            higher fff preserves high-frequency info; lower fff smoothes images;            |
|    ffs     |   Fourier<br>scale    |            4             |     4, 8, 12     |                                  Less important parameter                                  |
|   Epochs   |                       |           15e3           |   5e3 to 20e4    |                  Tuned in parallel with LR; <br>(10e3 for faster tuning)                   |
|     LR     |   Learning<br>rate    |           5e-5           | 1e-3, 5e-4, 5e-5 |                           Tuned in parallel with epochs and fff                            |
| $\omega_0$ |      WIRE omega       |           20.0           |  20, 25, 30, 35  | Not tuned independently; <br>Original paper suggests not very important to carefully tune; |
| $\sigma_0$ |      WIRE sigma       |           10.0           |    10, 15, 20    | Not tuned independently; <br>Original paper suggests not very important to carefully tune; |
|   layers   |     hidden layers     |            5             |     3, 5, 7      |                         tuned in parallel with LR, hidden features                         |
|  features  |    hidden features    |           128            |     128, 256     |                          tuned in parallel with LR, hidden layers                          |


