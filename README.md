# 4DRaL: Bridging 4D Radar with LiDAR for Place Recognition using Knowledge Distillation

## Quick start

1. **Baidu Cloud**: [Download Link](https://pan.baidu.com/s/1MsxPBH-N5cPJjhdGjgeQoA?pwd=7777)  (extraction Code: `7777`)

2. Create a conda environment and install Pytorch according to your Cuda version. 

3. Install the dependencies by

   ```
   pip install -r requirements.txt
   ```

4. Configure Paths in `config/config.py`

   Edit the following parameters in `config/config.py` to match your system:

   ```
   data_arg.add_argument('--dataset_path', 
                        type=str,  
                        default='/path/to/your/dataset')  # Set to dataset root directory
   
   data_arg.add_argument('--checkpoint', 
                        type=str,  
                        default='/path/to/pretrained/weights')  # Set to model weights directory
   
   data_arg.add_argument('--logPath', 
                        type=str,  
                        default='/path/to/log/files')  # Set to log output directory
   ```

   

## Evaluation 

1. Evaluating radar-to-radar place recognition (R2R) performance by

   ```
   python Evaluate_R2L.py
   ```

2. Evaluating radar-to-LiDAR place recognition (R2L) performance by

   ```
   python Evaluate_R2L.py
   ```

## Training

Currently under organization. We will release the training code and instructions as soon as possible.