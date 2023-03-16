著作权：华东理工大学信息科学与工程学院
哔站demo视频：https://www.bilibili.com/video/BV11m4y1w7wH/#reply135077881808
联系方式：xiaxiaowu2022@163.com

建议先参考grcnn的源代码和readme文件，再看本项目的代码
```

## Requirements
- numpy
- opencv-python
- matplotlib
- scikit-image
- imageio
- torch
- torchvision
- torchsummary
- tensorboardX
- pyrealsense2
- Pillow

#### Cornell Grasping Dataset

1. Download and extract the cornell Dataset
链接：https://pan.baidu.com/s/1rHz-79Mt47Dv_od-7uuEKg 提取码：8888
2. Convert the PCD files to depth images by running `python -m utils.dataset_processing.generate_cornell_depth <Path To Dataset>`

#### Jacquard Dataset

1. Download and extract the Jacquard Dataset
链接：https://pan.baidu.com/s/1524HrVAoHNlc6-9lcZaGew 提取码：8888

## Model Training

A model can be trained using the `train_network.py` script.  Run `train_network.py --help` to see a full list of options.

Example for Jacquard dataset:

```bash
python train_network.py - --dataset-path <Path To Dataset> --description training_jacquard 
```

## Model Evaluation

The trained network can be evaluated using the `evaluate.py` script.  Run `evaluate.py --help` for a full set of options.

```
Example for Jacquard dataset:

```bash
python evaluate.py --network "trained-models/jacquard-d-grconvnet3-drop0-ch32/epoch_50_iou_0.94" --iou-eval --input-size 300 --use-rgb 0
```
python evaluate.py --network "trained-models/jacquard-rgbd-grconvnet3-drop0-ch32/epoch_42_iou_0.93" --iou-eval --input-size 300

## Run Tasks
A task can be executed using the relevant run script. All task scripts are named as `run_<task name>.py`. For example, to run the grasp generator run:
```bash
python plane_grasp_real.py
```

## Run on a Robot
```bash
python run_plane_grasp_real.py
```
