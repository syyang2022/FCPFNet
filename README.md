# 1.Project Name and Introduction
# 2.Folder List
The FCPFNet.py in./model corresponds to my model code  
The corresponding indicator calculation code for./utils/metrics. py can also be ignored  
The parameter settings corresponding to Config-FCPF.json are  
# 3.Environment settings
Server  Conda activate tensorflow 1.9  
torch==1.1.0       
torchvision==0.3.0  
tqdm==4.32.2  
tensorboard==1.14.0  
Pillow==6.2.0  
opencv-python==4.1.0.25  
# 4.Training method
Run the code input: Python train.py -- config config-FCPF.json to run. The default is the VOC dataset. If you want to change COCO, change "type" and "data_dir". I have already placed the dataset on the server under the data folder in the FCPFNet folder.  
The project validation is integrated into the training.  
To view the visualization, you can call tensorboard and go to the runs folder under "save_dir": "save/". You can open it using tensorboard -- logdir xxx, which needs to be viewed locally on your own computer and cannot be viewed on the server.  
Among them, xxx is the folder selected under runs.  
Parameter situation: The backbone uses ResNet, which can be changed to ResNet50 or ResNet101.  
Data_dir is the address of the dataset. Put the dataset into it to run the training code.  
