Chapter 8 - Behavioral Cloning
===

There is a dataset of labeled images in **dataset_in**.
Optionally you can build your own dataset or augment the current dataset:
- **01_manual_control_recording.py** can run as a client for Carla that allows you to record images from 3 cameras, when pressing 'r'. In its current form, it needs to run from the installation directory of Carla, in PythonAPI/examples
- Remember to record not only correct driving, but also stints to "correct the position", where you go from the border toward the center
- You might also need to record more than once the turns that you want the car to take

In addition:
- You have to run **02_prepare_dataset.py** to prepare the dataset and change the name of the files with the intended steering correction.
- **03_train_behavioral_cloning_generator.py** can train the network with generators
- **04_visualize.py** can show the saliency maps
- **05_manual_control_drive.py** uses the network trained to drive Carla, when pressing 'd'.  In its current form, it needs to run from the installation directory of Carla, in PythonAPI/examples



