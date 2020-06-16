Chapter 7 - Detecting pedestrians and traffic lights
===

There is a dataset of traffic lights in the **traffic_dataset** directory.
Optionally you can build your own dataset or augment the current dataset:
* Save images (e.g. from Carla) of roads with traffic lights as .png
* Copy the images in **traffic_input_full**
* Run **01_extract_traffic_lights.py**
* The Python file will detect traffic lights and will save the images cropped in **traffic_input_cropped**
* Now you have to separate them by color, and put them in the appropriate directory under **traffic_dataset**: *0_green* (green lights), *1_yellow* (yellow lights), *2_red* (red lights) and *3_not* (not a traffic light, or the back of a traffic light)

To train a Neural Network to detect traffic lights color, using Transfer Learning from Inception V3, simply run **02_train_traffic.py**.
For your convenience, the validation dataset is processed and saved in out_valid, so you can check how the network is behaving in the validation dataset.
Please consider that you should delete the directory every time, or old files can still be there.
The best model will be saved as **traffic.h5**.

To execute the full detection, you can run **03_detect.py**. It will use SSD plus the network saved as **traffic.h5** to process the files in the directory **images**, and it will output annotated images on the directory **out_images**. 

    
