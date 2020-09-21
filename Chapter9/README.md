Chapter 9 - Semantic Segmentation
===

There is a dataset of labeled images in **dataset**.
Optionally you can build your own dataset or augment the current dataset using **01_synchronous_mode_recording.py**, which can run as a client for Carla secording automatically in RGB, raw semantic segmentation and colored semantic segmentation.
There is also a test dataset in **dataset_test**.
There are some examples of output:
- **dataset_good**: example from the network we are building
- **dataset_bad**: example from a network that has not been trained well 

In addition:
- **02_fc_densenet.py** will train the neural network; please adjust the parameters (size, batch size, groups) according to yout hardware capabilities
- **03_inference.py** will process the test dataset, saving the results in **dataset_out**



