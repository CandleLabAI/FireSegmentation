## A Deep Learning based Approach for Semantic Segmentation of Small Fires from UAV Image

In this paper, we propose four novel techniques for fire segmentation from UAV images:
1. **CIELAB Thresholding**: Detects fire pixels in CIELAB color space by comparing their hue and chroma values to threshold values.
2. **ObjectDetection+CIELAB Thresholding**: Uses an object detector (e.g., YOLO) for obtaining fire bounding-boxes and then applies CIELAB thresholding only inside the bounding boxes. This technique has high speed since it converts the segmentation task into an object detection task.
3. **SEG-4CHANNEL**: Creates a mask of fire pixels using the ObjectDetection+CIELAB thresholding technique. This channel is passed as the fourth channel to various segmentation networks, and it helps in giving attention to fire while ignoring the background.
4. **AttentionSeg**: Uses an attention module and a segmentation model (e.g., SegFormer-B5) that takes four channels as input. It combines the advantages of CIELAB color models, CNNs, and transformer architecture.

We explore a large design space consisting of various networks and backbones and evaluate our techniques on the FLAME segmentation dataset. Our best model, AttentionSeg-B5, segments fire with an intersection-over-union (IoU) score of 84.15% and 91.39% F1-score.

**Keywords**: Deep neural networks, fire segmentation, UAV images, CIELAB color space, attention, transformer
