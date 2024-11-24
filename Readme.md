## A Deep Learning based Approach for Semantic Segmentation of Small Fires from UAV Image

In this paper, we propose four novel techniques for fire segmentation from UAV images:
1. **CIELAB Thresholding**: Detects fire pixels in CIELAB color space by comparing their hue and chroma values to threshold values.
2. **ObjectDetection+CIELAB Thresholding**: Uses an object detector (e.g., YOLO) for obtaining fire bounding-boxes and then applies CIELAB thresholding only inside the bounding boxes. This technique has high speed since it converts the segmentation task into an object detection task.
3. **SEG-4CHANNEL**: Creates a mask of fire pixels using the ObjectDetection+CIELAB thresholding technique. This channel is passed as the fourth channel to various segmentation networks, and it helps in giving attention to fire while ignoring the background.
4. **AttentionSeg**: Uses an attention module and a segmentation model (e.g., SegFormer-B5) that takes four channels as input. It combines the advantages of CIELAB color models, CNNs, and transformer architecture.

We explore a large design space consisting of various networks and backbones and evaluate our techniques on the FLAME segmentation dataset. Our best model, AttentionSeg-B5, segments fire with an intersection-over-union (IoU) score of 84.15% and 91.39% F1-score.

**Keywords**: Deep neural networks, fire segmentation, UAV images, CIELAB color space, attention, transformer

## Repository Setup

To set up the repository, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/CandleLabAI/FireSegmentation.git
    cd FireSegmentation
    ```

2. **Create a virtual environment and activate it**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Download the FLAME segmentation dataset** and place it in the `dataset` directory. The `dataset` directory must contain two folders named `Images` and `Masks`, which should contain the images and masks downloaded from [FLAME dataset](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs).

5. **Run the data preprocessing notebook**:
    ```sh
    jupyter notebook notebooks/01_data_preprocessing.ipynb
    ```
    This notebook handles:
    - Resizing images and masks to 512x512
    - Converting segmentation masks to YOLO format bounding boxes (handling multiple fires)
    - Saving annotations in the required format (including empty annotations)

    
6. **Run the dataset splitting notebook**:
    ```sh
    jupyter notebook notebooks/02_dataset_split.ipynb
    ```
    This notebook splits the processed dataset into train/val/test sets with the specified distribution:
    - Train: 1645 images
    - Val: 151 images
    - Test: 207 images

7. **Run the multichannel image creation notebook**:
    ```sh
    jupyter notebook notebooks/05_create_multichannel_images.ipynb
    ```
    This notebook creates 8-channel images from RGB images and bounding boxes using CIELAB color space and bounding box masks.

8. **Run the YOLO training script**:
    ```sh
    python src/train_yolo.py
    ```
   
9. **Run the SegFormer training script**:
    ```sh
    python src/train_attentionseg.py
    ```
