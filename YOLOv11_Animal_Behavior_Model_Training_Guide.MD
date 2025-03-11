# Step-by-Step Guide for YOLO Training Data Preparation

|Youtube Video Tutorial Is Available for Those Who Want to Follow Along|
|---|
|[![Deer Mouse Behavior Analysis Tutorial](http://img.youtube.com/vi/eO-gpvaay-4/0.jpg)](https://www.youtube.com/watch?v=eO-gpvaay-4)|


This guide provides a detailed walkthrough for preparing your data for training a YOLOv8 and YOLOv11 object detection model. It covers extracting frames from videos, setting up the necessary folder structure, annotating images using LabelMe, and training your model.

#### Install Anaconda Navigator or Miniconda Before Proceeding - (https://www.anaconda.com/products/navigator)

## 1. Set Up Your Environment

*   **Create Anaconda Environment:**
    ```bash
    conda create -n yolo python=3.11 
    conda activate yolo
    ```
    *   **Explanation:** These two lines of code creates an isolated environment for your project, preventing conflicts with other Python installations. Activating the environment ensures you're working within it.

## 2. Extract Frames from Behavior Videos

* Gather a representative sample of behavior videos to create a traning and model validation dataset.
* Alternatively, you can also use your experimental videos to extract representative video frames of behavior interest.
* 
*   **Using FFmpeg:**
*   To Install FFmpeg on Windows, follow this guide here: https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/
*   Once FFmpeg is install, use the following command in your command terminal to extract video frames form your recordings
    ```bash
    ffmpeg -i input.mp4 -vf "fps=1/2" -frames:v 40 output_%02d.png
    ```
    *   **Explanation:**
        *   `-i input.mp4`: Specifies the input video file.
        *   `-vf "fps=1/2"`: Sets the frame rate to 1 frame every 2 seconds. Adjust `1/2` to your desired frame rate (e.g., `1/5` for 1 frame every 5 seconds, `1` for every second, `30` for 30 frames per second).
        *   `-frames:v 40`: Limits the extraction to 40 frames. Remove this if you want to extract all frames at the specified frame rate.
        *   `output_%02d.png`: Specifies the output file name pattern. `%02d` creates a sequence of numbers (e.g., `output_01.png`, `output_02.png`).
    *   **Best Practices:**
        *   **Frame Rate:** Choose a frame rate that captures the behaviors you want to detect. Too low, and you might miss important movements; too high, and you'll have a lot of redundant frames.
        *   **Frame Count:** If you have long videos, consider extracting frames at intervals rather than all frames to reduce the dataset size.
        *   **File Format:** `.png` is a good lossless format for image annotation.

## 3. Create Folder Structure for YOLO Training

*   **Main Project Folder:** Create a main folder for your project (e.g., `yolo_project`).
*   **Data Folders:** Inside the project folder, create the following subfolders:
    ```
    yolo_project/
        data/
            images/
                train/
                val/
            labels/
                train/
                val/
    ```
    *   **Explanation:**
        *   `data/`: This folder will contain all your training and validation data.
        *   `images/`: This folder will contain the extracted image frames.
        *   `labels/`: This folder will contain the annotation files (labels) corresponding to the images.
        *   `train/`: This folder will contain the images and labels used for training the model.
        *   `val/`: This folder will contain the images and labels used for validating the model.
*   **Move Extracted Frames:** Move the extracted frames (e.g., `output_01.png`, `output_02.png`, etc.) into the `data/images/train/` folder.

## 4. Install LabelMe

*   **Using pip:**
    ```bash
    pip install labelme
    ```
    *   **Explanation:** This installs the LabelMe annotation tool.

## 5. Run LabelMe

*   **From the command line:**
    ```bash
    labelme
    ```
    *   **Explanation:** This launches the LabelMe application.

## 6. Configure LabelMe

*   **Open Images:**
    *   Click "Open Dir" and select the `data/images/train/` folder. This will load all your extracted frames into LabelMe.
*   **Change Save Directory:**
    *   Click "Change Save Dir" and select the `data/labels/train/` folder. This will ensure that the annotation files are saved in the correct location.
*   **Set Annotation Format:**
    *   Click "View" -> "YOLO" to set the annotation format to YOLO. This is crucial for compatibility with YOLO training.

## 7. Annotate Your Objects

*   **Create Bounding Boxes:**
    *   Click "Create RectBox" and draw a rectangle around the object you want to annotate (e.g., the mouse).
*   **Label the Object:**
    *   A dialog box will appear asking you to label the object. Enter the appropriate label (e.g., "Mouse", "Jump", "WallHugging").
*   **Save Annotations:**
    *   Click "Save" to save the annotation. This will create a `.txt` file with the same name as the image in the `data/labels/train/` folder. The `.txt` file will contain the bounding box coordinates and class label in the YOLO format.
*   **Repeat:** Repeat steps 7 for all the images in your training set.

## 8. Install Ultralytics (YOLOv8)

*   **Using pip:**
    ```bash
    pip install ultralytics
    ```
    *   **Explanation:** This installs the Ultralytics library, which includes YOLOv8.

## 9. Install PyTorch (with CUDA if available or use the CPU pytorch if you don't have a GPU)

*   **Follow the instructions on the PyTorch website:**
    *   Go to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
    *   Select your operating system, package manager (pip), Python version, and CUDA version (if you have an NVIDIA GPU).
    *   Copy the command provided and run it in your Anaconda environment.
    *   **Example (CUDA 11.8):**
        ```bash
        pip3 install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```
    *   **Explanation:** This installs PyTorch with the appropriate CUDA support if you have an NVIDIA GPU. Using a GPU will significantly speed up training.

## 10. Create Validation Data

*   **Create Validation Folders:** Inside the `data/images/` and `data/labels/` folders, create `val/` subfolders if they don't exist.
*   **Move Validation Images and Labels:**
    *   Move a subset of your annotated images (e.g., 10-20% of your total images) from `data/images/train/` to `data/images/val/`.
    *   Move the corresponding annotation files from `data/labels/train/` to `data/labels/val/`.
    *   **Important:** Delete the moved images and labels from the `train` folders to avoid data leakage during training.

## 11. Create `data_custom.yaml`

*   **Using a text editor:** Create a new file named `data_custom.yaml` in your `yolo_project` folder.
*   **Add the following content:**
    ```yaml
    train: "path/to/your/yolo_project/data/images/train"
    val: "path/to/your/yolo_project/data/images/val"
    nc: 3  # Replace with the number of classes you have

    names: ["Jump", "Mouse", "WallHugging"] # Replace with your class names
    ```
    *   **Explanation:**
        *   `train`: Specifies the path to your training images folder.
        *   `val`: Specifies the path to your validation images folder.
        *   `nc`: Specifies the number of classes you are detecting.
        *   `names`: Specifies the names of your classes.
    *   **Important:** Replace the placeholder paths with the actual paths to your folders.

## 12. Train Your YOLO Model

*   **Using the command line:**
    ```bash
    yolo task=detect mode=train epochs=500 data=data_custom.yaml model=yolov8m.pt imgsz=640 batch=8
    ```
    *   **Explanation:**
        *   `task=detect`: Specifies that you are performing object detection.
        *   `mode=train`: Specifies that you are training a model.
        *   `epochs=500`: Specifies the number of training epochs. Adjust this based on your needs.
        *   `data=data_custom.yaml`: Specifies the path to your data configuration file.
        *   `model=yolov8m.pt`: Specifies the pre-trained YOLOv8 model to use. You can choose other models (e.g., `yolov8n.pt`, `yolov8s.pt`, `yolov8l.pt`, `yolov8x.pt`) based on your needs.
        *   `imgsz=640`: Specifies the image size for training.
        *   `batch=8`: Specifies the batch size for training. Adjust this based on your GPU memory.
    *   **Best Practices:**
        *   **Epochs:** Start with a reasonable number of epochs (e.g., 500) and adjust based on your training results.
        *   **Model Choice:** Choose a model size that balances accuracy and speed. Smaller models train faster but may be less accurate.
        *   **Batch Size:** Adjust the batch size based on your GPU memory. If you run out of memory, reduce the batch size.

## 13. Run YOLO Custom Model on New Videos (with Labeled Predictions)

*   **Using the command line:**
    ```bash
    yolo task=detect mode=predict model=runs/detect/train/weights/best.pt show=True conf=0.5 save_txt=True source=path/to/your/new_video.mp4
    ```
    *   **Explanation:**
        *   `model=runs/detect/train/weights/best.pt`: Specifies the path to your trained model weights. The path will vary depending on where you saved your training results.
        *   `show=True`: Displays the predictions on the video.
        *   `conf=0.5`: Sets the confidence threshold for detections.
        *   `save_txt=True`: Saves the detection results as text files.
        *   `source=path/to/your/new_video.mp4`: Specifies the path to your new video.

## 14. Run YOLO Custom Model on New Videos (without Labels and Confidences)

*   **Using the command line:**
    ```bash
    yolo task=detect mode=predict model='runs/detect/train/weights/best.pt' conf=0.7 save_txt=True source='path/to/your/new_video.mp4' project='path/to/your/output/folder'
    ```
    *   **Addition command line arguments to use:**
        *   `hide_labels=True`: Hides the class labels on the video.
        *   `hide_conf=True`: Hides the confidence scores on the video.
        *   `show=Ture`: Displays frame-by-frame inferences occuring in real-time (can be slow. I recommend leaving it as defualt [False] if running inference on long videos)

## Important Notes:

*   **Paths:** Make sure to replace all placeholder paths with the actual paths to your folders and files.
*   **Class Names:** Ensure that the class names in your `data_custom.yaml` file match the labels you used in LabelMe.
*   **GPU:** If you have an NVIDIA GPU, make sure you have installed the correct CUDA drivers and PyTorch with CUDA support to speed up training.
*   **Experimentation:** The training process may require some experimentation with different parameters (e.g., epochs, model size, batch size) to achieve the best results.
