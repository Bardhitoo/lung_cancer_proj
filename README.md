# lung_cancer_proj
This projects aims to explore and design an effective 3-step pipeline for lung cancer 1) detection, 2) segmentation, and 3) 3D reconstruction.

Why is this problem worth solving? 
* Problem
  * Solution
* Problem
  * Solution
* ...


# 1. Detection


### 1.0 Data  
In order to be able to give the data to detections algorithm, the data needs to be: processed, collected, standardized, and normalized.
This project uses a subset of data from Lung Image Database Consortium and Infectious Disease Research Institute ([LIDC/IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)) database,
specifically from the available online dataset of [LUNA16](https://luna16.grand-challenge.org).

This dataset consists of 888 CT scans with annotations (detection and segmentation) describing coordinates and ground truth labels.
The first step is to process the data and create an image database for training.

## 1.1 Preprocessing
### 1.1.1 Dataset processing
- Select patients with nodule >= 3mm
- Tackling the problem in 2D (image)
    - Positive_slices = X positive instances (slices from 3D CT Scan of patience with    lung cancer)
    - Negative_slices = 250-X negative instances (slices from 3D CT Scan of patience without lung cancer)


### Useful Commands
*Launch Training: 
`python model_main_tf2.py --pipeline_config_path="C:\Users\bardh\tensorflow\model_zoo\efficientdet_d0_coco17_tpu-32\pipeline.config"   --model_dir="C:\Users\bardh\tensorflow\model_zoo\efficientdet_d0_coco17_tpu-32\v0.1"   --checkpoint_every_n=1000   --alsologtostderr  --sample_1_of_n_eval_examples=1`

*Evaluate Model:
`python model_main_tf2.py --pipeline_config_path="C:\Users\bardh\tensorflow\model_zoo\efficientdet_d0_coco17_tpu-32\pipeline.config" --model_dir="C:\Users\bardh\tensorflow\model_zoo\efficientdet_d0_coco17_tpu-32\v0.3"   --checkpoint_dir="C:\Users\bardh\tensorflow\model_zoo\efficientdet_d0_coco17_tpu-32\v0.3" --run_once`

*Track Model:
`tensorboard --logdir="C:\Users\bardh\tensorflow\model_zoo\efficientdet_d0_coco17_tpu-32" // <path to a directory with your experiment / experiments>`

*Export Model:
`python exporter_main_v2.py --input_type=image_tensor --trained_checkpoint_dir="C:\Users\bardh\tensorflow\model_zoo\efficientdet_d0_coco17_tpu-32\v0.6_3D_2_gamma_2.5_alpha_0.75" --pipeline_config_path="C:\Users\bardh\tensorflow\model_zoo\efficientdet_d0_coco17_tpu-32\pipeline_3D_2.config" --output_directory="C:\Users\bardh\tensorflow\model_zoo\efficientdet_d0_coco17_tpu-32\v0.6_3D_2_gamma_2.5_alpha_0.75\exported_models"`
- TODO:
    - Separate positive & negative cases by patient id's. Dont want to have samples of patients split into train and test

- Figure this out:
  - ~~Why is localization_loss 0.0?~~
  - What is the difference between classification_loss and localization_loss? 
  - WARNING:tensorflow:Gradients do not exist for variables ['top_bn/gamma:0', 'top_bn/beta:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss` argument? 
  - Failed to compile generated PTX with ptxas. Falling back to compilation by driver.
  - ![img_3.png](img_3.png)


- Discussions:
    - Dataset preparation:
      - Doesn't know which way is up/down (Does that matter)?
      - ~~Extract bounding-boxes from segmentation map~~
        - I have diameter of positive classes, will stick with this for now.
      - Dataset is greatly imbalanced - 3 negative for every positive lung detection
      - x,y,z coordinate spacing is not consistent throughout different scans 
        
    - Training approaches for handling non-independent data (3-D):
      - Naive approach 
        - 1 slice per detection
        - 3 slices per detection
      - LSTM Object Detection? 
      - U-Net for segmentation
      - Find and follow already documented & working approaches      

    - Evaluation metrics?
      - Avg Precision/ Avg Recall

  

# Results: 
1. Iteration with independent slices
 ![img_1.png](img_1.png)

2. Iteration with independent slices
 ![img_2.png](img_2.png)

3. Iteration w/o independent slices
 ![img_4.png](img_4.png)

Sources: 
- https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/ 
- https://neptune.ai/blog/how-to-train-your-own-object-detector-using-tensorflow-object-detection-api
- https://neptune.ai/blog/tensorflow-object-detection-api-best-practices-to-training-evaluation-deployment 
- https://github.com/swethasubramanian/LungCancerDetection/
- https://github.com/vessemer/LungCancerDetection/
- https://github.com/tensorflow/tensorflow/issues/56927#issuecomment-1236291285

Libraries:
- https://simpleitk.readthedocs.io/en/master/gettingStarted.html