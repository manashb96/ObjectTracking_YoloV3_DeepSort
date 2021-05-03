# ObjectTracking_YoloV3_DeepSort
Working: object_tracker1_v1.py  
On Going:object_tracker1.py  

To function:
1. Create environment using .yml file if needed
2.  Download yolo weights
3.  Convert weights to .tf file using 
   ```
    #yolov3
    python load_weights.py  
    #yolov3-tiny  
    python load_weights.py --weights ./weights/yolov3-tiny.weights --output ./weights/yolov3-tiny.tf --tiny
   ```
 4. Make sure to choose which algorithm to use in object_tracker1.py
 5. Run object_tracker1.py
 ```
 python object_tracker.py --video ./data/video/<VIDEO_FILE_NAME> --output ./data/video/<OUTPUT_VIDEO_FILE_NAME>
 ```

On going project of Object Tracking using YOLOv3 and YOLOv3-tiny along with deepsort to track objects  
TODO:
1. Determine way to count object moving in different direction
2. Determine way to find out speed of object
