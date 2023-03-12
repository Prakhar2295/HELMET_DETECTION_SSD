
# Steps to setup TFOD1.x environment in Local PC

### Step 1 :-
####  Download Repository as zip
https://github.com/tensorflow/models/tree/v1.13.0

OR

`wget https://github.com/tensorflow/models/archive/v1.13.0.zip`

### Step 2 :-
#### Creating virtual env using conda

`conda create -n your_env_name python=3.6`

`conda activate your_env_name`

### Step 3 :-
#### Installing necessary Packages for TFOD1.x

`pip install pillow lxml Cython contextlib2 jupyter matplotlib pandas opencv-python tensorflow==1.14.0`

#### Installing protobuf using conda package manager
`conda install -c anaconda protobuf`

### Step 4 :-
#### Conversion of Protos files to Python files

#### Execute From Research Folder

##### Linux or Mac
`protoc object_detection/protos/*.proto --python_out=.`

##### Windows
`protoc object_detection/protos/*.proto --python_out=.`

### Step 5 :-
#### Execute From Research Folder
#### Install Object Detection

`python setup.py install`

### Step 6 :-
##### Command to open Jupiter notebook from terminal.

`jupyter notebook`

then select the notebook   <i><b>object_detection_tutorial.ipynb</b></i>



### Other Important Links
#### Model Zoo Link
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md

### Working with Webcam

##### Code to get the pop camera
 ```python
import cv2

cap = cv2.VideoCapture(0)
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      ret, image_np = cap.read()
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)

      cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
```
Finally in the next cell please execute the next Command

`cap.release()`      
