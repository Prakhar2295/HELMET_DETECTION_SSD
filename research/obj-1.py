import numpy as np
import os
#import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import base64

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt

from PIL import Image

def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath,"rb") as f:
        return base64.b64encode(f.read())


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'my_model'
#MODEL_FILE = MODEL_NAME + '.tar.gz'
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = 'research/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('research','data', 'labelmap.pbtxt')
NUM_CLASSES = 1
#print(PATH_TO_FROZEN_GRAPH)
#print(PATH_TO_LABELS)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
#print(category_index)
class_names_mapping = { 1: "Helmets"}
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    #print(serialized_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')


sess= tf.Session(graph=detection_graph)

image = cv2.imread("research/inputImage.jpg")
#image = tf.gfile.FastGFile('research\inputImage.jpg', 'rb').read()
#image = tf.image.decode_jpeg(image, channels=3)
image_expanded = np.expand_dims(image, axis=0)
print(image.shape)
print(image_expanded.shape)
#img_tensor = tf.cast(image_expanded, dtype=tf.float32)
#print(img_tensor)

(boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

result = scores.flatten()
#print(len(result))
res = []
for idx in range(0, len(result)):
            if result[idx] > .40:
                res.append(idx)
#print(res)
top_classes = classes.flatten()
        # Selecting class 2 and 3
        #top_classes = top_classes[top_classes > 1]
res_list = [top_classes[i] for i in res]
#print(top_classes)
#print(res_list)
class_final_names = [class_names_mapping[x] for x in res_list]
top_scores = [e for l2 in scores for e in l2 if e > 0.30]
final_output = list(zip(class_final_names, top_scores))
#print(class_final_names)
#print(top_scores)
#print(final_output)

new_scores = scores.flatten()
#print(new_scores)        ##print(new_scores)

new_boxes = boxes.reshape(300, 4)
#print(new_boxes)
        # get all boxes from an array
max_boxes_to_draw = new_boxes.shape[0]
print(max_boxes_to_draw)
# this is set as a default but feel free to adjust it to your needs
min_score_thresh = .30

listOfOutput = []
for (name, score, i) in zip(class_final_names, top_scores, range(min(max_boxes_to_draw, new_boxes.shape[0]))):
    valDict = {}
    valDict["className"] = name
    valDict["confidence"] = str(score)
    if new_scores is None or new_scores[i] > min_score_thresh:
        val = list(new_boxes[i])
        valDict["yMin"] = str(val[0])
        valDict["xMin"] = str(val[1])
        valDict["yMax"] = str(val[2])
        valDict["xMax"] = str(val[3])
        listOfOutput.append(valDict)
print(listOfOutput)

vis_util.visualize_boxes_and_labels_on_image_array(
             image,
             np.squeeze(boxes),
             np.squeeze(classes).astype(np.int32),
             np.squeeze(scores),
             category_index,
             use_normalized_coordinates=True,
             line_thickness=8,
             min_score_thresh=0.4)
#output_filename = 'output4.jpg'
#cv2.imwrite(output_filename,image)
#openencodebase64= encodeImageIntoBase64('output4.jpg')

#cv2.imshow('Object detector', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()