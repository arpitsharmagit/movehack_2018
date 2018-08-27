import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2 as cv;

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = os.path.join('data', 'frozen_inference_graph_15K.pb')
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'label_map_15k.pbtxt')
NUM_CLASSES = 13

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

IMAGE_PATH = 'input/'
SAVE_PATH = "output/"
TEST_IMAGE_PATHS = [os.path.join(r,file) for r,d,f in os.walk(IMAGE_PATH) for file in f]

# Size, in inches, of the output images.

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    for image_path in TEST_IMAGE_PATHS:
      print('Input Image => {}'.format(image_path))
      image = cv.imread(image_path,cv.IMREAD_COLOR )
      encode_param = [int(cv.IMWRITE_JPEG_QUALITY), 90]
      result, encimg = cv.imencode('.jpg', image, encode_param)
      print('Decoding Image => {}'.format(result))
      image_np = cv.imdecode(encimg, 1)	
      image_np_expanded = np.expand_dims(image_np, axis=0)
      print("Detecting...")
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      print("Labeling image...")
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=2)
      pngResult,pngEncodedImg = cv.imencode(".png", image_np)
      print('Encoding Image => {}'.format(pngResult));
      save_path =os.path.join(SAVE_PATH, os.path.basename(os.path.normpath(image_path)))
      cv.imwrite(save_path,image_np)
      print('Saving Image => {}'.format(save_path)) 
      print('                               ') 
