import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2 as cv
import time

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from cv2 import VideoWriter, VideoWriter_fourcc

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

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

VIDEO_PATH = 'input/Mumbai.mp4'
SAVE_PATH = "Mumbai_Demo.mp4"
fourcc = VideoWriter_fourcc(*"MP4V")
fps=30.0
image_size=(1280,720)

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

    print('[INFO] Starting video feed...')
    video_capture = cv.VideoCapture(VIDEO_PATH)
    vid = VideoWriter(os.path.join(os.path.abspath("output"), SAVE_PATH),fourcc, fps, image_size)
    flag = True
    count = 0;
    while(flag):
        flag, frame = video_capture.read()
        # time.sleep(0.2)
        if flag==True:            
            image_np = cv.cvtColor(frame, cv.COLOR_RGB2BGR)            
            cv.imshow('Video', image_np)            	
            image_np_expanded = np.expand_dims(image_np, axis=0)
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=2)
            final_image =  cv.cvtColor(image_np, cv.COLOR_RGB2BGR)
            cv.imshow('Video', final_image)             
            save_path =os.path.join(SAVE_PATH, os.path.basename(os.path.normpath("frame%d.jpg" % count)))
            vid.write(final_image)
            # cv.imwrite(save_path,final_image)  
            count=count+1
        else:
            break
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    print('[INFO] Terminating Video feed...')
    video_capture.release()
    cv.destroyAllWindows()
