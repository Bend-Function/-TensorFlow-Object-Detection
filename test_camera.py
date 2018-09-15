# -*- coding: utf-8 -*-

# Imports
import time

start = time.time()
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
from matplotlib import pyplot as plt
import cmath
import math
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt

from PIL import Image, ImageDraw, ImageFont
import pandas as pd

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

os.chdir("/home/pi/tf/models-master/research/object_detection/")           
print(os.getcwd())
# Env setup
# This is needed to display the images.
# %matplotlib inline

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Object detection imports
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util



# Model preparation

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'             # 根据自己下载的改

'''
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
'''

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    # Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


mv = cv2.VideoCapture(0)  # 打开摄像头

width, height = 1920,1080
mv.set(3, width)     # 设置分辨率
mv.set(4, height)

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
        # TEST_IMAGE_PATHS = os.listdir(os.path.join(PATH_TO_TEST_IMAGES_DIR))
        # os.makedirs(output_image_path + image_folder)
        ii = 0
        while True:
            ii = ii + 1     # 保存图片的文件前名称
            ret, image_source = mv.read()  # 读取视频帧

            image_np = cv2.resize(image_source , (width, height), interpolation=cv2.INTER_CUBIC)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            name = "/home/pi/tf/img/" + str(ii) + ".png"        # 根据自己的需要改路径
            print(name)
            cv2.imwrite(name,image_np)
            s_boxes = boxes[scores > 0.5]
            s_classes = classes[scores > 0.5]
            s_scores = scores[scores > 0.5]

            for i in range(len(s_classes)):
                ymin = s_boxes[i][0] * height  # ymin
                xmin = s_boxes[i][1] * width  # xmin
                ymax = s_boxes[i][2] * height  # ymax
                xmax = s_boxes[i][3] * width  # xmax
                if s_classes[i] in category_index.keys():
                    class_name = category_index[s_classes[i]]['name']     # 得到英文class名称

                print("name = ", class_name, "--location =", int((ymin+xmin) / 2), ",", int((ymax + xmax) / 2))
            print("-------------------------------------")
end = time.time()
print("Execution Time: ", end - start)
