# By Bend_Function
# 可以放在任何文件夹下运行（前提正确配置API[环境变量]）
# 输出视频没有声音，pr可解决一切

import numpy as np
import os
import sys
import tensorflow as tf
import cv2
import time

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

start = time.time()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
cv2.setUseOptimized(True)           # 加速cv

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# 可能要改的内容
######################################################
PATH_TO_CKPT = 'model\\ssd_mobilenet_v1_graph.pb'   # 模型及标签地址
PATH_TO_LABELS = 'model\\mscoco_label_map.pbtxt'

video_PATH = "test_video\\cycling.mp4"              # 要检测的视频
out_PATH = "OUTPUT\\out_cycling1.mp4"            # 输出地址

NUM_CLASSES = 90            # 检测对象个数

fourcc = cv2.VideoWriter_fourcc(*'MJPG')            # 编码器类型（可选）
# 编码器： DIVX , XVID , MJPG , X264 , WMV1 , WMV2

######################################################

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
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# 读取视频
video_cap = cv2.VideoCapture(video_PATH)  
fps = int(video_cap.get(cv2.CAP_PROP_FPS))    # 帧率


width = int(video_cap.get(3))         # 视频长，宽
hight = int(video_cap.get(4))


videoWriter = cv2.VideoWriter(out_PATH, fourcc, fps, (width, hight)) 

config = tf.ConfigProto()
config.gpu_options.allow_growth = True    # 减小显存占用
with detection_graph.as_default():
  with tf.Session(graph=detection_graph, config=config) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    num = 0
    while True:
        ret, frame = video_cap.read()
        if ret == False:        # 没检测到就跳出
            break
        num += 1
        print(num)  # 输出检测到第几帧了
        # print(num/fps) # 检测到第几秒了
        
        image_np = frame

        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=4)

        # 写视频
        videoWriter.write(image_np)
        
videoWriter.release()
end = time.time()
print("Execution Time: ", end - start)
