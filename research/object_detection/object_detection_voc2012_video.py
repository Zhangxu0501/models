#coding:utf8
import numpy as np
import sys
import tensorflow as tf
from PIL import Image
import cv2
import imutils

sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util

def load_image_into_numpy_array(image):
    (im_width,im_height)=image.size
    return np.array(image.getdata()).reshape(im_height,im_width,3).astype(np.uint8)

NUM_CLASS=20

detection_graph=tf.Graph()
with detection_graph.as_default():
    od_graph_def=tf.GraphDef()
    with tf.gfile.GFile("/root/models/research/object_detection/AnzhiModelVoc2012/frozen_inference_graph.pb", 'rb') as fid:
        serialized_graph=fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap("/root/voc2012/pascal_label_map.pbtxt")
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASS, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


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



  PATH="/Users/zx142489/Desktop/01.mp4"
  camera = cv2.VideoCapture(PATH)

  while True:
      (result,frame)=camera.read()
      if result:
          image_np_expanded = np.expand_dims(frame, axis=0)
          # Actual detection.
          (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          # Visualization of the results of a detection.
          vis_util.visualize_boxes_and_labels_on_image_array(
              frame,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8)
          frame = imutils.resize(frame, width=960)
          cv2.imshow("video", frame)



          key = cv2.waitKey(1) & 0xFF
            # 按'q'健退出循环
          if key == ord('q'):
            break


  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
