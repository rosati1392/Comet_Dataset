#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:56:16 2019

@author: riccardorosati
"""

import numpy as np
import os
import pandas as pd
import six.moves.urllib as urllib
import sys
import tensorflow as tf

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

model_name = '/Comet_Dataset/Code/faster_rcnn_resnet/'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
path_to_frozen_graph = os.path.join(model_name,'frozen_inference_graph.pb')
# List of the strings that is used to add correct label for each box.
path_to_labels = '/Comet_Dataset/Code/COMET_label_map.pbtxt'

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  if image.format == "PNG":
     image = image.convert('RGB')
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
  
def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]  

data_path = '/Comet_Dataset/Datasets/Dataset_A/stdrossi/'
test_labels = '/Comet_Dataset/Datasets/Dataset_A/test_labels.csv'
test_labels = pd.read_csv(test_labels)
#duplicateRowsDF = test_labels[test_labels.duplicated()]
image_names = test_labels.loc[: , 'Image name']
image_names = unique(image_names.tolist())


def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

i = 0
for image_name in image_names:
    if i == 1:
        break
    image = Image.open(os.path.join(data_path,image_name))
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      
    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
      # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks'),
          use_normalized_coordinates=True,
          groundtruth_box_visualization_color='white',
          line_thickness=4)
    
#    fig = plt.figure(figsize=(6,4))
#    ax = fig.add_axes([0,0,1,1])
#    plt.imshow(image_np)  
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,15))
    fig.subplots_adjust(hspace = .2, wspace=.2)
    ax1.imshow(image)
    ax2.imshow(image_np)
    i = i+1
    print(i)

    # iterating over the image for different objects
    for _,row in test_labels[(test_labels['Image name'] == image_name)].iterrows():
        xmin = row.xmin
        xmax = row.xmax
        ymin = row.ymin
        ymax = row.ymax      
        
        width = xmax - xmin
        height = ymax - ymin
        
        # assign different color to different classes of objects
        if row.Class == 'low':
            edgecolor = 'w'
            ax1.annotate('low', xy=(xmax-40,ymin-5), color = 'w')
        elif row.Class == 'medium':
            edgecolor = 'chartreuse'
            ax1.annotate('medium', xy=(xmax-60,ymin-5), color = 'chartreuse')
        elif row.Class == 'high':
            edgecolor = 'aqua'
            ax1.annotate('high', xy=(xmax-40,ymin-5), color = 'aqua')
            
        # add bounding boxes to the image
        rect = patches.Rectangle((xmin,ymin), width, height, edgecolor = edgecolor, facecolor = 'none')
        
        ax1.add_patch(rect)