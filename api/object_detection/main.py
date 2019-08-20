import numpy as np
import os
import tensorflow as tf
import imutils
import sys
from api.object_detection.predict import detect_max_mask
import cv2

PATH_TO_FROZEN_GRAPH = "api/object_detection/frozen_graph_with_rotate.pb"

detection_graph = tf.Graph()
with detection_graph.as_default():
    detection_sess = tf.Session()
    with detection_sess.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

def reframe_box_masks_to_image_masks(box_masks, boxes, image_height,
                                     image_width):
    def reframe_box_masks_to_image_masks_default():
        """The default function when there are more than 0 box masks."""
        def transform_boxes_relative_to_boxes(boxes, reference_boxes):
            boxes = tf.reshape(boxes, [-1, 2, 2])
            min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
            max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
            transformed_boxes = (boxes - min_corner) / (max_corner - min_corner)
            return tf.reshape(transformed_boxes, [-1, 4])

        box_masks_expanded = tf.expand_dims(box_masks, axis=3)
        num_boxes = tf.shape(box_masks_expanded)[0]
        unit_boxes = tf.concat(
            [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
        reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)
        return tf.image.crop_and_resize(
            image=box_masks_expanded,
            boxes=reverse_boxes,
            box_ind=tf.range(num_boxes),
            crop_size=[image_height, image_width],
            extrapolation_value=0.0)
    image_masks = tf.cond(
        tf.shape(box_masks)[0] > 0,
        reframe_box_masks_to_image_masks_default,
        lambda: tf.zeros([0, image_height, image_width, 1], dtype=tf.float32))
    return tf.squeeze(image_masks, axis=3)

def run_inference_for_single_image(image, graph):
    with graph.as_default():
    # with tf.Session() as sess:
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
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
        if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        # Run inference
        output_dict = detection_sess.run(tensor_dict,
                                feed_dict={image_tensor: np.expand_dims(image, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

def get_roi(image_ori):
    if image_ori.shape[0] > image_ori.shape[1]:
        image_np = imutils.resize(image_ori, height=600)
    else:
        image_np = imutils.resize(image_ori, width=600)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
    
    height, width, _ = image_ori.shape
    boxes = output_dict['detection_boxes']

    stretch_ratio = int(25 / image_np.shape[1] * image_ori.shape[1])
    boxes = [
        int(boxes[0,0]*height),
        int(boxes[0,1]*width),
        int(boxes[0,2]*height),
        int(boxes[0,3]*width)
    ]

    # sketch boxes. 
    if boxes[3]-boxes[1] > boxes[2]-boxes[0]:
        boxes[3] += stretch_ratio
        boxes[1] = max(boxes[1]-stretch_ratio, 0)
    else:
        boxes[2] += stretch_ratio
        boxes[0] = max(boxes[0]-stretch_ratio, 0)


    box_detected = image_ori[boxes[0]:boxes[2], boxes[1]:boxes[3]]
    # import time
    # cv2.imwrite("{}.jpg".format(time.time()), box_detected)
    # get contour of mask
    masks = output_dict['detection_masks']
    masks = np.transpose(masks, (1,2,0))
    # resize masks 
    masks = imutils.resize(masks, height=image_ori.shape[0])
    # draw masks on image
    image_draw_mask = image_ori.copy()
    image_draw_mask[:,:,1] = masks*255


    masks = masks[boxes[0]:boxes[2], boxes[1]:boxes[3]]
    # dilate masks,
    # kernel = np.ones((5,20),np.uint8)
    # masks = cv2.dilate(masks, kernel,iterations = 1)
    # 
    cnts = cv2.findContours(masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnt = max(cnts, key = cv2.contourArea)
    
    # 
    if boxes[3]-boxes[1] > boxes[2]-boxes[0]:
        # sketch x
        center_x = cnt[:,0,0].mean()
        center_y = cnt[:,0,1].mean()
        cnt[:,0,0][cnt[:,0,0]<center_x] -= stretch_ratio
        cnt[:,0,0][cnt[:,0,0]<0] = 0
        cnt[:,0,0][cnt[:,0,0]>center_x] += stretch_ratio
        cnt[:,0,0][cnt[:,0,0]>=masks.shape[1]-1] = masks.shape[1] - 1
    else:
        center_x = cnt[:,0,0].mean()
        center_y = cnt[:,0,1].mean()
        cnt[:,0,1][cnt[:,0,1]<center_y] -= stretch_ratio
        cnt[:,0,1][cnt[:,0,1]<0] = 0 
        cnt[:,0,1][cnt[:,0,1]>center_y] += stretch_ratio
        cnt[:,0,1][cnt[:,0,1]>=masks.shape[0]-1] = masks.shape[0] - 1
    
    try:
        box_rotated = detect_max_mask(box_detected, cnt)
    except Exception as err:
        print('Error rotate')
        print(err)
        box_rotated = box_detected

    # 
    if box_rotated.shape[1] < box_rotated.shape[0]:
        box_rotated = imutils.rotate_bound(box_rotated, -90)
    return box_rotated, image_draw_mask

