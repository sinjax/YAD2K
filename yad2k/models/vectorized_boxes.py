import numpy as np
import keras.backend as K

from yad2k.models.keras_yolo import preprocess_true_boxes

true_boxes = np.array(
    [[0.16, 0.51355422, 0.296, 0.68975904, 14.],
     [0.428, 0.47590361, 0.276, 0.60240964, 14.],
     [0.664, 0.5813253, 0.432, 0.8373494, 14.],
     [0.863, 0.64457831, 0.274, 0.7108433, 14]])

image_size = (416, 416)
anchors = np.array([
    [0.738768, 0.874946],
    [2.42204, 2.65704],
    [4.30971, 7.04493],
    [10.246, 4.59428],
    [12.6868, 11.8741]
])
actual_dm, actual_mtb = preprocess_true_boxes(true_boxes, anchors, image_size)

def numpy_ptb():

    height, width = image_size
    num_anchors = len(anchors)
    # Downsampling factor of 5x 2-stride max_pools == 32.
    assert height % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    assert width % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    conv_height = height // 32
    conv_width = width // 32
    detectors_mask = np.zeros((conv_height, conv_width, num_anchors, 1))
    matching_true_boxes = np.zeros((conv_height, conv_width, num_anchors, 5))

    boxes_i = np.floor(true_boxes[:,1] * conv_height).astype('int')
    boxes_j = np.floor(true_boxes[:,0] * conv_width).astype('int')

    box_maxes = true_boxes[:,2:4] * np.array([conv_width, conv_height]) / 2.
    box_mins = -box_maxes
    anchor_maxes = (anchors / 2.)
    anchor_mins = -anchor_maxes

    box_mins = np.repeat(np.expand_dims(box_mins,1),anchor_mins.shape[0], 1)
    box_maxes = np.repeat(np.expand_dims(box_maxes,1),anchor_mins.shape[0], 1)
    intersect_mins = np.maximum(box_mins, anchor_mins)
    intersect_maxes = np.minimum(box_maxes, anchor_maxes)

    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[:,:,0] * intersect_wh[:,:,1]
    box_area = np.expand_dims(true_boxes[:,2] * true_boxes[:,3], 1)
    anchor_area = anchors[:,0] * anchors[:,1]
    iou = intersect_area / (box_area + anchor_area - intersect_area)

    max_anchor_index = iou.argmax(1)
    box_indexes = np.arange(0,true_boxes.shape[0])

    valid_anchor_index = iou[box_indexes, max_anchor_index] > 0

    valid_box_indexes = box_indexes[valid_anchor_index]
    valid_max_anchor_index = max_anchor_index[valid_anchor_index]
    valid_i = boxes_i[valid_box_indexes]
    valid_j = boxes_j[valid_box_indexes]

    detectors_mask[valid_i, valid_j, valid_max_anchor_index] = 1
    adjusted_boxes = np.array([
        true_boxes[valid_box_indexes,0] * conv_width - valid_j, true_boxes[valid_box_indexes,1] * conv_height - valid_i,
        np.log(true_boxes[valid_box_indexes,2] * conv_width / anchors[valid_max_anchor_index,0]),
        np.log(true_boxes[valid_box_indexes,3] * conv_height / anchors[valid_max_anchor_index,1]),
        true_boxes[valid_box_indexes,4]
    ]).T
    matching_true_boxes[valid_i, valid_j, valid_max_anchor_index] = adjusted_boxes

    assert np.all(detectors_mask == actual_dm)
    assert np.all(matching_true_boxes == actual_mtb)

def tf_ptb():
    height, width = image_size
    num_anchors = len(anchors)
    # Downsampling factor of 5x 2-stride max_pools == 32.
    assert height % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    assert width % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    conv_height = height // 32
    conv_width = width // 32
    detectors_mask = K.zeros((conv_height, conv_width, num_anchors, 1))
    matching_true_boxes = K.zeros((conv_height, conv_width, num_anchors, 5))

    boxes_i = K.cast(K.tf.floor(true_boxes[:, 1] * conv_height), "int32")
    boxes_j = K.cast(K.tf.floor(true_boxes[:, 0] * conv_width), "int32")

    box_maxes = true_boxes[:, 2:4] * K.variable(np.array([conv_width, conv_height])) / 2.
    box_mins = -box_maxes
    anchor_maxes = (anchors / 2.)
    anchor_mins = -anchor_maxes

    box_mins = K.tile(K.expand_dims(box_mins, 1), [1,K.shape(anchor_mins)[0],1])
    box_maxes = K.tile(K.expand_dims(box_maxes, 1), [1,K.shape(anchor_mins)[0], 1])
    intersect_mins = K.maximum(box_mins, anchor_mins)
    intersect_maxes = K.minimum(box_maxes, anchor_maxes)

    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[:, :, 0] * intersect_wh[:, :, 1]
    box_area = K.expand_dims(true_boxes[:, 2] * true_boxes[:, 3], 1)
    anchor_area = anchors[:, 0] * anchors[:, 1]
    iou = K.cast(intersect_area, "float32") / K.cast((K.cast(box_area,"float32") + K.cast(anchor_area, "float32") - intersect_area),"float32")

    max_anchor_index = K.cast(K.argmax(iou,1),"int32")
    box_indexes = K.arange(0, true_boxes.shape[0])
    indecies = K.stack([box_indexes, max_anchor_index])
    indecies = K.transpose(indecies)
    valid_anchor_index = K.tf.gather_nd(iou, indecies) > 0

    valid_anchor_index_gather = K.cast(K.tf.where(valid_anchor_index), "int32")
    valid_box_indexes = K.tf.gather(box_indexes, valid_anchor_index_gather)
    valid_max_anchor_index = K.tf.gather(max_anchor_index, valid_anchor_index_gather)
    valid_i = K.tf.gather(boxes_i,valid_anchor_index_gather)
    valid_j = K.tf.gather(boxes_j, valid_anchor_index_gather)

    valid_indecies = K.stack([valid_i, valid_j, valid_max_anchor_index])
    valid_indecies = K.transpose(valid_indecies)
    print(K.get_session().run(valid_indecies))
    K.tf.assign(detectors_mask, K.tf.scatter_nd(valid_indecies, K.variable([[1,1,1]]), K.shape(detectors_mask)))
    adjusted_boxes = K.stack([
        true_boxes[valid_box_indexes, 0] * conv_width - valid_j,
        true_boxes[valid_box_indexes, 1] * conv_height - valid_i,
        K.log(true_boxes[valid_box_indexes, 2] * conv_width / anchors[valid_max_anchor_index, 0]),
        K.log(true_boxes[valid_box_indexes, 3] * conv_height / anchors[valid_max_anchor_index, 1]),
        true_boxes[valid_box_indexes, 4]
    ]).T
    matching_true_boxes[valid_i, valid_j, valid_max_anchor_index] = adjusted_boxes


    assert np.all(K.get_session().run(detectors_mask) == actual_dm)
    assert np.all(K.get_session().run(matching_true_boxes) == actual_mtb)

numpy_ptb()
tf_ptb()