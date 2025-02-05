import numpy as np
import cv2


def get_center_point(coordinate_dict):
    di = dict()

    for key in coordinate_dict.keys():
        xmin, ymin, xmax, ymax = coordinate_dict[key]
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        di[key] = (x_center, y_center)

    return di


def find_miss_corner(coordinate_dict):
    position_name = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    position_index = np.array([0, 0, 0, 0])

    for name in coordinate_dict.keys():
        if name in position_name:
            position_index[position_name.index(name)] = 1

    index = np.argmin(position_index)

    return index


def calculate_missed_coord_corner(coordinate_dict):
    thresh = 0

    index = find_miss_corner(coordinate_dict)

    # calculate missed corner coordinate
    # case 1: missed corner is "top_left"
    if index == 0:
        midpoint = np.add(coordinate_dict['top_right'], coordinate_dict['bottom_left']) / 2
        y = 2 * midpoint[1] - coordinate_dict['bottom_right'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['bottom_right'][0] - thresh
        coordinate_dict['top_left'] = (x, y)
    elif index == 1:  # "top_right"
        midpoint = np.add(coordinate_dict['top_left'], coordinate_dict['bottom_right']) / 2
        y = 2 * midpoint[1] - coordinate_dict['bottom_left'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['bottom_left'][0] - thresh
        coordinate_dict['top_right'] = (x, y)
    elif index == 2:  # "bottom_left"
        midpoint = np.add(coordinate_dict['top_left'], coordinate_dict['bottom_right']) / 2
        y = 2 * midpoint[1] - coordinate_dict['top_right'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['top_right'][0] - thresh
        coordinate_dict['bottom_left'] = (x, y)
    elif index == 3:  # "bottom_right"
        midpoint = np.add(coordinate_dict['bottom_left'], coordinate_dict['top_right']) / 2
        y = 2 * midpoint[1] - coordinate_dict['top_left'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['top_left'][0] - thresh
        coordinate_dict['bottom_right'] = (x, y)

    return coordinate_dict


def perspective_transform(image, source_points):
    dest_points = np.float32([[0, 0], [500, 0], [500, 300], [0, 300]])
    M = cv2.getPerspectiveTransform(source_points, dest_points)
    dst = cv2.warpPerspective(image, M, (500, 300))

    return dst


def align_image(image, coordinate_dict):
    if len(coordinate_dict) < 3:
        raise ValueError('Please try again')

    # convert (xmin, ymin, xmax, ymax) to (x_center, y_center)
    coordinate_dict = get_center_point(coordinate_dict)

    if len(coordinate_dict) == 3:
        coordinate_dict = calculate_missed_coord_corner(coordinate_dict)

    top_left_point = coordinate_dict['top_left']
    top_right_point = coordinate_dict['top_right']
    bottom_right_point = coordinate_dict['bottom_right']
    bottom_left_point = coordinate_dict['bottom_left']

    source_points = np.float32([top_left_point, top_right_point, bottom_right_point, bottom_left_point])

    # transform image and crop
    crop = perspective_transform(image, source_points)

    return crop

def non_max_suppression_fast(boxes, scores, iou_threshold):
    """
    Perform Non-Maximum Suppression (NMS) on bounding boxes.
    
    Args:
        boxes (list): List of bounding boxes in the format [y_min, x_min, y_max, x_max].
        scores (list): List of confidence scores corresponding to each bounding box.
        iou_threshold (float): IoU threshold to use for NMS.
        
    Returns:
        List: Indices of boxes to keep after NMS.
    """
    if len(boxes) == 0:
        return []

    # Convert boxes to numpy array format
    boxes = np.array(boxes)
    scores = np.array(scores)

    # Extract coordinates for each box
    y_min = boxes[:, 0]
    x_min = boxes[:, 1]
    y_max = boxes[:, 2]
    x_max = boxes[:, 3]

    # Compute the area of each bounding box
    areas = (y_max - y_min) * (x_max - x_min)

    # Sort the bounding boxes by the scores in descending order
    order = np.argsort(scores)[::-1]

    keep_indices = []  # List of indices to keep

    while order.size > 0:
        # Get the index of the box with the highest score
        current_idx = order[0]
        keep_indices.append(current_idx)

        # Compute the coordinates of the intersection area
        yy_min = np.maximum(y_min[current_idx], y_min[order[1:]])
        xx_min = np.maximum(x_min[current_idx], x_min[order[1:]])
        yy_max = np.minimum(y_max[current_idx], y_max[order[1:]])
        xx_max = np.minimum(x_max[current_idx], x_max[order[1:]])

        # Compute the width and height of the intersection area
        width = np.maximum(0, xx_max - xx_min)
        height = np.maximum(0, yy_max - yy_min)

        # Compute the area of the intersection
        intersection_area = width * height

        # Compute the IoU (Intersection over Union)
        iou = intersection_area / (areas[current_idx] + areas[order[1:]] - intersection_area)

        # Keep boxes with IoU less than the threshold
        indices_to_keep = np.where(iou <= iou_threshold)[0]

        # Update order by only keeping indices with IoU less than the threshold
        order = order[indices_to_keep + 1]  # +1 because we already removed the current_idx

    return keep_indices




def sort_text(detection_boxes, detection_labels):
    detection_labels = np.array(detection_labels)
    id_boxes = detection_boxes[detection_labels == 1]
    name_boxes = detection_boxes[detection_labels == 2]
    birth_boxes = detection_boxes[detection_labels == 3]
    home_boxes = detection_boxes[detection_labels == 4]
    add_boxes = detection_boxes[detection_labels == 5]

    # arrange boxes
    id_boxes = sort_each_category(id_boxes)
    name_boxes = sort_each_category(name_boxes)
    birth_boxes = sort_each_category(birth_boxes)
    home_boxes = sort_each_category(home_boxes)
    add_boxes = sort_each_category(add_boxes)

    return id_boxes, name_boxes, birth_boxes, home_boxes, add_boxes


def get_y1(x):
    return x[0]


def get_x1(x):
    return x[1]


def sort_each_category(category_text_boxes):
    min_y1 = min(category_text_boxes, key=get_y1)[0]

    mask = np.where(category_text_boxes[:, 0] < min_y1 + 10, True, False)
    line1_text_boxes = category_text_boxes[mask]
    line2_text_boxes = category_text_boxes[np.invert(mask)]

    line1_text_boxes = sorted(line1_text_boxes, key=get_x1)
    line2_text_boxes = sorted(line2_text_boxes, key=get_x1)

    if len(line2_text_boxes) != 0:
        merged_text_boxes = [*line1_text_boxes, *line2_text_boxes]
    else:
        merged_text_boxes = line1_text_boxes

    return merged_text_boxes
