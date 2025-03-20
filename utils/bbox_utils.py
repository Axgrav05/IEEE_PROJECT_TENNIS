def get_center_of_bbox(bbox):
    """
    Calculates the center point of a bounding box.

    Parameters:
    - bbox (tuple): (x1, y1, x2, y2) coordinates of the bounding box.

    Returns:
    - tuple: (center_x, center_y)
    """
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y

def measure_distance(p1, p2):
    """
    Computes the Euclidean distance between two points.

    Parameters:
    - p1, p2 (tuple): (x, y) coordinates of the two points.

    Returns:
    - float: Distance between the points.
    """
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def get_foot_position(bbox):
    """
    Determines the foot (bottom-center) position of a bounding box.

    Parameters:
    - bbox (tuple): (x1, y1, x2, y2) coordinates of the bounding box.

    Returns:
    - tuple: (x, y) representing the bottom-center of the bounding box.
    """
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)

def get_closest_keypoint_index(point, keypoints, keypoint_indices):
   """
    Finds the index of the closest keypoint to a given point, based on vertical distance.

    Parameters:
    - point (tuple): (x, y) coordinates of the reference point.
    - keypoints (list): List of keypoint coordinates [x1, y1, x2, y2, ...].
    - keypoint_indices (list): Indices of the keypoints to consider.

    Returns:
    - int: Index of the closest keypoint.
    """
   closest_distance = float('inf')
   key_point_ind = keypoint_indices[0]
   for keypoint_indix in keypoint_indices:
       keypoint = keypoints[keypoint_indix*2], keypoints[keypoint_indix*2+1]
       distance = abs(point[1]-keypoint[1])

       if distance<closest_distance:
           closest_distance = distance
           key_point_ind = keypoint_indix
    
   return key_point_ind

def get_height_of_bbox(bbox):
    """
    Calculates the height of a bounding box.

    Parameters:
    - bbox (tuple): (x1, y1, x2, y2) coordinates of the bounding box.

    Returns:
    - int: Height of the bounding box.
    """
    return bbox[3]-bbox[1]

def measure_xy_distance(p1,p2):
    """
    Computes the absolute x and y distance between two points.

    Parameters:
    - p1, p2 (tuple): (x, y) coordinates of the two points.

    Returns:
    - tuple: (x_distance, y_distance)
    """
    return abs(p1[0]-p2[0]), abs(p1[1]-p2[1])

def get_center_of_bbox(bbox):
    return (int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2))