from ultralytics import YOLO 
import cv2
import pickle
import sys

sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self, model_path):
        """
        Initializes the PlayerTracker class with a YOLO model.

        Parameters:
        - model_path (str): Path to the trained YOLO model.
        """
        self.model = YOLO(model_path)

    def choose_and_filter_player(self, court_keypoints, player_detections):
        """
        Selects the two closest players to the court keypoints and filters their detections.

        Parameters:
        - court_keypoints (list): List of key points defining the court.
        - player_detections (list of dicts): Player detections per frame.

        Returns:
        - list of dicts: Filtered player detections for each frame.
        """
        # Get first frame player detections
        player_detections_first_frame = player_detections[0]

        # Isolate players 
        chosen_player = self.choose_player(court_keypoints, player_detections_first_frame)

        # Filter detections to keep only the chosen players
        player_detections_filtered = []

        for player_dict in player_detections:
            filtered_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            player_detections_filtered.append(filtered_dict)
        return player_detections_filtered

    def choose_player(self, court_keypoints, player_dict):
        """
        Determines the two closest players to the court keypoints.

        Parameters:
        - court_keypoints (list): List of court key points.
        - player_dict (dict): Dictionary of player detections in the format {track_id: bbox}.

        Returns:
        - list: Track IDs of the two selected players.
        """
        distances = []
        
        # Calculate the minimum distance of each player from court keypoints
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)

            min_distance = float('inf')
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))

        # Sort players by their proximity to the court and select the two closest
        distances = sorted(distances, key=lambda x: x[1])
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players



    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """
        Detects players in each frame or loads precomputed detections from a file.

        Parameters:
        - frames (list of numpy arrays): List of video frames.
        - read_from_stub (bool): Whether to load detections from a saved file.
        - stub_path (str): Path to the saved detections file.

        Returns:
        - list of dicts: Player detections for each frame.
        """
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections
        
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame):
        """
        Detects players in a single frame using YOLO.

        Parameters:
        - frame (numpy array): Input video frame.

        Returns:
        - dict: Detected players with track IDs as keys and bounding boxes as values.
        """
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0]) # Extract tracking ID
            result = box.xyxy.tolist()[0] # Extract bounding box coordinates
            object_cls_id = box.cls.tolist()[0] # Extract object class ID
            object_cls_name = id_name_dict[object_cls_id] # Extract object class name

            if object_cls_name == "person":
                player_dict[track_id] = result # Add player to dictionary

        return player_dict
    
    def draw_bounding_boxes(self, video_frames, player_detections):
        """
        Draws bounding boxes around detected players in video frames.

        Parameters:
        - video_frames (list of numpy arrays): List of video frames.
        - player_detections (list of dicts): Detected player positions.

        Returns:
        - list of numpy arrays: Frames with bounding boxes drawn.
        """
        output_frames = []

        # Draw bounding box and player ID label 
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            output_frames.append(frame)

        return output_frames