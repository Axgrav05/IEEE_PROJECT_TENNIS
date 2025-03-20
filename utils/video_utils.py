import cv2

def read_video(video_path):
    """
    Reads a video file and extracts its frames.

    Parameters:
    - video_path (str): Path to the input video file.

    Returns:
    - list: A list of frames (each frame is a NumPy array).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    # Read all frames from input video
    while True:
        ret, frame = cap.read()
        if not ret: # If there are no more frames to read, exit loop
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    """
    Saves a list of frames as a video file.

    Parameters:
    - output_video_frames (list): List of frames (NumPy arrays) to save.
    - output_video_path (str): Path to save the output video.
    - fps (int, optional): Frames per second. Default is 30.

    Returns:
    - None
    """

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    
    # Write each frame to the video
    for frame in output_video_frames:
        out.write(frame)
    out.release()