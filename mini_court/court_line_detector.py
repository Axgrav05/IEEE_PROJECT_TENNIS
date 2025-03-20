# Import Necessary Libraries
import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
import numpy as np

class CourtLineDetector:
    def __init__(self, model_path):
        """
        Initializes the keypoint detection model.

        Parameters:
        - model_path (str): Path to the trained model file.
        """
        # Load a pre-trained ResNet-50 model
        self.model = models.resnet50(pretrained=True)
        
        # Modify the fully connected layer to output 28 values (14 keypoints with (x, y) coordinates)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14 * 2)

        # Load the trained weights
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()  # Set the model to evaluation mode

        # Define image preprocessing transformations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # Resize image to match model input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
        ])

    def predict(self, image):
        """
        Predicts keypoints from an input image.

        Parameters:
        - image (numpy array): Input image in BGR format.

        Returns:
        - keypoints (numpy array): Predicted keypoints (x, y) coordinates.
        """
        # Convert BGR to RGB (OpenCV loads images in BGR format)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations and add batch dimension
        image_tensor = self.transform(image_rgb).unsqueeze(0)

        # Perform inference without computing gradients
        with torch.no_grad():
            outputs = self.model(image_tensor)

        # Convert output to NumPy array
        keypoints = outputs.squeeze().cpu().numpy()

        # Rescale keypoints back to the original image size
        original_h, original_w = image.shape[:2]
        keypoints[::2] *= original_w / 224.0  # Scale x-coordinates
        keypoints[1::2] *= original_h / 224.0  # Scale y-coordinates

        return keypoints

    def draw_keypoints(self, image, keypoints):
        """
        Draws keypoints on an image.

        Parameters:
        - image (numpy array): Input image.
        - keypoints (numpy array): Array of keypoint coordinates.

        Returns:
        - image (numpy array): Image with keypoints drawn.
        """
        for i in range(0, len(keypoints), 2):
            x, y = int(keypoints[i]), int(keypoints[i + 1])

            # Draw index number for each keypoint
            cv2.putText(image, str(i // 2), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 255), 2)

            # Draw the keypoint as a small red circle
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

        return image

    def draw_keypoints_on_video(self, video_frames, keypoints):
        """
        Draws keypoints on each frame of a video.

        Parameters:
        - video_frames (list): List of video frames (numpy arrays).
        - keypoints (numpy array): Keypoints to be drawn on each frame.

        Returns:
        - output_video_frames (list): List of frames with keypoints drawn.
        """
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)

        return output_video_frames