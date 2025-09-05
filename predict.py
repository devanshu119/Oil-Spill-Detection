# scripts/predict.py

import os
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any

"""
This module contains the OilSpillDetector class, which encapsulates a pre-trained
Keras model for oil spill detection. It handles model loading and provides a
method for making predictions on preprocessed SAR images.
"""

class OilSpillDetector:
    """A class to load a Keras model and perform oil spill detection."""

    def __init__(self, model_path: str):
        """
        Initializes the detector by loading the pre-trained model.
        
        Args:
            model_path (str): The file path to the saved Keras model (e.g., 'model.keras').
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> tf.keras.Model:
        """
        Loads the Keras model from disk.
        
        For inference-only purposes, we set `compile=False`. This is a crucial
        optimization as it prevents the loading of the optimizer state and loss
        functions, which are not needed for prediction. This saves memory,
        reduces load time, and avoids potential errors if the model used
        custom loss functions during training.
        
        Args:
            model_path (str): The file path to the saved Keras model.
            
        Returns:
            tf.keras.Model: The loaded Keras model instance.
        """
        print(f"Loading model from {model_path}...")
        try:
            # Using compile=False for faster loading and prediction-only use cases
            model = tf.keras.models.load_model(model_path, compile=False)
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def make_prediction(self, image_array: np.ndarray, conf_threshold: float = 0.5) -> List:
        """
        Makes a prediction on a preprocessed image array.
        
        This function assumes the model is an object detector that outputs
        bounding boxes, confidence scores, and classes. The output format
        may need to be adjusted based on the specific model architecture.
        
        Args:
            image_array (np.ndarray): The preprocessed image tensor of shape (1, H, W, 1).
            conf_threshold (float): The confidence score threshold to filter detections.
            
        Returns:
            List]: A list of detected objects. Each object is a dictionary
                                  containing the bounding box coordinates and the score.
                                  Example: [{'box': [x1, y1, x2, y2], 'score': 0.95}]
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Cannot make predictions.")
            
        # The model.predict() method runs the forward pass
        raw_predictions = self.model.predict(image_array)
        
        # Post-process the raw model output
        processed_predictions = self._postprocess_output(raw_predictions, conf_threshold)
        
        return processed_predictions

    def _postprocess_output(self, raw_preds: np.ndarray, conf_threshold: float) -> List:
        """
        Parses the raw output from the model into a clean, structured format.
        
        NOTE: This is a placeholder implementation. The exact logic will depend
        heavily on the specific object detection model used (e.g., YOLO, Faster R-CNN, SSD).
        This example assumes a simple format where the output is a list of detections
        [num_detections, [x1, y1, x2, y2, score, class_id]].
        
        Args:
            raw_preds (np.ndarray): The raw tensor output from model.predict().
            conf_threshold (float): The confidence score threshold.
            
        Returns:
            List]: The filtered and formatted list of detections.
        """
        detections = []
        # Assuming the output is on the first element of the batch
        preds_for_image = raw_preds 
        
        for pred in preds_for_image:
            # Assuming the score is the 5th element (index 4)
            score = pred
            if score > conf_threshold:
                box = pred[0:4].astype(int).tolist() # Bounding box coordinates
                detection = {
                    'box': box, # [x_min, y_min, x_max, y_max]
                    'score': float(score)
                }
                detections.append(detection)
                
        return detections