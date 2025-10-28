#!/usr/bin/env python3
"""
FER2013 Test Using REAL Hailo-8 HEF via HailoRT Python API 
"""
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image
import time
import sys

# Redirect all output to test_emotion.log as well as stdout
import builtins
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()
sys.stdout = Tee(sys.stdout, open("test_emotion.log", "w"))

from pathlib import Path
from collections import defaultdict
from PIL import Image
import time

# Configuration
SCRIPT_DIR = Path(__file__).parent.parent  # Go up from scripts/

# Allow user to specify dataset directory via environment variable or command line
import sys
import os

if len(sys.argv) > 1:
    DATASET_DIR = Path(sys.argv[1])
else:
    DATASET_DIR = Path(os.environ.get('FER2013_DATASET_PATH', SCRIPT_DIR / "FER2013" / "test"))

EMOTION_HEF = Path("/path/to/emotion.hef")

# FER2013 emotion mapping
EMOTIONS_FER2013 = {
    "angry": "Angry",
    "disgust": "Disgust",
    "fear": "Fear",
    "happy": "Happy",
    "sad": "Sad",
    "surprise": "Surprise",
    "neutral": "Neutral"
}


MODEL_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]  
print("\n" + "="*70)
print("  FER2013 HAILO-8 HEF TEST (HailoRT Python API)")
print("="*70)
print(f"Dataset: {DATASET_DIR}")
print(f"HEF: {EMOTION_HEF}")
print(f"Labels: {MODEL_LABELS}")
print("="*70)
print("Usage: python3 test_fer2013_hailort.py [dataset_path]")
print("       or set FER2013_DATASET_PATH environment variable")
print("="*70 + "\n")

# Import HailoRT
try:
    from hailo_platform import (HEF, VDevice, HailoStreamInterface, 
                                 InferVStreams, ConfigureParams,
                                 InputVStreamParams, OutputVStreamParams,
                                 FormatType)
    print(" HailoRT Python API loaded")
except ImportError as e:
    print(f" Failed to import hailo_platform: {e}")
    print("   Install with: pip install hailort")
    exit(1)

# Load HEF and configure device
print(f"\n Loading HEF: {EMOTION_HEF}")
hef = HEF(str(EMOTION_HEF))

devices = VDevice()
print(f" Hailo device created")

configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
network_group = devices.configure(hef, configure_params)[0]
network_group_params = network_group.create_params()

print(f" Network configured: {network_group.name}")

# Get input/output info
input_vstreams_params = InputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.UINT8)
output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)

input_name = list(input_vstreams_params.keys())[0]
output_name = list(output_vstreams_params.keys())[0]

print(f" Input streams: {len(input_vstreams_params)}, name: {input_name}")
print(f" Output streams: {len(output_vstreams_params)}, name: {output_name}")

# Test dataset
confusion_matrix = defaultdict(lambda: defaultdict(int))
total_correct = 0
total_images = 0
emotion_stats = defaultdict(lambda: {"correct": 0, "total": 0})

print("\n" + "="*70)
print("STARTING DATASET TEST")
print("="*70 + "\n")

with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
    
    for emotion_folder in sorted(EMOTIONS_FER2013.keys()):
        true_emotion = EMOTIONS_FER2013[emotion_folder]
        emotion_dir = DATASET_DIR / emotion_folder
        
        if not emotion_dir.exists():
            print(f"  Skipping {emotion_folder}: directory not found")
            continue
        
        images = list(emotion_dir.glob("*.jpg")) + list(emotion_dir.glob("*.png"))
        
        if not images:
            print(f"  No images found in {emotion_folder}")
            continue
        
        print(f"\n Testing {true_emotion} ({len(images)} images)...")
        
        correct = 0
        
        for idx, img_path in enumerate(images, 1):
            total_images += 1
            
            # Load and preprocess image
            img = Image.open(img_path).convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img, dtype=np.uint8)
            
            # Add batch dimension if needed
            if len(img_array.shape) == 3:
                img_array = np.expand_dims(img_array, axis=0)
            
            # Run inference
            try:
                input_data = {input_name: img_array}
                
                with network_group.activate(network_group_params):
                    result = infer_pipeline.infer(input_data)
                
                # Get output
                output = result[output_name]
                
                # Process output
                if len(output.shape) > 1:
                    output = output.flatten()
                
                # Apply softmax
                output = output - np.max(output)  # Numerical stability
                exp_output = np.exp(output)
                probs = exp_output / np.sum(exp_output)
                
                # Get prediction
                predicted_idx = np.argmax(probs)
                confidence = probs[predicted_idx] * 100
                predicted_emotion = MODEL_LABELS[predicted_idx]
                
                # Check if correct
                is_correct = (predicted_emotion == true_emotion)
                
                if is_correct:
                    correct += 1
                    total_correct += 1
                    print(f"   [{total_images}] {img_path.name}: {predicted_emotion} ({confidence:.1f}%)")
                else:
                    print(f"   [{total_images}] {img_path.name}: {predicted_emotion} ({confidence:.1f}%) | True: {true_emotion}")
                
                # Update confusion matrix
                confusion_matrix[true_emotion][predicted_emotion] += 1
                emotion_stats[true_emotion]["total"] += 1
                if is_correct:
                    emotion_stats[true_emotion]["correct"] += 1
                
                # Progress indicator
                if idx % 100 == 0:
                    print(f"    Progress: {idx}/{len(images)} ({idx/len(images)*100:.1f}%)")
                
            except Exception as e:
                print(f"      Error processing {img_path.name}: {type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        accuracy = (correct / len(images) * 100) if len(images) > 0 else 0
        print(f"  📊 {true_emotion} Accuracy: {correct}/{len(images)} ({accuracy:.1f}%)")

# Print results
print("\n" + "="*70)
print("📊 RESULTS SUMMARY")
print("="*70)
print()

overall_accuracy = (total_correct / total_images * 100) if total_images > 0 else 0
print(f"Overall Accuracy: {total_correct}/{total_images} ({overall_accuracy:.1f}%)")
print()

# Print confusion matrix
print("Confusion Matrix (True Emotion → Predicted Emotion):")
print("-" * 70)

emotions = list(EMOTIONS_FER2013.values())
header = "True \\ Pred   " + "  ".join([e[:4] for e in emotions])
print(header)
print("-" * 70)

for true_emotion in emotions:
    row = f"{true_emotion:12s}  "
    for pred_emotion in emotions:
        count = confusion_matrix[true_emotion][pred_emotion]
        row += f"{count:6d} "
    print(row)

print("="*70)
print("\n Testing Complete with Hailo-8 HEF!")
print("="*70)
