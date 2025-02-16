import cv2
import numpy as np

# Read video
video_path = 'videos/init_nom.mp4'
cap = cv2.VideoCapture(video_path)

# Create Background Subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Determine the frame indices for capturing images
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
num_images = 20
frame_indices = np.linspace(0, total_frames - 1, num_images, dtype=int)

# Process frames and create the composite image
composite_image = None

for idx in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # Set the capture frame position
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Create composite image
    if composite_image is None:
        composite_image = np.copy(frame)
    else:
        composite_image[fg_mask > 0] = frame[fg_mask > 0]

cap.release()

# Save the composite image
output_image_path = 'output_images/composite_image.jpg'
cv2.imwrite(output_image_path, composite_image)

# Enhance the quality of the composite image
enhanced_composite_image = composite_image.copy()



# Denoising using Gaussian blur
enhanced_composite_image = cv2.GaussianBlur(enhanced_composite_image, (5, 5), 0)

# Save the enhanced composite image
output_enhanced_image_path = 'output_images/enhanced_composite_image.jpg'
cv2.imwrite(output_enhanced_image_path, enhanced_composite_image)