import cv2
import numpy as np

# Read video
video_path = 'videos/init_nom.mp4'
cap = cv2.VideoCapture(video_path)

# Create Background Subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Get the frame rate and dimensions of the input video
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter to save the output video with the path
output_path = 'videos/output_with_smoothed_path.avi'
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

# Initialize variables for object tracking and path drawing
object_position = None
path = []
smoothing_window = 20
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Update the object position and path based on the largest contour (assumes one moving object)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            object_position = (cX, cY)
            path.append(object_position)

    # Apply moving average smoothing to the path points
    if len(path) > smoothing_window:
        smoothed_path = []
        for i in range(len(path) - smoothing_window + 1):
            smoothed_point = np.mean(path[i : i + smoothing_window], axis=0)
            smoothed_path.append(smoothed_point.astype(int))

        # Draw the smoothed path on the frame
        for point in smoothed_path:
            cv2.circle(frame, tuple(point), 3, (255, 0, 0), -1)
        if len(smoothed_path) > 1:
            cv2.polylines(frame, [np.array(smoothed_path)], isClosed=False, color=(0, 255, 0), thickness=2)

    # Write the frame with the drawn path to the output video
    out.write(frame)

    # Display the frame with the path
    cv2.imshow('Frame with Smoothed Path', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()