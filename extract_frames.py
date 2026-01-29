import cv2
import os

def extract_frames(video_path, output_folder):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    
    if not video_capture.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    while True:
        # Read the next frame
        success, frame = video_capture.read()

        # If the frame was not retrieved, we've reached the end of the video
        if not success:
            break

        # Save the frame as a JPEG file
        frame_name = f"frame_{frame_count:04d}.jpg"
        frame_path = os.path.join(output_folder, frame_name)
        cv2.imwrite(frame_path, frame)

        frame_count += 1

    # Clean up
    video_capture.release()
    print(f"Done! Extracted {frame_count} frames to '{output_folder}'.")

# Usage
extract_frames('video_pokemon.mp4', 'extracted_frames')