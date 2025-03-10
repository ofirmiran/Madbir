### Libraries ###
import cv2
import os

### Parameter ###
# Update the paths according to your directory structure
source_folder = r"C:\Users\Mizui_Meida\Desktop\airTarget1"
destination_folder = r"C:\Users\Mizui_Meida\Desktop\airTarget1_frame"
skip_frames=25

def save_frames(video_path, target_folder, video_name, skip_frames=30):
    """
    Save frames from a video to a folder, naming each frame with the video name and frame number.

    :param video_path: Path to the video file.
    :param target_folder: Folder where images will be saved.
    :param video_name: Name of the video, used in naming saved frames.
    :param skip_frames: Number of frames to skip between saved images.
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    frame_idx = 0
    saved_image_idx = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_idx % skip_frames == 0:
            # Modify the naming pattern here to include the video name
            image_path = os.path.join(target_folder, f"{video_name}_frame_{saved_image_idx}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Saved {image_path}")
            saved_image_idx += 1

        frame_idx += 1

    video.release()
    print(f"Done processing video {video_path}.")

def process_video_folder(source_folder, target_folder_base, skip_frames=10):
    """
    Process all video files in a given folder, saving frames from each video to separate subfolders.

    :param source_folder: Folder containing video files.
    :param target_folder_base: Base folder where images will be saved. Subfolders will be created for each video.
    :param skip_frames: Number of frames to skip between saved images.
    """
    valid_formats = (".mp4", ".avi", ".mov")
    for filename in os.listdir(source_folder):
        if filename.endswith(valid_formats):
            video_name = os.path.splitext(filename)[0]  # Extract the video name without the extension
            video_path = os.path.join(source_folder, filename)
            target_folder = os.path.join(target_folder_base, video_name)
            save_frames(video_path, target_folder, video_name, skip_frames)

# Example usage
# Update the paths according to your directory structure
source_folder = r"C:\Users\Mizui_Meida\Desktop\airTarget1"
destination_folder = r"C:\Users\Mizui_Meida\Desktop\airTarget1_frame"

def __main__():
    process_video_folder(source_folder, destination_folder, skip_frames=25)
    return 0

