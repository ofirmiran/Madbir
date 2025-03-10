import os
import shutil
from sklearn.model_selection import train_test_split

def split_data(main_folder, dest_folder, image_ext='.jpg', label_ext='.txt', train_size=0.7, val_size=0.15, test_size=0.15):
    # Create directories for the train, validation, and test sets, with subdirectories for images and labels
    data_splits = ['train', 'valid', 'test']
    for split in data_splits:
        split_dir = os.path.join(dest_folder, split)
        images_dir = os.path.join(split_dir, 'images')
        labels_dir = os.path.join(split_dir, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

    # List all subfolders in the main folder
    subfolders = [os.path.join(main_folder, f) for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]

    all_files = []  # List to hold all image files across subfolders
    for folder in subfolders:
        # Add files from each subfolder
        files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.endswith(image_ext)]
        all_files.extend(files)

    total_files = len(all_files)

    if train_size + val_size + test_size != 1.0:
        raise ValueError("Train, validation, and test sizes must sum to 1.")

    # Randomly split the files into training, validation, and test sets
    train_files, temp_files = train_test_split(all_files, train_size=train_size, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=test_size / (val_size + test_size), random_state=42)

    # Function to copy files and their labels to a destination directory
    def copy_files(files, image_dir, label_dir):
        for file_path in files:
            shutil.copy(file_path, image_dir)
            label_file = file_path.rsplit('.', 1)[0] + label_ext
            if os.path.exists(label_file):
                shutil.copy(label_file, label_dir)

    # Copy files to the respective directories
    copy_files(train_files, os.path.join(dest_folder, 'train/images'), os.path.join(dest_folder, 'train/labels'))
    copy_files(val_files, os.path.join(dest_folder, 'valid/images'), os.path.join(dest_folder, 'valid/labels'))
    copy_files(test_files, os.path.join(dest_folder, 'test/images'), os.path.join(dest_folder, 'test/labels'))

    print(f"Files distributed: {len(train_files)} training, {len(val_files)} validation, {len(test_files)} testing.")

# Usage
main_folder = r'C:\Users\Mizui_Meida\Desktop\imagesDatabase3'  # Update this path
dest_folder = r'C:\Users\Mizui_Meida\Desktop\database_09.06.24'  # Update this path
split_data(main_folder, dest_folder)
