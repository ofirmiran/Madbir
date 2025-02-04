import math
import cv2
import numpy as np
from math import sin, radians

print("The version of cv2: ")
print(cv2.__version__)


def load_image(path):
    # Load an image from the given path in grayscale mode
    image = cv2.imread(path, 0)
    if image is None:
        print(f"Error: Unable to load image at {path}. Check the file path and integrity.")
    return image

# Load and define templates
template_image_path = r"C:\Users\Mizui_Meida\Desktop\1\1_frame_9.jpg"
template_image = load_image(template_image_path)
if template_image is None:
    exit()

# Define the coordinates for each character in the template image
template_positions = {
    '0': (100, 95, 113, 113), '1': (820, 128, 831, 146), '2': (820, 161, 833, 179),
    '3': (1803, 363, 1820, 381), '4': (1096, 1039, 1113, 1057), '5': (278, 1042, 292, 1061),
    '6': (867, 160, 882, 179), '7': (851, 128, 864, 146), '8': (130, 95, 145, 114),
    '9': (835, 127, 850, 147), '-': (1839, 461, 1850, 476), '.': (1869, 461, 1881, 478)
}
# Extract templates using the defined positions
templates = {char: template_image[y1:y2, x1:x2] for char, (x1, y1, x2, y2) in template_positions.items()}

# Open the video file
video_path = (r"C:\Users\Mizui_Meida\Desktop\05.02.24\Untitled_0038.mov")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_rate = cap.get(cv2.CAP_PROP_FPS)
print(f"Frame rate: {frame_rate} FPS")

def recognize_sequence(region, templates, debug=False):
    results = []
    x_offset = 0
    while x_offset < region.shape[1]:  # Process each segment of the region
        max_val = -1
        best_match = None
        best_x_offset = None

        for char, template in templates.items():
            if x_offset + template.shape[1] > region.shape[1]:
                continue  # Ensure the template fits within the current segment

            slice_region = region[:, x_offset:x_offset + template.shape[1]]
            if slice_region.shape[1] == 0:
                continue

            res = cv2.matchTemplate(slice_region, template, cv2.TM_CCOEFF_NORMED)
            loc_max_val = np.max(res)
            if loc_max_val > max_val:  # Find the best match with the highest correlation
                max_val = loc_max_val
                best_match = char
                best_x_offset = x_offset

        if best_match and max_val > 0.9:  # Accept the match if the correlation is high
            results.append(best_match)
            x_offset = best_x_offset + templates[best_match].shape[1]  # Move past the matched segment
        else:
            x_offset += 1  # Move one pixel over if no match is found

    return ''.join(results)  # Return the concatenated results

# Process each frame of the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    #analysis_region = (1835, 459, 1901, 479)  # Define the area of the frame to analyze
    analysis_region = (1835, 459, 1901, 479)  # Define the area of the frame to analyze

    x1, y1, x2, y2 = analysis_region
    EL_region = gray_frame[y1:y2, x1:x2]  # Extract the region of interest

    recognized_EL = recognize_sequence(EL_region, templates, debug=False)
    print(f"Recognized EL: {recognized_EL}")

    # Display the recognized sequence just above the analysis region
    EL_position = (x1, y1 - 10)  # Position text slightly above the analysis region
    cv2.putText(frame, recognized_EL, EL_position, cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    analysis_region = (815, 124, 890, 154)  # Define the area of the frame to analyze
    x1, y1, x2, y2 = analysis_region
    distance_region_km = gray_frame[y1:y2, x1:x2]  # Extract the region of interest
    recognized_distance_km = recognize_sequence(distance_region_km, templates, debug=False)
    print(f"Recognized distance km: {recognized_distance_km}")

    # Display the recognized sequence just above the analysis region
    text_position = (x1, y1 - 10)  # Position text slightly above the analysis region
    cv2.putText(frame, recognized_distance_km, text_position, cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)


    analysis_region = (815, 155, 890, 185)  # Define the area of the frame to analyze
    x1, y1, x2, y2 = analysis_region
    distance_region_yard = gray_frame[y1:y2, x1:x2]  # Extract the region of interest
    recognized_distance_yard = recognize_sequence(distance_region_yard, templates, debug=False)
    print(f"Recognized distance yard: {recognized_distance_yard}")

    # Display the recognized sequence just above the analysis region
    text_position = (x1, y2 + 30)  # Position text slightly above the analysis region
    cv2.putText(frame, recognized_distance_yard, text_position, cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    height = 0
    # Calculate height using the possibly corrected values
    if (recognized_EL is not None and recognized_EL is not ".") and (recognized_distance_km != '' and recognized_distance_km is not ".") and (recognized_distance_yard is not None and recognized_distance_yard is not ".") and (
            np.float64(recognized_distance_km) * 1.093 - np.float64(recognized_distance_yard) < 2):
        #height = 20 + np.float64(recognized_distance_km) * sin(radians(np.float64(recognized_EL)))
        height = 20 + np.float64(recognized_distance_km) * math.tan(radians(np.float64(recognized_EL)))

        print(f"Height: {height}")
        height = str(round(height, 2)) + " m"
    else:
        print("Invalid EL or distance, cannot calculate height.")


    x3_distance, y3_distance = 820, 70  # Top left corner for Distance

    if height != 0:
        # Adjust position for displaying the distance
        # Display distance with blue background and white font
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.rectangle(frame, (x3_distance, y3_distance - 20), (x3_distance + 170, y3_distance + 5), (255, 0, 0),
                      -1)  # Blue rectangle background
        cv2.putText(frame, f"height: {height}", (x3_distance, y3_distance), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    #cv2.imshow('region', region)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
