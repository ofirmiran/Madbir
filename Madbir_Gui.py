import sys
import subprocess
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QLineEdit, QFileDialog,QMessageBox, QCheckBox
from PyQt5.QtGui import QIcon, QPixmap,QFont
import os
import json


"""
Madbir Application
This application provides a graphical interface for running different modules of the Madbir project.
Modules include:
- Video to Frames conversion
- Labeling tool
- Model training
- Running the trained model

Author: [Ofir Miran]
Date: [5/3/2025]
"""

# Define sub Windows
class SubWindowBase(QWidget):
    """
    A base class for all sub-windows in the application.
    Provides a back button to return to the main window.
    """
    def __init__(self, main_window, title):
        super().__init__()
        self.main_window = main_window
        self.setGeometry(600, 100, 700, 1000)
        self.setWindowTitle(title)
        self.inputs = {}
        # Add the Madbir title
        title_label = QLabel(self)
        pixmap = QPixmap("icons/Madbir_title.jpg").scaled(400, 100)
        title_label.setPixmap(pixmap)
        title_label.move(150, 50)

        # Back button
        back_button = QPushButton(self)
        back_button.setIcon(QIcon(QPixmap(r"icons/back_icon.png").scaled(70, 70)))
        back_button.setGeometry(600, 900, 70, 70)
        back_button.clicked.connect(self.go_back)

    def go_back(self):
        """Saves inputs to JSON and returns to the main window."""
        self.save_to_json()
        self.close()
        self.main_window.show()

    def save_to_json(self):
        """Collects all input data and saves it to a JSON file."""
        data = {}

        # Initialize self.inputs if not already done
        if not hasattr(self, 'inputs'):
            self.inputs = {}

        # Collect data from input fields
        try:
            for key, widget in self.inputs.items():
                if isinstance(widget, QLineEdit):
                    data[key] = widget.text()
                elif isinstance(widget, QCheckBox):
                    data[key] = widget.isChecked()

            # Ensure the directory exists
            os.makedirs('parameters', exist_ok=True)

            # Save to a JSON file
            with open('parameters/developer_gui_parameters.json', 'w') as f:
                json.dump(data, f, indent=4)
                print("Data saved to 'developer_gui_parameters.json'")

        except Exception as e:
            self.show_error("Save Error", f"An error occurred while saving: {e}")

    def browse_directory(self):
        # Open a file dialog to choose a directory
        folder_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if folder_path:
            # If a directory was selected, set the path in the QLineEdit
            self.path_input.setText(folder_path)

    def browse_directory_line(self,label,y_pos):
        self.lable_folder = QLabel(label+":", self)
        self.lable_folder.setGeometry(50, y_pos, 130, 30)  # Set position and size of the label
        # Displaying the selected path
        self.path_input = QLineEdit(self)
        self.path_input.setGeometry(180,y_pos, 300, 30)
        self.path_input.setPlaceholderText("Select folder...")
        # Create a QPushButton to open the file dialog
        self.browse_button_input = QPushButton("...", self)
        self.browse_button_input.setGeometry(480,y_pos, 20, 30)
        self.browse_button_input.clicked.connect(self.browse_directory)

    def window_title(self,label):
        self.title = QLabel(label+":", self)
        self.title.setGeometry(50, 150, 200, 40)  # Set position and size of the label
        self.title.setFont(QFont("Arial", 14, QFont.Bold)) # Use Arial font, size 12, and make it bold

    def show_error(self,title, message):
        # Function to show error message using QMessageBox
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(message)
        msg.setWindowTitle(title)
        msg.exec_()

    def proper_value_check (self,input,min_value,max_value,integer_flag):
        # Check if input is a valid number
        try:
            value = float(input)  # Try to convert input to a float
        except ValueError:
            show_error("Invalid input", "Input must be a number.")
            return False

        # Check if input should be an integer and if it's an integer
        if integer_flag:
            if not value.is_integer():
                show_error("Invalid input", "Input must be an integer.")
                return False
            value = int(value)  # Convert to int if it's supposed to be an integer

        # Check if the value is within the specified range
        if value < min_value or value > max_value:
            show_error("Invalid input", f"Input must be between {min_value} and {max_value}.")
            return False

        # If all checks pass, return True
        return True

# Specific sub-windows for each module
class SubWindow1(SubWindowBase):
    def __init__(self, main_window):
        super().__init__(main_window, "Video to Frames")
        # Create a layout for the window

        # Title
        self.window_title("Video to frames")

        # Input folder
        self.browse_directory_line("Input video folder", 200)

        # Output folder
        self.browse_directory_line("Output frames folder", 250)

        # Frame skipping rate
        self.lable_frame_skipping_rate= QLabel("Frame skipping rate:  Save a frame out of every              frames", self)
        self.lable_frame_skipping_rate.setGeometry(50, 300, 400, 30)  # Set position and size of the label
        self.input_frame_skipping_rate = QLineEdit(self)
        self.input_frame_skipping_rate.setGeometry(330,300, 50, 30)
        self.input_frame_skipping_rate.setPlaceholderText("0-50")

class SubWindow2(SubWindowBase):
    def __init__(self, main_window):
        super().__init__(main_window, "Labeling")

        # Title
        self.window_title("Labeling")

        # Input folder
        self.browse_directory_line("Input frames folder", 200)

        # Output folder
        self.browse_directory_line("Output folder", 250)


class SubWindow3(SubWindowBase):
    def __init__(self, main_window):
        super().__init__(main_window, "Train Model")

        # Set Window Title
        self.setWindowTitle("Train model")

        # Input folder
        self.browse_directory_line("Input data folder", 200)

        # Output folder
        self.browse_directory_line("Output model folder", 250)

        # Output model name
        self.label_model_name = QLabel("Model name:", self)
        self.label_model_name.setGeometry(50, 300, 130, 30)  # Set position and size of the label
        self.input_model_name = QLineEdit(self)
        self.input_model_name.setGeometry(180, 300, 130, 30)
        self.input_model_name.setPlaceholderText("Enter model name")

        # Validation and test data
        self.label_valid_test_percentage = QLabel(
            "Validation data              %    Test data              %", self
        )
        self.label_valid_test_percentage.setGeometry(50, 350, 400, 30)

        self.input_validation_data_percentage = QLineEdit(self)
        self.input_validation_data_percentage.setGeometry(140, 350, 50, 30)
        self.input_validation_data_percentage.setPlaceholderText("0-40")
        self.input_validation_data_percentage.textChanged.connect(
            lambda: self.proper_value_check(
                self.input_validation_data_percentage.text(), 0, 40, True
            )
        )

        self.input_test_data_percentage = QLineEdit(self)
        self.input_test_data_percentage.setGeometry(277, 350, 50, 30)
        self.input_test_data_percentage.setPlaceholderText("0-15")
        self.input_test_data_percentage.textChanged.connect(
            lambda: self.proper_value_check(
                self.input_test_data_percentage.text(), 0, 15, True
            )
        )

        self.label_model_parameters = QLabel("Model parameters:", self)
        self.label_model_parameters.setGeometry(50, 400, 130, 30)  # Set position and size of the label




class SubWindow4(SubWindowBase):
    def __init__(self, main_window):
        super().__init__(main_window, "Run Model")

        # Set Window Title
        self.setWindowTitle("Run algorithem:")

        # Input video
        self.browse_directory_line("Input video:", 200)

        # Input model
        self.browse_directory_line("Input model:", 250)

        # choose if to save output video
        self.label_save_output = QLabel("Save output video:", self)
        self.label_save_output.setGeometry(50, 300, 130, 30)  # Set position and size of the label
        # Add a checkbox for saving the output video
        self.checkbox_save_output = QCheckBox(self)
        self.checkbox_save_output.setGeometry(180, 300, 20, 30)

        # Output video
        self.browse_directory_line("Output video:", 350)

class MainWindow(QWidget):
    """
    The main window of the Madbir application.
    Displays buttons for accessing each module and handles running scripts.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Madbir Main Window")
        self.setGeometry(600, 100, 700, 1000)
        self.window_classes = [SubWindow1, SubWindow2, SubWindow3, SubWindow4]

        # Add the Madbir title
        title_label = QLabel(self)
        pixmap = QPixmap("icons/Madbir_title.jpg").scaled(400, 100)
        title_label.setPixmap(pixmap)
        title_label.move(150, 50)

        # Add the top-right settings button
        top_right_button = QPushButton(self)
        top_right_button.setIcon(QIcon("icons/setting_icon.png"))
        top_right_button.setGeometry(600, 20, 70, 70)

        # Button configurations
        button_titles = ["Video to Frames", "Labeling", "Train Model", "Run Model"]
        button_positions = [(100, 200), (100, 275), (100, 350), (100, 425)]
        script_paths = ["VideoToImage.py", "label/YoloLabel.exe", "train.py", "toolbar.py"]

        # Create module buttons and their settings buttons
        for i, (title, pos, script) in enumerate(zip(button_titles, button_positions, script_paths)):
            # Button to run the associated script
            button = QPushButton(title, self)
            button.setGeometry(*pos, 250, 40)
            button.clicked.connect(lambda checked, s=script: self.run_script(s))

            # Settings button for additional configuration
            settings_button = QPushButton(self)
            settings_button.setIcon(QIcon("icons/setting_icon.png"))
            settings_button.setGeometry(pos[0] + 250, pos[1], 40, 40)
            settings_button.clicked.connect(lambda checked, num=i: self.open_window(num))

    def run_script(self, script_path: str):
        """
        Runs a script or executable based on the provided path.
        Supports both Python scripts (.py) and executables (.exe).
        """
        # Convert to an absolute path
        script_path = os.path.abspath(script_path)
        print(f"Resolved script path: {script_path}")

        # Check if the file exists
        if not os.path.exists(script_path):
            print(f"File not found: {script_path}")
            return

        # Run executable files
        if script_path.lower().endswith('.exe'):
            print(f"Running executable: {script_path}")
            try:
                os.startfile(script_path)
            except Exception as e:
                print(f"Failed to start executable: {e}")
        # Run Python scripts
        else:
            print(f"Running Python script: {script_path}")
            try:
                subprocess.run([sys.executable, script_path], check=True)
            except Exception as e:
                print(f"Error running script: {e}")

    def open_window(self, window_number: int):
        """Opens the selected sub-window based on the window number."""
        self.hide()
        self.new_window = self.window_classes[window_number](self)
        self.new_window.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


"""
TO DO: 
    - Save all the parameters
    - Load all the parameters
    - IN 0-50 allow only this numbers, numbers, full positive numbers
    
    - check about json file to save the parameters
    - Check QFormLayout
    
    
    -   Auto DL
"""