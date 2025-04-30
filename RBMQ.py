"""
Image Property Analyzer Application
----------------------------------
A PyQt5-based desktop application that analyzes various properties of images
across multiple categories including basic properties, pixel statistics,
histogram features, compression metrics, and advanced image characteristics.
"""

import sys
import os
import numpy as np
from PIL import Image, ImageStat, ImageFilter, ImageOps, ImageDraw
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QLabel, QVBoxLayout, QWidget, QMessageBox, QTextEdit,
    QScrollArea, QHBoxLayout, QGroupBox, QTabWidget, QFormLayout, QFrame
)
from PyQt5.QtGui import QPixmap, QFont, QColor, QPainter, QImage
from PyQt5.QtCore import Qt, QSize
import math
from collections import Counter
from skimage import filters, feature
import cv2
#from PIL.ImageQt import ImageQt


class ImageAnalyzerApp(QMainWindow):
    """
    Main application window for the Image Property Analyzer.
    
    Attributes:
        image (PIL.Image): Currently loaded image
        image_path (str): Path to the current image
        properties (dict): Dictionary to store all calculated image properties
    """
    
    def __init__(self):
        """Initialize the main application window and UI components."""
        super().__init__()
        
        # Window configuration
        self.setWindowTitle("📊 Image Property Analyzer Pro")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize image properties
        self.image = None
        self.image_path = ""
        self.properties = {}
        
        # Setup UI
        self.init_ui()
        self.apply_styles()
        
    def init_ui(self):
        """Initialize all UI components."""
        # Main widget with layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(20)
        
        # Left panel - Image display area
        self.setup_image_panel()
        
        # Right panel - Property tabs
        self.setup_property_tabs()
        
    def setup_image_panel(self):
        """Configure the left panel for image display."""
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setAlignment(Qt.AlignTop)
        
        # Image display frame
        self.image_frame = QFrame()
        self.image_frame.setFrameShape(QFrame.StyledPanel)
        self.image_layout = QVBoxLayout(self.image_frame)
        self.image_layout.setContentsMargins(10, 10, 10, 10)
        
        # Image label with placeholder
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(500, 500)
        self.setup_placeholder_image()
        
        # Load image button
        self.load_button = QPushButton("📁 Load Image")
        self.load_button.clicked.connect(self.load_image)
        
        # Image path label
        self.path_label = QLabel("No image loaded")
        self.path_label.setWordWrap(True)
        
        # Add widgets to left panel
        self.image_layout.addWidget(self.image_label)
        self.left_layout.addWidget(self.image_frame)
        self.left_layout.addWidget(self.load_button)
        self.left_layout.addWidget(self.path_label)
        
        # Add left panel to main layout
        self.main_layout.addWidget(self.left_panel, stretch=1)
        
    def setup_placeholder_image(self):
        """Create and set a placeholder image when no image is loaded."""
        placeholder = QPixmap(500, 500)
        placeholder.fill(QColor(248, 250, 252))
        painter = QPainter(placeholder)
        painter.setPen(QColor(209, 213, 219))
        painter.drawText(placeholder.rect(), Qt.AlignCenter, "No Image Loaded")
        painter.end()
        self.image_label.setPixmap(placeholder)
        
    def setup_property_tabs(self):
        """Configure the right panel with property tabs."""
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setAlignment(Qt.AlignTop)
        
        # Create tab widget
        self.tabs = QTabWidget()
        
        # Add tabs for different property categories
        self.setup_basic_properties_tab()
        self.setup_pixel_statistics_tab()
        self.setup_histogram_tab()
        self.setup_compression_tab()
        self.setup_advanced_tab()
        
        # Add tabs to right panel
        self.right_layout.addWidget(self.tabs)
        self.main_layout.addWidget(self.right_panel, stretch=2)
        
    def setup_basic_properties_tab(self):
        """Create and configure the Basic Properties tab."""
        self.basic_tab = QWidget()
        self.basic_layout = QFormLayout(self.basic_tab)
        
        basic_group = QGroupBox("📋 Basic Image Properties")
        basic_group_layout = QFormLayout()
        
        # Define basic properties to display
        self.basic_properties = [
            ("Dimensions", "1. Width × Height (pixels)"),
            ("File Size", "2. File Size (KB or MB)"),
            ("File Format", "3. File Format"),
            ("Bit Depth", "4. Bits per channel"),
            ("Color Mode", "5. Color Mode"),
            ("Aspect Ratio", "6. Width / Height"),
            ("Image Mode", "7. Color, Grayscale, Binary")
        ]
        
        # Create UI elements for each property
        for prop, desc in self.basic_properties:
            label = QLabel(desc)
            value = QLabel("N/A")
            value.setObjectName(f"basic_{prop.replace(' ', '_')}")
            basic_group_layout.addRow(label, value)
        
        basic_group.setLayout(basic_group_layout)
        self.basic_layout.addWidget(basic_group)
        self.tabs.addTab(self.basic_tab, "Basic Properties")
        
    def setup_pixel_statistics_tab(self):
        """Create and configure the Pixel Statistics tab."""
        self.pixel_tab = QWidget()
        self.pixel_layout = QFormLayout(self.pixel_tab)
        
        pixel_group = QGroupBox("🔍 Pixel Value Statistics")
        pixel_group_layout = QFormLayout()
        
        # Define pixel statistics properties
        self.pixel_properties = [
            ("Mean Intensity", "8. Average brightness"),
            ("Std Deviation", "9. Contrast measure"),
            ("Min/Max Value", "10. Brightness extremes"),
            ("Channel Averages", "11. Mean of R, G, B separately"),
            ("Unique Colors", "12. Number of unique colors"),
            ("Brightness Score", "13. Scalar brightness"),
            ("Contrast Score", "14. Standard deviation"),
            ("Color Dominance", "15. Most influential color")
        ]
        
        # Create UI elements
        for prop, desc in self.pixel_properties:
            label = QLabel(desc)
            value = QLabel("N/A")
            value.setObjectName(f"pixel_{prop.replace(' ', '_')}")
            pixel_group_layout.addRow(label, value)
        
        pixel_group.setLayout(pixel_group_layout)
        self.pixel_layout.addWidget(pixel_group)
        self.tabs.addTab(self.pixel_tab, "Pixel Statistics")
        
    def setup_histogram_tab(self):
        """Create and configure the Histogram tab."""
        self.histogram_tab = QWidget()
        self.histogram_layout = QVBoxLayout(self.histogram_tab)
        
        histogram_group = QGroupBox("📊 Histogram and Frequency Features")
        histogram_group_layout = QVBoxLayout()
        
        # Define histogram properties
        self.histogram_properties = [
            ("Intensity Histogram", "16. Histogram of grayscale pixel values"),
            ("Byte Frequency", "17. Byte value distribution"),
            ("Redundancy Ratio", "19. 1 − (entropy / log₂(n))")
        ]
        
        # Create form layout for properties
        form_layout = QFormLayout()
        for prop, desc in self.histogram_properties:
            label = QLabel(desc)
            value = QLabel("N/A")
            value.setObjectName(f"histogram_{prop.replace(' ', '_')}")
            form_layout.addRow(label, value)
        
        # Histogram visualization
        self.histogram_label = QLabel()
        self.histogram_label.setAlignment(Qt.AlignCenter)
        self.histogram_label.setMinimumHeight(200)
        
        # Add components to tab
        histogram_group_layout.addLayout(form_layout)
        histogram_group_layout.addWidget(QLabel("Grayscale Histogram:"))
        histogram_group_layout.addWidget(self.histogram_label)
        
        histogram_group.setLayout(histogram_group_layout)
        self.histogram_layout.addWidget(histogram_group)
        self.tabs.addTab(self.histogram_tab, "Histogram")
        
    def setup_compression_tab(self):
        """Create and configure the Compression/Entropy tab."""
        self.compression_tab = QWidget()
        self.compression_layout = QFormLayout(self.compression_tab)
        
        compression_group = QGroupBox("🗜️ Compression/Entropy Features")
        compression_group_layout = QFormLayout()
        
        # Define compression-related properties
        self.compression_properties = [
            ("Shannon Entropy", "20. Information randomness"),
            ("Repetition Rate", "21. Detect repeating patterns (RLE suitability)"),
            ("Smoothness Score", "22. Local variance, edge flatness"),
            ("Noise Estimation", "23. Using local variance / filters"),
            ("Gradient Strength", "25. Measures texture"),
            ("Edge Density", "26. % of edge pixels using Sobel/Canny")
        ]
        
        # Create UI elements
        for prop, desc in self.compression_properties:
            label = QLabel(desc)
            value = QLabel("N/A")
            value.setObjectName(f"compression_{prop.replace(' ', '_')}")
            compression_group_layout.addRow(label, value)
        
        compression_group.setLayout(compression_group_layout)
        self.compression_layout.addWidget(compression_group)
        self.tabs.addTab(self.compression_tab, "Compression")
        
    def setup_advanced_tab(self):
        """Create and configure the Advanced tab."""
        self.advanced_tab = QWidget()
        self.advanced_layout = QFormLayout(self.advanced_tab)
        
        advanced_group = QGroupBox("🔬 Advanced Features")
        advanced_group_layout = QFormLayout()
        
        # Define advanced properties
        self.advanced_properties = [
            ("Compression Artifacts", "27. Detect blockiness or ringing"),
            ("Heuristic Image Type", "28. Photo, cartoon, etc.")
        ]
        
        # Create UI elements
        for prop, desc in self.advanced_properties:
            label = QLabel(desc)
            value = QLabel("N/A")
            value.setObjectName(f"advanced_{prop.replace(' ', '_')}")
            advanced_group_layout.addRow(label, value)
        
        advanced_group.setLayout(advanced_group_layout)
        self.advanced_layout.addWidget(advanced_group)
        self.tabs.addTab(self.advanced_tab, "Advanced")
        
    def apply_styles(self):
        """Apply consistent styling to all UI components."""
        self.setStyleSheet("""
            /* Main window styling */
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                          stop:0 #f5f7fa, stop:1 #c3cfe2);
            }
            
            /* Button styling */
            QPushButton {
                background-color: #4f46e5;
                color: white;
                font-weight: bold;
                padding: 12px;
                border-radius: 6px;
                font-size: 14px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #4338ca;
            }
            QPushButton:pressed {
                background-color: #3730a3;
            }
            
            /* Tab widget styling */
            QTabWidget::pane {
                border: 1px solid #d1d5db;
                border-radius: 8px;
                background: white;
                margin-top: 10px;
            }
            QTabBar::tab {
                background: #e5e7eb;
                color: #4b5563;
                padding: 8px 16px;
                border: 1px solid #d1d5db;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 4px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: white;
                color: #4f46e5;
                border-bottom: 2px solid #4f46e5;
            }
            QTabBar::tab:hover {
                background: #d1d5db;
            }
            
            /* Group box styling */
            QGroupBox {
                border: 1px solid #d1d5db;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 20px;
                background: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px;
                color: #4f46e5;
                font-weight: bold;
            }
            
            /* Label styling */
            QLabel {
                color: #374151;
                margin: 2px;
            }
            
            /* Image frame styling */
            QFrame {
                background: white;
                border-radius: 8px;
                border: 1px solid #d1d5db;
            }
            
            /* Image label styling */
            #image_label {
                background: #f8fafc;
                border: 2px dashed #d1d5db;
                border-radius: 6px;
            }
            
            /* Histogram label styling */
            #histogram_label {
                border: 1px solid #d1d5db;
                background-color: white;
                border-radius: 4px;
            }
            
            /* Path label styling */
            #path_label {
                color: #6b7280;
                font-style: italic;
                padding: 5px;
            }
        """)
        
        # Apply specific object names for styling
        self.image_label.setObjectName("image_label")
        self.histogram_label.setObjectName("histogram_label")
        self.path_label.setObjectName("path_label")
        
        # Set button cursor
        self.load_button.setCursor(Qt.PointingHandCursor)
        
    def load_image(self):
        """
        Open a file dialog to select an image and load it into the application.
        
        Shows an error message if the image cannot be loaded.
        """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",
            options=options
        )
        
        if file_path:
            try:
                self.image_path = file_path
                self.path_label.setText(os.path.basename(file_path))
                
                # Load image with PIL
                self.image = Image.open(file_path)
                
                # Display image
                self.display_loaded_image(file_path)
                
                # Analyze image properties
                self.analyze_image()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
                
    def display_loaded_image(self, file_path):
        """Display the loaded image in the image label."""
        pixmap = QPixmap(file_path)
        scaled_pixmap = pixmap.scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        
    def analyze_image(self):
        """
        Analyze the loaded image and calculate all properties.
        
        Calculates properties across all categories and updates the UI.
        Shows an error message if analysis fails.
        """
        if not self.image:
            return
        
        try:
            # Convert image to RGB if not already
            if self.image.mode != 'RGB':
                img_rgb = self.image.convert('RGB')
            else:
                img_rgb = self.image
            
            # Convert to numpy array
            img_array = np.array(img_rgb)
            
            # Convert to grayscale for some calculations
            img_gray = np.array(self.image.convert('L'))
            
            # Calculate properties for each category
            self.calculate_basic_properties(img_array, img_gray)
            self.calculate_pixel_statistics(img_array, img_gray)
            self.calculate_histogram_features(img_gray)
            self.calculate_compression_features(img_array, img_gray)
            self.calculate_advanced_features(img_array, img_gray)
            
            # Update UI with calculated properties
            self.update_property_display()
            
            # Generate and display histogram
            self.display_histogram(img_gray)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to analyze image: {str(e)}")
            print(f"Error: {str(e)}")
            
    def calculate_basic_properties(self, img_array, img_gray):
        """Calculate and store basic image properties."""
        self.properties['Dimensions'] = f"{self.image.width} × {self.image.height}"
        file_size = os.path.getsize(self.image_path)
        self.properties['File Size'] = self.format_file_size(file_size)
        self.properties['File Format'] = os.path.splitext(self.image_path)[1].upper().replace('.', '')
        self.properties['Bit Depth'] = self.get_bit_depth(self.image)
        self.properties['Color Mode'] = self.image.mode
        self.properties['Aspect Ratio'] = self.calculate_aspect_ratio(self.image.width, self.image.height)
        self.properties['Image Mode'] = self.get_image_mode_category(self.image)
        
    def calculate_pixel_statistics(self, img_array, img_gray):
        """Calculate and store pixel statistics."""
        self.properties['Mean Intensity'] = f"{np.mean(img_gray):.2f}"
        self.properties['Std Deviation'] = f"{np.std(img_gray):.2f}"
        self.properties['Min/Max Value'] = f"{np.min(img_gray)} / {np.max(img_gray)}"
        
        # Channel averages if color image
        if len(img_array.shape) == 3:
            channel_avgs = [f"{np.mean(img_array[:,:,i]):.2f}" for i in range(3)]
            self.properties['Channel Averages'] = f"R: {channel_avgs[0]}, G: {channel_avgs[1]}, B: {channel_avgs[2]}"
        else:
            self.properties['Channel Averages'] = "N/A (Grayscale)"
        
        self.properties['Unique Colors'] = self.count_unique_colors(img_array)
        self.properties['Brightness Score'] = f"{self.calculate_brightness(img_gray):.2f}"
        self.properties['Contrast Score'] = f"{np.std(img_gray):.2f}"
        self.properties['Color Dominance'] = self.get_dominant_color(img_array)
        
    def calculate_histogram_features(self, img_gray):
        """Calculate and store histogram-related features."""
        self.properties['Intensity Histogram'] = "Calculated"  # Displayed visually
        self.properties['Byte Frequency'] = "Calculated"  # Not displayed numerically
        self.properties['Redundancy Ratio'] = f"{self.calculate_redundancy_ratio(img_gray):.4f}"
        
    def calculate_compression_features(self, img_array, img_gray):
        """Calculate and store compression-related features."""
        self.properties['Shannon Entropy'] = f"{self.calculate_shannon_entropy(img_gray):.4f}"
        self.properties['Repetition Rate'] = f"{self.calculate_repetition_rate(img_array):.4f}"
        self.properties['Smoothness Score'] = f"{self.calculate_smoothness_score(img_gray):.4f}"
        self.properties['Noise Estimation'] = f"{self.estimate_noise_level(img_gray):.4f}"
        self.properties['Gradient Strength'] = f"{self.calculate_gradient_strength(img_gray):.4f}"
        self.properties['Edge Density'] = f"{self.calculate_edge_density(img_gray):.2f}%"
        
    def calculate_advanced_features(self, img_array, img_gray):
        """Calculate and store advanced image features."""
        self.properties['Compression Artifacts'] = self.detect_compression_artifacts(img_gray)
        self.properties['Heuristic Image Type'] = self.determine_image_type(img_array)
        
    def update_property_display(self):
        """Update all UI elements with the calculated property values."""
        # Update basic properties
        for prop, _ in self.basic_properties:
            value = self.properties.get(prop, "N/A")
            label = self.findChild(QLabel, f"basic_{prop.replace(' ', '_')}")
            if label:
                label.setText(str(value))
        
        # Update pixel statistics
        for prop, _ in self.pixel_properties:
            value = self.properties.get(prop, "N/A")
            label = self.findChild(QLabel, f"pixel_{prop.replace(' ', '_')}")
            if label:
                label.setText(str(value))
        
        # Update histogram properties
        for prop, _ in self.histogram_properties:
            value = self.properties.get(prop, "N/A")
            label = self.findChild(QLabel, f"histogram_{prop.replace(' ', '_')}")
            if label:
                label.setText(str(value))
        
        # Update compression properties
        for prop, _ in self.compression_properties:
            value = self.properties.get(prop, "N/A")
            label = self.findChild(QLabel, f"compression_{prop.replace(' ', '_')}")
            if label:
                label.setText(str(value))
        
        # Update advanced properties
        for prop, _ in self.advanced_properties:
            value = self.properties.get(prop, "N/A")
            label = self.findChild(QLabel, f"advanced_{prop.replace(' ', '_')}")
            if label:
                label.setText(str(value))
    
    def display_histogram(self, img_gray):
        """
        Generate and display a grayscale histogram of the image.
        
        Args:
            img_gray (numpy.ndarray): Grayscale image array
        """
        try:
            # Calculate histogram
            hist, bins = np.histogram(img_gray.flatten(), bins=256, range=[0,256])
            
            # Create histogram image
            hist_img = Image.new('RGB', (256, 100), (255, 255, 255))
            draw = ImageDraw.Draw(hist_img)
            
            # Normalize histogram to fit in display
            max_val = max(hist)
            if max_val > 0:
                hist_normalized = [int(h * 100 / max_val) for h in hist]
            else:
                hist_normalized = [0] * 256
            
            # Draw histogram bars
            for i in range(256):
                draw.line([(i, 100), (i, 100 - hist_normalized[i])], fill=(0, 0, 255))
            
            # Convert to QPixmap and display
            qimage = ImageQt.ImageQt(hist_img)
            pixmap = QPixmap.fromImage(qimage)
            self.histogram_label.setPixmap(pixmap.scaled(
                self.histogram_label.width(), self.histogram_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            
        except Exception as e:
            print(f"Error generating histogram: {str(e)}")
    
    # =====================================================================
    # Helper methods for property calculations
    # =====================================================================
    
    def format_file_size(self, size):
        """
        Convert file size in bytes to a human-readable string.
        
        Args:
            size (int): File size in bytes
            
        Returns:
            str: Formatted file size (e.g., "1.23 MB")
        """
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.2f} KB"
        else:
            return f"{size / (1024 * 1024):.2f} MB"
    
    def get_bit_depth(self, image):
        """
        Determine the bit depth of an image.
        
        Args:
            image (PIL.Image): Image to analyze
            
        Returns:
            int or str: Bit depth or "Unknown" if undetermined
        """
        if image.mode in ('1', 'L', 'P'):
            return 8  # Assuming 8-bit for these modes
        elif image.mode == 'I':
            return 32
        elif image.mode == 'F':
            return 32
        elif image.mode in ('RGB', 'RGBA', 'CMYK', 'YCbCr'):
            return 8  # Per channel
        else:
            return "Unknown"
    
    def calculate_aspect_ratio(self, width, height):
        """
        Calculate and simplify an image's aspect ratio.
        
        Args:
            width (int): Image width in pixels
            height (int): Image height in pixels
            
        Returns:
            str: Simplified aspect ratio (e.g., "16:9 (1.78)")
        """
        def gcd(a, b):
            """Calculate greatest common divisor using Euclidean algorithm."""
            while b:
                a, b = b, a % b
            return a
        
        divisor = gcd(width, height)
        return f"{width//divisor}:{height//divisor} ({width/height:.2f})"
    
    def get_image_mode_category(self, image):
        """
        Categorize an image as color, grayscale, or binary.
        
        Args:
            image (PIL.Image): Image to categorize
            
        Returns:
            str: Image category ("Color", "Grayscale", or "Binary")
        """
        if image.mode == '1':
            return "Binary"
        elif image.mode in ('L', 'LA'):
            return "Grayscale"
        else:
            return "Color"
    
    def count_unique_colors(self, img_array):
        """
        Count the number of unique colors in an image.
        
        Args:
            img_array (numpy.ndarray): Image pixel array
            
        Returns:
            str: String representation of unique color count
        """
        if len(img_array.shape) == 3:  # Color image
            # Reshape to list of RGB tuples
            pixels = img_array.reshape(-1, img_array.shape[2])
            unique_colors = len(np.unique(pixels, axis=0))
        else:  # Grayscale
            unique_colors = len(np.unique(img_array))
        
        return str(unique_colors)
    
    def calculate_brightness(self, img_gray):
        """
        Calculate the average brightness of a grayscale image.
        
        Args:
            img_gray (numpy.ndarray): Grayscale image array
            
        Returns:
            float: Average brightness value (0-255)
        """
        return np.mean(img_gray)
    
    def get_dominant_color(self, img_array):
        """
        Identify the most dominant color in an image.
        
        Args:
            img_array (numpy.ndarray): Image pixel array
            
        Returns:
            str: String representation of dominant color
        """
        if len(img_array.shape) == 3:  # Color image
            pixels = img_array.reshape(-1, img_array.shape[2])
            # Count color frequencies
            color_counts = Counter(map(tuple, pixels))
            dominant_color = color_counts.most_common(1)[0][0]
            return f"RGB({dominant_color[0]}, {dominant_color[1]}, {dominant_color[2]})"
        else:  # Grayscale
            return f"Gray({np.argmax(np.bincount(img_array.flatten()))})"
    
    def calculate_shannon_entropy(self, img_gray):
        """
        Calculate the Shannon entropy of an image.
        
        Args:
            img_gray (numpy.ndarray): Grayscale image array
            
        Returns:
            float: Shannon entropy value
        """
        hist, _ = np.histogram(img_gray.flatten(), bins=256, range=[0,256])
        hist = hist[hist > 0]  # Remove bins with zero counts
        prob = hist / hist.sum()
        return -np.sum(prob * np.log2(prob))
    
    def calculate_redundancy_ratio(self, img_gray):
        """
        Calculate the redundancy ratio of an image.
        
        Args:
            img_gray (numpy.ndarray): Grayscale image array
            
        Returns:
            float: Redundancy ratio (1 - entropy/log2(n))
        """
        entropy = self.calculate_shannon_entropy(img_gray)
        max_entropy = np.log2(256)  # For 8-bit image
        return 1 - (entropy / max_entropy)
    
    def calculate_repetition_rate(self, img_array):
        """
        Estimate the repetition rate in an image (for RLE suitability).
        
        Args:
            img_array (numpy.ndarray): Image pixel array
            
        Returns:
            float: Repetition rate (changes/total_pixels)
        """
        if len(img_array.shape) == 3:  # Color image
            # Convert to grayscale for this calculation
            img_gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            img_gray = img_array
        
        # Calculate run lengths (simplified approach)
        diff = np.diff(img_gray.flatten())
        changes = np.sum(diff != 0)
        total_pixels = img_gray.size
        return changes / total_pixels
    
    def calculate_smoothness_score(self, img_gray):
        """
        Calculate a smoothness score based on local variance.
        
        Args:
            img_gray (numpy.ndarray): Grayscale image array
            
        Returns:
            float: Smoothness score (0-1, higher = less smooth)
        """
        # Calculate local variance using Gaussian blur
        blurred = cv2.GaussianBlur(img_gray.astype(np.float32), (5,5), 0)
        variance = np.mean((img_gray - blurred) ** 2)
        # Normalize to 0-1 range
        return variance / (255**2)
    
    def estimate_noise_level(self, img_gray):
        """
        Estimate the noise level in an image using MAD.
        
        Args:
            img_gray (numpy.ndarray): Grayscale image array
            
        Returns:
            float: Normalized noise level estimate (0-1)
        """
        # Median absolute deviation approach
        median = np.median(img_gray)
        mad = np.median(np.abs(img_gray - median))
        return mad / 255.0
    
    def calculate_gradient_strength(self, img_gray):
        """
        Calculate gradient strength as a texture measure.
        
        Args:
            img_gray (numpy.ndarray): Grayscale image array
            
        Returns:
            float: Normalized gradient strength (0-1)
        """
        # Use Sobel operator to get gradients
        grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        return np.mean(grad_mag) / 255.0
    
    def calculate_edge_density(self, img_gray):
        """
        Calculate the percentage of edge pixels in an image.
        
        Args:
            img_gray (numpy.ndarray): Grayscale image array
            
        Returns:
            float: Percentage of edge pixels
        """
        # Use Canny edge detection
        edges = feature.canny(img_gray, sigma=2)
        edge_pixels = np.sum(edges)
        total_pixels = img_gray.size
        return (edge_pixels / total_pixels) * 100
    
    def detect_compression_artifacts(self, img_gray):
        """
        Heuristically detect compression artifacts in an image.
        
        Args:
            img_gray (numpy.ndarray): Grayscale image array
            
        Returns:
            str: Artifact level description
        """
        # Calculate variance of 8x8 blocks
        h, w = img_gray.shape
        block_vars = []
        
        for i in range(0, h-8, 8):
            for j in range(0, w-8, 8):
                block = img_gray[i:i+8, j:j+8]
                block_vars.append(np.var(block))
        
        avg_block_var = np.mean(block_vars)
        overall_var = np.var(img_gray)
        
        if avg_block_var < 0.1 * overall_var:
            return "High (likely compressed)"
        elif avg_block_var < 0.3 * overall_var:
            return "Medium (possibly compressed)"
        else:
            return "Low (likely uncompressed)"
    
    def determine_image_type(self, img_array):
        """
        Heuristically determine the image type (photo, cartoon, etc.).
        
        Args:
            img_array (numpy.ndarray): Image pixel array
            
        Returns:
            str: Image type classification
        """
        if len(img_array.shape) == 3:  # Color image
            img_gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            img_gray = img_array
        
        # Calculate classification metrics
        unique_colors = len(np.unique(img_gray))
        edge_density = self.calculate_edge_density(img_gray)
        smoothness = self.calculate_smoothness_score(img_gray)
        
        if unique_colors < 50 and edge_density > 5:
            return "Cartoon/Drawing"
        elif smoothness < 0.01 and edge_density < 2:
            return "Synthetic/Computer-generated"
        else:
            return "Natural/Photograph"


if __name__ == "__main__":
    # Create and run the application
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Use modern Fusion style
    
    # Create and show main window
    window = ImageAnalyzerApp()
    window.show()
    
    sys.exit(app.exec_())