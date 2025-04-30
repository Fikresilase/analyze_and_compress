import sys
import os
import numpy as np
from PIL import Image, ImageStat, ImageFilter, ImageOps
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QLabel, QVBoxLayout, QWidget, QMessageBox, QTextEdit, 
    QScrollArea, QHBoxLayout, QGroupBox, QTabWidget, QFormLayout
)
from PyQt5.QtGui import QPixmap, QFont, QColor, QPainter
from PyQt5.QtCore import Qt, QSize
import math
from collections import Counter
from skimage import filters, feature
import cv2

class ImageAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Property Analyzer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)
        
        # Left panel for image display
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setAlignment(Qt.AlignTop)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setStyleSheet("border: 2px solid #ccc;")
        self.left_layout.addWidget(self.image_label)
        
        # Load image button
        self.load_button = QPushButton("Load Image")
        self.load_button.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        self.load_button.clicked.connect(self.load_image)
        self.left_layout.addWidget(self.load_button)
        
        # Image path label
        self.path_label = QLabel("No image loaded")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("color: #666; font-style: italic;")
        self.left_layout.addWidget(self.path_label)
        
        # Add left panel to main layout
        self.main_layout.addWidget(self.left_panel, stretch=1)
        
        # Right panel for properties display
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setAlignment(Qt.AlignTop)
        
        # Create tabs for different property categories
        self.tabs = QTabWidget()
        
        # Basic Properties Tab
        self.basic_tab = QWidget()
        self.basic_layout = QFormLayout(self.basic_tab)
        self.create_basic_properties_section()
        
        # Pixel Statistics Tab
        self.pixel_tab = QWidget()
        self.pixel_layout = QFormLayout(self.pixel_tab)
        self.create_pixel_statistics_section()
        
        # Histogram Tab
        self.histogram_tab = QWidget()
        self.histogram_layout = QVBoxLayout(self.histogram_tab)
        self.create_histogram_section()
        
        # Compression/Entropy Tab
        self.compression_tab = QWidget()
        self.compression_layout = QFormLayout(self.compression_tab)
        self.create_compression_section()
        
        # Advanced Tab
        self.advanced_tab = QWidget()
        self.advanced_layout = QFormLayout(self.advanced_tab)
        self.create_advanced_section()
        
        # Add tabs
        self.tabs.addTab(self.basic_tab, "Basic Properties")
        self.tabs.addTab(self.pixel_tab, "Pixel Statistics")
        self.tabs.addTab(self.histogram_tab, "Histogram")
        self.tabs.addTab(self.compression_tab, "Compression")
        self.tabs.addTab(self.advanced_tab, "Advanced")
        
        self.right_layout.addWidget(self.tabs)
        self.main_layout.addWidget(self.right_panel, stretch=2)
        
        # Initialize image properties
        self.image = None
        self.image_path = ""
        self.properties = {}
        
        # Style the app
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QTabWidget::pane {
                border: 1px solid #ccc;
                padding: 5px;
            }
            QTabBar::tab {
                padding: 8px;
                background: #e0e0e0;
                border: 1px solid #ccc;
            }
            QTabBar::tab:selected {
                background: #f5f5f5;
                border-bottom: 2px solid #4CAF50;
            }
            QGroupBox {
                border: 1px solid #ccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
            QLabel {
                margin: 2px;
            }
        """)
    
    def create_basic_properties_section(self):
        # Group for basic properties
        basic_group = QGroupBox("Basic Image Properties")
        basic_group_layout = QFormLayout()
        
        # Add properties to the group
        self.basic_properties = [
            ("Dimensions", "1. Width × Height (pixels)"),
            ("File Size", "2. File Size (KB or MB)"),
            ("File Format", "3. File Format"),
            ("Bit Depth", "4. Bits per channel"),
            ("Color Mode", "5. Color Mode"),
            ("Aspect Ratio", "6. Width / Height"),
            ("Image Mode", "7. Color, Grayscale, Binary")
        ]
        
        for prop, desc in self.basic_properties:
            label = QLabel(desc)
            value = QLabel("N/A")
            value.setStyleSheet("font-weight: bold;")
            value.setObjectName(f"basic_{prop.replace(' ', '_')}")
            basic_group_layout.addRow(label, value)
        
        basic_group.setLayout(basic_group_layout)
        self.basic_layout.addWidget(basic_group)
    
    def create_pixel_statistics_section(self):
        # Group for pixel statistics
        pixel_group = QGroupBox("Pixel Value Statistics")
        pixel_group_layout = QFormLayout()
        
        # Add properties to the group
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
        
        for prop, desc in self.pixel_properties:
            label = QLabel(desc)
            value = QLabel("N/A")
            value.setStyleSheet("font-weight: bold;")
            value.setObjectName(f"pixel_{prop.replace(' ', '_')}")
            pixel_group_layout.addRow(label, value)
        
        pixel_group.setLayout(pixel_group_layout)
        self.pixel_layout.addWidget(pixel_group)
    
    def create_histogram_section(self):
        # Group for histogram properties
        histogram_group = QGroupBox("Histogram and Frequency Features")
        histogram_group_layout = QVBoxLayout()
        
        # Add properties to the group
        self.histogram_properties = [
            ("Intensity Histogram", "16. Histogram of grayscale pixel values"),
            ("Byte Frequency", "17. Byte value distribution"),
            ("Redundancy Ratio", "19. 1 − (entropy / log₂(n))")
        ]
        
        form_layout = QFormLayout()
        for prop, desc in self.histogram_properties:
            label = QLabel(desc)
            value = QLabel("N/A")
            value.setStyleSheet("font-weight: bold;")
            value.setObjectName(f"histogram_{prop.replace(' ', '_')}")
            form_layout.addRow(label, value)
        
        # Histogram display
        self.histogram_label = QLabel()
        self.histogram_label.setAlignment(Qt.AlignCenter)
        self.histogram_label.setMinimumHeight(200)
        self.histogram_label.setStyleSheet("border: 1px solid #ccc; background-color: white;")
        
        histogram_group_layout.addLayout(form_layout)
        histogram_group_layout.addWidget(QLabel("Grayscale Histogram:"))
        histogram_group_layout.addWidget(self.histogram_label)
        
        histogram_group.setLayout(histogram_group_layout)
        self.histogram_layout.addWidget(histogram_group)
    
    def create_compression_section(self):
        # Group for compression properties
        compression_group = QGroupBox("Compression-Relevant / Entropy Features")
        compression_group_layout = QFormLayout()
        
        # Add properties to the group
        self.compression_properties = [
            ("Shannon Entropy", "20. Information randomness"),
            ("Repetition Rate", "21. Detect repeating patterns (RLE suitability)"),
            ("Smoothness Score", "22. Local variance, edge flatness"),
            ("Noise Estimation", "23. Using local variance / filters"),
            ("Gradient Strength", "25. Measures texture"),
            ("Edge Density", "26. % of edge pixels using Sobel/Canny")
        ]
        
        for prop, desc in self.compression_properties:
            label = QLabel(desc)
            value = QLabel("N/A")
            value.setStyleSheet("font-weight: bold;")
            value.setObjectName(f"compression_{prop.replace(' ', '_')}")
            compression_group_layout.addRow(label, value)
        
        compression_group.setLayout(compression_group_layout)
        self.compression_layout.addWidget(compression_group)
    
    def create_advanced_section(self):
        # Group for advanced properties
        advanced_group = QGroupBox("Advanced/Optional Context-Aware Features")
        advanced_group_layout = QFormLayout()
        
        # Add properties to the group
        self.advanced_properties = [
            ("Compression Artifacts", "27. Detect blockiness or ringing"),
            ("Heuristic Image Type", "28. Photo, cartoon, etc.")
        ]
        
        for prop, desc in self.advanced_properties:
            label = QLabel(desc)
            value = QLabel("N/A")
            value.setStyleSheet("font-weight: bold;")
            value.setObjectName(f"advanced_{prop.replace(' ', '_')}")
            advanced_group_layout.addRow(label, value)
        
        advanced_group.setLayout(advanced_group_layout)
        self.advanced_layout.addWidget(advanced_group)
    
    def load_image(self):
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
                pixmap = QPixmap(file_path)
                scaled_pixmap = pixmap.scaled(
                    self.image_label.width(), self.image_label.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
                
                # Analyze image properties
                self.analyze_image()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
    
    def analyze_image(self):
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
            
            # 1. Basic Properties
            self.properties['Dimensions'] = f"{self.image.width} × {self.image.height}"
            file_size = os.path.getsize(self.image_path)
            self.properties['File Size'] = self.format_file_size(file_size)
            self.properties['File Format'] = os.path.splitext(self.image_path)[1].upper().replace('.', '')
            self.properties['Bit Depth'] = self.get_bit_depth(self.image)
            self.properties['Color Mode'] = self.image.mode
            self.properties['Aspect Ratio'] = self.calculate_aspect_ratio(self.image.width, self.image.height)
            self.properties['Image Mode'] = self.get_image_mode_category(self.image)
            
            # 2. Pixel Value Statistics
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
            
            # 3. Histogram and Frequency Features
            self.properties['Intensity Histogram'] = "Calculated"  # Will be displayed visually
            self.properties['Byte Frequency'] = "Calculated"  # Not displayed numerically
            self.properties['Redundancy Ratio'] = f"{self.calculate_redundancy_ratio(img_gray):.4f}"
            
            # 4. Compression/Entropy Features
            self.properties['Shannon Entropy'] = f"{self.calculate_shannon_entropy(img_gray):.4f}"
            self.properties['Repetition Rate'] = f"{self.calculate_repetition_rate(img_array):.4f}"
            self.properties['Smoothness Score'] = f"{self.calculate_smoothness_score(img_gray):.4f}"
            self.properties['Noise Estimation'] = f"{self.estimate_noise_level(img_gray):.4f}"
            self.properties['Gradient Strength'] = f"{self.calculate_gradient_strength(img_gray):.4f}"
            self.properties['Edge Density'] = f"{self.calculate_edge_density(img_gray):.2f}%"
            
            # 5. Advanced Features
            self.properties['Compression Artifacts'] = self.detect_compression_artifacts(img_gray)
            self.properties['Heuristic Image Type'] = self.determine_image_type(img_array)
            
            # Update UI with calculated properties
            self.update_property_display()
            
            # Generate and display histogram
            self.display_histogram(img_gray)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to analyze image: {str(e)}")
            print(f"Error: {str(e)}")
    
    def update_property_display(self):
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
        try:
            # Calculate histogram
            hist, bins = np.histogram(img_gray.flatten(), bins=256, range=[0,256])
            
            # Create a simple histogram plot
            hist_img = Image.new('RGB', (256, 100), (255, 255, 255))
            draw = ImageDraw.Draw(hist_img)
            
            # Normalize histogram to fit in 100px height
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
    
    # Helper methods for property calculations
    def format_file_size(self, size):
        """Convert file size to KB or MB string"""
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.2f} KB"
        else:
            return f"{size / (1024 * 1024):.2f} MB"
    
    def get_bit_depth(self, image):
        """Get bit depth of the image"""
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
        """Calculate and simplify aspect ratio"""
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        divisor = gcd(width, height)
        return f"{width//divisor}:{height//divisor} ({width/height:.2f})"
    
    def get_image_mode_category(self, image):
        """Categorize image as color, grayscale, or binary"""
        if image.mode == '1':
            return "Binary"
        elif image.mode in ('L', 'LA'):
            return "Grayscale"
        else:
            return "Color"
    
    def count_unique_colors(self, img_array):
        """Count number of unique colors in image"""
        if len(img_array.shape) == 3:  # Color image
            # Reshape to list of RGB tuples
            pixels = img_array.reshape(-1, img_array.shape[2])
            unique_colors = len(np.unique(pixels, axis=0))
        else:  # Grayscale
            unique_colors = len(np.unique(img_array))
        
        return str(unique_colors)
    
    def calculate_brightness(self, img_gray):
        """Calculate brightness score (0-255)"""
        return np.mean(img_gray)
    
    def get_dominant_color(self, img_array):
        """Get the most dominant color in the image"""
        if len(img_array.shape) == 3:  # Color image
            pixels = img_array.reshape(-1, img_array.shape[2])
            # Count color frequencies
            color_counts = Counter(map(tuple, pixels))
            dominant_color = color_counts.most_common(1)[0][0]
            return f"RGB({dominant_color[0]}, {dominant_color[1]}, {dominant_color[2]})"
        else:  # Grayscale
            return f"Gray({np.argmax(np.bincount(img_array.flatten()))})"
    
    def calculate_shannon_entropy(self, img_gray):
        """Calculate Shannon entropy of the image"""
        hist, _ = np.histogram(img_gray.flatten(), bins=256, range=[0,256])
        hist = hist[hist > 0]  # Remove bins with zero counts
        prob = hist / hist.sum()
        return -np.sum(prob * np.log2(prob))
    
    def calculate_redundancy_ratio(self, img_gray):
        """Calculate redundancy ratio (1 - entropy/log2(n))"""
        entropy = self.calculate_shannon_entropy(img_gray)
        max_entropy = np.log2(256)  # For 8-bit image
        return 1 - (entropy / max_entropy)
    
    def calculate_repetition_rate(self, img_array):
        """Estimate repetition rate in the image (for RLE suitability)"""
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
        """Calculate smoothness score based on local variance"""
        # Calculate local variance using a simple approach
        blurred = cv2.GaussianBlur(img_gray.astype(np.float32), (5,5), 0)
        variance = np.mean((img_gray - blurred) ** 2)
        # Normalize to 0-1 range (higher = less smooth)
        return variance / (255**2)
    
    def estimate_noise_level(self, img_gray):
        """Estimate noise level in the image"""
        # Simple approach using median absolute deviation
        median = np.median(img_gray)
        mad = np.median(np.abs(img_gray - median))
        return mad / 255.0
    
    def calculate_gradient_strength(self, img_gray):
        """Calculate gradient strength (texture measure)"""
        # Use Sobel operator to get gradients
        grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        return np.mean(grad_mag) / 255.0
    
    def calculate_edge_density(self, img_gray):
        """Calculate percentage of edge pixels"""
        # Use Canny edge detection
        edges = feature.canny(img_gray, sigma=2)
        edge_pixels = np.sum(edges)
        total_pixels = img_gray.size
        return (edge_pixels / total_pixels) * 100
    
    def detect_compression_artifacts(self, img_gray):
        """Heuristic to detect compression artifacts"""
        # Simple approach looking for blockiness
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
        """Heuristic to determine image type (photo, cartoon, etc.)"""
        if len(img_array.shape) == 3:  # Color image
            img_gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            img_gray = img_array
        
        # Calculate metrics
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
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show main window
    window = ImageAnalyzerApp()
    window.show()
    
    sys.exit(app.exec_())