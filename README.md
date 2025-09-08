# Document Processing System

A complete pipeline for classifying and extracting information from driver's licenses, passports, and invoices using computer vision and machine learning.

## Document Classification

### Feature Extraction Approach
We implemented a comprehensive feature extraction system that analyzes documents using 7 different types of features:

1. **Histogram Features (256 features)**: Analyzes pixel brightness distribution
2. **Statistical Features (5 features)**: Calculates mean, standard deviation, variance, median, and percentiles
3. **Local Binary Pattern (LBP) Features (256 features)**: Detects texture patterns using 8-point circular patterns
4. **Edge Density Features (1 feature)**: Uses Canny edge detection to measure structural complexity
5. **Shape Features (4 features)**: Analyzes document contours (area, perimeter, aspect ratio, solidity)
6. **Spatial Features (3 features)**: Captures image dimensions and proportions
7. **Frequency Domain Features (64 features)**: Uses DCT coefficients to analyze layout patterns

**Total: 589 features per document**

### Classifier Choice
We tested two classifiers and selected the best performing one:
- **Support Vector Machine (SVM)** with RBF kernel
- **Random Forest** with 100 estimators

Both were evaluated using 5-fold cross-validation, and the best performer was chosen based on test accuracy.

### Classification Performance
The system uses:
- 80/20 train-test split with stratified sampling
- StandardScaler for feature normalization
- Cross-validation for model stability assessment
- Confusion matrix and classification report for detailed analysis

## Information Extraction

### Region Extraction Method
We defined specific coordinate-based regions for each document type:

- **Driver's License**: Single region (x: 400-800, y: 150-600) containing all text fields
- **Passport**: Single region (x: 350-1100, y: 90-590) for comprehensive text extraction
- **Invoice**: Three distinct regions:
  - Header information (x: 20-220, y: 140-230)
  - Main content (x: 320-550, y: 50-230)
  - Footer/total information (x: 0-570, y: 650-750)

Coordinates are automatically scaled to match actual image dimensions.

### OCR Approach
We implemented a dual-engine OCR system:

1. **Primary Engine**: PyTesseract
   - Optimized for alphanumeric characters
   - Custom configuration for better text recognition
   - PSM mode 6 for uniform text blocks

2. **Fallback Engine**: EasyOCR
   - Handles complex layouts and rotated text
   - Better performance on low-quality images
   - Automatic language detection

### Text Post-Processing
We applied field-specific validation and cleaning:

- **Dates**: Recognizes multiple formats (DD/MM/YYYY, DD-MM-YYYY, etc.)
- **Email**: Regex validation with domain checking
- **ID Numbers**: Alphanumeric pattern matching
- **Names**: Title case formatting and special character handling
- **Addresses**: Multi-line text cleaning and formatting
- **Monetary Amounts**: Currency symbol recognition and number formatting

## Challenges and Solutions

### Main Challenges Encountered

1. **OpenCV DCT Limitation**
   - **Problem**: DCT function fails on odd-sized images
   - **Solution**: Added dimension checking and cropping to even sizes before DCT
   - **Result**: Maintained frequency domain features while ensuring stability

2. **OCR Engine Compatibility**
   - **Problem**: EasyOCR had numpy array format issues
   - **Solution**: Proper image format conversion (grayscale to RGB) and data type validation
   - **Result**: Reliable text extraction across different image formats

3. **Region Definition Precision**
   - **Problem**: Accurate text field localization across document variations
   - **Solution**: User-provided coordinate-based regions with proportional scaling
   - **Result**: High-precision text extraction with minimal false positives

### Innovative Approaches Used
- Multi-modal feature extraction combining traditional and modern techniques
- Dual-engine OCR with intelligent fallback
- Coordinate-based region extraction with adaptive scaling
- Comprehensive error handling and graceful degradation

## Potential Improvements

### Information Extraction Improvements
- **Better OCR**: Implement transformer-based OCR models (TrOCR, PaddleOCR)
- **Smart Region Detection**: Use object detection models (YOLO, R-CNN) for automatic field localization
- **Enhanced Post-Processing**: Add Named Entity Recognition (NER) for better field validation

### Edge Case Handling
- **Document Variations**: Add more document templates and formats
- **Quality Issues**: Implement better image preprocessing for low-quality scans
- **New Document Types**: Create flexible framework for adding new document types easily

## Production Considerations

### Deployment
The system can be deployed as:
- **REST API**: For web and mobile applications
- **Microservice**: In containerized environments (Docker/Kubernetes)
- **Batch Processing**: For large document processing workflows

### Scalability and Performance
- **Horizontal Scaling**: Process multiple documents in parallel
- **Caching**: Cache preprocessed features and model predictions

### Security and Privacy
- **Data Encryption**: Encrypt documents during transmission and storage
- **Audit Logging**: Track all document processing activities
- **Secure Storage**: Use encrypted storage for processed documents

## To install dependencies

`pip install -r requirements.txt`


## Dependencies

- opencv-python==4.12.0.88
- pillow==10.4.0
- numpy==1.25.2
- pandas==2.2.3
- scikit-learn==1.5.2
- scipy==1.13.1
- matplotlib==3.9.3
- pytesseract==0.3.13
- easyocr==1.7.2