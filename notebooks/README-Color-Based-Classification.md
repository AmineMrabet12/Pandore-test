# Color-Based Document Classification

A machine learning approach for classifying driver's licenses, passports, and invoices using color-based features extracted from document images.

## Overview

This notebook implements a color-based classification system that analyzes the color characteristics of document images to distinguish between three document types: driver's licenses, passports, and invoices. The approach focuses on extracting meaningful color features that capture the visual characteristics unique to each document type.

## Dataset

The system works with a balanced dataset of 300 images:
- **Driver's License**: 100 images
- **Passport**: 100 images  
- **Invoice**: 100 images

## Color Feature Extraction

### Feature Types
The system extracts comprehensive color-based features from each document:

1. **Dominant Color Analysis (10 features)**: Identifies the most frequent colors in the image
2. **Color Ratios (10 features)**: Calculates the proportion of each dominant color
3. **Mean RGB Values (3 features)**: Average red, green, and blue channel values
4. **RGB Standard Deviations (3 features)**: Color variation across the image
5. **Color Entropy (1 feature)**: Measures color diversity and complexity
6. **Color Similarity Matching**: Compares dominant colors against class-specific color patterns

**Total: 27+ color-based features per document**

### Color Analysis Process
1. **Image Preprocessing**: Converts images to RGB format for consistent analysis
2. **Pixel Sampling**: Samples every 50th pixel for computational efficiency
3. **Color Quantization**: Groups similar colors by rounding to nearest 20 RGB values
4. **Frequency Analysis**: Identifies most common color patterns
5. **Statistical Computation**: Calculates mean, standard deviation, and entropy metrics

## Classification Approach

### Classifier Design
The system uses a custom color-based classifier that learns distinctive color patterns for each document type:

- **Passport Class**: 
  - Mean RGB: [195.2, 185.6, 174.0]
  - Color Entropy: 2.719 (high diversity)
  - Dominant Colors: 4 unique color patterns

- **Driver's License Class**:
  - Mean RGB: [219.9, 206.0, 205.6] 
  - Color Entropy: 2.189 (moderate diversity)
  - Dominant Colors: 3 unique color patterns

- **Invoice Class**:
  - Mean RGB: [205.4, 198.0, 209.9]
  - Color Entropy: 0.825 (low diversity)
  - Dominant Colors: 4 unique color patterns

### Classification Algorithm
The classifier uses a multi-criteria scoring system:

1. **RGB Distance Similarity**: Compares mean RGB values using Euclidean distance
2. **Entropy Similarity**: Matches color diversity patterns
3. **Dominant Color Matching**: Counts similar colors within a 30-unit RGB threshold
4. **Weighted Scoring**: Combines all criteria with appropriate weights
5. **Confidence Calculation**: Normalizes scores to provide confidence levels

## Performance Analysis

### Classification Results
The system achieves high accuracy on the test dataset:
- **Overall Accuracy**: High accuracy on the 300-image test set
- **Confidence Scoring**: Provides confidence levels for each prediction
- **Detailed Metrics**: Tracks individual component scores for analysis


## Implementation Features

### Core Functions

1. **`create_image_dataset()`**: Creates a structured dataset from image folders
2. **`analyze_pixel_distribution()`**: Comprehensive color analysis and visualization
3. **`extract_color_features()`**: Extracts all color-based features from images
4. **`create_color_classifier()`**: Builds the classification model from training data
5. **`classify_image()`**: Classifies new images with confidence scoring
6. **`classify_and_add_image()`**: Adds new images to dataset based on confidence threshold

### Visualization Capabilities
- **RGB Channel Histograms**: Shows color distribution across channels
- **Dominant Color Analysis**: Displays top 20 most frequent colors
- **Color Palette Visualization**: Visual representation of dominant colors
- **Sample Image Display**: Shows representative images from each class

## Challenges and Solutions

### Main Challenges Encountered

1. **Computational Efficiency**
   - **Problem**: Processing full-resolution images is computationally expensive
   - **Solution**: Implemented pixel sampling (every 50th pixel) for feature extraction
   - **Result**: Maintained accuracy while significantly reducing processing time



### Innovative Approaches Used
- **Multi-criteria Color Analysis**: Combines statistical and frequency-based color features
- **Adaptive Color Quantization**: Groups similar colors to reduce noise
- **Confidence-based Classification**: Provides reliability measures for predictions
- **Visual Analysis Tools**: Comprehensive visualization for understanding color patterns

## Potential Improvements

### Performance Optimization
- **GPU Acceleration**: Use CUDA for faster color processing
- **Feature Selection**: Identify most discriminative color features
- **Caching**: Cache computed features for repeated analysis
- **Parallel Processing**: Process multiple images simultaneously


## Dependencies

- pandas==2.2.3
- numpy==1.25.2
- matplotlib==3.9.3
- seaborn
- pillow==10.4.0
- pathlib

This color-based classification approach provides a robust foundation for document classification while maintaining computational efficiency and providing interpretable results through detailed color analysis.
