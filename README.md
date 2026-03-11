# Set Solver

A real-time computer vision application that detects, classifies, and solves the card game **Set** using a live webcam feed. Given a board of cards, the system identifies each card's four attributes and finds valid sets automatically.

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green) ![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-orange)

## Demo

| Live Detection | Solution Overlay |
|:-:|:-:|
| Cards outlined in **red** as they are detected | Valid sets highlighted in **green** |

> Point a webcam at a Set board, press `r` to scan, and the solver finds valid sets instantly.

## How It Works

The pipeline processes each frame through three stages:

### 1. Card Detection
- Applies **Gaussian blur** and **Otsu's adaptive thresholding** to segment cards from the background
- Extracts card boundaries via **contour detection** with polygon approximation
- Normalizes each card using a **perspective transform** to correct for rotation and skew

### 2. Feature Classification
Each card is classified across four attributes using traditional computer vision techniques:

| Attribute | Technique |
|-----------|-----------|
| **Shape** | Template matching with **Hu Moments** (`cv2.matchShapes`) against reference images |
| **Color** | **K-means clustering** (K=2) to extract dominant colors, then BGR ratio analysis to distinguish red, green, and purple |
| **Shading** | Pixel distribution analysis on a center crop — solid, striped, or empty based on black/white pixel ratios |
| **Count** | Contour counting with area-based filtering (shapes must be >80% of the largest detected shape) |

### 3. Set-Finding Algorithm
Uses a **constraint-satisfaction approach** — for each pair of cards, it calculates the exact attributes a third card would need (each attribute must be all the same or all different across three cards) and searches the board for a match. Runs in **O(n²)** time.

## Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python** | Core application language |
| **OpenCV** | Image processing, contour detection, template matching, perspective transforms, GUI rendering |
| **NumPy** | Matrix operations, pixel manipulation, K-means implementation |

## Project Structure

```
├── game.py        # Game loop, set-finding algorithm, and OpenCV GUI
├── card.py        # Card feature extraction (shape, color, shading, count)
├── scanner.py     # Image preprocessing and card boundary detection
└── cards/         # Shape reference templates and test images
```

## Getting Started

### Prerequisites
- Python 3.8+
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)

### Usage
```bash
python game.py
```
- Press **`r`** to scan the current frame and find sets
- Press **`q`** to quit

The application also supports a static test mode using a sample image — toggle by changing the mode parameter in `game.py`.

## Key Takeaways

- **Traditional CV is powerful** — Template matching with Hu Moments and K-means clustering proved effective for structured card classification without needing deep learning
- **Image preprocessing matters** — Getting reliable card detection required careful tuning of blur kernels, threshold methods, and contour filtering parameters
- **Perspective correction is essential** — Cards at an angle caused misclassification until perspective transforms were added to normalize each card's orientation
- **Color spaces are tricky** — Distinguishing red from purple under varying lighting conditions was one of the harder problems, solved through color ratio analysis rather than fixed thresholds

## Author
Derek Jain
