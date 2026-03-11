# Set Solver

A computer vision-based application to play the card game **Set**.

## Features
- **Real-time Card Detection**: Uses OpenCV to detect cards from a webcam feed.
- **AI-Powered Recognition**: Employs a custom Convolutional Neural Network (CNN) to identify the four features of each card (Shape, Color, Number, Shading).
- **Automatic Solution Finding**: Automatically identifies and highlights valid "Sets" of three cards from the board.
- **Interactive Interface**: Clean GUI to display the camera feed, detected cards, and solutions.

## Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package installer)

### Installation
1.  **Clone the repository** (or download the source code).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Usage
1.  Run the application from your terminal:
    ```bash
    python main.py
    ```
2.  Point your webcam at a valid Set game board.
3.  The application will automatically detect cards and display solutions in real-time.

## Project Structure
- `main.py`: Main entry point for the application.
- `camera.py`: Handles webcam access and image capture.
- `card_detector.py`: Logic for detecting card boundaries in the image.
- `card_recognizer.py`: Neural network model for classifying card features.
- `solver.py`: Algorithm to find valid sets on the board.
- `gui.py`: User interface for displaying the game state.

## Author
Derek Jain
