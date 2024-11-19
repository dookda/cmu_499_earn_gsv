from flask import Flask, Response, send_file
import cv2
import numpy as np
import rasterio
import os
from matplotlib import pyplot as plt
from io import BytesIO

app = Flask(__name__)


def process_geotiff(filepath):
    """Process a GeoTIFF file and return visualization"""
    with rasterio.open(filepath) as dataset:
        # Read the first band
        band1 = dataset.read(1)

        # Normalize the data for visualization
        min_val = np.percentile(band1, 2)
        max_val = np.percentile(band1, 98)
        normalized = np.clip((band1 - min_val) * 255 /
                             (max_val - min_val), 0, 255).astype(np.uint8)

        # Apply OpenCV enhancement
        enhanced = cv2.equalizeHist(normalized)

        # Create a color map
        plt.figure(figsize=(10, 10))
        plt.imshow(enhanced, cmap='viridis')
        plt.colorbar(label='Pixel Value')
        plt.title('Processed GeoTIFF with OpenCV Enhancement')

        # Save to bytes buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()

        return buffer


@app.route('/')
def index():
    return """
    <h1>Geo-Processing Demo</h1>
    <p>Available endpoints:</p>
    <ul>
        <li><a href="/opencv-demo">OpenCV Demo</a></li>
        <li><a href="/process-geotiff">Process Sample GeoTIFF</a> (requires a GeoTIFF file in /data)</li>
    </ul>
    """


@app.route('/opencv-demo')
def opencv_demo():
    # Create a black image
    img = np.zeros((480, 640, 3), np.uint8)

    # Add some OpenCV drawing
    cv2.putText(img, 'OpenCV + GDAL Demo', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Draw a circle
    cv2.circle(img, (320, 240), 100, (0, 255, 0), 3)

    # Encode the image
    _, buffer = cv2.imencode('.jpg', img)
    return Response(buffer.tobytes(), mimetype='image/jpeg')


@app.route('/process-geotiff')
def process_sample_geotiff():
    # Assuming there's a sample.tif in the data directory
    sample_path = '/data/sample.tif'

    if not os.path.exists(sample_path):
        return "Please place a GeoTIFF file named 'sample.tif' in the data directory"

    buffer = process_geotiff(sample_path)
    return send_file(buffer, mimetype='image/png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
