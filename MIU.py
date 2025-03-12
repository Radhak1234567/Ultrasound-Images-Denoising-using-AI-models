import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from flask import Flask, request, render_template_string
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as compare_ssim
from flask import Flask, render_template, Response, request, redirect, url_for, session, flash, jsonify
from flask import Flask, render_template, url_for
from flask import Flask, render_template
from flask import Flask, render_template, request, redirect, url_for, flash, Response

# Constants
ADMIN_ID = "radha"
ADMIN_PASSWORD = "1234"

# Flask web app
app = Flask(__name__)
app.secret_key = 'supersecretkey'
# Paths for static files
STATIC_FOLDER = 'static'
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load dataset (now includes Poisson noise)
def load_dataset():
    images = [np.random.rand(100, 100) for _ in range(300)]  # Replace with actual image data
    labels = [0] * 100 + [1] * 100 + [2] * 100  # 0: Gaussian, 1: Salt-and-Pepper, 2: Poisson
    return np.array(images), np.array(labels)

# Function to add Poisson noise (if needed elsewhere)
def add_poisson_noise(image):
    noisy_image = np.random.poisson(image * 255) / 255
    return np.clip(noisy_image, 0, 1)

# Preprocess image (resize and flatten)
def preprocess_image(image):
    resized = cv2.resize(image, (100, 100))  # Resize to standard size
    return resized.flatten()  # Flatten for ML models

# Denoising algorithms
def denoise_image(image):
    denoised_gaussian = cv2.GaussianBlur(image, (5, 5), 0)
    denoised_median = cv2.medianBlur(image, 5)
    denoised_bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    return denoised_gaussian, denoised_median, denoised_bilateral

# Train classifiers
def train_classifiers():
    images, labels = load_dataset()
    images = np.array([preprocess_image(img) for img in images])

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    # KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # SVM Classifier
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)

    # Evaluate classifiers
    knn_acc = knn.score(X_test, y_test)
    svm_acc = svm.score(X_test, y_test)
    
    return knn, svm, X_test, y_test

# Train classifiers
knn_model, svm_model, X_test, y_test = train_classifiers()

# Flask routes

# Routes
@app.route('/')
def login():
    """Admin Login Page."""
    return render_template('login.html')

@app.route('/authenticate', methods=['POST'])
def authenticate():
    """Handle Admin Login."""
    user_id = request.form.get('user_id')
    password = request.form.get('password')

    if user_id == ADMIN_ID and password == ADMIN_PASSWORD:
        session['admin'] = True
        return redirect(url_for('dashboard'))
    else:
        return "Invalid credentials! Please try again."

@app.route('/dashboard')
def dashboard():
    """Admin Dashboard."""
    if 'admin' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'admin' not in session:
        return redirect(url_for('login'))

    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Save and preprocess the uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    preprocessed_image = preprocess_image(image)  # Preprocessing function to match training dataset

    # Predict noise type using KNN and SVM
    knn_pred = knn_model.predict([preprocessed_image])[0]
    svm_pred = svm_model.predict([preprocessed_image])[0]

    noise_types = {0: "Gaussian Noise", 1: "Salt-and-Pepper Noise", 2: "Poisson Noise"}
    noise_type_knn = noise_types[knn_pred]
    noise_type_svm = noise_types[svm_pred]

    # Denoise the image
    denoised_gaussian, denoised_median, denoised_bilateral = denoise_image(image)

    # Calculate MSE and SSIM for each denoised image
    mse_gaussian = np.mean((image - denoised_gaussian) ** 2)
    mse_median = np.mean((image - denoised_median) ** 2)
    mse_bilateral = np.mean((image - denoised_bilateral) ** 2)

    ssim_gaussian = compare_ssim(image, denoised_gaussian)
    ssim_median = compare_ssim(image, denoised_median)
    ssim_bilateral = compare_ssim(image, denoised_bilateral)

    # Assuming the true label is provided or can be determined
    true_label = knn_pred  # Replace with actual ground truth if available
    
    # Update the test dataset dynamically with the true label
    global X_test, y_test
    X_test = np.vstack((X_test, [preprocessed_image]))
    y_test = np.append(y_test, true_label)  # Use the true label, not the predicted label

    # Recompute classification reports after adding the new image
    knn_predictions = knn_model.predict(X_test)
    svm_predictions = svm_model.predict(X_test)
    
    # Now compute classification reports
    knn_report = classification_report(y_test, knn_predictions, output_dict=True)
    svm_report = classification_report(y_test, svm_predictions, output_dict=True)

    # Denoise the image and select the best method based on SSIM
    denoised_methods = denoise_image(image)
    
    ssims = [
        compare_ssim(image, denoised, data_range=denoised.max() - denoised.min())
        for denoised in denoised_methods
    ]

    if noise_type_knn == "Gaussian Noise" or noise_type_svm == "Gaussian Noise":
        best_denoised_image, best_method = max(
            zip(denoised_methods, ['Gaussian', 'Median', 'Bilateral'], ssims),
            key=lambda x: x[2]  # SSIM comparison
        )[:2]
    else:
        best_denoised_image, best_method = max(
            zip(denoised_methods, ['Median', 'Bilateral'], ssims[1:]),
            key=lambda x: x[2]  # SSIM comparison excluding Gaussian
        )[:2]

    # Save images and generate graphs
    original_path = os.path.join(STATIC_FOLDER, 'original.png')
    best_denoised_path = os.path.join(STATIC_FOLDER, f'denoised_{best_method}.png')
    cv2.imwrite(original_path, image)
    cv2.imwrite(best_denoised_path, best_denoised_image)

    create_graphs(
        mse_gaussian, mse_median, mse_bilateral, ssim_gaussian, ssim_median, ssim_bilateral, knn_report, svm_report, image, best_denoised_image
    )

    return f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Processing Results</title>
    <link rel="stylesheet" href="/static/styles1.css">
</head>
<body>

    <!-- Main Content -->
    <div class="container">
        <h2>Image Processing Results</h2>

        <div class="card">
            <h3>Noise Classification</h3>
            <p>KNN: <strong>{noise_type_knn}</strong>, SVM: <strong>{noise_type_svm}</strong></p>
        </div>

        <div class="card">
            <h3>Best Denoising Method: <strong>{best_method}</strong></h3>
        </div>

        <div class="card">
            <h3>Classification Reports</h3>
            <div class="card">
                <h4>KNN Classifier Report</h4>
                <pre>{classification_report(y_test, knn_predictions)}</pre>
            </div>
            <div class="card">
                <h4>SVM Classifier Report</h4>
                <pre>{classification_report(y_test, svm_predictions)}</pre>
            </div>
        </div>

        <div class="graph-container">
            <div>
                <h4>Classification Report Graph</h4>
                <img src="/static/mse_ssim_graph.png" alt="MSE and SSIM Graph">
                <img src="/static/classification_report_graph.png" alt="Classification Report Graph">
            </div>
        </div>

        <div class="result-images">
            <div>
                <h4>Original Image</h4>
                <img src="/static/original.png" alt="Original Image">
            </div>
            <div>
                <h4>Best Denoised Image ({best_method})</h4>
                <img src="/static/denoised_{best_method}.png" alt="Best Denoised Image">
            </div>
        </div>

        <div class="pixel-distribution">
            <h4>Pixel Intensity Distribution</h4>
            <img src="/static/pixel_histogram.png" alt="Pixel Intensity Distribution">
        </div>

        <a href="/dashboard" class="back-button">Back to Dashboard</a>
    </div>

    <!-- Footer -->
    <footer class="footer">
        <p> Image Processing Dashboard</p>
    </footer>
    <!-- Print Button -->
    <button onclick="printPage()">Print Report</button>

    <script>
        function printPage() {{"window.print();"}}
    </script>
</body>
</html>
    '''

@app.route('/add_noise', methods=['POST'])
def add_noise():
    """Add noise to the uploaded image."""
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    noise_type = request.form.get('noise_type')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Load image
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    noisy_image = image.copy()

    # Add selected noise type
    if noise_type == 'gaussian':
        mean = 0
        stddev = 25
        gaussian_noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, gaussian_noise)
    elif noise_type == 'salt_and_pepper':
        s_vs_p = 0.5
        amount = 0.04
        noisy_image = image.copy()
        num_salt = np.ceil(amount * image.size * s_vs_p)
        num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
        
        # Salt
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_image[coords[0], coords[1]] = 255

        # Pepper
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_image[coords[0], coords[1]] = 0
    elif noise_type == 'poisson':
        noisy_image = np.random.poisson(image / 255.0 * 1000).astype(np.float32) / 1000
        noisy_image = (noisy_image * 255).astype(np.uint8)

    # Calculate noise percentage
    difference = cv2.absdiff(image, noisy_image)
    noise_percentage = (np.sum(difference) / np.sum(image)) * 100

    # Save images
    original_path = os.path.join(STATIC_FOLDER, 'original.png')
    noisy_path = os.path.join(STATIC_FOLDER, 'noisy_image.png')
    cv2.imwrite(original_path, image)
    cv2.imwrite(noisy_path, noisy_image)

    return f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Noise Added</title>
    <style>
        body {{
            font-family: 'Poppins', sans-serif;
            text-align: center;
            background: #f7f7f7;
            color: #333;
        }}
        img {{
            max-width: 100%;
            height: auto;
        }}
    </style>
</head>
<body>
    <h1>Noise Added Successfully</h1>
    <p>Noise Type: <strong>{noise_type}</strong></p>
    <p>Noise Percentage: <strong>{noise_percentage:.2f}%</strong></p>
    <div>
        <h3>Original Image:</h3>
        <img src="/static/original.png" alt="Original Image">
    </div>
    <div>
        <h3>Noisy Image:</h3>
        <img src="/static/noisy_image.png" alt="Noisy Image">
    </div>
    <a href="/dashboard">Back to Dashboard</a>
</body>
</html>
    '''


# Function to create graphs and pixel intensity histograms
def create_graphs(mse_gaussian, mse_median, mse_bilateral, ssim_gaussian, ssim_median, ssim_bilateral, knn_report, svm_report, original_image, denoised_image):

    # MSE and SSIM graphs
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].bar(['Gaussian', 'Median', 'Bilateral'], [mse_gaussian, mse_median, mse_bilateral], color='red')
    ax[0].set_title('MSE Comparison')
    ax[0].set_ylabel('MSE')

    ax[1].bar(['Gaussian', 'Median', 'Bilateral'], [ssim_gaussian, ssim_median, ssim_bilateral], color='green')
    ax[1].set_title('SSIM Comparison')
    ax[1].set_ylabel('SSIM')

    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_FOLDER, 'mse_ssim_graph.png'))

    # Classification report graph
    fig, ax = plt.subplots(figsize=(10, 6))

    knn_f1 = knn_report['accuracy']
    svm_f1 = svm_report['accuracy']

    ax.bar(['KNN', 'SVM'], [knn_f1, svm_f1], color='blue')
    ax.set_title('F1-Score Comparison')
    ax.set_ylabel('F1-Score')

    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_FOLDER, 'classification_report_graph.png'))

    # Pixel Intensity Distribution (Histogram)
    plt.figure(figsize=(8, 6))

    plt.hist(original_image.ravel(), bins=256, color='blue', alpha=0.5, label='Original')
    plt.hist(denoised_image.ravel(), bins=256, color='red', alpha=0.5, label='Denoised')
    plt.title('Pixel Intensity Distribution')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_FOLDER, 'pixel_histogram.png'))
    
@app.route('/logout')
def logout():
    """Logout the admin."""
    session.pop('admin', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
