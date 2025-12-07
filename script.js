// ============================================
// Global Variables
// ============================================
let isVideoActive = false;
let stream = null;
let video = null;
let canvas = null;
let classificationResult = null;
let CaptureButton = null;
let videoContainer = null;

// ============================================
// DOM Initialization
// ============================================
document.addEventListener('DOMContentLoaded', function() {
    // Initialize capture button
    CaptureButton = document.getElementById('CaptureBTN');
    if (CaptureButton) {
        CaptureButton.innerHTML = "Open<br>Camera";
        CaptureButton.style.zIndex = '10';
    }

    // Create video container
    videoContainer = document.getElementById('videoContainer');
    if (!videoContainer) {
        videoContainer = document.createElement('div');
        videoContainer.id = 'videoContainer';
        videoContainer.style.display = 'none';
        videoContainer.style.position = 'relative';
        videoContainer.style.margin = '80px auto';
        videoContainer.style.width = 'fit-content';
        videoContainer.style.zIndex = '1000';
        document.body.appendChild(videoContainer);
    }
});

// ============================================
// Navigation Functions
// ============================================
function hideAllSections() {
    document.getElementById('homeSection').style.display = 'none';
    document.getElementById('captureSection').style.display = 'none';
    document.getElementById('gallerySection').style.display = 'none';
    document.getElementById('analyticsSection').style.display = 'none';
}

function showHome() {
    console.log("Navigating to Home");
    hideAllSections();
    
    // Stop video if it's running
    if (isVideoActive) {
        stopVideo();
    }
    
    // Hide results box
    const resultsBox = document.getElementById('resultsBox');
    if (resultsBox) {
        resultsBox.style.display = 'none';
    }
    
    document.getElementById('homeSection').style.display = 'block';
        // Hide and disable capture button
        CaptureButton.style.display = 'none';
        CaptureButton.disabled = true;
}
        // Hide and disable capture button
        CaptureButton.style.display = 'none';
        CaptureButton.disabled = true;

function showCapture() {
    console.log("Navigating to Capture");
    hideAllSections();
    document.getElementById('captureSection').style.display = 'block';
        // Show and enable capture button
        CaptureButton.style.display = 'block';
        CaptureButton.disabled = false;
}
        // Show and enable capture button
        CaptureButton.style.display = 'block';
        CaptureButton.disabled = false;

function showGallery() {
    console.log("Navigating to Gallery");
    hideAllSections();
    
    // Stop video if it's running
    if (isVideoActive) {
        stopVideo();
    }
    
    // Hide results box
    const resultsBox = document.getElementById('resultsBox');
    if (resultsBox) {
        resultsBox.style.display = 'none';
    }
    
    document.getElementById('gallerySection').style.display = 'block';
        // Hide and disable capture button
        CaptureButton.style.display = 'none';
        CaptureButton.disabled = true;
}
        // Hide and disable capture button
        CaptureButton.style.display = 'none';
        CaptureButton.disabled = true;

function showAnalytics() {
    console.log("Navigating to Analytics");
    hideAllSections();
    
    // Stop video if it's running
    if (isVideoActive) {
        stopVideo();
    }
    
    // Hide results box
    const resultsBox = document.getElementById('resultsBox');
    if (resultsBox) {
        resultsBox.style.display = 'none';
    }
    
    document.getElementById('analyticsSection').style.display = 'block';
        // Hide and disable capture button
        CaptureButton.style.display = 'none';
        CaptureButton.disabled = true;
}
        // Hide and disable capture button
        CaptureButton.style.display = 'none';
        CaptureButton.disabled = true;

// ============================================
// Camera & Photo Capture
// ============================================
function clicked() {
    if (!isVideoActive) {
        startVideo();
    } else {
        takePhoto();
        stopVideo();
    }
}

function startVideo() {
    // Create video and canvas elements
    video = document.createElement('video');
    canvas = document.createElement('canvas');

    video.width = 640;
    video.height = 480;
    video.style.border = '2px solid rgb(237,209,142)';
    video.style.display = 'block';
    video.autoplay = true;
    video.playsInline = true;

    canvas.width = 640;
    canvas.height = 480;

    // Add video and button to container
    videoContainer.innerHTML = '';
    videoContainer.appendChild(video);
    videoContainer.appendChild(CaptureButton);
    
    // Center button horizontally and position at bottom of video
    CaptureButton.style.position = 'absolute';
    CaptureButton.style.bottom = '10px';
    CaptureButton.style.left = '50%';
    CaptureButton.style.transform = 'translateX(-50%)';
    CaptureButton.style.marginLeft = '0';
    
    videoContainer.style.display = 'block';
    CaptureButton.innerHTML = "Take<br>Photo";

    // Request camera access
    navigator.mediaDevices.getUserMedia({ video: true })
        .then((mediaStream) => {
            stream = mediaStream;
            video.srcObject = stream;
            video.play();
            isVideoActive = true;
            CaptureButton.innerHTML = "Capture<br>Photo";
        })
        .catch((err) => {
            console.error("Error accessing camera: ", err);
            alert("Unable to access camera. Please check permissions.");
        });
}

function takePhoto() {
    if (!video || !canvas || !stream) {
        console.error("Video or canvas not initialized");
        return;
    }

    // Capture video frame to canvas
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas to blob and send for classification
    canvas.toBlob((blob) => {
        savePhoto(blob);
    }, 'image/jpeg', 0.9);
}

function savePhoto(blob) {
    classifyImage(blob);
}

function stopVideo() {
    if (stream) {
        // Stop all camera tracks
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        isVideoActive = false;
        videoContainer.style.display = 'none';
        
        // Reset button styling
        CaptureButton.style.position = 'static';
        CaptureButton.style.bottom = '';
        CaptureButton.style.left = '';
        CaptureButton.style.transform = '';
        
        document.body.appendChild(CaptureButton);
        CaptureButton.innerHTML = "Open<br>Camera";
        
        // Move results box back to capture section
        const resultsBox = document.getElementById('resultsBox');
        const captureSection = document.getElementById('captureSection');
        if (resultsBox && captureSection && resultsBox.style.display === 'block') {
            captureSection.appendChild(resultsBox);
        }
    }
}

// ============================================
// AI Image Classification
// ============================================
async function classifyImage(blob) {
    try {
        // Create FormData with image blob
        const formData = new FormData();
        formData.append("file", blob, `photo_${Date.now()}.jpg`);

        // Send to classification API
        const response = await fetch("http://localhost:8000/predict", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();
        classificationResult = data;
        
        console.log("Classification result:", data);
        appendResultToBox(data);
        
    } catch (err) {
        console.error("Error calling classification API:", err);
        appendResultToBox({ error: `Error: ${err.message}` });
    }
}

// ============================================
// Results Display
// ============================================
function appendResultToBox(result) {
    const resultsBox = document.getElementById('resultsBox');
    if (!resultsBox) return;

    // Show results box on first result
    if (resultsBox.style.display === 'none') {
        resultsBox.style.display = 'block';
        
        // Position below video if visible
        if (videoContainer && videoContainer.style.display === 'block') {
            resultsBox.style.marginTop = '20px';
            videoContainer.appendChild(resultsBox);
        }
    }

    // Format and append result
    const timestamp = new Date().toLocaleTimeString();
    let resultHTML = `<div style="margin-bottom: 15px; padding-bottom: 15px; border-bottom: 1px solid rgba(255,255,255,0.3);">`;
    resultHTML += `<strong>[${timestamp}]</strong><br>`;
    
    if (result.error) {
        resultHTML += `<span style="color: #ffcccc;">${result.error}</span>`;
    } else {
        const confidencePercent = (result.confidence * 100).toFixed(1);
        resultHTML += `<strong>Category:</strong> ${result.class_name}<br>`;
        resultHTML += `<strong>Confidence:</strong> ${confidencePercent}%`;
    }
    
    resultHTML += `</div>`;
    resultsBox.innerHTML += resultHTML;
}

// ============================================
// Utility Functions
// ============================================
function downloadPhoto(blob) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `photo_${Date.now()}.jpg`;
    
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    URL.revokeObjectURL(url);
}
