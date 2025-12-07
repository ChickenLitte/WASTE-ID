let isVideoActive = false;
let stream = null;
let video = null;
let canvas = null;
let classificationResult = null;
let CaptureButton = null;
let videoContainer = null;

// Initialize after DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    CaptureButton = document.getElementById('CaptureBTN');
    if (CaptureButton) {
        CaptureButton.innerHTML = "Open<br>Camera";
        CaptureButton.style.zIndex = '10';
    }

    // Create a container for the video if it doesn't exist
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

// Function to hide all sections
function hideAllSections() {
    document.getElementById('homeSection').style.display = 'none';
    document.getElementById('captureSection').style.display = 'none';
    document.getElementById('gallerySection').style.display = 'none';
    document.getElementById('analyticsSection').style.display = 'none';
}

// Navigation functions
function showHome() {
    console.log("Navigating to Home");
    hideAllSections();
    document.getElementById('homeSection').style.display = 'block';
}

function showCapture() {
    console.log("Navigating to Capture");
    hideAllSections();
    document.getElementById('captureSection').style.display = 'block';
}

function showGallery() {
    console.log("Navigating to Gallery");
    hideAllSections();
    document.getElementById('gallerySection').style.display = 'block';
}

function showAnalytics() {
    console.log("Navigating to Analytics");
    hideAllSections();
    document.getElementById('analyticsSection').style.display = 'block';
}

function clicked() {
    if (!isVideoActive) {
        startVideo();
    } else {
        takePhoto();
        stopVideo();
    }
}
function startVideo() {
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

    videoContainer.innerHTML = '';
    videoContainer.appendChild(video);
    videoContainer.appendChild(CaptureButton);
    
    // Apply absolute positioning to button inside video container
    CaptureButton.style.position = 'absolute';
    CaptureButton.style.bottom = '5px';
    CaptureButton.style.marginLeft = '50%';
    CaptureButton.style.transform = 'translateX(-70%)';
    
    videoContainer.style.display = 'block';

    CaptureButton.innerHTML = "Take<br>Photo";

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

    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
        savePhoto(blob);
    }, 'image/jpeg', 0.9);
}

function savePhoto(blob) {
    // Send the blob to the AI image analysis API
    classifyImage(blob);
}

async function classifyImage(blob) {
    try {
        // Create FormData to send the image
        const formData = new FormData();
        formData.append("file", blob, `photo_${Date.now()}.jpg`);

        // Send to your AI API endpoint
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
        
        // Display the result in the results box
        appendResultToBox(data);
        
        // Optionally save the image after classification
        // downloadPhoto(blob);
    } catch (err) {
        console.error("Error calling classification API:", err);
        appendResultToBox({ error: `Error: ${err.message}` });
    }
}

function appendResultToBox(result) {
    const resultsBox = document.getElementById('resultsBox');
    if (!resultsBox) return;

    // Show the results box if it's hidden
    if (resultsBox.style.display === 'none') {
        resultsBox.style.display = 'block';
        
        // Position it below the video container if it's visible
        if (videoContainer && videoContainer.style.display === 'block') {
            resultsBox.style.marginTop = '20px';
            videoContainer.appendChild(resultsBox);
        }
    }

    // Create a result entry
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
    
    // Append to results box
    resultsBox.innerHTML += resultHTML;
}

function downloadPhoto(blob) {
    // Optional: download the photo along with classification
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `photo_${Date.now()}.jpg`;
    
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    URL.revokeObjectURL(url);
}

function stopVideo() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        isVideoActive = false;
        videoContainer.style.display = 'none';
        
        // Remove absolute positioning from button
        CaptureButton.style.position = 'static';
        CaptureButton.style.bottom = '';
        CaptureButton.style.left = '';
        CaptureButton.style.transform = '';
        
        document.body.appendChild(CaptureButton);
        CaptureButton.innerHTML = "Open<br>Camera";
        
        // Move results box back to capture section if it exists
        const resultsBox = document.getElementById('resultsBox');
        const captureSection = document.getElementById('captureSection');
        if (resultsBox && captureSection && resultsBox.style.display === 'block') {
            captureSection.appendChild(resultsBox);
        }
    }
}
