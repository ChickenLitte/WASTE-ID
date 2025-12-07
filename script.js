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

// 🔑 Backend API base URL (FastAPI on localhost:8000)
const API_URL = "http://localhost:8000";

// ============================================
// DOM Initialization
// ============================================
document.addEventListener('DOMContentLoaded', function() {
    // Initialize capture button
    CaptureButton = document.getElementById('CaptureBTN');
    if (CaptureButton) {
        CaptureButton.innerHTML = "Open<br>Camera";
        CaptureButton.style.zIndex = '10';
        // Hide button by default (not on Capture tab)
        CaptureButton.style.display = 'none';
        CaptureButton.disabled = true;
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
    
    // Hide and disable capture button
    if (CaptureButton) {
        CaptureButton.style.display = 'none';
        CaptureButton.disabled = true;
    }
    
    document.getElementById('homeSection').style.display = 'block';
}

function showCapture() {
    console.log("Navigating to Capture");
    hideAllSections();
    
    // Show and enable capture button
    if (CaptureButton) {
        CaptureButton.style.display = 'block';
        CaptureButton.disabled = false;
    }
    
    document.getElementById('captureSection').style.display = 'block';
}

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
    
    // Hide and disable capture button
    if (CaptureButton) {
        CaptureButton.style.display = 'none';
        CaptureButton.disabled = true;
    }
    
    document.getElementById('gallerySection').style.display = 'block';
}

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
    
    // Hide and disable capture button
    if (CaptureButton) {
        CaptureButton.style.display = 'none';
        CaptureButton.disabled = true;
    }
    
    document.getElementById('analyticsSection').style.display = 'block';
}

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
// Waste Bin Categorization
// ============================================
function getWasteBinCategory(className) {
    // Categorize waste items into bins: Recycling, Compost, Trash
    const recyclables = [
        'aerosol_cans',
        'aluminum_food_cans',
        'aluminum_soda_cans',
        'cardboard_boxes',
        'cardboard_packaging',
        'glass_beverage_bottles',
        'glass_cosmetic_containers',
        'glass_food_jars',
        'magazines',
        'newspaper',
        'office_paper',
        'paper_cups',
        'plastic_cup_lids',
        'plastic_detergent_bottles',
        'plastic_food_containers',
        'plastic_shopping_bags',
        'plastic_soda_bottles',
        'plastic_straws',
        'plastic_trash_bags',
        'plastic_water_bottles',
        'steel_food_cans'
    ];

    const compostables = [
        'coffee_grounds',
        'eggshells',
        'food_waste',
        'tea_bags'
    ];

    const trash = [
        'clothing',
        'disposable_plastic_cutlery',
        'shoes',
        'styrofoam_cups',
        'styrofoam_food_containers'
    ];

    if (recyclables.includes(className)) {
        return {
            bin: '♻️ Recycling Bin',
            binColor: '#4CAF50',
            description: 'This item can be recycled'
        };
    } else if (compostables.includes(className)) {
        return {
            bin: '🌱 Compost Bin',
            binColor: '#8B4513',
            description: 'This item can be composted'
        };
    } else {
        return {
            bin: '🗑️ Trash Bin',
            binColor: '#f44336',
            description: 'This item goes to trash'
        };
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

        // ✅ Call FastAPI backend on localhost:8000 /predict
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();
        classificationResult = data;
        
        console.log("Classification result:", data);
        
        // Get waste bin categorization
        const binCategory = getWasteBinCategory(data.class_name);
        
        appendResultToBox(data, binCategory);
        
    } catch (err) {
        console.error("Error calling classification API:", err);
        appendResultToBox({ error: `Error: ${err.message}` });
    }
}

// ============================================
// Results Display
// ============================================
function appendResultToBox(result, binCategory) {
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
        resultHTML += `<strong>Item:</strong> ${result.class_name}<br>`;
        resultHTML += `<strong>Confidence:</strong> ${confidencePercent}%<br>`;
        
        // Display bin categorization
        if (binCategory) {
            resultHTML += `<div style="margin-top: 8px; padding: 8px; background-color: ${binCategory.binColor}40; border-left: 4px solid ${binCategory.binColor}; border-radius: 4px;">`;
            resultHTML += `<strong style="color: ${binCategory.binColor};">${binCategory.bin}</strong><br>`;
            resultHTML += `<span style="font-size: 0.9em; color: #333;">${binCategory.description}</span>`;
            resultHTML += `</div>`;
        }
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

