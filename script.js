let isVideoActive = false;
let stream = null;
let video = null;
let canvas = null;

const CaptureButton = document.getElementById('CaptureBTN');
CaptureButton.innerHTML = "Open<br>Camera";

// Create a container for the video if it doesn't exist
let videoContainer = document.getElementById('videoContainer');
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

// Style the button for positioning
CaptureButton.style.zIndex = '10';

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
    // Create object URL for the blob
    const url = URL.createObjectURL(blob);
    
    // Create a download link
    const a = document.createElement('a');
    a.href = url;
    a.download = `photo_${Date.now()}.jpg`;
    
    // Trigger download
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    // Clean up the object URL
    URL.revokeObjectURL(url);
    
    console.log("Photo saved:", a.download);
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
    }
}
