let isVideoActive = false;

async function TakePhoto() {// a simple function that displays a live image then takes a photo
    if (isVideoActive) return; // Prevent multiple video feeds
    isVideoActive = true;
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        const video = document.createElement('video');
        video.srcObject = stream;
        video.autoplay = true;
        video.playsInline = true;
        document.body.appendChild(video);

        // Wait for video metadata to load
        await new Promise((resolve) => {
            video.onloadedmetadata = () => {
                resolve();
            };
        });

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const captureBtn = document.createElement('button');
        captureBtn.textContent = 'Capture Photo';
        captureBtn.onclick = () => {
            const context = canvas.getContext('2d');
            context.drawImage(video, 0, 0);
            const photoData = canvas.toDataURL('image/png');
            
            // Convert base64 to blob and download
            fetch(photoData)
                .then(res => res.blob())
                .then(blob => {
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `photo_${Date.now()}.png`;
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    document.body.removeChild(a);
                });
            
            stream.getTracks().forEach(track => track.stop());
            video.remove();
            captureBtn.remove();
            isVideoActive = false;
            
            console.log('Photo captured:', photoData);
        };
        document.body.appendChild(captureBtn);

    } catch (error) {
        console.error('Camera permission denied or error:', error);
        isVideoActive = false;
    }
}