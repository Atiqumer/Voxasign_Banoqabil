import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";
import { extractPslV2Features, scalePslV2Features } from "./psl_v2_features.js";

// Canvas elements
const video = document.getElementById("webcam");
const liveCanvas = document.getElementById("output_canvas");
const liveCtx = liveCanvas.getContext("2d");
const imageCanvas = document.getElementById("image_canvas");

// State
let handLandmarker;
const models = { asl: null, psl: null };
let pslAssets = null;
let signLanguage = "asl";
let currentPrediction = "";
let lastVideoTime = -1;
let activeMode = null; // 'live' | 'upload'
let cameraStream = null;

// Detect if mobile and set default camera
const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
let currentFacingMode = isMobile ? "environment" : "user"; // Default to back camera on mobile, front on desktop
let availableCameras = [];

// Auto-detection state
let detectionStartTime = null;
let lastDetectedLetter = "";
const HOLD_DURATION = 1500; // 1.5 seconds
const AUTO_CONFIDENCE_THRESHOLD = 0.85; // 85% confidence for auto-add

// Expose prediction to parent UI
window.currentPrediction = "";
window.clearCurrentPrediction = () => {
    currentPrediction = "";
    window.currentPrediction = "";
    if (window.updatePrediction) window.updatePrediction("", 0);
};

const aslLabelMap = [
    "A","B","Blank","C","D","E","F","G","H","I",
    "J","K","L","M","N","O","P","Q","R","S",
    "T","U","V","W","X","Y","Z"
];

const modelConfig = {
    asl: {
        modelPath: "./web_model/model.json",
        weightsPath: "./web_model/weights.json",
        inputSize: 63,
        confidenceThreshold: 0.75,
        hasBlankClass: true,
    },
    psl: {
        modelPath: "./web_model/psl_v2/model.json",
        weightsPath: "./web_model/psl_v2/weights.json",
        scalerPath: "./web_model/psl_v2/scaler.json",
        classMapPath: "./web_model/psl_v2/class_map.json",
        inputSize: 115,
        confidenceThreshold: 0.75,
        hasBlankClass: false,
    },
};

function base64ToArrayBuffer(base64) {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
        bytes[index] = binary.charCodeAt(index);
    }
    return bytes.buffer;
}

async function loadModelFromJsonWeights(config, expectedWeightBytes) {
    const [modelResponse, weightsResponse] = await Promise.all([
        fetch(config.modelPath, { cache: "no-store" }),
        fetch(config.weightsPath, { cache: "no-store" }),
    ]);
    if (!modelResponse.ok || !weightsResponse.ok) {
        throw new Error("Model assets are missing. Run the web-weight export command first.");
    }

    const [modelJSON, weightsJSON] = await Promise.all([
        modelResponse.json(),
        weightsResponse.json(),
    ]);
    const weightData = base64ToArrayBuffer(weightsJSON.data);
    if (weightsJSON.encoding !== "base64" || weightData.byteLength !== expectedWeightBytes) {
        throw new Error(
            `Model weights are invalid: expected ${expectedWeightBytes} bytes, received ${weightData.byteLength}.`
        );
    }

    return tf.loadLayersModel({
        load: async () => ({
            modelTopology: modelJSON.modelTopology,
            format: modelJSON.format,
            generatedBy: modelJSON.generatedBy,
            convertedBy: modelJSON.convertedBy,
            weightSpecs: modelJSON.weightsManifest[0].weights,
            weightData,
        }),
    });
}

async function loadModel(language) {
    if (models[language]) return models[language];

    const config = modelConfig[language];
    if (!config) throw new Error(`Unsupported sign language: ${language}`);

    if (language === "psl") {
        const [model, scalerResponse, classMapResponse] = await Promise.all([
            loadModelFromJsonWeights(config, 298896),
            fetch(config.scalerPath),
            fetch(config.classMapPath),
        ]);
        if (!scalerResponse.ok || !classMapResponse.ok) {
            throw new Error("PSL assets are missing. Export and convert the verified PSL V2 model first.");
        }
        pslAssets = {
            scaler: await scalerResponse.json(),
            classMap: await classMapResponse.json(),
        };
        if (pslAssets.scaler.feature_count !== 115) {
            throw new Error("PSL scaler must contain the verified 115-feature contract.");
        }
        models.psl = model;
    } else {
        models.asl = await loadModelFromJsonWeights(config, 237164);
    }
    console.log(`✅ ${language.toUpperCase()} model loaded`);
    return models[language];
}

async function setSignLanguage(language) {
    if (!modelConfig[language] || language === signLanguage) return;

    try {
        if (window.updatePrediction) window.updatePrediction("", 0);
        resetAutoDetection();
        await loadModel(language);
        signLanguage = language;
        window.currentPrediction = "";
        console.log(`Switched to ${language.toUpperCase()} inference.`);
    } catch (error) {
        console.error(`Could not switch to ${language.toUpperCase()}:`, error);
        alert(error.message);
        window.dispatchEvent(new CustomEvent("signLanguageRejected", { detail: signLanguage }));
    }
}

window.addEventListener("signLanguageChange", (event) => setSignLanguage(event.detail));

// ── INIT ──
async function initialize() {
    console.log("🚀 Initializing VoxaSign...");
    try {
        console.log("Loading MediaPipe HandLandmarker...");
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            },
            runningMode: "VIDEO",
            numHands: 1
        });
        console.log("✅ MediaPipe HandLandmarker loaded");
        
        console.log("Loading ASL TensorFlow.js model...");
        await loadModel("asl");
        console.log("✅ AI Engine Ready - You can now use the app!");
    } catch (error) {
        console.error("❌ Initialization failed:", error);
        const reason = error instanceof Error ? error.message : String(error);
        alert(
            "Failed to load AI models.\n\n" +
            `Reason: ${reason}\n\n` +
            "Make sure the server was started from D:\\voxasign and that you opened http://localhost:8000/studio.html."
        );
    }
}

// ── CAMERA ──
async function getCameraDevices() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        availableCameras = devices.filter(device => device.kind === 'videoinput');
        console.log(`Found ${availableCameras.length} camera(s)`);
        return availableCameras;
    } catch (err) {
        console.error("Error enumerating devices:", err);
        return [];
    }
}

async function startCamera(facingMode = currentFacingMode) {
    if (cameraStream) {
        stopCamera(); // Stop existing stream first
    }
    
    try {
        // Update current facing mode
        currentFacingMode = facingMode;
        
        const constraints = {
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: { ideal: facingMode }
            }
        };
        
        console.log(`Starting camera with facingMode: ${facingMode}`);
        cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = cameraStream;
        
        video.onloadedmetadata = () => {
            video.play();
            predictWebcam();
            
            // Show/update camera switch button
            if (window.updateCameraSwitchButton) {
                window.updateCameraSwitchButton(true);
            }
        };
        
        // Get actual camera info
        const videoTrack = cameraStream.getVideoTracks()[0];
        const settings = videoTrack.getSettings();
        console.log(`✅ Camera started: ${settings.facingMode || 'unknown'} (${settings.width}x${settings.height})`);
        
    } catch (err) {
        console.error("Camera access denied:", err);
        alert("Camera access denied. Please grant camera permission and try again.");
    }
}

async function switchCamera() {
    const newFacingMode = currentFacingMode === "user" ? "environment" : "user";
    console.log(`Switching camera from ${currentFacingMode} to ${newFacingMode}`);
    await startCamera(newFacingMode);
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
        video.srcObject = null;
        
        // Hide camera switch button
        if (window.updateCameraSwitchButton) {
            window.updateCameraSwitchButton(false);
        }
    }
}

// Expose camera switch function globally
window.switchCamera = switchCamera;

// ── MODE CHANGE LISTENER ──
window.addEventListener('modeChange', async (e) => {
    activeMode = e.detail;

    if (activeMode === 'live') {
        await startCamera();
    } else {
        stopCamera();
        currentPrediction = "";
        window.currentPrediction = "";
        if (window.updatePrediction) window.updatePrediction("", 0);
    }
});

// ── DRAW UI on canvas ──
function drawLandmarkBox(ctx, landmarks, w, h) {
    const x = landmarks.map(l => l.x * w);
    const y = landmarks.map(l => l.y * h);
    const minX = Math.min(...x), maxX = Math.max(...x);
    const minY = Math.min(...y), maxY = Math.max(...y);
    const pad = 20;

    // Glow box
    ctx.shadowColor = "#00c6ff";
    ctx.shadowBlur = 12;
    ctx.strokeStyle = "#00c6ff";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    // Use rect instead of roundRect for better browser compatibility
    ctx.rect(minX - pad, minY - pad, (maxX - minX) + pad * 2, (maxY - minY) + pad * 2);
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Landmark dots
    ctx.fillStyle = "rgba(0,198,255,0.8)";
    landmarks.forEach(l => {
        ctx.beginPath();
        ctx.arc(l.x * w, l.y * h, 3, 0, Math.PI * 2);
        ctx.fill();
    });
}

// ── PROCESS LANDMARKS ──
function processLandmarks(landmarks) {
    const wrist = landmarks[0];
    let coords = [];
    for (let i = 0; i < landmarks.length; i++) {
        coords.push(landmarks[i].x - wrist.x);
        coords.push(landmarks[i].y - wrist.y);
        coords.push(landmarks[i].z - wrist.z);
    }
    const maxVal = Math.max(...coords.map(Math.abs)) || 1;
    return coords.map(c => c / maxVal);
}

// ── INFERENCE ──
async function runInference(landmarks) {
    const model = models[signLanguage];
    if (!model) {
        console.error("Model not loaded yet");
        return;
    }

    const config = modelConfig[signLanguage];
    let inputData;
    let label;
    if (signLanguage === "psl") {
        inputData = scalePslV2Features(extractPslV2Features(landmarks), pslAssets.scaler);
    } else {
        inputData = processLandmarks(landmarks);
    }

    const inputTensor = tf.tensor2d(inputData, [1, config.inputSize]);
    const prediction = model.predict(inputTensor);
    const scores = await prediction.data();
    const maxIdx = scores.indexOf(Math.max(...scores));
    const confidence = scores[maxIdx];
    label = signLanguage === "psl" ? pslAssets.classMap[String(maxIdx)] : aslLabelMap[maxIdx];

    console.log(`Prediction: ${label} (${(confidence * 100).toFixed(1)}%)`);

    // PSL has no Blank class. Its useful diagnostic output is the top class plus
    // confidence, including uncertain results (notably the Tuey/Daal pair).
    // ASL retains its existing confidence/Blank filter.
    const shouldDisplay = signLanguage === "psl"
        || (confidence > config.confidenceThreshold && (!config.hasBlankClass || label !== "Blank"));

    if (shouldDisplay) {
        currentPrediction = label;
        window.currentPrediction = label;
        if (window.updatePrediction) window.updatePrediction(label, confidence);
        
        // Check for auto-detection
        checkAutoAdd(label, confidence);
    } else {
        currentPrediction = "";
        window.currentPrediction = "";
        if (window.updatePrediction) window.updatePrediction("", 0);
        
        // Reset auto-detection
        resetAutoDetection();
    }

    inputTensor.dispose();
    prediction.dispose();
}

// ── AUTO-DETECTION LOGIC ──
function checkAutoAdd(letter, confidence) {
    // Only in live mode and if auto-detection is enabled
    if (activeMode !== 'live' || !window.isAutoDetectionEnabled || !window.isAutoDetectionEnabled()) {
        resetAutoDetection();
        return;
    }
    
    // Check if confidence is high enough
    if (confidence < AUTO_CONFIDENCE_THRESHOLD || letter === "Blank") {
        resetAutoDetection();
        return;
    }
    
    // Check if same letter as before
    if (letter === lastDetectedLetter && detectionStartTime !== null) {
        // Calculate hold time
        const holdTime = Date.now() - detectionStartTime;
        const progress = Math.min(holdTime / HOLD_DURATION, 1);
        
        // Update progress bar
        if (window.updateProgressBar) {
            window.updateProgressBar(progress);
        }
        
        // Auto-add if held long enough
        if (holdTime >= HOLD_DURATION) {
            if (window.autoAddLetter && window.autoAddLetter(letter)) {
                // Successfully added, reset detection
                resetAutoDetection();
            }
        }
    } else {
        // New letter detected, start timer
        lastDetectedLetter = letter;
        detectionStartTime = Date.now();
        
        if (window.updateProgressBar) {
            window.updateProgressBar(0);
        }
    }
}

function resetAutoDetection() {
    lastDetectedLetter = "";
    detectionStartTime = null;
    
    if (window.updateProgressBar) {
        window.updateProgressBar(0);
    }
}

// ── LIVE WEBCAM LOOP ──
async function predictWebcam() {
    if (activeMode !== 'live') return;

    if (video.currentTime !== lastVideoTime) {
        liveCanvas.width = video.videoWidth;
        liveCanvas.height = video.videoHeight;
        lastVideoTime = video.currentTime;

        const result = handLandmarker.detectForVideo(video, performance.now());
        liveCtx.clearRect(0, 0, liveCanvas.width, liveCanvas.height);

        if (result.landmarks && result.landmarks.length > 0) {
            drawLandmarkBox(liveCtx, result.landmarks[0], liveCanvas.width, liveCanvas.height);
            await runInference(result.landmarks[0]);
        } else {
            currentPrediction = "";
            window.currentPrediction = "";
            if (window.updatePrediction) window.updatePrediction("", 0);
        }
    }

    if (activeMode === 'live') {
        window.requestAnimationFrame(predictWebcam);
    }
}

// ── IMAGE UPLOAD INFERENCE ──
document.getElementById("imageUpload").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file || !handLandmarker || !models[signLanguage]) {
        console.log("Upload skipped - not ready:", { file: !!file, handLandmarker: !!handLandmarker, model: !!models[signLanguage] });
        return;
    }

    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = async () => {
        try {
            // The display canvas is handled by studio.html
            // We run inference on the original image
            await handLandmarker.setOptions({ runningMode: "IMAGE" });
            const result = await handLandmarker.detect(img);

            if (result.landmarks && result.landmarks.length > 0) {
                console.log("✓ Hand detected in uploaded image");
                // Draw box on image_canvas after it's been drawn by studio.html
                setTimeout(() => {
                    const ctx2 = imageCanvas.getContext('2d');
                    drawLandmarkBox(ctx2, result.landmarks[0], imageCanvas.width, imageCanvas.height);
                }, 100);
                await runInference(result.landmarks[0]);
            } else {
                console.log("⚠ No hand detected in uploaded image");
                if (window.updatePrediction) window.updatePrediction("", 0);
            }

            // Switch back to VIDEO mode for live camera
            await handLandmarker.setOptions({ runningMode: "VIDEO" });
        } catch (error) {
            console.error("Image inference error:", error);
            if (window.updatePrediction) window.updatePrediction("", 0);
        }
    };
});

initialize();


