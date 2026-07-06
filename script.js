/* ============================================
   CornVision — YOLOv8 TensorFlow.js Detection
   ============================================ */

const MODEL_PATH = './model/model.json';
const INPUT_SIZE = 640;
const CONF_THRESHOLD = 0.35;
const IOU_THRESHOLD = 0.45;
const CLASS_NAMES = ['blight', 'common_rust', 'gray_leaf_spot', 'healthy'];
const CLASS_COLORS = {
    blight: { bg: '#8B4513', text: '#FFFFFF' },
    common_rust: { bg: '#E67E22', text: '#FFFFFF' },
    gray_leaf_spot: { bg: '#7F8C8D', text: '#FFFFFF' },
    healthy: { bg: '#27AE60', text: '#FFFFFF' },
};

const DISEASE_DATA = {
    blight: {
        title: "Hawar Daun (Blight)",
        description: "Disebabkan oleh jamur <em>Exserohilum turcicum</em>. Gejala berupa bercak kecil berbentuk oval yang memanjang seperti cerutu berwarna coklat keabu-abuan.",
        recommendations: [
            "Gunakan varietas jagung yang tahan terhadap hawar daun.",
            "Lakukan pergiliran tanaman (rotasi) dengan tanaman non-graminae.",
            "Bersihkan sisa-sisa tanaman yang terinfeksi setelah panen.",
            "Semprotkan fungisida berbahan aktif mankozeb atau benomil sesuai dosis jika serangan meluas."
        ]
    },
    common_rust: {
        title: "Karat Daun (Common Rust)",
        description: "Disebabkan oleh jamur <em>Puccinia sorghi</em>. Terlihat sebagai bintik-bintik pustula berwarna oranye-coklat pada permukaan daun.",
        recommendations: [
            "Pilih waktu tanam yang tepat untuk menghindari puncak kelembapan tinggi.",
            "Atur jarak tanam agar sirkulasi udara di area pertanaman lancar.",
            "Aplikasi fungisida sistemik berbahan aktif triazol jika intensitas serangan tinggi.",
            "Pastikan sistem drainase lahan berjalan dengan baik."
        ]
    },
    gray_leaf_spot: {
        title: "Bercak Daun Abu-abu (Gray Leaf Spot)",
        description: "Disebabkan oleh jamur <em>Cercospora zeae-maydis</em>. Bercak berbentuk persegi panjang mengikuti alur tulang daun.",
        recommendations: [
            "Lakukan pengolahan tanah yang sempurna untuk memendam sisa tanaman sakit.",
            "Kurangi kelembapan lingkungan pertanaman.",
            "Gunakan fungisida pelindung sebelum infeksi meluas ke daun di atas tongkol.",
            "Pilih benih yang memiliki ketahanan genetik terhadap GLS."
        ]
    },
    healthy: {
        title: "Tanaman Sehat",
        description: "Daun jagung dalam kondisi prima tanpa tanda-tanda patogen yang terdeteksi.",
        recommendations: [
            "Pertahankan nutrisi tanaman dengan pemupukan berimbang (N-P-K).",
            "Lakukan pemantauan rutin untuk deteksi dini hama dan penyakit.",
            "Pastikan ketersediaan air cukup terutama pada fase generatif."
        ]
    }
};

let model = null;
let selectedFile = null;
let currentDetections = [];
let stream = null;

// ============================================
// DOM Elements
// ============================================
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingStatus = document.getElementById('loadingStatus');
const loadingBarFill = document.getElementById('loadingBarFill');
const modelStatusEl = document.getElementById('modelStatus');
const uploadZone = document.getElementById('uploadZone');
const imageInput = document.getElementById('imageInput');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const btnRemove = document.getElementById('btnRemove');
const btnDetect = document.getElementById('btnDetect');
const canvasWrapper = document.getElementById('canvasWrapper');
const outputCanvas = document.getElementById('outputCanvas');
const resultPlaceholder = document.getElementById('resultPlaceholder');
const detectionInfo = document.getElementById('detectionInfo');
const detectionTime = document.getElementById('detectionTime');
const detectionList = document.getElementById('detectionList');

// Camera Elements
const tabUpload = document.getElementById('tabUpload');
const tabCamera = document.getElementById('tabCamera');
const inputDesc = document.getElementById('inputDesc');
const cameraZone = document.getElementById('cameraZone');
const cameraVideo = document.getElementById('cameraVideo');
const cameraCanvas = document.getElementById('cameraCanvas');
const btnCapture = document.getElementById('btnCapture');

// New Elements
const modalOverlay = document.getElementById('modalOverlay');
const modalBody = document.getElementById('modalBody');
const modalTitle = document.getElementById('modalTitle');
const btnCloseModal = document.getElementById('btnCloseModal');
const btnRecommendation = document.getElementById('btnRecommendation');
const btnDownload = document.getElementById('btnDownload');

// ============================================
// Configure TF.js Backend (Fix Mobile Precision)
// ============================================
async function configureTFBackend() {
    try {
        // Paksa float32 (32-bit) — mencegah mobile GPU pakai 16-bit
        // yang menyebabkan koordinat bounding box meleset
        tf.env().set('WEBGL_FORCE_F16_TEXTURES', false);

        // Pastikan WebGL bisa render float32
        if (tf.env().getBool('WEBGL_RENDER_FLOAT32_CAPABLE')) {
            tf.env().set('WEBGL_RENDER_FLOAT32_ENABLED', true);
        }

        // Nonaktifkan packing agar output tensor lebih konsisten antar device
        tf.env().set('WEBGL_PACK', false);

        await tf.setBackend('webgl');
        await tf.ready();
        console.log('✅ TF Backend:', tf.getBackend(), '| Float32 capable:', tf.env().getBool('WEBGL_RENDER_FLOAT32_CAPABLE'));
    } catch (e) {
        console.warn('⚠️ WebGL config gagal, fallback ke default:', e.message);
    }
}

// ============================================
// Load Model
// ============================================
async function loadModel() {
    try {
        loadingBarFill.style.width = '10%';
        loadingStatus.textContent = 'Mengkonfigurasi sistem...';

        // Konfigurasi backend terlebih dahulu untuk presisi float32
        await configureTFBackend();

        loadingBarFill.style.width = '20%';
        loadingStatus.textContent = 'Mengunduh komponen deteksi...';

        model = await tf.loadGraphModel(MODEL_PATH, {
            onProgress: (fraction) => {
                const pct = 20 + fraction * 60;
                loadingBarFill.style.width = pct + '%';
                loadingStatus.textContent = `Memuat sistem... ${Math.round(fraction * 100)}%`;
            }
        });

        loadingBarFill.style.width = '85%';
        loadingStatus.textContent = 'Menyiapkan sistem deteksi...';

        const dummy = tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3]);
        await model.executeAsync(dummy);
        dummy.dispose();

        loadingBarFill.style.width = '100%';
        loadingStatus.textContent = 'Sistem siap!';

        setTimeout(() => {
            loadingOverlay.classList.add('hidden');
        }, 600);

        modelStatusEl.innerHTML = '<span class="status-dot"></span><span>Siap</span>';
        console.log('✅ Model loaded successfully');
    } catch (err) {
        console.error('❌ Error loading model:', err);
        loadingStatus.textContent = '❌ Gagal memuat sistem: ' + err.message;
        loadingBarFill.style.width = '100%';
        loadingBarFill.style.background = '#c0392b';
        modelStatusEl.innerHTML = '<span class="status-dot error"></span><span>Error</span>';
    }
}

// ============================================
// Image Upload Handling
// ============================================
uploadZone.addEventListener('click', () => imageInput.click());

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleImageSelect(file);
    }
});

imageInput.addEventListener('change', (e) => {
    if (e.target.files[0]) {
        handleImageSelect(e.target.files[0]);
    }
});

btnRemove.addEventListener('click', () => {
    resetUpload();
});

// Camera Logic
tabUpload?.addEventListener('click', () => {
    tabUpload.classList.add('active');
    tabUpload.style.opacity = '1';
    tabCamera.classList.remove('active');
    tabCamera.style.opacity = '0.6';
    uploadZone.style.display = 'block';
    cameraZone.style.display = 'none';
    inputDesc.textContent = 'Pilih atau seret foto daun jagung';
    stopCamera();
});

tabCamera?.addEventListener('click', async () => {
    tabCamera.classList.add('active');
    tabCamera.style.opacity = '1';
    tabUpload.classList.remove('active');
    tabUpload.style.opacity = '0.6';
    uploadZone.style.display = 'none';
    cameraZone.style.display = 'block';
    inputDesc.textContent = 'Arahkan kamera ke daun jagung';
    await startCamera();
});

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' }
        });
        cameraVideo.srcObject = stream;
    } catch (err) {
        console.error('Error accessing camera:', err);
        alert('Tidak dapat mengakses kamera. Pastikan Anda telah memberikan izin di browser.');
        tabUpload.click();
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
}

btnCapture?.addEventListener('click', () => {
    if (!stream) return;

    cameraCanvas.width = cameraVideo.videoWidth;
    cameraCanvas.height = cameraVideo.videoHeight;
    const ctx = cameraCanvas.getContext('2d');
    ctx.drawImage(cameraVideo, 0, 0);

    cameraCanvas.toBlob((blob) => {
        if (blob) {
            const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
            stopCamera();
            handleImageSelect(file);
        }
    }, 'image/jpeg', 0.9);
});

function handleImageSelect(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewContainer.style.display = 'block';
        uploadZone.style.display = 'none';
        btnDetect.disabled = false;
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    selectedFile = null;
    previewImage.src = '';
    previewContainer.style.display = 'none';

    // Reset back to upload zone if camera was used
    if (tabUpload) tabUpload.click();
    else uploadZone.style.display = 'block';

    btnDetect.disabled = true;
    imageInput.value = '';

    canvasWrapper.style.display = 'none';
    resultPlaceholder.style.display = 'block';
    detectionInfo.style.display = 'none';
    currentDetections = [];
}

// ============================================
// Detection
// ============================================
btnDetect.addEventListener('click', () => runDetection());

async function runDetection() {
    if (!model || !selectedFile) return;

    btnDetect.disabled = true;
    btnDetect.classList.add('detecting');
    btnDetect.innerHTML = '<span class="btn-icon">⏳</span><span>Mendeteksi...</span>';

    const startTime = performance.now();

    try {
        const img = new Image();
        img.src = previewImage.src;
        await new Promise((resolve) => {
            if (img.complete) resolve();
            else img.onload = resolve;
        });

        const { tensor, scale, padX, padY } = preprocessImage(img);
        const rawOutput = await model.executeAsync(tensor);
        tensor.dispose();

        const detections = await postprocess(rawOutput, scale, padX, padY, img.naturalWidth, img.naturalHeight);
        currentDetections = detections;

        if (Array.isArray(rawOutput)) {
            rawOutput.forEach(t => t.dispose());
        } else {
            rawOutput.dispose();
        }

        const elapsed = (performance.now() - startTime).toFixed(0);

        drawResults(img, detections);
        showDetectionInfo(detections, elapsed);
        saveToHistory(img, detections);

    } catch (err) {
        console.error('Detection error:', err);
        alert('Terjadi kesalahan saat deteksi: ' + err.message);
    }

    btnDetect.disabled = false;
    btnDetect.classList.remove('detecting');
    btnDetect.innerHTML = '<span class="btn-icon">🔍</span><span>Deteksi Ulang</span>';
}

// ============================================
// Preprocessing — Letterbox
// ============================================
function preprocessImage(img) {
    const origW = img.naturalWidth;
    const origH = img.naturalHeight;

    const scale = Math.min(INPUT_SIZE / origW, INPUT_SIZE / origH);
    const newW = Math.round(origW * scale);
    const newH = Math.round(origH * scale);

    // Gunakan Math.floor agar padX & padY selalu integer
    // Padding desimal (misal 16.5) menyebabkan bounding box meleset di mobile
    const padX = Math.floor((INPUT_SIZE - newW) / 2);
    const padY = Math.floor((INPUT_SIZE - newH) / 2);

    const offCanvas = document.createElement('canvas');
    offCanvas.width = INPUT_SIZE;
    offCanvas.height = INPUT_SIZE;
    // Nonaktifkan image smoothing untuk hasil pixel yang identik di semua device
    const ctx = offCanvas.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingEnabled = false;

    ctx.fillStyle = '#808080';
    ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
    // Gambar di posisi integer agar tidak ada sub-pixel rendering
    ctx.drawImage(img, padX, padY, newW, newH);

    // Aktifkan imageSmoothingEnabled kembali hanya untuk konversi tensor
    ctx.imageSmoothingEnabled = true;

    const tensor = tf.browser.fromPixels(offCanvas)
        .toFloat()
        .div(255.0)
        .expandDims(0);

    return { tensor, scale, padX, padY };
}

// ============================================
// Postprocessing — Parse YOLOv8 output + NMS
// ============================================
async function postprocess(rawOutput, scale, padX, padY, origW, origH) {
    // YOLOv8 output shape: [1, 4+nc, 8400]
    // where nc = number of classes
    let outputTensor;
    if (Array.isArray(rawOutput)) {
        outputTensor = rawOutput[0];
    } else {
        outputTensor = rawOutput;
    }

    // Get shape to understand output format
    const shape = outputTensor.shape;
    console.log('Output shape:', shape);

    // Squeeze batch dimension and transpose to [8400, 4+nc]
    let data;
    if (shape.length === 3 && shape[1] === (4 + CLASS_NAMES.length)) {
        // Shape [1, 8, 8400] → transpose to [1, 8400, 8] → squeeze → [8400, 8]
        data = await outputTensor.squeeze(0).transpose().array();
    } else if (shape.length === 3 && shape[2] === (4 + CLASS_NAMES.length)) {
        // Shape [1, 8400, 8] → squeeze → [8400, 8]
        data = await outputTensor.squeeze(0).array();
    } else {
        // Try to handle other shapes
        console.warn('Unexpected output shape:', shape);
        data = await outputTensor.squeeze().array();
    }

    const numDetections = data.length;
    const numClasses = CLASS_NAMES.length;

    let boxes = [];
    let scores = [];
    let classIds = [];

    for (let i = 0; i < numDetections; i++) {
        const row = data[i];

        // First 4 values: cx, cy, w, h (in 640x640 space)
        const cx = row[0];
        const cy = row[1];
        const w = row[2];
        const h = row[3];

        // Remaining values: class scores
        let maxScore = 0;
        let maxClassId = 0;
        for (let c = 0; c < numClasses; c++) {
            if (row[4 + c] > maxScore) {
                maxScore = row[4 + c];
                maxClassId = c;
            }
        }

        if (maxScore < CONF_THRESHOLD) continue;

        // Konversi koordinat letterbox ke koordinat gambar asli
        // Gunakan Math.round agar semua koordinat pixel bulat (integer)
        // — mencegah sub-pixel artifact pada canvas mobile
        const x1 = Math.round((cx - w / 2 - padX) / scale);
        const y1 = Math.round((cy - h / 2 - padY) / scale);
        const x2 = Math.round((cx + w / 2 - padX) / scale);
        const y2 = Math.round((cy + h / 2 - padY) / scale);

        // Clamp agar tidak melebihi batas gambar
        const clampX1 = Math.max(0, Math.min(x1, origW));
        const clampY1 = Math.max(0, Math.min(y1, origH));
        const clampX2 = Math.max(0, Math.min(x2, origW));
        const clampY2 = Math.max(0, Math.min(y2, origH));

        boxes.push([clampX1, clampY1, clampX2, clampY2]);
        scores.push(maxScore);
        classIds.push(maxClassId);
    }

    // Apply NMS
    const nmsResults = nms(boxes, scores, classIds, IOU_THRESHOLD);

    return nmsResults;
}

// ============================================
// NMS (Non-Maximum Suppression)
// ============================================
function nms(boxes, scores, classIds, iouThreshold) {
    const results = [];

    // Group by class
    const classGroups = {};
    for (let i = 0; i < boxes.length; i++) {
        const cls = classIds[i];
        if (!classGroups[cls]) classGroups[cls] = [];
        classGroups[cls].push(i);
    }

    for (const cls in classGroups) {
        let indices = classGroups[cls];

        // Sort by score descending
        indices.sort((a, b) => scores[b] - scores[a]);

        const keep = [];
        while (indices.length > 0) {
            const current = indices[0];
            keep.push(current);

            const remaining = [];
            for (let j = 1; j < indices.length; j++) {
                const iou = computeIoU(boxes[current], boxes[indices[j]]);
                if (iou < iouThreshold) {
                    remaining.push(indices[j]);
                }
            }
            indices = remaining;
        }

        for (const idx of keep) {
            results.push({
                box: boxes[idx],
                score: scores[idx],
                classId: classIds[idx],
                className: CLASS_NAMES[classIds[idx]],
            });
        }
    }

    // Sort final results by score
    results.sort((a, b) => b.score - a.score);
    return results;
}

function computeIoU(boxA, boxB) {
    const x1 = Math.max(boxA[0], boxB[0]);
    const y1 = Math.max(boxA[1], boxB[1]);
    const x2 = Math.min(boxA[2], boxB[2]);
    const y2 = Math.min(boxA[3], boxB[3]);

    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
    const areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);
    const union = areaA + areaB - intersection;

    return union > 0 ? intersection / union : 0;
}

// ============================================
// UI Rendering
// ============================================
function drawResults(img, detections) {
    const canvas = outputCanvas;
    const ctx = canvas.getContext('2d');
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);

    for (const det of detections) {
        const [x1, y1, x2, y2] = det.box;
        const color = CLASS_COLORS[det.className] || { bg: '#2d6a4f', text: '#fff' };
        const label = `${formatClassName(det.className)} ${(det.score * 100).toFixed(0)}%`;

        const lineWidth = Math.max(2, Math.min(4, Math.round(canvas.width / 300)));
        ctx.lineWidth = lineWidth;
        ctx.strokeStyle = color.bg;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        ctx.fillStyle = color.bg + '20';
        ctx.fillRect(x1, y1, x2 - x1, y2 - y1);

        const fontSize = Math.max(12, Math.min(18, Math.round(canvas.width / 50)));
        ctx.font = `bold ${fontSize}px Inter, sans-serif`;
        const textWidth = ctx.measureText(label).width;
        const textHeight = fontSize + 8;
        const labelY = y1 > textHeight ? y1 - textHeight : y1;

        ctx.fillStyle = color.bg;
        ctx.fillRect(x1, labelY, textWidth + 12, textHeight);
        ctx.fillStyle = color.text;
        ctx.fillText(label, x1 + 6, labelY + fontSize + 1);
    }

    canvasWrapper.style.display = 'block';
    resultPlaceholder.style.display = 'none';
}

function formatClassName(name) {
    return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function showDetectionInfo(detections, elapsed) {
    detectionTime.textContent = `⚡ ${elapsed}ms`;
    detectionList.innerHTML = '';

    if (detections.length === 0) {
        detectionList.innerHTML = `<div class="no-detection"><p>😕 Tidak ada penyakit yang terdeteksi.</p></div>`;
        btnRecommendation.disabled = true;
    } else {
        btnRecommendation.disabled = false;
        detections.forEach(det => {
            const color = CLASS_COLORS[det.className] || { bg: '#2d6a4f' };
            const pct = (det.score * 100).toFixed(1);
            const item = document.createElement('div');
            item.className = `detection-item disease-${det.className}`;
            item.innerHTML = `
                <div class="detection-item-left" style="flex: 1;">
                    <span class="detection-dot" style="background:${color.bg}"></span>
                    <span class="detection-item-name">${formatClassName(det.className)}</span>
                    <div class="conf-bar-container">
                        <div class="conf-bar-fill" style="width: ${pct}%; background: ${color.bg}"></div>
                    </div>
                </div>
                <span class="detection-item-conf">${pct}%</span>
            `;
            detectionList.appendChild(item);
        });
    }
    detectionInfo.style.display = 'block';
}

// ============================================
// New Features Logic
// ============================================

// History Management
function saveToHistory(img, detections) {
    if (detections.length === 0) return;

    // Get primary detection (highest score)
    const primary = detections[0];

    // Create mini thumbnail
    const thumbCanvas = document.createElement('canvas');
    thumbCanvas.width = 200;
    thumbCanvas.height = 150;
    const tCtx = thumbCanvas.getContext('2d');

    // Draw cropped center or just scaled
    const scale = Math.max(thumbCanvas.width / img.naturalWidth, thumbCanvas.height / img.naturalHeight);
    const w = img.naturalWidth * scale;
    const h = img.naturalHeight * scale;
    tCtx.drawImage(img, (thumbCanvas.width - w) / 2, (thumbCanvas.height - h) / 2, w, h);

    const historyItem = {
        id: Date.now(),
        date: new Date().toLocaleString('id-ID', { dateStyle: 'medium', timeStyle: 'short' }),
        className: primary.className,
        conf: (primary.score * 100).toFixed(1),
        image: thumbCanvas.toDataURL('image/jpeg', 0.7)
    };

    let history = JSON.parse(localStorage.getItem('cornvision_history') || '[]');
    history.unshift(historyItem);
    history = history.slice(0, 8); // Keep last 8
    localStorage.setItem('cornvision_history', JSON.stringify(history));
}


// Modal Logic
btnRecommendation.addEventListener('click', () => {
    if (currentDetections.length === 0) return;

    const primary = currentDetections[0].className;
    const data = DISEASE_DATA[primary];

    modalTitle.textContent = data.title;
    modalBody.innerHTML = `
        <div class="rec-item">
            <h4><span class="logo-icon">🧐</span> Deskripsi</h4>
            <p>${data.description}</p>
        </div>
        <div class="rec-item">
            <h4><span class="logo-icon">💡</span> Rekomendasi Penanganan</h4>
            <ul>
                ${data.recommendations.map(r => `<li>${r}</li>`).join('')}
            </ul>
        </div>
    `;

    modalOverlay.classList.add('active');
});

btnCloseModal.addEventListener('click', () => modalOverlay.classList.remove('active'));
modalOverlay.addEventListener('click', (e) => {
    if (e.target === modalOverlay) modalOverlay.classList.remove('active');
});

// Export Logic
btnDownload.addEventListener('click', () => {
    if (currentDetections.length === 0) return;

    const link = document.createElement('a');
    link.download = `CornVision_Result_${Date.now()}.png`;
    link.href = outputCanvas.toDataURL('image/png');
    link.click();
});

// Initialize
loadModel();