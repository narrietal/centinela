const loadingBar = document.getElementById('loading-bar');
const uploadSection = document.getElementById('upload-section');
const loadingMessage = document.getElementById('loading-message');
const loadingSection = document.getElementById('loading-section');
const result = document.getElementById('result');
const imageUpload = document.getElementById('image-upload');
const submitButton = document.getElementById('submit-button');
const imagePreview = document.getElementById('image-preview');
let identification_model;
let detection_model; // Object detection model

// Mapping prediction indices to whale names
const whaleMapping = {
    0: 'ALEX',
    1: 'AURO',
    2: 'CHEE',
    3: 'COPERNICO',
    4: 'CUCHI',
    5: 'DANIEL',
    6: 'DOMI',
    7: 'DORI',
    8: 'ERNEST',
    9: 'FURK',
    10: 'GRAHAM',
    11: 'HUGO',
    12: 'INDIO',
    13: 'MARK',
    14: 'MATIA',
    15: 'MIRNUKI',
    16: 'NICO',
    17: 'SAMI',
    18: 'SIERRA',
    19: 'YANI'
};

const messages = ["Loading...", "Grab a coffe and relax...", "Almost there..."];
let messageIndex = 0;

// Function to load the models with a progress simulation that lasts at least 2 seconds
async function loadModelsWithProgress() {
    // Start the message cycling every 5 seconds
    const messageInterval = setInterval(() => {
        messageIndex = (messageIndex + 1) % messages.length;
        loadingMessage.textContent = messages[messageIndex];
    }, 5000); // Change message every 5 seconds

    try {
        // Load the classification and object detection models asynchronously
        identification_model = await tf.loadLayersModel('fin_identification_web_model/model.json');
        console.log("Identification model loaded successfully!");

        detection_model = await tf.loadGraphModel('fin_detection_web_model/model.json'); // Load object detection model
        console.log("Detection model loaded successfully!");

    } catch (error) {
        console.error("Error loading models:", error);
    }

    // Once the models are loaded, stop changing the message
    clearInterval(messageInterval);

    // Hide the loading section and show the upload section
    loadingSection.classList.add('hidden');
    uploadSection.classList.remove('hidden');
}

// Load the models when the page loads
loadModelsWithProgress();

// Hide the button until a file is selected
imageUpload.addEventListener('change', function () {
    if (this.files.length > 0) {
        submitButton.classList.remove('hidden');
        const file = this.files[0];
        const reader = new FileReader();
        reader.onload = function (event) {
            imagePreview.src = event.target.result;
            imagePreview.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    } else {
        submitButton.classList.add('hidden');
        imagePreview.classList.add('hidden');
        result.classList.add('hidden');
    }
});

// Function to draw bounding boxes on the image
function drawBoundingBoxes(image, validBoxes) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = image.width;
    canvas.height = image.height;
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

    validBoxes.forEach((box) => {
        const [x, y, w, h, score] = box;
        ctx.strokeStyle = "red";
        ctx.lineWidth = 4;
        ctx.strokeRect(
          x * canvas.width,
          y * canvas.height,
          w * canvas.width,
          h * canvas.height
        );
        //const text = `${(score * 100).toFixed(2)}%`;
        //ctx.fillText(text, x * canvas.width, y * canvas.height - 10);
    });

    return canvas;
}

// Function to crop the detected object from the image
function cropBoundingBox(image, box) {
    let [x, y, w, h, score] = box;

    // Calculate the absolute dimensions relative to the original image size
    x = Math.max(0, Math.round(x * image.width));
    y = Math.max(0, Math.round(y * image.height));
    w = Math.max(1, Math.round(w * image.width)); // Ensure minimum width is 1
    h = Math.max(1, Math.round(h * image.height)); // Ensure minimum height is 1

    // Ensure the box fits within the image bounds
    w = Math.min(image.width - x, w);
    h = Math.min(image.height - y, h);
    
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = w;
    canvas.height = h;
    ctx.drawImage(image, x, y, w, h, 0, 0, w, h);
    return canvas;
}

document.getElementById('upload-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    result.classList.remove('hidden');
    result.textContent = 'Detecting fins and classifying...';
    submitButton.setAttribute('aria-busy', 'true');

    const file = imageUpload.files[0];
    if (!file) {
        result.textContent = 'Please upload an image first.';
        submitButton.setAttribute('aria-busy', 'false');
        return;
    }

    // Load the image
    const reader = new FileReader();
    reader.onload = async function (event) {
        const img = new Image();
        img.src = event.target.result;
        img.onload = async function () {
            // Convert image to tensor
            const tensorImg = tf.browser.fromPixels(img).resizeNearestNeighbor([640, 640]).expandDims(0).toFloat().div(tf.scalar(255));

            // Run object detection model
            const detectionResults = await detection_model.executeAsync(tensorImg);

            //  predictions shape is [1, 5, 8400]
            const reshapedPredictions = detectionResults.reshape([5, 8400]);
            const boxes = reshapedPredictions.slice([0, 0], [4, 8400]).arraySync();
            const scores = reshapedPredictions.slice([4, 0], [1, 8400]).squeeze().arraySync();

            //console.log('boxesTensor:', boxes.length, boxes[0]);
            //console.log('scoresTensor:', scores.length, scores[0]);

            // Filter out low-confidence detections (you can adjust the threshold)
            const confidenceThreshold = 0.5; // Example threshold
            const validBoxes = [];
            let maxScore = -Infinity;
            let bestBox = [];
            
            for (let i = 0; i < scores.length; i++) {
                // For multiple bounding boxes (it requires fine-tuning the threshold)
                /*if (scores[i] > confidenceThreshold) {
                    const x_c = boxes[0][i] / 640;
                    const y_c = boxes[1][i] / 640;
                    const w = boxes[2][i] / 640;
                    const h = boxes[3][i] / 640;
                    const x = x_c - w / 2;
                    const y = y_c - h / 2;

                    validBoxes.push([x, y, w, h, scores[i]]);
                    console.log(`Score:${scores[i]} > ${confidenceThreshold}`)
                }*/

                // For a single bounding-box approach
                if (scores[i] > maxScore) {
                    maxScore = scores[i];                    
                    const x_c = boxes[0][i] / 640;
                    const y_c = boxes[1][i] / 640;
                    const w = boxes[2][i] / 640;
                    const h = boxes[3][i] / 640;
                    const x = x_c - w / 2;
                    const y = y_c - h / 2;

                    bestBox = [x, y, w, h, maxScore]
                }
            }

            // Only use the bounding box if confidence is greater than threshold (Only use for single bounding-box mode)
            if (bestBox[4] > 0.4){
                validBoxes.push(bestBox);
            }
            

            if (validBoxes.length > 0) {

                console.log(`Score best box:${bestBox[4]}`)

                // Draw bounding boxes on the image
                const canvasWithBoxes = drawBoundingBoxes(img, validBoxes);
                imagePreview.src = canvasWithBoxes.toDataURL();
                result.textContent = '';
                // For each detected object, crop and classify
                for (const box of validBoxes) {
                    const croppedCanvas = cropBoundingBox(img, box);
                    const croppedTensor = tf.browser.fromPixels(croppedCanvas)
                        .resizeNearestNeighbor([224, 224])
                        .toFloat()
                        .div(tf.scalar(255))
                        .sub(tf.scalar(0.5)).mul(tf.scalar(2.0))
                        .expandDims();

                    // Run classification on cropped region
                    const predictions = await identification_model.predict(croppedTensor).data();
                    let whaleName = 'UNKNOWN';
                    let topPred = Math.max(...predictions)
                    const predictedClass = predictions.indexOf(Math.max(...predictions));

                    console.log("Identification confidence:", topPred, Math.max(...predictions))
                    if ( topPred >= 0.95) {
                        whaleName = whaleMapping[predictedClass] + ' [ALMOST CERTAIN]';
                    }else if (topPred >= 0.90 && topPred < 0.95){
                        whaleName = whaleMapping[predictedClass] + ' [LIKELY]';
                    }else if (topPred >= 0.85 && topPred < 0.90){
                        whaleName = whaleMapping[predictedClass] + ' [HIGHLY UNCERTAIN]';
                    }

                    // Display the classification result
                    result.innerHTML += `RESULT: ${whaleName}`;
                }
            } else {
                result.textContent = 'Could not confidently find any fin in the image';
            }
            submitButton.setAttribute('aria-busy', 'false');
        };
    };
    reader.readAsDataURL(file);
});

