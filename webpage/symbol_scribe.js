document.addEventListener('DOMContentLoaded', function () {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');

    const resetButton = document.getElementById('resetButton');
    const undoButton = document.getElementById('undoButton');
    const darkmodeButton = document.getElementById("darkmode");

    const resizedCanvas = document.createElement('canvas');
    const resizedCtx = resizedCanvas.getContext('2d');
    resizedCanvas.width = 32;
    resizedCanvas.height = 32;

    const resultsDiv = document.getElementById('results');


    let strokeStyle = document.documentElement.getAttribute("saved-theme") === "dark" ? "white" : "black";
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;
    let lineHistory = [];
    let ortSession = null; // Store the ONNX session

    // Path for small latex preview images
    const imagePath = './files/symbols/';

    // Labels for symbol recognition
    const labels = [
        ["A", "Upper_A"],
        ["B", "Upper_B"],
        ["C", "Upper_C"],
        ["D", "Upper_D"],
        ["E", "Upper_E"],
        ["F", "Upper_F"],
        ["G", "Upper_G"],
        ["H", "Upper_H"],
        ["I", "Upper_I"],
        ["J", "Upper_J"],
        ["K", "Upper_K"],
        ["L", "Upper_L"],
        ["M", "Upper_M"],
        ["N", "Upper_N"],
        ["O", "Upper_O"],
        ["P", "Upper_P"],
        ["Q", "Upper_Q"],
        ["R", "Upper_R"],
        ["S", "Upper_S"],
        ["T", "Upper_T"],
        ["U", "Upper_U"],
        ["V", "Upper_V"],
        ["W", "Upper_W"],
        ["X", "Upper_X"],
        ["Y", "Upper_Y"],
        ["Z", "Upper_Z"],
        ["\\Delta", "Upper_Delta"],
        ["\\Downarrow", "Upper_Downarrow"],
        ["\\Gamma", "Upper_Gamma"],
        ["\\Im", "Upper_Im"],
        ["\\Lambda", "Upper_Lambda"],
        ["\\Leftarrow", "Upper_Leftarrow"],
        ["\\Leftrightarrow", "Upper_Leftrightarrow"],
        ["\\Omega", "Upper_Omega"],
        ["\\Phi", "Upper_Phi"],
        ["\\Pi", "Upper_Pi"],
        ["\\Psi", "Upper_Psi"],
        ["\\Re", "Upper_Re"],
        ["\\Rightarrow", "Upper_Rightarrow"],
        ["\\Sigma", "Upper_Sigma"],
        ["\\Theta", "Upper_Theta"],
        ["\\Uparrow", "Upper_Uparrow"],
        ["\\Updownarrow", "Upper_Updownarrow"],
        ["\\Upsilon", "Upper_Upsilon"],
        ["\\Xi", "Upper_Xi"],
        ["\\alpha", "alpha"],
        ["\\approx", "approx"],
        ["\\beta", "beta"],
        ["\\blacksquare", "blacksquare"],
        ["\\boxtimes", "boxtimes"],
        ["\\cap", "cap"],
        ["\\cdot", "cdot"],
        ["\\cdots", "cdots"],
        ["\\chi", "chi"],
        ["\\complement", "complement"],
        ["\\cong", "cong"],
        ["\\cup", "cup"],
        ["\\delta", "delta"],
        ["\\div", "div"],
        ["\\downarrow", "downarrow"],
        ["\\emptyset", "emptyset"],
        ["\\epsilon", "epsilon"],
        ["\\equiv", "equiv"],
        ["\\eta", "eta"],
        ["\\exists", "exists"],
        ["\\forall", "forall"],
        ["\\gamma", "gamma"],
        ["\\geq", "geq"],
        ["\\in", "in"],
        ["\\infty", "infty"],
        ["\\int", "int"],
        ["\\iota", "iota"],
        ["\\kappa", "kappa"],
        ["\\lambda", "lambda"],
        ["\\leftarrow", "leftarrow"],
        ["\\leftharpoondown", "leftharpoondown"],
        ["\\leftharpoonup", "leftharpoonup"],
        ["\\leftrightarrow", "leftrightarrow"],
        ["\\leq", "leq"],
        ["\\longmapsto", "longmapsto"],
        ["\\mapsto", "mapsto"],
        ["\\mu", "mu"],
        ["\\nabla", "nabla"],
        ["\\nearrow", "nearrow"],
        ["\\neg", "neg"],
        ["\\neq", "neq"],
        ["\\nexists", "nexists"],
        ["\\notin", "notin"],
        ["\\nu", "nu"],
        ["\\nwarrow", "nwarrow"],
        ["\\omega", "omega"],
        ["\\oplus", "oplus"],
        ["\\otimes", "otimes"],
        ["\\partial", "partial"],
        ["\\perp", "perp"],
        ["\\phi", "phi"],
        ["\\pi", "pi"],
        ["\\psi", "psi"],
        ["\\rho", "rho"],
        ["\\rightarrow", "rightarrow"],
        ["\\rightharpoondown", "rightharpoondown"],
        ["\\rightharpoonup", "rightharpoonup"],
        ["\\rightleftharpoons", "rightleftharpoons"],
        ["\\searrow", "searrow"],
        ["\\sigma", "sigma"],
        ["\\simeq", "simeq"],
        ["\\subset", "subset"],
        ["\\sum", "sum"],
        ["\\swarrow", "swarrow"],
        ["\\tau", "tau"],
        ["\\theta", "theta"],
        ["\\times", "times"],
        ["\\triangle", "triangle"],
        ["\\uparrow", "uparrow"],
        ["\\upsilon", "upsilon"],
        ["\\varepsilon", "varepsilon"],
        ["\\varnothing", "varnothing"],
        ["\\varphi", "varphi"],
        ["\\varrho", "varrho"],
        ["\\vartheta", "vartheta"],
        ["\\vee", "vee"],
        ["\\wedge", "wedge"],
        ["\\wp", "wp"],
        ["\\xi", "xi"],
        ["\\zeta", "zeta"],
        ["a", "a"],
        ["b", "b"],
        ["c", "c"],
        ["d", "d"],
        ["e", "e"],
        ["f", "f"],
        ["g", "g"],
        ["h", "h"],
        ["i", "i"],
        ["j", "j"],
        ["k", "k"],
        ["l", "l"],
        ["m", "m"],
        ["n", "n"],
        ["o", "o"],
        ["p", "p"],
        ["q", "q"],
        ["r", "r"],
        ["s", "s"],
        ["t", "t"],
        ["u", "u"],
        ["v", "v"],
        ["w", "w"],
        ["x", "x"],
        ["y", "y"],
        ["z", "z"],
        ["~", "space"],
    ];
    // Function to initialize ONNX session
    async function initONNX() {
        // Skip ONNX Model loading if already loaded.
        if (ortSession) {
            console.log("already loaded");
            console.log(ortSession);
            return;
        }

        try {
            ortSession = await ort.InferenceSession.create('files/SymbolCNN.onnx');
            console.log('ONNX session loaded successfully.');
        } catch (e) {
            console.error("Failed to load ONNX session:", e);
            displayResults("Error loading SymbolScribe model. Please reload the page.");
        }
    }

    function resizeCanvas() {
        // Maximum Width of Canvas = (ScreenWidth - (2*Padding + 2*border) or 500
        const viewportWidth = Math.min(document.documentElement.clientWidth - 36, 500);
        canvas.width = viewportWidth;
        canvas.height = viewportWidth * (3 / 5);
        redrawLines();
    }

    window.addEventListener('resize', resizeCanvas);

    // Touch-friendly event listeners
    function getTouchPos(canvasDom, touchEvent) {
        var rect = canvasDom.getBoundingClientRect();
        return {
            x: touchEvent.touches[0].clientX - rect.left,
            y: touchEvent.touches[0].clientY - rect.top
        };
    }

    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault(); // Prevent default touch behaviors (e.g., scrolling)
        const pos = getTouchPos(canvas, e);
        isDrawing = true;
        [lastX, lastY] = [pos.x, pos.y];
        lineHistory.push([]);
    }, { passive: false });

    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        if (!isDrawing) return;

        const pos = getTouchPos(canvas, e);
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(pos.x, pos.y);
        ctx.strokeStyle = strokeStyle;
        ctx.lineWidth = 16;
        ctx.lineCap = 'round';
        ctx.stroke();

        lineHistory[lineHistory.length - 1].push({ x: lastX, y: lastY }, { x: pos.x, y: pos.y });
        [lastX, lastY] = [pos.x, pos.y];
    }, { passive: false });

    canvas.addEventListener('touchend', (e) => {
        e.preventDefault();
        isDrawing = false;
        if (lineHistory.length > 0 && lineHistory[lineHistory.length - 1].length === 0) {
            lineHistory.pop();
        }
        infer();
    }, { passive: false });

    canvas.addEventListener('touchcancel', (e) => {
        e.preventDefault();
        isDrawing = false;
        if (lineHistory.length > 0 && lineHistory[lineHistory.length - 1].length === 0) {
            lineHistory.pop();
        }
        infer();
    }, { passive: false });


    // Start drawing
    canvas.addEventListener('mousedown', (e) => {
        isDrawing = true;
        [lastX, lastY] = [e.offsetX, e.offsetY];
        lineHistory.push([]);
    });

    // Draw lines
    canvas.addEventListener('mousemove', (e) => {
        if (!isDrawing) return;

        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.strokeStyle = strokeStyle;
        ctx.lineWidth = 16;
        ctx.lineCap = 'round';
        ctx.stroke();

        lineHistory[lineHistory.length - 1].push({ x: lastX, y: lastY }, { x: e.offsetX, y: e.offsetY });
        [lastX, lastY] = [e.offsetX, e.offsetY];
    });

    // Stop drawing and perform inference
    canvas.addEventListener('mouseup', (e) => {
        isDrawing = false;
        if (lineHistory.length > 0 && lineHistory[lineHistory.length - 1].length === 0) {
            lineHistory.pop();
        }
        infer();
    });

    // Reset canvas
    resetButton.addEventListener('click', (e) => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        resizedCtx.clearRect(0, 0, resizedCanvas.width, resizedCanvas.height);
        lineHistory = [];
        displayResults("");
    });

    // Undo last line
    undoButton.addEventListener('click', (e) => {
        if (lineHistory.length === 0) return;
        lineHistory.pop();
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        redrawLines();
        infer();
    });

    // switch Dark / Light mode
    darkmodeButton.addEventListener('click', (e) => {
        strokeStyle = strokeStyle === "black" ? "white" : "black";
        redrawLines();
        infer();
    });


    // Redraw lines from history
    function redrawLines() {
        lineHistory.forEach(line => {
            for (let i = 0; i < line.length - 1; i += 2) {
                ctx.beginPath();
                ctx.moveTo(line[i].x, line[i].y);
                ctx.lineTo(line[i + 1].x, line[i + 1].y);
                ctx.strokeStyle = strokeStyle;
                ctx.lineWidth = 16;
                ctx.lineCap = 'round';
                ctx.stroke();
            }
        });
    }

    // Get bounding box of drawn content
    function getBoundingBox() {
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        if (lineHistory.length === 0) return {}; // Nothing drawn

        lineHistory.forEach(line => {
            line.forEach(point => {
                minX = Math.min(minX, point.x);
                minY = Math.min(minY, point.y);
                maxX = Math.max(maxX, point.x);
                maxY = Math.max(maxY, point.y);
            });
        });

        return { minX, minY, maxX, maxY };
    }

    // Prepare canvas for inference
    function prepareCanvasForInference(minX, minY, maxX, maxY) {
        resizedCtx.imageSmoothingEnabled = true;
        resizedCtx.clearRect(0, 0, resizedCanvas.width, resizedCanvas.height);
        resizedCtx.drawImage(
            canvas,
            minX, minY, maxX - minX, maxY - minY,
            0, 0, resizedCanvas.width, resizedCanvas.height
        );
    }

    // Get image data from resized canvas
    function getImageData() {
        const imageData = resizedCtx.getImageData(0, 0, resizedCanvas.width, resizedCanvas.height);
        const data = imageData.data;
        const grayscaleData = new Float32Array(resizedCanvas.width * resizedCanvas.height);
        for (let i = 3; i <= data.length; i += 4) {
            grayscaleData[(i - 3) / 4] = Math.trunc(data[i] / 255); // Use the red channel for grayscale
        }
        return grayscaleData;
    }

    // Run inference
    async function runInference(grayscaleData, width, height) {
        if (!ortSession) {
            displayResults("SymbolScribe session not loaded. Please reload the page.");
            return null;
        }
        try {
            const dataTensor = new ort.Tensor('float32', grayscaleData, [1, 1, 32, 32]);
            const widthTensor = new ort.Tensor('int64', [width], [1]);
            const heightTensor = new ort.Tensor('int64', [height], [1]);
            const feeds = { "l_x_": dataTensor, "l_widths_": widthTensor, "l_heights_": heightTensor };
            return await ortSession.run(feeds);
        } catch (e) {
            console.error("Inference error:", e);
            displayResults("Error during inference.  Please reload the page.");
            return null;
        }
    }

    // Process inference results
    function processResults(results) {
        if (!results) return;
        const output = results.fc2_1.cpuData;
        const probabilities = softmax(output);

        const topIndices = [];
        for (let i = 0; i < probabilities.length; i++) {
            topIndices.push({ index: i, prob: probabilities[i] });
        }

        // Sort by probability in descending order
        topIndices.sort((a, b) => b.prob - a.prob);

        displayResults(topIndices.slice(0, 5));
    }

    // Calculate softmax
    const softmax = (arr) => {
        const max = Math.max(...arr);
        const exp = arr.map(x => Math.exp(x - max));
        const sum = exp.reduce((a, b) => a + b, 0);
        return exp.map(x => x / sum);
    };

    // Display results
    function displayResults(results) {
        resultsDiv.innerHTML = "";

        if (typeof results === 'string') {
            if (0 < results.length) {
                resultsDiv.innerHTML = `<blockquote class="callout bug" data-callout="bug">
    <div class="callout-title">
        <div class="callout-icon"></div>
        <div class="callout-title-inner"><p>Error</p></div>          
    </div>
    <div class="callout-content">
    <p>${results}</p>
    </div>
</blockquote>`;
            }
            return;
        }
        console.log(results);

        imageSuffix = strokeStyle === "black" ? "" : "_dark";
        results.forEach(({ index, prob }) => {
            const [label, imageFile] = labels[index];
            const resultItem = document.createElement('div');
            resultItem.classList.add('result-item');
            resultItem.innerHTML = `
            <div class="result-content">
                <img src="${imagePath + imageFile}${imageSuffix}.png" alt="${label}" class="symbol-image" width=32 height=32>
                <span class="label">${label}</span>
                <span class="probability">${(prob * 100).toFixed(2)}%</span>
            </div>
        `;
            resultItem.addEventListener('click', () => {
                navigator.clipboard.writeText(label)
                    .then(() => {
                        resultItem.classList.add('copied');
                        setTimeout(() => resultItem.classList.remove('copied'), 1000); // Remove 'copied' class after 1 second
                    })
                    .catch(err => {
                        console.error('Failed to copy text: ', err);
                        // Optionally, display an error to the user
                    });
            });
            resultsDiv.appendChild(resultItem);
        });
    }
    // Main inference function
    async function infer() {
        const { minX, minY, maxX, maxY } = getBoundingBox();
        if (!minX) {
            displayResults("");
            return;
        } // Nothing drawn

        prepareCanvasForInference(minX, minY, maxX, maxY);
        const grayscaleData = getImageData();
        const results = await runInference(grayscaleData, Math.trunc(maxX - minX), Math.trunc(maxY - minY));
        processResults(results);
    }

    // Initialize ONNX session when window is loaded
    console.log("Loading SymbolScribe ONNX Model.");
    resizeCanvas();
    initONNX();
});