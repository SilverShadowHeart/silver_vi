// static/script.js
document.addEventListener('DOMContentLoaded', () => {
    const socket = io(window.location.protocol + '//' + window.location.hostname + ':' + window.location.port + '/live');

    const yoloDataEl = document.getElementById('yoloDataPanel');
    const heuristicDataEl = document.getElementById('heuristicDataPanel');
    const captionDataEl = document.getElementById('captionDataPanel');
    const ocrDataEl = document.getElementById('ocrDataPanel');
    const onlinePrefCheckbox = document.getElementById('onlinePrefCheckbox');
    const requestOfflineCaptionBtn = document.getElementById('requestOfflineCaptionBtn');
    const requestOfflineOCRBtn = document.getElementById('requestOfflineOCRBtn');
    const clearInfoPanelsBtn = document.getElementById('clearInfoPanelsBtn');
    const connectionStatusEl = document.getElementById('connectionStatus');
    const onlineStatusTextEl = document.getElementById('onlineStatusText');

    let speechEngine = window.speechSynthesis;
    let speechQueue = []; // Simpler queue, just strings for now
    let isCurrentlySpeaking = false;

    console.log("JS: Script loaded. Speech engine:", speechEngine ? "Available" : "Not Available");

    function cleanTextForSpeech(text) {
        if (typeof text !== 'string') return "";
        let cleanedText = text;
        cleanedText = cleanedText.replace(/\[.*?\]/g, '').trim();
        cleanedText = cleanedText.replace(/\(Conf: \d\.\d+\)/g, '').trim();
        cleanedText = cleanedText.replace(/\(Less Reliable\)/gi, ', which may be less reliable,').trim();
        cleanedText = cleanedText.replace(/\(User Request\)/gi, '').trim();
        cleanedText = cleanedText.replace(/\(Internet Permitting\)/gi, 'if internet is available').trim();
        cleanedText = cleanedText.replace(/\(User Choice\)/gi, 'by user choice').trim();
        cleanedText = cleanedText.replace(/\(No text found by online OCR\)/gi, 'No clear text was found online.').trim();
        cleanedText = cleanedText.replace(/\(No text found online\)/gi, 'No clear text was found online.').trim();
        cleanedText = cleanedText.replace(/No clear text found by offline OCR/gi, 'No clear text was found by offline OCR.').trim();
        cleanedText = cleanedText.replace(/No significant objects detected by YOLO/gi, 'No significant objects detected.').trim();
        cleanedText = cleanedText.replace(/Area appears open or no distinct objects detected/gi, 'The area appears open.').trim();
        cleanedText = cleanedText.replace(/\.\s*\./g, '.'); 
        cleanedText = cleanedText.replace(/^Text:\s*/i, '').trim();
        cleanedText = cleanedText.replace(/\s+/g, ' '); // Normalize whitespace
        return cleanedText;
    }
    
    function speakNextInQueue() {
        console.log("JS: speakNextInQueue called. isCurrentlySpeaking:", isCurrentlySpeaking, "Queue length:", speechQueue.length);
        if (isCurrentlySpeaking || speechQueue.length === 0) {
            if(isCurrentlySpeaking) console.log("JS: Still speaking, will not start new utterance.");
            if(speechQueue.length === 0 && !isCurrentlySpeaking) console.log("JS: Queue empty and not speaking.");
            return;
        }

        isCurrentlySpeaking = true;
        let rawTextToSpeak = speechQueue.shift(); // Get the oldest message
        let textToSpeak = cleanTextForSpeech(rawTextToSpeak);

        console.log("JS: Preparing to speak (raw):", rawTextToSpeak.substring(0,100));
        console.log("JS: Preparing to speak (cleaned):", textToSpeak.substring(0,100));

        if (speechEngine && textToSpeak && textToSpeak.trim() !== "") {
            const utterance = new SpeechSynthesisUtterance(textToSpeak);
            
            utterance.onstart = () => {
                console.log("JS: Speech started for:", textToSpeak.substring(0,50));
            };
            utterance.onend = () => {
                console.log("JS: Speech ended for:", textToSpeak.substring(0,50));
                isCurrentlySpeaking = false;
                setTimeout(speakNextInQueue, 50); // Small delay before processing next, helps prevent race conditions
            };
            utterance.onerror = (event) => {
                console.error('JS: SpeechSynthesisUtterance.onerror for:', textToSpeak.substring(0,50), event);
                isCurrentlySpeaking = false;
                setTimeout(speakNextInQueue, 50); // Try next item even on error
            };
            
            console.log("JS: Calling speechEngine.speak().");
            speechEngine.speak(utterance);
        } else {
            if (!speechEngine) console.log("JS: speechEngine not available for this message.");
            if (!textToSpeak || textToSpeak.trim() === "") console.log("JS: textToSpeak is empty after cleaning, not speaking.");
            isCurrentlySpeaking = false; // Reset flag
            setTimeout(speakNextInQueue, 50); // Process queue even if this one wasn't spoken
        }
    }

    function addToSpeechQueue(text, important = false) {
        if (!text || typeof text !== 'string' || text.trim() === "") {
            console.log("JS: addToSpeechQueue - Text is empty, not adding.");
            return;
        }
        
        let lowerText = text.toLowerCase();
        if (!important && (
            lowerText.includes("no significant objects") || 
            lowerText.includes("no critical objects") ||
            lowerText.includes("area appears open") || // Broader match
            text.trim() === "---" 
           )) {
            console.log("JS: addToSpeechQueue - Filtered out non-important 'no objects' type message:", text.substring(0,50));
            return; 
        }

        console.log("JS: addToSpeechQueue:", text.substring(0,50), "Important:", important);

        if (important && speechEngine) {
            console.log("JS: Important message. Cancelling current speech and adding to front of queue.");
            speechEngine.cancel(); // Stop any current speech immediately
            isCurrentlySpeaking = false; // Reset flag as current speech is cancelled
            speechQueue.unshift(text); // Add important messages to the front
        } else {
            speechQueue.push(text); // Add normal messages to the back
        }
        speakNextInQueue(); // Attempt to process the queue
    }

    socket.on('connect', () => {
        console.log('JS: Connected to backend (/live).');
        connectionStatusEl.textContent = 'Connected';
        connectionStatusEl.className = 'status-online';
        socket.emit('request_initial_status_event'); 
        addToSpeechQueue("Connected to the vision system.", true);
    });

    socket.on('disconnect', () => {
        console.log('JS: Disconnected from backend.');
        connectionStatusEl.textContent = 'Disconnected';
        connectionStatusEl.className = 'status-offline';
        addToSpeechQueue("Disconnected from server.", true);
    });

    socket.on('connect_error', (err) => {
        console.error('JS: Connection Error:', err);
        connectionStatusEl.textContent = 'Connection Error!';
        connectionStatusEl.className = 'status-offline';
        addToSpeechQueue("Error connecting to server.", true);
    });

    socket.on('ai_update', (data) => {
        // console.log('JS: Received AI update:', data); 
        if (data.yolo_summary !== undefined) yoloDataEl.textContent = data.yolo_summary;
        if (data.scene_heuristic !== undefined) {
            heuristicDataEl.textContent = data.scene_heuristic;
            addToSpeechQueue(data.scene_heuristic, false); // Normal priority
        }
        if (data.detailed_caption !== undefined) {
            captionDataEl.textContent = data.detailed_caption;
            addToSpeechQueue(data.detailed_caption, false); // Normal priority
        }
        if (data.ocr_results !== undefined) {
            ocrDataEl.textContent = data.ocr_results;
            addToSpeechQueue(data.ocr_results, false); // Normal priority
        }
    });
    
    socket.on('mode_update_status', (data) => {
        console.log('JS: Mode update status received:', data);
        onlinePrefCheckbox.checked = data.is_online;
        onlineStatusTextEl.textContent = data.status_text_print;
        addToSpeechQueue(data.status_text_speak, true); // Mode changes are important
    });

    socket.on('manual_ai_result', (data) => {
        console.log('JS: Manual AI Result:', data);
        let messageToDisplay = data.data;
        if (data.type === 'caption_offline_user_req') captionDataEl.textContent = messageToDisplay;
        else if (data.type === 'ocr_offline_user_req') ocrDataEl.textContent = messageToDisplay; 
        else if (data.type.includes('error') || data.type.includes('status')) {
            if (data.type.includes('ocr')) ocrDataEl.textContent = messageToDisplay;
            else captionDataEl.textContent = messageToDisplay;
        }
        addToSpeechQueue(messageToDisplay, true); // User requested this, so it's important
    });

    onlinePrefCheckbox.addEventListener('change', () => {
        socket.emit('toggle_online_preference_event', { is_online: onlinePrefCheckbox.checked });
    });

    requestOfflineCaptionBtn.addEventListener('click', () => {
        console.log('JS: Requesting offline detailed caption (BLIP)...');
        socket.emit('request_offline_caption_event'); 
        captionDataEl.textContent = "Requesting offline detailed caption (BLIP)..."; 
    });

    requestOfflineOCRBtn.addEventListener('click', () => {
        console.log('JS: Requesting offline OCR (Tesseract)...');
        socket.emit('request_offline_ocr_event'); 
        ocrDataEl.textContent = "Requesting offline OCR (Tesseract)..."; 
    });

    clearInfoPanelsBtn.addEventListener('click', () => {
        console.log('JS: Clearing info panels and speech queue.');
        yoloDataEl.textContent = "---"; 
        heuristicDataEl.textContent = "---";
        captionDataEl.textContent = "---";
        ocrDataEl.textContent = "---";
        if (speechEngine) speechEngine.cancel(); 
        speechQueue = []; // Clear speech buffer
        isCurrentlySpeaking = false; // Reset flag
        addToSpeechQueue("Information panels cleared.", true);
    });
});