# ComprehendVision: Real-Time AI-Powered Visual Assistant

**ComprehendVision** is an innovative project leveraging state-of-the-art machine learning and computer vision to provide real-time environmental understanding, primarily designed as an assistive technology for visually impaired individuals. This system processes live camera input to detect objects, describe scenes, and read text, aiming to enhance situational awareness and independence.

**(Note: This project is currently a functional prototype. Real-world testing with target users has not yet been conducted.)**

## Table of Contents
1.  [Project Goal](#project-goal)
2.  [Core Philosophy](#core-philosophy)
3.  [Key Features](#key-features)
4.  [System Architecture](#system-architecture)
    *   [Offline Core](#offline-core)
    *   [Online Enhancement Layer](#online-enhancement-layer)
    *   [User Interaction](#user-interaction)
5.  [Technology Stack](#technology-stack)
6.  [Development Journey & Key Decisions](#development-journey--key-decisions)
    *   [Initial Challenges with Edge Deployment](#initial-challenges-with-edge-deployment)
    *   [The Shift to a Hybrid Approach](#the-shift-to-a-hybrid-approach)
    *   [Addressing Model Reliability](#addressing-model-reliability)
    *   [Leveraging Google's Gemini API](#leveraging-googles-gemini-api)
7.  [Current Status & Performance Insights](#current-status--performance-insights)
8.  [Setup Instructions](#setup-instructions)
9.  [Usage](#usage)
10. [Future Work & Potential Enhancements](#future-work--potential-enhancements)
11. [Acknowledgements](#acknowledgements) (Optional)

## Project Goal
The primary objective of ComprehendVision is to translate complex visual information into accessible textual and auditory feedback. It aims to empower visually impaired users by providing them with a dynamic understanding of their surroundings, including object identification, scene context, and text present in their environment.

## Core Philosophy
This project is built upon the following principles:
*   **Reliability First:** Prioritizing dependable information, especially in the offline core.
*   **User-Centric Design:** Decisions are guided by the needs of assistive technology users, focusing on actionable and trustworthy output.
*   **Offline Capability:** Ensuring essential functionality works without internet access.
*   **Intelligent Enhancement:** Leveraging powerful online AI services (like Google's Gemini) when available to provide richer, more accurate understanding, while gracefully handling their absence.
*   **Pragmatic Development:** Acknowledging resource constraints (individual developer, $0 runtime budget for commercial APIs beyond free tiers) and focusing on achievable, impactful solutions.

## Key Features
*   **Real-Time Object Detection:** Utilizes YOLOv8 to identify and locate a wide range of common objects.
    *   Provides object class, confidence, category (living/non-living), and a heuristic for prominence/closeness.
*   **Dynamic Scene Understanding:**
    *   **Offline:** Generates robust, simple scene heuristics based on detected objects.
    *   **Offline (User-Requested):** Offers more detailed (but potentially less reliable) scene captions using an offline BLIP model, clearly prefaced.
    *   **Online (Gemini Powered):** Provides significantly more accurate and nuanced scene descriptions using Google's Gemini 1.5 Flash model when an internet connection is available and preferred by the user.
*   **Text Recognition (OCR):**
    *   **Offline:** Flags potential text-bearing objects detected by YOLO. Allows user-requested attempts to read text using Tesseract OCR (experimental, with reliability caveats).
    *   **Online (Gemini Powered):** Offers high-accuracy text extraction from image crops using Google's Gemini 1.5 Flash model.
*   **Hybrid Operational Mode:** Seamlessly switches between offline core functionality and enhanced online capabilities based on internet availability and user preference.
*   **User Controls:** Keyboard commands to toggle online preference, request detailed offline captions, and attempt offline OCR.
*   **Web-Based UI (for sighted demonstration):** A clean, pitch-black themed web interface built with Flask, SocketIO, HTML, CSS, and JavaScript to display the video feed and AI-generated information.
*   **Auditory Feedback (via Browser TTS):** The web UI uses browser-based Text-to-Speech to announce system outputs, making it accessible.

## System Architecture
ComprehendVision employs a modular, hybrid architecture:

![Conceptual Architecture Diagram - Placeholder: You can create a simple diagram for this]
*(Conceptual: Camera -> Python Backend (YOLO -> [Offline Heuristics/BLIP | Online Gemini]) -> SocketIO -> Web UI (Display & TTS))*

### Offline Core
*   **Input:** Live video feed from a webcam.
*   **Processing:**
    *   **YOLOv8 (`yolov8s.pt`):** Performs object detection on each frame.
    *   **Information Extraction:** Derives object class, confidence, prominence (heuristic), living/non-living category.
    *   **Text Flagging:** Identifies objects likely to contain text based on their class.
    *   **Simple Scene Heuristics:** Generates a basic scene description from the types and density of detected objects.
    *   **User-Invoked Offline BLIP:** On user command ('d'), generates a detailed caption using a local Salesforce BLIP-base model (prefaced as less reliable).
    *   **User-Invoked Offline Tesseract:** On user command ('r'), attempts OCR on flagged text regions using local Tesseract (prefaced as less reliable and experimental).
*   **Output:** Summaries of detected objects, scene heuristics, and text flags, presented via the UI and TTS.

### Online Enhancement Layer
*   **Trigger:** Activates if internet is detected AND the user has "Prefer Online Features" enabled.
*   **Processing (Google Gemini 1.5 Flash API):**
    *   **Enhanced Scene Captioning:** Sends the current frame to Gemini API for a high-quality, detailed scene description.
    *   **Enhanced OCR:** Sends image crops of YOLO-flagged text-bearing objects to Gemini API for accurate text extraction.
*   **Fallback:** If online calls fail or are unavailable (e.g., API key issue, quota limit, service down), the system gracefully informs the user and relies on the offline core or user-invoked offline options.

### User Interaction
*   **Web Interface:** A Flask-SocketIO powered web UI displays the camera feed, control toggles, and output panels for YOLO/heuristics, detailed captions, and OCR results.
*   **Keyboard Controls (in the original console app, adaptable for UI if needed):**
    *   `o`: Toggles preference for online features.
    *   `d`: Requests a detailed offline BLIP scene caption.
    *   `r`: Requests an attempt at offline Tesseract OCR on flagged text regions.
*   **Auditory Feedback:** The web UI utilizes browser-based Text-to-Speech for key information.

## Technology Stack
*   **Python 3.x**
*   **Core AI/ML:**
    *   **Object Detection:** Ultralytics YOLOv8 (`yolov8s.pt`)
    *   **Offline Image Captioning:** Hugging Face Transformers (`Salesforce/blip-image-captioning-base` via PyTorch)
    *   **Offline OCR:** `pytesseract` (wrapper for Tesseract OCR engine)
    *   **Online Image Captioning & OCR:** Google Gemini API (`gemini-1.5-flash-latest` via `google-generativeai` SDK)
*   **Image Processing:** OpenCV
*   **Web Backend:** Flask, Flask-SocketIO, Eventlet
*   **Web Frontend:** HTML5, CSS3, JavaScript (Vanilla JS, Socket.IO client)
*   **Key Python Libraries:** `torch`, `torchvision`, `Pillow`, (See `requirements.txt` for full list)
*   **Development Environment:** Windows, VS Code, PowerShell, Python Virtual Environment (`venv`)

## Development Journey & Key Decisions

This project evolved significantly based on iterative testing and realistic assessment of technological capabilities and resource constraints.

### Initial Challenges with Edge Deployment
*   The initial vision involved potential deployment on edge devices like Raspberry Pi.
*   **Observation:** Realized significant performance limitations (4-5 FPS for YOLO) due to lack of GPU, limited RAM on Raspberry Pi for models like YOLOv8 (n/s variants were considered the limit).
*   **Consideration:** Solutions like Google Coral USB accelerator (~$60 USD) or custom PCBs were deemed outside the $0 budget and current project scope.
*   **Decision:** Shifted primary development and testing to a more capable laptop (RTX 3050 GPU), achieving much better performance (27-30 FPS for YOLO). Model quantization (ONNX, TFLite, OpenVINO) was noted as a future path for potential edge deployment if pursued.

### The Shift to a Hybrid Approach
*   **Observation (YOLO's Limits):** While YOLO excels at detection, it's not designed for rich, nuanced descriptions of scenes or objects.
*   **Observation (Offline Model Reliability):** Initial tests with offline captioning (BLIP-base) showed ~60% reliability with some misinformation. Offline OCR (Tesseract) struggled significantly with real-world fonts and non-document text.
*   **Strategic Decision:** Adopted a **hybrid online/offline architecture**.
    *   **Offline First:** Prioritize a robust offline core for essential awareness.
    *   **Online Enhancement:** Use online services to "boost performance" (accuracy and detail) when internet is available and preferred. This ensures the system is not "completely useless" without internet.
*   **Rationale:** This approach was deemed far more efficient and achievable for an individual developer than attempting to train SOTA offline models from scratch or extensively fine-tune them, which requires vast datasets and computational resources typically available only to large organizations.

### Addressing Model Reliability
*   **Challenge:** Overly "creative" or inaccurate AI outputs can be detrimental in an assistive technology context.
*   **Solution for Offline:**
    *   **YOLO + Heuristics as Primary:** The most reliable offline information comes from direct YOLO detections and simple heuristics derived from them.
    *   **User-Invoked Less Reliable Features:** Offline BLIP captions and offline Tesseract OCR are made available only upon explicit user request ('d' and 'r' keys respectively) and are clearly prefaced as "Potentially Less Reliable." This gives the user control and manages expectations.
    *   **Text Flagging vs. Reading Offline:** The system defaults to only *flagging* potential text regions offline, rather than attempting to read them with the less reliable Tesseract.

### Leveraging Google's Gemini API
*   **Challenge:** Accessing high-quality online AI services without incurring costs or requiring credit card for initial setup.
*   **Solution:** Identified Google AI Studio and the Gemini API (specifically `gemini-1.5-flash-latest`) as a viable option.
    *   The Gemini API offers a free tier accessible via API keys generated through Google AI Studio, often without initial credit card requirements for basic developer usage.
    *   This allows for integration of SOTA online captioning and OCR capabilities for the "online enhancement" layer.
*   **Status:** Online captioning with Gemini is now functional and significantly more descriptive than offline BLIP. Online OCR with Gemini is the next integration step.

## Current Status & Performance Insights
*(As of latest testing)*

*   **Offline Core:**
    *   YOLOv8s object detection is stable and provides real-time identification of common objects, prominence heuristics, and living/non-living categorization.
    *   Simple scene heuristics based on YOLO output offer a basic, reliable understanding of the environment.
    *   Offline BLIP captioning (user-invoked) shows ~60-80% contextual accuracy but can introduce erroneous details.
    *   Offline Tesseract OCR (user-invoked) is highly dependent on text clarity and font; struggles with stylized or non-standard text.
*   **Online Enhancement (Google Gemini 1.5 Flash):**
    *   **Image Captioning:** Successfully integrated and functional. Provides significantly more detailed and contextually relevant scene descriptions compared to offline BLIP. Occasional minor misinterpretations (e.g., "hair as dog") noted, typical of current generative AI.
    *   **OCR:** Implementation in progress. Expected to provide high accuracy on real-world text.
*   **Web UI:**
    *   Functional Flask/SocketIO backend serving a pitch-black themed HTML/CSS/JS frontend.
    *   Live video feed displayed.
    *   Real-time updates of AI analysis (YOLO, heuristics, Gemini captions) in designated UI panels.
    *   Browser-based Text-to-Speech (TTS) for auditory feedback is functional, with a basic queuing mechanism.
    *   User controls for online preference and manual offline requests are implemented.
*   **Overall:** The hybrid system demonstrates the core concept effectively. The online Gemini features provide a substantial uplift in descriptive power.

**(Note: The system has not yet been tested in diverse, uncontrolled real-world environments with target users.)**

## Setup Instructions

These instructions will guide you through setting up the necessary Python environment to run this project.

**Prerequisites:**
*   Python 3.9+ recommended.
*   `pip` (Python package installer).
*   Git installed.
*   **Tesseract OCR Engine:** (For optional user-invoked offline OCR attempts)
    *   **Windows:** Download from [UB Mannheim Tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki). **Ensure "Add Tesseract to system PATH" is checked and "English" language data is included.**
    *   **Linux (Debian/Ubuntu):** `sudo apt update && sudo apt install tesseract-ocr tesseract-ocr-eng`
    *   **macOS:** `brew install tesseract tesseract-lang`
    *   Verify with `tesseract --version` in a new terminal.
*   **Google Account:** For accessing Google AI Studio and generating a Gemini API Key.

**Steps:**

1.  **Clone the Repository:**
    ```bash
    git clone [URL_OF_YOUR_GIT_REPOSITORY]
    cd comprehend_vision # Or your repo name
    ```

2.  **Create and Activate a Python Virtual Environment:**
    (Navigate to the project root directory first)
    ```bash
    python -m venv venv
    ```
    *   Windows (PowerShell): `.\venv\Scripts\Activate.ps1`
    *   Windows (CMD): `.\venv\Scripts\activate.bat`
    *   macOS / Linux: `source venv/bin/activate`
    Your terminal prompt should now be prefixed with `(venv)`.

3.  **Install Python Packages:**
    ```bash
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt 
    ```
    *(Note: If `pip install -r requirements.txt` fails on PyTorch, you may need to comment out `torch`, `torchvision`, `torchaudio` lines in `requirements.txt`, run the command again, then install PyTorch manually using the appropriate command from [pytorch.org](https://pytorch.org/get-started/locally/) for your CUDA version or CPU, then re-run `python -m pip freeze > requirements.txt`.)*

4.  **Set Up Google Gemini API Key:**
    *   Go to [aistudio.google.com](https://aistudio.google.com/).
    *   Follow the instructions to "Get API key." This will associate it with a Google Cloud Project.
    *   Copy the generated API key.
    *   **Set an environment variable named `GOOGLE_API_KEY` with your copied key.**
        *   PowerShell (current session): `$env:GOOGLE_API_KEY="YOUR_API_KEY_HERE"`
        *   Bash/Zsh (current session): `export GOOGLE_API_KEY="YOUR_API_KEY_HERE"`
        *   For permanent setup, add it to your system's environment variables.

5.  **Verify Tesseract Path (If Needed):**
    *   In `ocr_module.py`, ensure the line `pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'` (or equivalent for your OS) is correct if Tesseract isn't found automatically by `pytesseract`.

## Usage

1.  Ensure your virtual environment is active.
2.  Ensure the `GOOGLE_API_KEY` environment variable is set if you want to use online Gemini features.
3.  Run the Flask web server:
    ```bash
    python app_server.py
    ```
4.  Open your web browser and navigate to `http://127.0.0.1:5000` or `http://localhost:5000`.
5.  Interact with the UI:
    *   The video feed should start automatically.
    *   Information panels will update with AI analysis.
    *   Use the "Prefer Online Features" checkbox to toggle online Gemini enhancements.
    *   Use the "Request Offline..." buttons to manually trigger offline BLIP captioning or Tesseract OCR.
    *   Listen for auditory feedback via your browser's TTS.

## Future Work & Potential Enhancements
*   **Full Online OCR Integration with Gemini:** Complete and thoroughly test.
*   **Robust Text-to-Speech Management:** More sophisticated queuing, prioritization, and naturalness for audio output.
*   **Advanced Preprocessing for Offline OCR:** Improve Tesseract's chances on challenging text.
*   **Depth Perception/Distance Estimation:** Integrate monocular depth models or explore hardware solutions for more accurate distance information.
*   **User Customization:** Allow users to select critical objects, preferred voice, speech rate, verbosity.
*   **Specific Hazard Detection:** Fine-tune YOLO or use specialized models for detecting specific hazards (e.g., wet floors, low-hanging obstacles).
*   **Edge Device Deployment Research:** Investigate optimized models (TensorFlow Lite for YOLO) and hardware (Coral AI) for potential on-device processing for the offline core.
*   **Rigorous Real-World User Testing:** Essential for validating utility and gathering feedback from visually impaired individuals.

---
