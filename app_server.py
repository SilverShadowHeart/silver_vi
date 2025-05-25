# app_server.py
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
import time
import threading
import os
from ultralytics import YOLO

# --- IMPORT YOUR EXISTING MODULES ---
try:
    import caption_module # Should have Gemini integration
    import ocr_module     # Should have Gemini integration
    import utils          # Should be checking real internet
except ImportError as e:
    print(f"CRITICAL Error importing custom modules: {e}"); exit()


# --- CONFIGURATIONS ---
MODEL_NAME = 'yolov8s.pt'
CAMERA_INDEX = 0
YOLO_CONFIDENCE_THRESHOLD = 0.5
CRITICAL_OBJECT_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck', 'traffic light', 'stop sign', 'chair', 'couch', 'bed', 'dining table', 'laptop', 'cell phone', 'book', 'bottle', 'cup', 'door', 'backpack', 'handbag', 'suitcase']
TEXT_BEARING_CLASSES = ['book', 'stop sign', 'laptop', 'cell phone', 'tv', 'bottle']
LIVING_CLASSES = ['person', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']

YOLO_OUTPUT_INTERVAL = 1.0  # Seconds for how often AI thread sends YOLO summary
SCENE_HEURISTIC_INTERVAL = 5
DETAILED_CAPTION_INTERVAL = 10 
ONLINE_OCR_INTERVAL = 8

# --- Initialize Models (Load once) ---
print(f"Loading YOLO model: {MODEL_NAME}...")
try:
    yolo_model = YOLO(MODEL_NAME)
    print(f"YOLO loaded. Classes: {len(yolo_model.names)}")
except Exception as e:
    print(f"CRITICAL Error loading YOLO: {e}"); yolo_model = None

# Modules (caption_module, ocr_module) already print their Gemini config status
# Ensure GOOGLE_API_KEY is set in the environment where this Flask app runs.
if not os.environ.get("GOOGLE_API_KEY") and (caption_module.GEMINI_API_CONFIGURED_CAPTION or ocr_module.OCR_GEMINI_API_CONFIGURED):
    # This case should ideally not happen if modules print their own warnings
    print("SERVER WARNING: GOOGLE_API_KEY env var might be missing, but modules reported Gemini configured. This is unusual.")
elif not os.environ.get("GOOGLE_API_KEY"):
     print("SERVER WARNING: GOOGLE_API_KEY environment variable not set. Online Gemini features will be unavailable.")


app = Flask(__name__)
app.config['SECRET_KEY'] = 'ComprehendVisionSecret!' # Change this!
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*") 

# --- Global State for Camera and Threads ---
camera_object_server = None # Renamed to avoid conflict if you run Silver_detection.py separately
video_frame_for_ai_server = None 
frame_lock_server = threading.Lock() 
ai_processing_thread_server = None
stop_threads_event = threading.Event() # Single event to stop all threads
user_prefers_online_server = True 

def get_simple_scene_heuristic(detected_object_infos):
    if not detected_object_infos: return "Area appears open or no distinct objects detected."
    names_only = [item['name'] for item in detected_object_infos]
    person_count = names_only.count('person')
    vehicle_count = sum(1 for name in names_only if name in ['car', 'bus', 'truck', 'motorcycle', 'bicycle'])
    indoor_item_count = sum(1 for name in names_only if name in ['chair', 'table', 'couch', 'bed', 'laptop', 'tv', 'book'])
    if person_count > 2: return "Multiple people detected; area might be crowded."
    if vehicle_count > 0 and indoor_item_count == 0 : return "Street scene likely. Vehicles detected."
    if indoor_item_count > 0 and vehicle_count == 0: return "Indoor setting likely. Furniture or electronics detected."
    if person_count == 1: return f"1 person detected."
    if person_count > 1 : return f"{person_count} persons detected."
    unique_prominent_objects = sorted(list(set(item['name'] for item in detected_object_infos if item['prominence'] in ['very prominent', 'prominent'])))
    if len(unique_prominent_objects) > 2: return f"Various objects detected including: {', '.join(unique_prominent_objects[:2])}."
    elif unique_prominent_objects: return f"Objects detected including: {', '.join(unique_prominent_objects)}."
    return "General environment observed."

def camera_capture_thread_func():
    global camera_object_server, video_frame_for_ai_server, frame_lock_server, stop_threads_event
    print("SERVER: Camera capture thread started.")
    
    camera_object_server = cv2.VideoCapture(CAMERA_INDEX)
    if not camera_object_server.isOpened():
        print(f"SERVER ERROR: Could not open camera {CAMERA_INDEX}.")
        stop_threads_event.set(); return

    while not stop_threads_event.is_set():
        success, frame = camera_object_server.read()
        if not success:
            print("SERVER ERROR: Camera frame capture failed."); socketio.sleep(0.1); continue
        with frame_lock_server: video_frame_for_ai_server = frame.copy()
        socketio.sleep(1.0 / 30.0) # Update AI frame at ~30fps

    if camera_object_server: camera_object_server.release()
    print("SERVER: Camera capture thread stopped.")

def generate_mjpeg_stream_from_global_frame():
    global video_frame_for_ai_server, frame_lock_server, stop_threads_event # Use server-specific names
    print("SERVER: MJPEG stream generator started.")
    while not stop_threads_event.is_set():
        frame_to_encode = None
        with frame_lock_server:
            if video_frame_for_ai_server is not None:
                frame_to_encode = video_frame_for_ai_server.copy()
        
        if frame_to_encode is None:
            # Create a placeholder "Waiting for Camera" image if needed
            placeholder = cv2.imread("placeholder_nocamera.jpg") # You'd need to create this image
            if placeholder is None:
                placeholder = cv2.UMat(480, 640, cv2.CV_8UC3).get() # Black image
                cv2.putText(placeholder, "Camera Error", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            ret, buffer = cv2.imencode('.jpg', placeholder)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            socketio.sleep(0.5)
            continue
            
        ret, buffer = cv2.imencode('.jpg', frame_to_encode)
        if not ret: socketio.sleep(0.1); continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        socketio.sleep(1.0 / 25.0) 
    print("SERVER: MJPEG stream generator stopped.")

def ai_processing_background_task():
    global video_frame_for_ai_server, frame_lock_server, user_prefers_online_server, stop_threads_event, yolo_model
    
    last_yolo_output_time, last_scene_heuristic_time = 0, 0
    last_detailed_caption_time, last_online_ocr_time = 0, 0
    current_text_flags_for_ui = [] # Specific to this thread for UI updates

    print("SERVER: AI Processing background task started.")
    if yolo_model is None: print("SERVER: YOLO model not loaded, AI processing will be limited."); #return

    while not stop_threads_event.is_set():
        frame_to_process = None
        with frame_lock_server:
            if video_frame_for_ai_server is not None:
                frame_to_process = video_frame_for_ai_server.copy()
        
        if frame_to_process is None: socketio.sleep(0.2); continue

        current_time = time.time()
        processed_data_this_cycle = {}
        internet_available = utils.check_internet_connection()

        if (current_time - last_yolo_output_time > YOLO_OUTPUT_INTERVAL):
            last_yolo_output_time = current_time
            yolo_summary_lines = []; current_text_flags_for_ui.clear(); objects_detected_infos = []
            if yolo_model:
                yolo_results = yolo_model(frame_to_process, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)[0]
                if yolo_results.boxes:
                    for box in yolo_results.boxes:
                        # ... (same object_info population as your Silver_detection.py)
                        class_id=int(box.cls[0]); obj_name=yolo_model.names[class_id]; conf=float(box.conf[0])
                        bbox=box.xyxy[0].tolist(); cat="living" if obj_name in LIVING_CLASSES else "non-living"
                        h=bbox[3]-bbox[1]; fh=frame_to_process.shape[0]; prom="unknown"
                        if fh>0:
                            if h>fh*0.5: prom="very prominent"
                            elif h>fh*0.25: prom="prominent"
                            elif h>fh*0.1: prom="medium"
                            else: prom="less prominent"
                        obj_info={"name":obj_name,"category":cat,"prominence":prom,"confidence":conf,"bbox":bbox}
                        objects_detected_infos.append(obj_info)
                        if obj_name in CRITICAL_OBJECT_CLASSES: yolo_summary_lines.append(f"- {prom} {obj_name} ({cat}, {conf:.2f})")
                        if obj_name in TEXT_BEARING_CLASSES:
                            current_text_flags_for_ui.append(obj_info) # Use thread-local list
                            yolo_summary_lines.append(f"  (Potential text on {obj_name})")
            processed_data_this_cycle['yolo_summary'] = "\n".join(yolo_summary_lines) if yolo_summary_lines else "No critical objects detected by YOLO."
            if (current_time - last_scene_heuristic_time > SCENE_HEURISTIC_INTERVAL):
                last_scene_heuristic_time = current_time
                processed_data_this_cycle['scene_heuristic'] = get_simple_scene_heuristic(objects_detected_infos)

        if (current_time - last_detailed_caption_time > DETAILED_CAPTION_INTERVAL):
            last_detailed_caption_time = current_time
            caption_to_send = ""
            if user_prefers_online_server and internet_available:
                raw_caption, cap_src = caption_module.get_caption(frame_to_process.copy(), prefer_online=True)
                if cap_src == "online_gemini_caption": caption_to_send = f"[Online Gemini Scene] {raw_caption}"
                elif cap_src == "offline_blip": caption_to_send = f"[Online N/A - Offline BLIP (Less Reliable)] {raw_caption}"
                elif "error_config" in cap_src: caption_to_send = "[SYSTEM NOTE] Online captioning not configured. Press 'D' for offline BLIP."
                elif "error" in cap_src.lower(): caption_to_send = f"[Caption Service Error: {cap_src}] {raw_caption}"
                else: caption_to_send = "[System Note] Online caption status unclear. Press 'D' for offline BLIP."
            elif user_prefers_online_server and not internet_available:
                caption_to_send = "[System Note] No internet for online caption. Press 'D' for offline BLIP."
            if caption_to_send: processed_data_this_cycle['detailed_caption'] = caption_to_send

        if user_prefers_online_server and internet_available and current_text_flags_for_ui and \
           (current_time - last_online_ocr_time > ONLINE_OCR_INTERVAL):
            last_online_ocr_time = current_time
            ocr_results_ui = []
            for region_info in current_text_flags_for_ui:
                obj_name=region_info["name"]; obj_bbox=region_info["bbox"]
                x1,y1,x2,y2=[int(c) for c in obj_bbox];fh,fw=frame_to_process.shape[:2];x1c,y1c,x2c,y2c=max(0,x1),max(0,y1),min(fw-1,x2),min(fh-1,y2)
                if x1c < x2c and y1c < y2c:
                    text_crop = frame_to_process[y1c:y2c,x1c:x2c].copy()
                    text_read, ocr_src = ocr_module.get_ocr_text(text_crop, prefer_online=True) # This will use online Gemini
                    if ocr_src == "online_gemini_ocr": ocr_results_ui.append(f"On {obj_name}: '{text_read}'")
                    elif ocr_src == "online_no_text_ocr": ocr_results_ui.append(f"On {obj_name}: (No text found by online OCR)")
                    elif "error_config" in ocr_src: ocr_results_ui.append(f"On {obj_name}: (Online OCR not configured)")
                    else: ocr_results_ui.append(f"On {obj_name}: (OCR Status: {ocr_src} - {text_read})")
            if ocr_results_ui: processed_data_this_cycle['ocr_results'] = "[Online Gemini OCR]\n" + "\n".join(ocr_results_ui)
        
        if processed_data_this_cycle:
            socketio.emit('ai_update', processed_data_this_cycle, namespace='/live')
        socketio.sleep(0.2)
    print("SERVER: AI Processing background task stopped.")

@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed_route')
def video_feed_route_func(): return Response(generate_mjpeg_stream_from_global_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect', namespace='/live')
def handle_connect():
    print('SERVER: Client connected to /live')
    global ai_processing_thread_server # Use server-specific name
    if ai_processing_thread_server is None or not ai_processing_thread_server.is_alive():
        print("SERVER: Starting AI processing task...")
        stop_threads_event.clear() # Clear the global stop event
        ai_processing_thread_server = socketio.start_background_task(target=ai_processing_background_task)
    # Send initial status immediately upon connection
    internet_now = utils.check_internet_connection()
    mode_print = "ONLINE preferred" if user_prefers_online_server else "OFFLINE preferred"
    speak_mode = "Online mode preferred" if user_prefers_online_server else "Offline mode preferred"
    emit('mode_update_status', {
        'is_online': user_prefers_online_server,
        'status_text_print': f"Internet: {'DETECTED' if internet_now else 'NOT DETECTED'} | Mode: {mode_print}",
        'status_text_speak': f"Current mode is {speak_mode}. Internet is {'available' if internet_now else 'not available'}."
    }, namespace='/live')

@socketio.on('disconnect', namespace='/live')
def handle_disconnect(): print('SERVER: Client disconnected from /live')

@socketio.on('request_initial_status_event', namespace='/live')
def handle_request_initial_status():
    print('SERVER: Client requested initial status.')
    # Re-send initial status, this ensures new clients get it
    internet_now = utils.check_internet_connection()
    mode_print = "ONLINE preferred" if user_prefers_online_server else "OFFLINE preferred"
    speak_mode = "Online mode preferred" if user_prefers_online_server else "Offline mode preferred"
    emit('mode_update_status', {
        'is_online': user_prefers_online_server,
        'status_text_print': f"Internet: {'DETECTED' if internet_now else 'NOT DETECTED'} | Mode: {mode_print}",
        'status_text_speak': f"Current mode is {speak_mode}. Internet is {'available' if internet_now else 'not available'}."
    }, namespace='/live')


@socketio.on('toggle_online_preference_event', namespace='/live')
def handle_toggle_online(message):
    global user_prefers_online_server
    user_prefers_online_server = message.get('is_online', True)
    mode_print = "ONLINE preferred" if user_prefers_online_server else "OFFLINE preferred"
    speak_mode = "Online mode preferred" if user_prefers_online_server else "Offline mode preferred"
    print(f"SERVER: UI Update: User toggled mode to: {mode_print}")
    internet_now = utils.check_internet_connection()
    emit('mode_update_status', {
        'is_online': user_prefers_online_server,
        'status_text_print': f"Internet: {'DETECTED' if internet_now else 'NOT DETECTED'} | Mode: {mode_print}",
        'status_text_speak': f"Mode switched to {speak_mode}. Internet is {'available' if internet_now else 'not available'}."
    }, broadcast=True, namespace='/live')

@socketio.on('request_offline_caption_event', namespace='/live')
def handle_request_offline_caption():
    global video_frame_for_ai_server, frame_lock_server
    print("SERVER: UI Event: User requested offline BLIP caption.")
    frame_to_process = None; result_message = "No frame available for offline caption."
    with frame_lock_server:
        if video_frame_for_ai_server is not None: frame_to_process = video_frame_for_ai_server.copy()
    if frame_to_process is not None:
        preface = "[OFFLINE DETAIL - User Request - Potentially Less Reliable] "
        caption, cap_src = caption_module.generate_caption_offline(frame_to_process)
        result_message = preface + caption if "Error:" not in caption else "[OFFLINE CAPTION ERROR - User Request] " + caption
    emit('manual_ai_result', {'type': 'caption_offline_user_req', 'data': result_message}, namespace='/live')
    print(f"SERVER: Sent manual offline caption: {result_message[:70]}...")

@socketio.on('request_offline_ocr_event', namespace='/live')
def handle_request_offline_ocr():
    global video_frame_for_ai_server, frame_lock_server, yolo_model
    print("SERVER: UI Event: User requested offline Tesseract OCR.")
    frame_to_process = None; result_message = "No frame for offline OCR."
    with frame_lock_server:
        if video_frame_for_ai_server is not None: frame_to_process = video_frame_for_ai_server.copy()

    if frame_to_process is not None and yolo_model:
        yolo_results = yolo_model(frame_to_process, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)[0]
        local_text_flags = []
        if yolo_results.boxes:
            for box in yolo_results.boxes:
                if yolo_model.names[int(box.cls[0])] in TEXT_BEARING_CLASSES:
                    local_text_flags.append({"name": yolo_model.names[int(box.cls[0])], "bbox": box.xyxy[0].tolist()})
        
        if not local_text_flags: result_message = "[OFFLINE OCR] No text regions flagged by YOLO in current view to attempt OCR on."
        else:
            ocr_parts = ["[Offline Tesseract (User Req - Unreliable) Results]:"]
            for region in local_text_flags:
                obj_name, bbox = region["name"], region["bbox"]
                x1,y1,x2,y2=[int(c) for c in bbox];fh,fw=frame_to_process.shape[:2];x1c,y1c,x2c,y2c=max(0,x1),max(0,y1),min(fw-1,x2),min(fh-1,y2)
                if x1c < x2c and y1c < y2c:
                    crop = frame_to_process[y1c:y2c,x1c:x2c].copy()
                    text, src = ocr_module.extract_text_from_image_offline(crop, psm_mode=6)
                    ocr_parts.append(f"On {obj_name}: {text if src=='offline_tesseract' else ('No text found' if src=='offline_no_text' else src)}")
                else: ocr_parts.append(f"On {obj_name}: Invalid crop.")
            result_message = " ".join(ocr_parts)
    else: result_message = "[OFFLINE OCR ERROR] Frame or YOLO model unavailable."
    
    emit('manual_ai_result', {'type': 'ocr_offline_user_req', 'data': result_message}, namespace='/live')
    print(f"SERVER: Sent manual offline OCR: {result_message[:70]}...")


if __name__ == '__main__':
    print("Starting Flask-SocketIO server on http://127.0.0.1:5000 or http://localhost:5000")
    
    # Start the camera capture thread that updates video_frame_for_ai_server
    # The AI processing thread will be started when a client connects.
    cam_feed_thread = threading.Thread(target=camera_capture_thread_func, daemon=True)
    cam_feed_thread.start()
    
    # Use eventlet as the WSGI server for SocketIO
    print("Attempting to run with eventlet...")
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False) 
    except Exception as e:
        print(f"Could not run with eventlet: {e}. Ensure eventlet is installed (`pip install eventlet`)")
        print("Falling back to Flask's default development server (Werkzeug)...")
        print("NOTE: Werkzeug may not fully support all SocketIO async features.")
        app.run(host='0.0.0.0', port=5000, debug=False) # Fallback without socketio.run if eventlet fails
    
    print("Attempting to shut down server and threads...")
    stop_threads_event.set() 
    if camera_object_server and camera_object_server.isOpened(): camera_object_server.release()
    if cam_feed_thread.is_alive(): cam_feed_thread.join(timeout=1)
    # ai_processing_thread_server is started by socketio, its management on shutdown is tricky
    # For now, rely on daemon=True for threads and the main process exiting.
    if ai_processing_thread_server and ai_processing_thread_server.is_alive(): 
        print("Waiting for AI thread to join...")
        ai_processing_thread_server.join(timeout=1)
    print("Server shut down process initiated.")