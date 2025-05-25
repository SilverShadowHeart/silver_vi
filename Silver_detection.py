# Silver_detection.py
from ultralytics import YOLO
import cv2
import time
import os
import pyttsx3 # Import pyttsx3

try:
    import caption_module
    import ocr_module
    import utils
except ImportError as e:
    print(f"CRITICAL Error importing custom modules: {e}"); exit()

# --- TTS ENGINE INITIALIZATION ---
try:
    tts_engine = pyttsx3.init()
    # Optional: Adjust properties
    # tts_engine.setProperty('rate', 150)  # Speed percent (can go over 100)
    # tts_engine.setProperty('volume', 0.9) # Volume 0-1
    # voices = tts_engine.getProperty('voices')
    # tts_engine.setProperty('voice', voices[1].id) # Example: Change voice if multiple are available
    print("TTS Engine Initialized.")
except Exception as e:
    print(f"CRITICAL Error initializing TTS Engine: {e}. Speech output will be unavailable.")
    tts_engine = None

# --- CONFIGURATIONS --- (Keep these as they were)
MODEL_NAME = 'yolov8s.pt'
CAMERA_INDEX = 0
# ... (rest of your configurations: CAMERA_INDEX, YOLO_CONFIDENCE_THRESHOLD, class lists, intervals) ...
YOLO_CONFIDENCE_THRESHOLD = 0.5
CRITICAL_OBJECT_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck', 'traffic light', 'stop sign', 'chair', 'couch', 'bed', 'dining table', 'laptop', 'cell phone', 'book', 'bottle', 'cup', 'door', 'backpack', 'handbag', 'suitcase']
TEXT_BEARING_CLASSES = ['book', 'stop sign', 'laptop', 'cell phone', 'tv', 'bottle'] 
LIVING_CLASSES = ['person', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
YOLO_OUTPUT_INTERVAL = 0.5      
SCENE_HEURISTIC_INTERVAL = 5  
DETAILED_CAPTION_INTERVAL = 12 
ONLINE_OCR_INTERVAL = 7  

# --- TTS HELPER FUNCTION ---
def speak(text_to_speak):
    if tts_engine:
        try:
            # Simple approach: Wait if busy, then speak.
            # For more advanced, queue messages if it's speaking too fast.
            # while tts_engine.isBusy(): # This might cause stutter if new info comes too fast
            #     time.sleep(0.05)
            tts_engine.say(text_to_speak)
            tts_engine.runAndWait() # Blocks until speaking is done
        except Exception as e:
            print(f"TTS Error: Could not speak text - {e}")
    else:
        print(f"TTS Disabled (Print Fallback): {text_to_speak}")


# --- Helper function for simple offline scene heuristics --- (Keep as is)
def get_simple_scene_heuristic(detected_object_infos):
    # ... (your existing heuristic logic) ...
    if not detected_object_infos: return "Area appears open or no distinct objects detected."
    names_only = [item['name'] for item in detected_object_infos]
    person_count = names_only.count('person')
    vehicle_count = sum(1 for name in names_only if name in ['car', 'bus', 'truck', 'motorcycle', 'bicycle'])
    indoor_item_count = sum(1 for name in names_only if name in ['chair', 'table', 'couch', 'bed', 'laptop', 'tv', 'book'])
    if person_count > 2: return "Multiple people detected; area might be crowded."
    if vehicle_count > 0 and indoor_item_count == 0 : return "Street scene likely. Vehicles detected." # Made more natural
    if indoor_item_count > 0 and vehicle_count == 0: return "Indoor setting likely. Furniture or electronics detected." # Made more natural
    if person_count == 1: return f"1 person detected."
    if person_count > 1 : return f"{person_count} persons detected."
    unique_prominent_objects = sorted(list(set(item['name'] for item in detected_object_infos if item['prominence'] in ['very prominent', 'prominent'])))
    if len(unique_prominent_objects) > 2: return f"Various objects detected including {unique_prominent_objects[0]} and {unique_prominent_objects[1]}."
    elif unique_prominent_objects: return f"Objects detected including {', '.join(unique_prominent_objects)}."
    return "General environment observed."


def live_detect_objects(model_name, camera_index, yolo_conf_threshold):
    global USER_PREFERS_ONLINE # Allow modification by key press
    # Initialize time trackers and user preference inside the function scope
    last_yolo_output_time = 0
    last_scene_heuristic_time = 0
    last_detailed_caption_time = 0
    last_online_ocr_time = 0
    USER_PREFERS_ONLINE = True # Default

    try:
        print(f"Loading YOLO model: {model_name}..."); model = YOLO(model_name)
        print(f"YOLO loaded. Classes: {len(model.names)}")
    except Exception as e: print(f"CRITICAL Error loading YOLO: {e}"); speak(f"Critical error loading vision model."); return

    print(f"Opening camera: {camera_index}..."); cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened(): print(f"CRITICAL Error opening camera {camera_index}."); speak(f"Critical error opening camera."); return
    print("Webcam opened.")

    fps_calc_prev_time = 0
    current_text_flags = [] 

    # --- Startup Announcements ---
    print("\n--- System Status at Startup ---")
    startup_message_print = ""
    startup_message_speak = ""
    initial_internet_available = utils.check_internet_connection()
    if USER_PREFERS_ONLINE:
        if initial_internet_available:
            startup_message_print = "Mode: ONLINE preferred. Internet Connection: DETECTED."
            startup_message_speak = "Online mode active. Internet detected."
        else:
            startup_message_print = "Mode: ONLINE preferred. Internet Connection: NOT DETECTED. Using offline core."
            startup_message_speak = "Online mode preferred, but no internet. Using offline features."
    else:
        startup_message_print = "Mode: OFFLINE preferred by user setting."
        startup_message_speak = "Offline mode active by user preference."
    print(startup_message_print)
    speak(startup_message_speak)
    print("--------------------------------\n")

    while True:
        success, frame_bgr = cap.read()
        if not success: print("Error: Failed to capture frame."); speak("Error, failed to capture frame."); break
        
        current_time = time.time()

        # ... (FPS calculation as before) ...
        fps_calc_new_time = time.time()
        if (fps_calc_new_time - fps_calc_prev_time) > 0: fps = 1 / (fps_calc_new_time - fps_calc_prev_time)
        else: fps = 0
        fps_calc_prev_time = fps_calc_new_time
        fps_text = f"FPS: {int(fps)}"


        yolo_results_list = model(frame_bgr, conf=yolo_conf_threshold, verbose=False)
        yolo_result_data = yolo_results_list[0]
        annotated_display_frame = yolo_result_data.plot()
        cv2.putText(annotated_display_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        online_mode_text_display = "Online Mode: ON" if USER_PREFERS_ONLINE else "Online Mode: OFF (User)"
        cv2.putText(annotated_display_frame, online_mode_text_display, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        objects_detected_this_frame = [] 
        if yolo_result_data.boxes:
            for box in yolo_result_data.boxes:
                # ... (object_info population as before) ...
                class_id = int(box.cls[0]); obj_name = model.names[class_id]; confidence = float(box.conf[0])
                bbox_coords = box.xyxy[0].tolist(); category = "living" if obj_name in LIVING_CLASSES else "non-living"
                box_height = bbox_coords[3] - bbox_coords[1]; frame_height = frame_bgr.shape[0]; prominence = "unknown"
                if frame_height > 0:
                    if box_height > frame_height * 0.5: prominence = "very prominent"
                    elif box_height > frame_height * 0.25: prominence = "prominent"
                    elif box_height > frame_height * 0.1: prominence = "medium prominence"
                    else: prominence = "less prominent"
                objects_detected_this_frame.append({"name": obj_name, "category": category, "prominence": prominence, "confidence": confidence, "bbox": bbox_coords})

        # --- Periodic OFFLINE CORE Output (YOLO Summary & Scene Heuristic) ---
        if (current_time - last_yolo_output_time > YOLO_OUTPUT_INTERVAL):
            last_yolo_output_time = current_time
            print(f"\n--- [{time.strftime('%H:%M:%S')}] OFFLINE YOLO SUMMARY ---")
            current_text_flags.clear()
            
            # --- Generate text for TTS from YOLO detections ---
            yolo_speech_output = []

            if not objects_detected_this_frame:
                no_objects_msg = "[YOLO] No significant objects detected." # Simplified
                print(no_objects_msg)
                yolo_speech_output.append("No significant objects detected.")
            else:
                for obj_info in objects_detected_this_frame:
                    if obj_info['name'] in CRITICAL_OBJECT_CLASSES:
                        yolo_print_str = f"[YOLO] {obj_info['name']} ({obj_info['category']}) - {obj_info['prominence']} (Conf: {obj_info['confidence']:.2f})"
                        print(yolo_print_str)
                        yolo_speech_output.append(f"{obj_info['prominence']} {obj_info['name']}.") # More natural speech

                    if obj_info['name'] in TEXT_BEARING_CLASSES:
                        current_text_flags.append(obj_info)
                        text_flag_print_str = f"    [TEXT_FLAG] Potential text on {obj_info['name']}."
                        print(text_flag_print_str)
                        yolo_speech_output.append(f"Potential text on {obj_info['name']}.")
             
            # Speak YOLO summary (if any)
            if yolo_speech_output:
                speak(". ".join(yolo_speech_output)) # Join multiple detections into one speech segment

            # Simple Offline Scene Heuristic
            if (current_time - last_scene_heuristic_time > SCENE_HEURISTIC_INTERVAL):
                last_scene_heuristic_time = current_time
                scene_heuristic_text = get_simple_scene_heuristic(objects_detected_this_frame)
                print(f"[SCENE_HEURISTIC_OFFLINE] {scene_heuristic_text}")
                speak(f"Overall, {scene_heuristic_text}")

        # --- Periodic DETAILED SCENE CAPTIONING ---
        internet_available = utils.check_internet_connection() # Check once per main loop iter
        
        if (current_time - last_detailed_caption_time > DETAILED_CAPTION_INTERVAL):
            last_detailed_caption_time = current_time
            caption_print_message = ""
            caption_speak_message = ""

            if USER_PREFERS_ONLINE and internet_available:
                print(f"\n--- [{time.strftime('%H:%M:%S')}] ATTEMPTING ONLINE DETAILED SCENE DESCRIPTION ---")
                raw_caption, caption_source = caption_module.get_caption(frame_bgr.copy(), prefer_online=True)
                
                if caption_source == "online_gemini_caption": 
                    caption_print_message = "[ENHANCED ONLINE SCENE] " + raw_caption
                    caption_speak_message = "Online description: " + raw_caption
                elif caption_source == "offline_blip": 
                    caption_print_message = "[ONLINE UNAVAILABLE - Using Offline Detail - Potentially Less Reliable] " + raw_caption
                    caption_speak_message = "Online description unavailable. Offline detail, which may be less reliable: " + raw_caption
                elif "error" in caption_source.lower(): 
                    caption_print_message = f"[CAPTIONING SERVICE ERROR - Source: {caption_source}] {raw_caption}"
                    caption_speak_message = f"Captioning service error: {raw_caption.split('-')[-1].strip()}" # Speak only the core error
                else: 
                    caption_print_message = f"[UNEXPECTED CAPTION SOURCE: {caption_source}] {raw_caption}. Basic scene understanding provided."
                    caption_speak_message = "Detailed caption status unclear. Basic scene understanding provided."
            
            elif USER_PREFERS_ONLINE and not internet_available:
                print(f"\n--- [{time.strftime('%H:%M:%S')}] DETAILED SCENE DESCRIPTION ATTEMPT (NO INTERNET) ---")
                caption_print_message = "[SYSTEM NOTE] No internet for online detailed description. Basic heuristic active. Press 'D' for optional less reliable offline caption."
                caption_speak_message = "No internet for enhanced description. Press D for optional offline detail."
            
            if caption_print_message: 
                print(caption_print_message); print("---------------------------------------------------")
                speak(caption_speak_message)

        # --- ONLINE TEXT READING ---
        if USER_PREFERS_ONLINE and internet_available and current_text_flags and \
           (current_time - last_online_ocr_time > ONLINE_OCR_INTERVAL):
            last_online_ocr_time = current_time
            print(f"\n--- [{time.strftime('%H:%M:%S')}] ATTEMPTING ONLINE OCR ---")
            ocr_speech_output = []
            for region_info in current_text_flags:
                obj_name = region_info["name"]; obj_bbox = region_info["bbox"]
                print(f"  Reading text on flagged: {obj_name}")
                x1,y1,x2,y2=[int(c) for c in obj_bbox];fh,fw=frame_bgr.shape[:2];x1c,y1c,x2c,y2c=max(0,x1),max(0,y1),min(fw-1,x2),min(fh-1,y2)
                if x1c < x2c and y1c < y2c:
                    text_crop = frame_bgr[y1c:y2c,x1c:x2c].copy()
                    text_read, ocr_source = ocr_module.get_ocr_text(text_crop, prefer_online=True)
                    
                    ocr_print_msg, ocr_speak_msg = "", ""
                    if ocr_source == "online_gemini_ocr":
                        ocr_print_msg = f"[ONLINE OCR - {obj_name}] Text: {text_read.replace(chr(10),' ').replace(chr(13),' ')}"
                        ocr_speak_msg = f"Online text on {obj_name} reads: {text_read}"
                    elif ocr_source == "online_no_text_ocr":
                        ocr_print_msg = f"[ONLINE OCR - {obj_name}] {text_read}"
                        ocr_speak_msg = f"Online OCR found no clear text on {obj_name}."
                    elif "error" in ocr_source.lower():
                        ocr_print_msg = f"[ONLINE OCR ERROR - {obj_name} - Source: {ocr_source}] {text_read}"
                        ocr_speak_msg = f"Online OCR error for {obj_name}: {text_read.split('-')[-1].strip()}"
                    else:
                        ocr_print_msg = f"[OCR STATUS - {obj_name} - Source: {ocr_source}] {text_read}"
                        ocr_speak_msg = f"OCR status for {obj_name}: {text_read}"
                    
                    print(f"    {ocr_print_msg}")
                    ocr_speech_output.append(ocr_speak_msg)
                else: print(f"    Skipping online OCR for {obj_name}, invalid crop.")
            if ocr_speech_output:
                speak(". ".join(ocr_speech_output))

        cv2.imshow("ComprehendVision - Assistive Tech", annotated_display_frame)

        # --- Handle Key Press ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): print("Exiting application..."); speak("Exiting application."); break
        elif key == ord('o'): 
            USER_PREFERS_ONLINE = not USER_PREFERS_ONLINE
            mode_text_print = "ONLINE preferred (internet permitting)" if USER_PREFERS_ONLINE else "OFFLINE preferred (user choice)"
            mode_text_speak = "Online mode preferred" if USER_PREFERS_ONLINE else "Offline mode preferred"
            print(f"\n--- User toggled mode to: {mode_text_print} ---")
            speak(f"Mode switched to {mode_text_speak}")
        elif key == ord('d'): 
            print_msg_d = f"\n--- [{time.strftime('%H:%M:%S')}] USER REQUEST: DETAILED OFFLINE SCENE DESCRIPTION ---"
            print(print_msg_d)
            preface_speak = "Attempting detailed offline description, which may be less reliable: "
            preface_print = "[OFFLINE DETAIL - Potentially Less Reliable] "
            offline_detail_caption, cap_src = caption_module.generate_caption_offline(frame_bgr.copy()) 
            print(f"{preface_print}{offline_detail_caption}")
            speak(preface_speak + offline_detail_caption)
        elif key == ord('r'): 
            print_msg_r = f"\n--- [{time.strftime('%H:%M:%S')}] USER REQUEST: ATTEMPT OFFLINE OCR ---"
            print(print_msg_r)
            if not current_text_flags: 
                no_flags_print = "[OCR REQUEST] No text regions flagged in last YOLO cycle."
                print(no_flags_print); speak(no_flags_print)
            else:
                ocr_r_speak_intro = "Attempting offline OCR on recent flags. Results may be inaccurate: "
                print(ocr_r_speak_intro.replace(": ", ":\n")) # Print with newline
                # speak(ocr_r_speak_intro) # Speak intro once
                ocr_r_results_speak = []
                for region_info in current_text_flags:
                    obj_name=region_info["name"];obj_bbox=region_info["bbox"];
                    print_ocr_r_obj = f"  Reading text on flagged: {obj_name}"
                    print(print_ocr_r_obj)
                    x1,y1,x2,y2=[int(c) for c in obj_bbox];fh,fw=frame_bgr.shape[:2];x1c,y1c,x2c,y2c=max(0,x1),max(0,y1),min(fw-1,x2),min(fh-1,y2)
                    if x1c < x2c and y1c < y2c:
                        text_crop_bgr = frame_bgr[y1c:y2c, x1c:x2c].copy()
                        text_read, ocr_source = ocr_module.extract_text_from_image_offline(text_crop_bgr, psm_mode=6)
                        ocr_preface_print = f"[OFFLINE OCR - {obj_name} - Potentially Unreliable] "
                        ocr_speak_item = ""
                        if ocr_source == "offline_tesseract": 
                            print(f"    {ocr_preface_print}{text_read.replace(chr(10),' ').replace(chr(13),' ')}")
                            ocr_speak_item = f"Text on {obj_name} reads: {text_read}."
                        elif ocr_source == "offline_no_text": 
                            print(f"    {ocr_preface_print}No clear text found by offline OCR.")
                            ocr_speak_item = f"No clear text found by offline OCR on {obj_name}."
                        else: # Error cases
                            print(f"    {ocr_preface_print}{text_read}")
                            ocr_speak_item = f"Offline OCR error for {obj_name}: {text_read.split('-')[-1].strip()}."
                        ocr_r_results_speak.append(ocr_speak_item)
                    else: print(f"    Skipping offline OCR for {obj_name}, invalid crop.")
                if ocr_r_results_speak: speak(ocr_r_speak_intro + " ".join(ocr_r_results_speak))
                print("------------------------------------------------------")

    cap.release(); cv2.destroyAllWindows()
    print("Application shut down cleanly.")
    speak("Application shut down.")

if __name__ == "__main__":
    print("Starting ComprehendVision System...")
    speak("Starting Comprehend Vision System.") # Initial speak
    live_detect_objects(MODEL_NAME, CAMERA_INDEX, YOLO_CONFIDENCE_THRESHOLD)