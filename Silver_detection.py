# Silver_detection.py
from ultralytics import YOLO
import cv2
import time
import os

try:
    import caption_module
    import ocr_module
    import utils
except ImportError as e:
    print(f"CRITICAL Error importing custom modules: {e}"); exit()

# --- CONFIGURATIONS ---
MODEL_NAME = 'yolov8s.pt'
CAMERA_INDEX = 0
YOLO_CONFIDENCE_THRESHOLD = 0.5 

CRITICAL_OBJECT_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck',
    'traffic light', 'stop sign', 'chair', 'couch', 'bed', 'dining table',
    'laptop', 'cell phone', 'book', 'bottle', 'cup', 'door', 'backpack', 'handbag', 'suitcase'
]
TEXT_BEARING_CLASSES = ['book', 'stop sign', 'laptop', 'cell phone', 'tv', 'bottle'] # Add more as needed
LIVING_CLASSES = ['person', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']

# Analysis Intervals
YOLO_OUTPUT_INTERVAL = 0.5      # How often to process and print/speak YOLO summary
SCENE_HEURISTIC_INTERVAL = 5  # How often for basic offline scene heuristic
DETAILED_CAPTION_INTERVAL = 12 # For online caption or user-requested offline BLIP
ONLINE_OCR_INTERVAL = 7       # For attempting online OCR on flagged text

# Time trackers - To be managed by the live_detect_objects function
# last_yolo_output_time = 0 # Initialized inside function
# last_scene_heuristic_time = 0 # Initialized inside function
# last_detailed_caption_time = 0 # Initialized inside function
# last_online_ocr_time = 0 # Initialized inside function

# User preference - To be managed by the live_detect_objects function
# USER_PREFERS_ONLINE = True # Default, initialized inside function

# --- Helper function for simple offline scene heuristics ---
def get_simple_scene_heuristic(detected_object_infos): # Takes list of object_info dicts
    if not detected_object_infos:
        return "Area appears open or no distinct objects detected."
    
    names_only = [item['name'] for item in detected_object_infos]
    person_count = names_only.count('person')
    vehicle_count = sum(1 for name in names_only if name in ['car', 'bus', 'truck', 'motorcycle', 'bicycle'])
    indoor_item_count = sum(1 for name in names_only if name in ['chair', 'table', 'couch', 'bed', 'laptop', 'tv', 'book'])

    if person_count > 2: return "Multiple people detected; area might be crowded."
    if vehicle_count > 0 and indoor_item_count == 0 : return "Street scene likely (vehicles detected)."
    if indoor_item_count > 0 and vehicle_count == 0: return "Indoor setting likely (furniture/electronics detected)."
    if person_count == 1: return f"1 person detected."
    if person_count > 1 : return f"{person_count} persons detected."
        
    unique_prominent_objects = sorted(list(set(
        item['name'] for item in detected_object_infos if item['prominence'] in ['very prominent', 'prominent']
    )))
    if len(unique_prominent_objects) > 2:
        return f"Various objects detected including: {', '.join(unique_prominent_objects[:2])}."
    elif unique_prominent_objects:
        return f"Objects detected: {', '.join(unique_prominent_objects)}."
    return "General environment observed."

def live_detect_objects(model_name, camera_index, yolo_conf_threshold):
    # Initialize time trackers and user preference inside the function scope
    # This makes the function more self-contained if called multiple times, though for this app it's called once.
    last_yolo_output_time = 0
    last_scene_heuristic_time = 0
    last_detailed_caption_time = 0
    last_online_ocr_time = 0
    USER_PREFERS_ONLINE = True # Default

    try:
        print(f"Loading YOLO model: {model_name}...");model = YOLO(model_name)
        print(f"YOLO loaded. Classes: {len(model.names)}")
    except Exception as e: print(f"CRITICAL Error loading YOLO: {e}"); return

    print(f"Opening camera: {camera_index}..."); cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened(): print(f"CRITICAL Error opening camera {camera_index}."); return
    print("Webcam opened.")

    fps_calc_prev_time = 0
    current_text_flags = [] # Stores info about text regions for 'r' key and online OCR

    # Initial status print based on USER_PREFERS_ONLINE and actual internet
    print("\n--- System Status at Startup ---")
    initial_internet_available = utils.check_internet_connection()
    if USER_PREFERS_ONLINE:
        if initial_internet_available:
            print("Mode: ONLINE preferred. Internet Connection: DETECTED.")
            # TODO: engine.say("Online mode active. Internet detected.")
        else:
            print("Mode: ONLINE preferred. Internet Connection: NOT DETECTED. Using offline core.")
            # TODO: engine.say("Online mode preferred, but no internet. Using offline features.")
    else:
        print("Mode: OFFLINE preferred by user setting.")
        # TODO: engine.say("Offline mode active by user preference.")
    print("--------------------------------\n")


    while True:
        success, frame_bgr = cap.read()
        if not success: print("Error: Failed to capture frame."); break
        
        current_time = time.time()

        fps_calc_new_time = time.time()
        if (fps_calc_new_time - fps_calc_prev_time) > 0: fps = 1 / (fps_calc_new_time - fps_calc_prev_time)
        else: fps = 0
        fps_calc_prev_time = fps_calc_new_time
        fps_text = f"FPS: {int(fps)}"

        yolo_results_list = model(frame_bgr, conf=yolo_conf_threshold, verbose=False)
        yolo_result_data = yolo_results_list[0]
        annotated_display_frame = yolo_result_data.plot()
        cv2.putText(annotated_display_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        online_mode_text = "Online Mode: ON (Internet Preferred)" if USER_PREFERS_ONLINE else "Online Mode: OFF (User Prefers Offline)"
        cv2.putText(annotated_display_frame, online_mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # --- DATA COLLECTION from current frame's YOLO results (happens every frame) ---
        objects_detected_this_frame = [] 
        if yolo_result_data.boxes:
            for box in yolo_result_data.boxes:
                class_id = int(box.cls[0])
                obj_name = model.names[class_id]
                confidence = float(box.conf[0])
                bbox_coords = box.xyxy[0].tolist()
                category = "living" if obj_name in LIVING_CLASSES else "non-living"
                box_height = bbox_coords[3] - bbox_coords[1]
                frame_height = frame_bgr.shape[0]
                prominence = "unknown"
                if frame_height > 0:
                    if box_height > frame_height * 0.5: prominence = "very prominent"
                    elif box_height > frame_height * 0.25: prominence = "prominent"
                    elif box_height > frame_height * 0.1: prominence = "medium prominence"
                    else: prominence = "less prominent"
                objects_detected_this_frame.append({
                    "name": obj_name, "category": category, "prominence": prominence, 
                    "confidence": confidence, "bbox": bbox_coords
                })

        # --- Periodic OFFLINE CORE Output (YOLO Summary & Scene Heuristic) ---
        if (current_time - last_yolo_output_time > YOLO_OUTPUT_INTERVAL):
            last_yolo_output_time = current_time
            print(f"\n--- [{time.strftime('%H:%M:%S')}] OFFLINE YOLO SUMMARY ---")
            
            current_text_flags.clear() # Clear previous flags before processing new ones

            if not objects_detected_this_frame:
                print("[YOLO] No significant objects detected in this cycle.")
            else:
                for obj_info in objects_detected_this_frame:
                    if obj_info['name'] in CRITICAL_OBJECT_CLASSES:
                        yolo_output_str = f"[YOLO] {obj_info['name']} ({obj_info['category']}) - {obj_info['prominence']} (Conf: {obj_info['confidence']:.2f})"
                        print(yolo_output_str)
                        # TODO: engine.say(yolo_output_str)

                    if obj_info['name'] in TEXT_BEARING_CLASSES:
                        current_text_flags.append(obj_info)
                        text_flag_str = f"    [TEXT_FLAG] Potential text on {obj_info['name']}."
                        print(text_flag_str)
                        # TODO: engine.say(text_flag_str)
            
            if (current_time - last_scene_heuristic_time > SCENE_HEURISTIC_INTERVAL):
                last_scene_heuristic_time = current_time
                scene_heuristic_text = get_simple_scene_heuristic(objects_detected_this_frame)
                print(f"[SCENE_HEURISTIC_OFFLINE] {scene_heuristic_text}")
                # TODO: engine.say(f"Overall, {scene_heuristic_text}")


        # --- Periodic DETAILED SCENE CAPTIONING ---
        internet_available = utils.check_internet_connection()
        
        if (current_time - last_detailed_caption_time > DETAILED_CAPTION_INTERVAL):
            last_detailed_caption_time = current_time
            detailed_caption_message_to_user = "" # Initialize

            if USER_PREFERS_ONLINE and internet_available:
                print(f"\n--- [{time.strftime('%H:%M:%S')}] ATTEMPTING ONLINE DETAILED SCENE DESCRIPTION ---")
                raw_caption, caption_source = caption_module.get_caption(frame_bgr.copy(), prefer_online=True)
                
                if caption_source == "online_cloud": 
                    detailed_caption_message_to_user = "[ENHANCED ONLINE SCENE] " + raw_caption
                elif caption_source == "offline_blip": # Fallback from failed/unimplemented online
                    detailed_caption_message_to_user = "[ONLINE UNAVAILABLE - Using Offline Detail - Potentially Less Reliable] " + raw_caption
                elif caption_source == "online_placeholder":
                    detailed_caption_message_to_user = "[SYSTEM NOTE] Online detailed description service not yet fully active. Basic scene understanding provided. Press 'D' for a less reliable detailed offline attempt."
                else: # Error
                    detailed_caption_message_to_user = "[CAPTIONING ERROR] Could not generate a detailed scene description. Basic scene understanding provided."
            
            elif USER_PREFERS_ONLINE and not internet_available:
                # This case for when user wants online, but internet is out.
                print(f"\n--- [{time.strftime('%H:%M:%S')}] DETAILED SCENE DESCRIPTION ATTEMPT (NO INTERNET) ---")
                detailed_caption_message_to_user = "[SYSTEM NOTE] No internet for online detailed description. Basic scene understanding provided. Press 'D' for a less reliable detailed offline attempt."
            
            # If not USER_PREFERS_ONLINE, this automatic block does nothing for detailed captions by default.
            # User must press 'd' to get offline BLIP. SCENE_HEURISTIC_OFFLINE is the auto offline info.

            if detailed_caption_message_to_user:
                print(detailed_caption_message_to_user)
                # TODO: engine.say(detailed_caption_message_to_user)
                print("---------------------------------------------------")

        # --- ONLINE TEXT READING (if text flagged, internet available, and user prefers online) ---
        if USER_PREFERS_ONLINE and internet_available and current_text_flags and \
           (current_time - last_online_ocr_time > ONLINE_OCR_INTERVAL):
            last_online_ocr_time = current_time
            print(f"\n--- [{time.strftime('%H:%M:%S')}] ATTEMPTING ONLINE OCR ---")
            for region_info in current_text_flags:
                obj_name = region_info["name"]; obj_bbox = region_info["bbox"]
                print(f"  Reading text on flagged: {obj_name}")
                x1,y1,x2,y2 = [int(c) for c in obj_bbox]; fh,fw = frame_bgr.shape[:2]
                x1c,y1c,x2c,y2c = max(0,x1),max(0,y1),min(fw-1,x2),min(fh-1,y2)
                if x1c < x2c and y1c < y2c:
                    text_crop = frame_bgr[y1c:y2c, x1c:x2c].copy()
                    text_read, ocr_source = ocr_module.get_ocr_text(text_crop, prefer_online=True)
                    
                    ocr_preface = ""
                    if ocr_source == "online_cloud_ocr": ocr_preface = f"[ONLINE OCR - {obj_name}] "
                    elif ocr_source == "online_placeholder_ocr": ocr_preface = f"[SYSTEM NOTE - {obj_name}] "; text_read = "Online OCR not yet implemented."
                    elif ocr_source == "offline_not_attempted": ocr_preface = f"[SYSTEM NOTE - {obj_name}] "; text_read = "Offline OCR not attempted by default for this item."
                    else: ocr_preface = f"[OCR INFO - {obj_name}] "

                    full_ocr_message = ocr_preface + text_read.replace(chr(10),' ').replace(chr(13),' ')
                    print(f"    {full_ocr_message}")
                    # TODO: engine.say(full_ocr_message)
                else: print(f"    Skipping online OCR for {obj_name}, invalid crop.")

        cv2.imshow("ComprehendVision - Assistive Tech", annotated_display_frame)

        # --- Handle Key Press ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): print("Exiting application..."); break
        elif key == ord('o'): 
            USER_PREFERS_ONLINE = not USER_PREFERS_ONLINE
            mode_text = "ONLINE preferred (will use internet if available)" if USER_PREFERS_ONLINE else "OFFLINE preferred (internet features disabled by user)"
            print(f"\n--- User toggled mode to: {mode_text} ---")
            # TODO: engine.say(f"Mode switched to {mode_text}")
        elif key == ord('d'): 
            print(f"\n--- [{time.strftime('%H:%M:%S')}] USER REQUEST: DETAILED OFFLINE SCENE DESCRIPTION ---")
            preface = "[OFFLINE DETAIL - Potentially Less Reliable] "
            offline_detail_caption, cap_src = caption_module.generate_caption_offline(frame_bgr.copy()) 
            full_spoken_message = preface + offline_detail_caption
            print(f"{full_spoken_message}")
            # TODO: engine.say(full_spoken_message)
        elif key == ord('r'): # Request to attempt OFFLINE OCR on flagged text
            print(f"\n--- [{time.strftime('%H:%M:%S')}] USER REQUEST: ATTEMPT OFFLINE OCR ---")
            if not current_text_flags:
                no_flags_msg = "[OCR REQUEST] No text regions were flagged in the last analysis cycle to attempt OCR on."
                print(no_flags_msg)
                # TODO: engine.say(no_flags_msg)
            else:
                print("Attempting offline OCR on recently flagged text regions (results may be inaccurate):")
                # TODO: engine.say("Attempting offline OCR on recently flagged text regions. Results may be inaccurate.")
                for region_info in current_text_flags:
                    obj_name = region_info["name"]
                    obj_bbox = region_info["bbox"]
                    print(f"  Reading text on flagged: {obj_name}")
                    
                    x1,y1,x2,y2 = [int(c) for c in obj_bbox]; fh,fw = frame_bgr.shape[:2]
                    x1c,y1c,x2c,y2c = max(0,x1),max(0,y1),min(fw-1,x2),min(fh-1,y2)

                    if x1c < x2c and y1c < y2c:
                        text_crop_bgr = frame_bgr[y1c:y2c, x1c:x2c].copy()
                        # Directly call the offline OCR function from ocr_module
                        # psm_mode=1 (Auto page seg with OSD) or 6 (single block) might be better for isolated text on objects
                        text_read, ocr_source = ocr_module.extract_text_from_image_offline(text_crop_bgr, psm_mode=6) 
                        
                        ocr_preface = f"[OFFLINE OCR - {obj_name} - Potentially Unreliable] "
                        
                        if ocr_source == "offline_tesseract":
                            full_ocr_message = ocr_preface + text_read.replace(chr(10),' ').replace(chr(13),' ')
                        elif ocr_source == "offline_no_text":
                            full_ocr_message = ocr_preface + "No clear text found by offline OCR."
                        else: # Error cases
                            full_ocr_message = ocr_preface + text_read # text_read will contain the error message
                        
                        print(f"    {full_ocr_message}")
                        # TODO: engine.say(full_ocr_message)
                    else:
                        print(f"    Skipping offline OCR for {obj_name}, invalid crop region.")
                print("------------------------------------------------------")

    cap.release(); cv2.destroyAllWindows()
    print("Application shut down cleanly.")

if __name__ == "__main__":
    # These are effectively global for the script if live_detect_objects is called directly
    # but initializing them here is fine as they are passed or managed by the function.
    # The function itself now initializes them for its scope.
    
    print("Starting ComprehendVision System...")
    # Initial check for user info, live_detect_objects will also print its own status
    is_online_initially = utils.check_internet_connection()
    user_prefers_online_initially = True # Matches the default in live_detect_objects

    if user_prefers_online_initially:
        if is_online_initially:
            print("System will start with: ONLINE preferred. Internet Connection: DETECTED.")
        else:
            print("System will start with: ONLINE preferred, but NO INTERNET detected. Using offline core.")
    else: # This case won't be hit with current default
        print("System will start with: OFFLINE preferred by user setting.")

    live_detect_objects(MODEL_NAME, CAMERA_INDEX, YOLO_CONFIDENCE_THRESHOLD)