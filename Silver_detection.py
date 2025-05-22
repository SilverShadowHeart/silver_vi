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
YOLO_CONFIDENCE_THRESHOLD = 0.5 # Adjusted based on previous tests

CRITICAL_OBJECT_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck',
    'traffic light', 'stop sign', 'chair', 'couch', 'bed', 'dining table',
    'laptop', 'cell phone', 'book', 'bottle', 'cup', 'door', 'backpack', 'handbag', 'suitcase'
]
TEXT_BEARING_CLASSES = ['book', 'stop sign', 'laptop', 'cell phone', 'tv', 'bottle']
LIVING_CLASSES = ['person', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']

# Analysis Intervals
YOLO_OUTPUT_INTERVAL = 0.5      # How often to process and print/speak YOLO summary
SCENE_HEURISTIC_INTERVAL = 5  # How often for basic offline scene heuristic
DETAILED_CAPTION_INTERVAL = 12 # For online caption or user-requested offline BLIP
ONLINE_OCR_INTERVAL = 7       # For attempting online OCR on flagged text

# Time trackers
last_yolo_output_time = 0
last_scene_heuristic_time = 0
last_detailed_caption_time = 0
last_online_ocr_time = 0

# User preference
USER_PREFERS_ONLINE = True # Default

# --- Helper function for simple offline scene heuristics ---
def get_simple_scene_heuristic(detected_object_names_with_prominence):
    if not detected_object_names_with_prominence:
        return "Area appears open or no distinct objects detected."
    
    names_only = [item['name'] for item in detected_object_names_with_prominence]
    person_count = names_only.count('person')
    vehicle_count = sum(1 for name in names_only if name in ['car', 'bus', 'truck', 'motorcycle', 'bicycle'])
    indoor_item_count = sum(1 for name in names_only if name in ['chair', 'table', 'couch', 'bed', 'laptop', 'tv', 'book'])

    if person_count > 2: return "Multiple people detected; area might be crowded."
    if vehicle_count > 0 and indoor_item_count == 0 : return "Street scene likely (vehicles detected)."
    if indoor_item_count > 0 and vehicle_count == 0: return "Indoor setting likely (furniture/electronics detected)."
    if person_count > 0: return f"{person_count} person(s) detected."
        
    unique_prominent_objects = sorted(list(set(
        item['name'] for item in detected_object_names_with_prominence if item['prominence'] in ['very prominent', 'prominent']
    )))
    if len(unique_prominent_objects) > 2:
        return f"Various objects detected including: {', '.join(unique_prominent_objects[:2])}."
    elif unique_prominent_objects:
        return f"Objects detected: {', '.join(unique_prominent_objects)}."
    return "General environment observed."

def live_detect_objects(model_name, camera_index, yolo_conf_threshold):
    global last_yolo_output_time, last_scene_heuristic_time, \
           last_detailed_caption_time, last_online_ocr_time, USER_PREFERS_ONLINE

    try:
        print(f"Loading YOLO model: {model_name}...");model = YOLO(model_name)
        print(f"YOLO loaded. Classes: {len(model.names)}")
    except Exception as e: print(f"CRITICAL Error loading YOLO: {e}"); return

    print(f"Opening camera: {camera_index}..."); cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened(): print(f"CRITICAL Error opening camera {camera_index}."); return
    print("Webcam opened.")

    fps_calc_prev_time = 0
    current_text_flags = [] # Store info about text regions from current YOLO cycle

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
        cv2.putText(annotated_display_frame, f"Online Mode: {'ON' if USER_PREFERS_ONLINE else 'OFF'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


        # --- OFFLINE CORE: YOLO Analysis & Output ---
        if (current_time - last_yolo_output_time > YOLO_OUTPUT_INTERVAL):
            last_yolo_output_time = current_time
            print(f"\n--- [{time.strftime('%H:%M:%S')}] OFFLINE YOLO SUMMARY ---")
            
            detected_objects_for_heuristic = []
            current_text_flags.clear() # Clear flags from previous cycle

            if not yolo_result_data.boxes:
                print("[YOLO] No objects detected in this cycle.")
            
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
                
                object_info = {"name": obj_name, "category": category, "prominence": prominence, "confidence": confidence, "bbox": bbox_coords}
                detected_objects_for_heuristic.append(object_info)

                if obj_name in CRITICAL_OBJECT_CLASSES:
                    yolo_output_str = f"[YOLO] {obj_name} ({category}) - {prominence} (Conf: {confidence:.2f})"
                    print(yolo_output_str)
                    # TODO: engine.say(yolo_output_str)

                if obj_name in TEXT_BEARING_CLASSES:
                    current_text_flags.append(object_info) # Store full info for potential OCR
                    text_flag_str = f"    [TEXT_FLAG] Potential text on {obj_name}."
                    print(text_flag_str)
                    # TODO: engine.say(text_flag_str)
            
            # --- Simple Offline Scene Heuristic (less frequent) ---
            if (current_time - last_scene_heuristic_time > SCENE_HEURISTIC_INTERVAL):
                last_scene_heuristic_time = current_time
                scene_heuristic_text = get_simple_scene_heuristic(detected_objects_for_heuristic)
                print(f"[SCENE_HEURISTIC_OFFLINE] {scene_heuristic_text}")
                # TODO: engine.say(f"Overall, {scene_heuristic_text}")


        # --- Periodic DETAILED SCENE CAPTIONING (Online or User-Requested Offline BLIP) ---
        internet_available = utils.check_internet_connection()
        
        # This block is for automatic detailed captions IF online is preferred AND available
        if USER_PREFERS_ONLINE and internet_available and \
           (current_time - last_detailed_caption_time > DETAILED_CAPTION_INTERVAL):
            last_detailed_caption_time = current_time
            print(f"\n--- [{time.strftime('%H:%M:%S')}] ATTEMPTING ONLINE DETAILED SCENE DESCRIPTION ---")
            
            raw_caption, caption_source = caption_module.get_caption(frame_bgr.copy(), prefer_online=True)
            preface = ""
            
            if caption_source == "online_cloud": # This will be true when you implement actual cloud calls
                preface = "[ENHANCED ONLINE SCENE] "
            elif caption_source == "offline_blip": # Fallback from failed/unimplemented online
                preface = "[ONLINE UNAVAILABLE - Offline Detail - Potentially Less Reliable] "
            elif caption_source == "online_placeholder":
                preface = "[SYSTEM NOTE] "
                raw_caption = "Online detailed description not yet fully implemented."
            else: # Error
                preface = "[CAPTIONING ERROR] "
                raw_caption = "Could not generate detailed scene description."
            
            full_message = preface + raw_caption
            print(full_message)
            # TODO: engine.say(full_message)

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
                    # Force online attempt. get_ocr_text handles the placeholder if not implemented.
                    text_read, ocr_source = ocr_module.get_ocr_text(text_crop, prefer_online=True)
                    
                    ocr_preface = ""
                    if ocr_source == "online_cloud_ocr": ocr_preface = f"[ONLINE OCR - {obj_name}] "
                    elif ocr_source == "online_placeholder_ocr": ocr_preface = f"[SYSTEM NOTE - {obj_name}] "; text_read = "Online OCR not yet implemented."
                    else: ocr_preface = f"[OCR INFO - {obj_name}] " # Covers errors or offline if it somehow ran

                    full_ocr_message = ocr_preface + text_read.replace(chr(10),' ').replace(chr(13),' ')
                    print(f"    {full_ocr_message}")
                    # TODO: engine.say(full_ocr_message)
                else: print(f"    Skipping online OCR for {obj_name}, invalid crop.")
            # current_text_flags.clear() # Decide if flags persist or clear each OCR cycle

        cv2.imshow("ComprehendVision - Assistive Tech", annotated_display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): print("Exiting application..."); break
        elif key == ord('o'): # Toggle Online/Offline Preference
            USER_PREFERS_ONLINE = not USER_PREFERS_ONLINE
            mode_text = "ONLINE preferred (will use internet if available)" if USER_PREFERS_ONLINE else "OFFLINE preferred (internet features disabled by user)"
            print(f"\n--- User toggled mode to: {mode_text} ---")
            # TODO: engine.say(f"Mode switched to {mode_text}")
        elif key == ord('d'): # Request detailed OFFLINE caption
            print(f"\n--- [{time.strftime('%H:%M:%S')}] USER REQUEST: DETAILED OFFLINE SCENE DESCRIPTION ---")
            if not internet_available and not USER_PREFERS_ONLINE:
                 print("Attempting detailed offline caption as requested (no internet or offline mode)...")
            elif internet_available and not USER_PREFERS_ONLINE:
                 print("Attempting detailed offline caption as per user preference (internet available but disabled)...")
            else: # internet_available and USER_PREFERS_ONLINE (but they pressed 'd' anyway)
                 print("User requested detailed offline caption specifically...")

            preface = "[OFFLINE DETAIL - Potentially Less Reliable] "
            offline_detail_caption, cap_src = caption_module.generate_caption_offline(frame_bgr.copy()) 
            full_spoken_message = preface + offline_detail_caption
            print(f"{full_spoken_message}")
            # TODO: engine.say(full_spoken_message)
        elif key == ord('n'): # Notification if trying online feature without internet
             if USER_PREFERS_ONLINE and not internet_available:
                  no_internet_msg = "Cannot perform enhanced online functions: No internet connection detected."
                  print(f"\n--- {no_internet_msg} ---")
                  # TODO: engine.say(no_internet_msg)


    cap.release(); cv2.destroyAllWindows()
    print("Application shut down cleanly.")

if __name__ == "__main__":
    last_yolo_output_time = 0
    last_scene_heuristic_time = 0
    last_detailed_caption_time = 0
    last_online_ocr_time = 0
    USER_PREFERS_ONLINE = True # Default
    
    # Initial check to inform user about internet status at startup
    if USER_PREFERS_ONLINE:
        if utils.check_internet_connection():
            print("System started in ONLINE preferred mode. Internet connection detected.")
        else:
            print("System started in ONLINE preferred mode, but NO INTERNET detected. Falling back to offline core.")
    else:
        print("System started in OFFLINE preferred mode.")

    live_detect_objects(MODEL_NAME, CAMERA_INDEX, YOLO_CONFIDENCE_THRESHOLD)