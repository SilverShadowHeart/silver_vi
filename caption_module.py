# caption_module.py
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import cv2
import os

try:
    import utils
except ImportError:
    print("CRITICAL Error: utils.py not found in caption_module.")
    class MockUtils:
        def check_internet_connection(self): return False
    utils = MockUtils()

OFFLINE_CAPTION_MODEL_ID = "Salesforce/blip-image-captioning-base"
caption_processor_offline = None
caption_model_offline = None
offline_model_device = "cpu"

try:
    print(f"CaptionModule: Loading OFFLINE processor: {OFFLINE_CAPTION_MODEL_ID}...")
    caption_processor_offline = BlipProcessor.from_pretrained(OFFLINE_CAPTION_MODEL_ID)
    print(f"CaptionModule: Loading OFFLINE model: {OFFLINE_CAPTION_MODEL_ID}...")
    caption_model_offline = BlipForConditionalGeneration.from_pretrained(OFFLINE_CAPTION_MODEL_ID)
    if torch.cuda.is_available():
        offline_model_device = "cuda"
    caption_model_offline.to(offline_model_device)
    print(f"CaptionModule: Offline model loaded to: {offline_model_device}")
except Exception as e:
    print(f"CRITICAL ERROR loading OFFLINE captioning model: {e}. Offline captioning unavailable.")

def generate_caption_offline(image_frame_bgr):
    if caption_model_offline is None or caption_processor_offline is None:
        return "Error: Offline captioning model not loaded.", "offline_error"
    try:
        rgb_image = cv2.cvtColor(image_frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        inputs = caption_processor_offline(images=pil_image, return_tensors="pt").to(offline_model_device)
        output_ids = caption_model_offline.generate(**inputs, max_new_tokens=50)
        caption = caption_processor_offline.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip(), "offline_blip"
    except Exception as e:
        print(f"Error during offline caption generation: {e}")
        return "Error: Could not generate offline caption.", "offline_error"

def generate_caption_online(image_frame_bgr):
    # TODO: Implement actual cloud API call
    print("Placeholder: generate_caption_online() called. (Cloud API not implemented)")
    # if cloud_call_successful:
    #    return "Actual Cloud Caption Text", "online_cloud"
    return "Placeholder: Online cloud caption pending.", "online_placeholder"

def get_caption(image_frame_bgr, prefer_online=True):
    caption_text = ""
    source_type = "unknown_error"

    if prefer_online and utils.check_internet_connection():
        print("CaptionModule: Attempting online captioning...")
        caption_text, source_type = generate_caption_online(image_frame_bgr.copy())
        
        # If online was just a placeholder or a true API error, try falling back
        if source_type == "online_placeholder" or source_type == "online_api_error": # online_api_error from actual impl.
            print(f"CaptionModule: Online attempt was '{source_type}', falling back to offline.")
            caption_text, source_type = generate_caption_offline(image_frame_bgr.copy())
    else: # Offline preferred or no internet
        if not prefer_online:
            print("CaptionModule: Offline captioning preferred by user.")
        else: # No internet
            print("CaptionModule: No internet, using offline captioning for detailed request.")
        caption_text, source_type = generate_caption_offline(image_frame_bgr.copy())
    
    return caption_text, source_type

if __name__ == '__main__':
    print("\n--- Testing Caption Module ---")
    test_image_path = 'test_image.jpg'
    if os.path.exists(test_image_path):
        img = cv2.imread(test_image_path)
        if img is not None:
            print("\nTest 1: Prefer Online (check internet)")
            cap, src = get_caption(img.copy(), prefer_online=True)
            print(f"Caption: {cap} (Source: {src})")
            
            print("\nTest 2: Force Offline")
            cap, src = get_caption(img.copy(), prefer_online=False)
            print(f"Caption: {cap} (Source: {src})")
        else: print(f"Could not read {test_image_path}")
    else: print(f"{test_image_path} not found for testing.")