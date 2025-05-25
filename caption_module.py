# caption_module.py
from PIL import Image as PILImage 
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import cv2
import os
import google.generativeai as genai

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

GEMINI_API_CONFIGURED_CAPTION = False # Specific flag for this module
try:
    api_key_caption = os.environ.get("GOOGLE_API_KEY")
    if not api_key_caption:
        print("CaptionModule (Online Setup): GOOGLE_API_KEY env var not set. Online Gemini captioning unavailable.")
    else:
        genai.configure(api_key=api_key_caption)
        GEMINI_API_CONFIGURED_CAPTION = True
        print("CaptionModule (Online Setup): Gemini API configured successfully for captions.")
except Exception as e:
    print(f"CaptionModule (Online Setup): ERROR configuring Gemini API - {type(e).__name__}: {e}. Online Gemini captioning unavailable.")

def generate_caption_offline(image_frame_bgr):
    if caption_model_offline is None or caption_processor_offline is None:
        return "Error: Offline captioning model not loaded.", "offline_error"
    try:
        rgb_image = cv2.cvtColor(image_frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_image)
        inputs = caption_processor_offline(images=pil_image, return_tensors="pt").to(offline_model_device)
        output_ids = caption_model_offline.generate(**inputs, max_new_tokens=50)
        caption = caption_processor_offline.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip(), "offline_blip"
    except Exception as e:
        print(f"Error during offline caption generation: {type(e).__name__} - {e}")
        return "Error: Could not generate offline caption.", "offline_error"

def generate_caption_online(image_frame_bgr):
    if not GEMINI_API_CONFIGURED_CAPTION:
        return "Error: Online Captioning (Gemini) API not configured.", "online_api_error_config"
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        rgb_frame = cv2.cvtColor(image_frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image_for_gemini = PILImage.fromarray(rgb_frame)
        prompt = "Describe this scene concisely for a visually impaired person. Focus on key objects, people, their immediate actions, and any potential obstacles or important environmental features. Avoid subjective interpretations unless very obvious."
        print(f"CaptionModule (Online): Sending image to Gemini with prompt: '{prompt[:70]}...'")
        response = model.generate_content([pil_image_for_gemini, prompt])
        if hasattr(response, 'text') and response.text:
            print("DEBUG (Caption Online): Gemini call succeeded, response has text.")
            return response.text.strip(), "online_gemini_caption"
        else:
            print(f"DEBUG (Caption Online): Gemini call may have succeeded, but response.text is missing or empty.")
            print(f"DEBUG (Caption Online): Full response object: {response}")
            print(f"DEBUG (Caption Online): response.parts: {response.parts if hasattr(response, 'parts') else 'N/A'}")
            print(f"DEBUG (Caption Online): response.prompt_feedback: {response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'N/A'}")
            block_reason_msg = "Unknown reason (no text in response)"
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason_msg = response.prompt_feedback.block_reason_message
            error_msg = f"No text in Gemini caption response. Possible block/reason: {block_reason_msg}"
            print(f"CaptionModule (Online): Error: {error_msg}")
            return f"Error: {error_msg}", "online_api_error_response"
    except Exception as e:
        detailed_error_msg = f"Exception during Gemini online captioning: {type(e).__name__} - {str(e)}"
        print(f"CaptionModule (Online): {detailed_error_msg}")
        return f"Error: {detailed_error_msg}", "online_api_error_exception"

def get_caption(image_frame_bgr, prefer_online=True):
    caption_text, source_type = "", "unknown_error"
    if prefer_online and utils.check_internet_connection():
        if GEMINI_API_CONFIGURED_CAPTION:
            print("CaptionModule: Attempting online captioning (Gemini)...")
            caption_text, source_type = generate_caption_online(image_frame_bgr.copy())
            # For Option B: If online Gemini call was not a clean success, we DON'T auto-fallback here.
            # Silver_detection.py will handle what to tell the user.
            if source_type not in ["online_gemini_caption"]:
                print(f"CaptionModule: Online Gemini captioning attempt was '{source_type}'. Not automatically falling back to offline BLIP in get_caption.")
                # Return the error/status from the online attempt.
        else: # Gemini not configured, but online was preferred and internet is on
            print("CaptionModule: Online Gemini API not configured for captioning. Cannot attempt online.")
            caption_text = "Error: Online captioning service not configured."
            source_type = "online_api_error_config"
    else: # Offline preferred or no internet
        if not prefer_online: print("CaptionModule: Offline captioning preferred by user (BLIP).")
        else: print("CaptionModule: No internet, using offline captioning for detailed request (BLIP).")
        caption_text, source_type = generate_caption_offline(image_frame_bgr.copy())
    return caption_text, source_type

if __name__ == '__main__':
    print("\n--- Testing Caption Module ---")
    # --- Setup for standalone testing ---
    print("CaptionModule Test: Ensuring GOOGLE_API_KEY is set in your environment for this test.")
    print("CaptionModule Test: For example, in PowerShell: $env:GOOGLE_API_KEY=\"YOUR_KEY_HERE\"")
    print("CaptionModule Test: And ensure utils.py is NOT simulating 'no internet'.")
    # --- End Setup for standalone testing ---

    test_image_path = 'test_image.jpg' # You need a test_image.jpg in the same directory
    if os.path.exists(test_image_path):
        img = cv2.imread(test_image_path)
        if img is not None:
            print(f"\nCaptionModule Test (Using {test_image_path}): Prefer Online (Actual Internet)")
            # Ensure genai is configured if GOOGLE_API_KEY is set for standalone test
            if GEMINI_API_CONFIGURED or (os.environ.get("GOOGLE_API_KEY") and not GEMINI_API_CONFIGURED) :
                try:
                    if not GEMINI_API_CONFIGURED: # If module level config failed but key is now set
                         genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
                         print("CaptionModule Test: Re-attempted Gemini API configuration for standalone test.")
                except Exception as e_cfg:
                     print(f"CaptionModule Test: Error configuring Gemini for standalone test: {e_cfg}")
            
            cap, src = get_caption(img.copy(), prefer_online=True)
            print(f"Caption: {cap} (Source: {src})")
            
            print("\nCaptionModule Test: Force Offline")
            cap_off, src_off = get_caption(img.copy(), prefer_online=False)
            print(f"Caption: {cap_off} (Source: {src_off})")
        else: print(f"Could not read {test_image_path}")
    else: print(f"{test_image_path} not found for testing.")