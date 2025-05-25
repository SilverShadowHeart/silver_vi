# ocr_module.py
import cv2
import pytesseract
import os
import google.generativeai as genai # For Gemini
import PIL.Image # For Gemini

try:
    import utils
except ImportError:
    print("CRITICAL Error: utils.py not found in ocr_module.")
    class MockUtils:
        def check_internet_connection(self): return False
    utils = MockUtils()

# --- TESSERACT CONFIGURATION ---
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # print(f"Pytesseract using Tesseract Engine: {pytesseract.get_tesseract_version()}") # Optional
except Exception as e:
    print(f"Note: Could not set tesseract_cmd or verify Tesseract. Error: {e}. Offline OCR may fail.")

# --- ONLINE GEMINI API CONFIGURATION (Attempt once when module loads) ---
OCR_GEMINI_API_CONFIGURED = False 
try:
    ocr_api_key = os.environ.get("GOOGLE_API_KEY")
    if not ocr_api_key:
        print("OCRModule (Online Setup): GOOGLE_API_KEY environment variable not set. Online Gemini OCR will be unavailable.")
    else:
        # genai.configure might have been called by caption_module if imported first.
        # Calling it again is safe and ensures this module configures it if it's the first one.
        genai.configure(api_key=ocr_api_key)
        OCR_GEMINI_API_CONFIGURED = True
        print("OCRModule (Online Setup): Gemini API configured successfully for OCR.")
except Exception as e:
    print(f"OCRModule (Online Setup): CRITICAL ERROR configuring Gemini API - {type(e).__name__}: {e}. Online Gemini OCR will be unavailable.")
    OCR_GEMINI_API_CONFIGURED = False


def extract_text_from_image_offline(image_frame_bgr, psm_mode=6):
    """
    Extracts text from an image using local Tesseract OCR.
    :param image_frame_bgr: A BGR image frame.
    :param psm_mode: Tesseract Page Segmentation Mode.
    :return: Tuple (extracted_text_string, source_type_string)
    """
    try:
        gray_image = cv2.cvtColor(image_frame_bgr, cv2.COLOR_BGR2GRAY)
        _, processed_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        custom_config = f'--oem 1 --psm {psm_mode}'
        text = pytesseract.image_to_string(processed_image, lang='eng', config=custom_config)
        if not text.strip(): # If Tesseract returns empty or only whitespace
            return "No clear text found by offline OCR.", "offline_no_text"
        return text.strip(), "offline_tesseract"
    except pytesseract.TesseractNotFoundError:
        print("CRITICAL ERROR: Tesseract engine not found by pytesseract. Offline OCR will fail.")
        return "Error: Tesseract OCR engine not found.", "offline_error_tesseract_missing"
    except Exception as e:
        print(f"Error during offline OCR: {type(e).__name__} - {e}")
        return "Error: Could not extract text offline.", "offline_error_general"

def extract_text_from_image_online(image_frame_bgr):
    """
    Extracts text from an image using Google Gemini API.
    :param image_frame_bgr: A BGR image frame (typically a crop).
    :return: Tuple (extracted_text_string, source_type_string)
    """
    global OCR_GEMINI_API_CONFIGURED # Refer to the module-level flag

    if not OCR_GEMINI_API_CONFIGURED:
        return "Error: Online OCR (Gemini) API not configured or failed to initialize.", "online_api_error_config"
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest') # Using flash for efficiency
        rgb_frame = cv2.cvtColor(image_frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image_for_gemini = PIL.Image.fromarray(rgb_frame)

        # Specific prompt for OCR
        prompt = "Extract all text clearly visible in this image. Respond only with the extracted text. If no text is clearly visible, respond with the exact phrase 'No clear text found'."
        print(f"OCRModule (Online): Sending image to Gemini for OCR with prompt: '{prompt}'")
        
        response = model.generate_content([pil_image_for_gemini, prompt])

        if hasattr(response, 'text') and response.text:
            print("DEBUG (OCR Online): Gemini call succeeded, response has text.")
            extracted_text = response.text.strip()
            
            # Check if Gemini explicitly said no text was found based on our prompt instruction
            if extracted_text == "No clear text found":
                return "No clear text found by online OCR.", "online_no_text_ocr"
            return extracted_text, "online_gemini_ocr" # Successful OCR
        else:
            # This block is reached if response.text is empty or missing
            print(f"DEBUG (OCR Online): Gemini call may have succeeded, but response.text is missing or empty.")
            print(f"DEBUG (OCR Online): Full response object: {response}")
            print(f"DEBUG (OCR Online): response.parts: {response.parts if hasattr(response, 'parts') else 'N/A'}")
            print(f"DEBUG (OCR Online): response.prompt_feedback: {response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'N/A'}")
            
            block_reason_msg = "Unknown reason (no text in response or response structure unexpected)"
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason_msg = response.prompt_feedback.block_reason_message
            
            error_msg = f"No text in Gemini OCR response. Possible block/reason: {block_reason_msg}"
            print(f"OCRModule (Online): Error: {error_msg}")
            # If blocked, it's still an API "error" from our perspective of getting usable text
            return f"Error: {error_msg}", "online_api_error_response"

    except Exception as e:
        detailed_error_msg = f"Exception during Gemini online OCR: {type(e).__name__} - {str(e)}"
        print(f"OCRModule (Online): {detailed_error_msg}")
        # from google.api_core import exceptions as google_exceptions
        # if isinstance(e, google_exceptions.ResourceExhausted):
        #     return f"Error: Gemini API quota reached for OCR.", "online_api_quota_exceeded"
        return f"Error: {detailed_error_msg}", "online_api_error_exception"

def get_ocr_text(image_frame_bgr, prefer_online=True, offline_psm_mode=6): # Default PSM 6 for offline
    text_result = ""
    source_type = "unknown_error" # Default

    if prefer_online and utils.check_internet_connection():
        if OCR_GEMINI_API_CONFIGURED:
            print("OCRModule: Attempting online OCR (Gemini)...")
            text_result, source_type = extract_text_from_image_online(image_frame_bgr.copy())
            
            # If the online attempt wasn't a clean success (e.g., an error, or even a "no text found" status from Gemini)
            # we do NOT automatically fall back to offline Tesseract here.
            # The Silver_detection.py will handle the message to the user based on source_type.
            if source_type not in ["online_gemini_ocr", "online_no_text_ocr"]:
                print(f"OCRModule: Online Gemini OCR attempt resulted in status '{source_type}'.")
                # text_result already contains the error message or status like "API not configured"
        else:
            print("OCRModule: Online Gemini API not configured for OCR. Cannot attempt online.")
            text_result = "Error: Online OCR service not configured."
            source_type = "online_api_error_config"
    else: 
        # This path is taken if user prefers offline OR no internet.
        # We DO NOT automatically run offline Tesseract. User must press 'r'.
        default_message = " Press 'R' in main app to attempt offline OCR (experimental)."
        if not prefer_online:
            print("OCRModule: Offline OCR preferred by user." + default_message)
            text_result = "Offline OCR mode selected by user." + default_message
            source_type = "offline_user_preference_no_attempt"
        else: # No internet
            print("OCRModule: No internet. Online OCR unavailable." + default_message)
            text_result = "Online OCR unavailable (no internet)." + default_message
            source_type = "offline_no_internet_no_attempt"
        
    return text_result, source_type

if __name__ == '__main__':
    print("\n--- Testing OCR Module ---")
    print("OCRModule Test: Ensure GOOGLE_API_KEY is set in your environment for this test.")
    print("OCRModule Test: And ensure utils.py is NOT simulating 'no internet'.")
    
    test_ocr_path = 'ocr_test_image.jpg' # CREATE AN IMAGE WITH CLEAR TEXT FOR TESTING
    if os.path.exists(test_ocr_path):
        img = cv2.imread(test_ocr_path)
        if img is not None:
            print(f"\nOCRModule Test (Using {test_ocr_path}):")
            
            print("\n--- Attempting Online OCR (Gemini) via get_ocr_text ---")
            # This simulates how Silver_detection.py would call it when online is preferred
            text_online, src_online = get_ocr_text(img.copy(), prefer_online=True)
            print(f"Online Attempt Text: {text_online} (Source: {src_online})")

            print("\n--- Attempting Offline OCR (Tesseract) directly for comparison ---")
            # This direct call is for testing Tesseract itself on a good image.
            text_offline, src_offline = extract_text_from_image_offline(img.copy(), psm_mode=6) 
            print(f"Offline Tesseract Text (PSM 6): {text_offline} (Source: {src_offline})")
        else: 
            print(f"Could not read '{test_ocr_path}'.")
    else: 
        print(f"'{test_ocr_path}' not found. Create an image with some clear text and save it as '{test_ocr_path}' to test this module.")