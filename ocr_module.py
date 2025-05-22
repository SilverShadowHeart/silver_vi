# ocr_module.py
import cv2
import pytesseract
import os

try:
    import utils
except ImportError:
    print("CRITICAL Error: utils.py not found in ocr_module.")
    class MockUtils:
        def check_internet_connection(self): return False
    utils = MockUtils()

try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # print(f"Pytesseract using Tesseract Engine: {pytesseract.get_tesseract_version()}")
except Exception as e:
    print(f"Note: Could not set tesseract_cmd or verify Tesseract. Error: {e}. Offline OCR may fail.")

def extract_text_from_image_offline(image_frame_bgr, psm_mode=3):
    try:
        gray_image = cv2.cvtColor(image_frame_bgr, cv2.COLOR_BGR2GRAY)
        _, processed_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        custom_config = f'--oem 1 --psm {psm_mode}'
        text = pytesseract.image_to_string(processed_image, lang='eng', config=custom_config)
        if not text.strip(): # If Tesseract returns empty or only whitespace
            return "No clear text found by offline OCR.", "offline_no_text"
        return text.strip(), "offline_tesseract"
    except pytesseract.TesseractNotFoundError:
        print("CRITICAL ERROR: Tesseract engine not found by pytesseract.")
        return "Error: Tesseract OCR engine not found.", "offline_error_tesseract_missing"
    except Exception as e:
        print(f"Error during offline OCR: {e}")
        return "Error: Could not extract text offline.", "offline_error_general"

def extract_text_from_image_online(image_frame_bgr):
    # TODO: Implement actual cloud API call for OCR
    print("Placeholder: extract_text_from_image_online() called. (Cloud API not implemented)")
    # if cloud_call_successful:
    #    return "Actual Cloud OCR Text", "online_cloud_ocr"
    return "Placeholder: Online cloud OCR pending.", "online_placeholder_ocr"

def get_ocr_text(image_frame_bgr, prefer_online=True, offline_psm_mode=3):
    text_result = ""
    source_type = "unknown_error"

    if prefer_online and utils.check_internet_connection():
        print("OCRModule: Attempting online OCR...")
        text_result, source_type = extract_text_from_image_online(image_frame_bgr.copy())
        
        # If online was just a placeholder or an API error, DO NOT fall back to offline Tesseract here
        # because we've established offline Tesseract is too unreliable for general in-the-wild text.
        # The main loop will decide if an offline attempt is ever made (likely not by default).
        if source_type == "online_placeholder_ocr" or source_type == "online_api_error_ocr":
            print(f"OCRModule: Online attempt was '{source_type}'. No automatic fallback to offline Tesseract here.")
            # Return the placeholder or error from online attempt. The calling function can decide next steps.
    else: # Offline preferred by user or no internet
        if not prefer_online:
            print("OCRModule: Offline OCR preferred by user. (Note: Generally unreliable for non-document text)")
        else: # No internet
            print("OCRModule: No internet. Offline OCR would be attempted if called directly.")
        # We will NOT call extract_text_from_image_offline by default here.
        # It should only be called if Silver_detection.py explicitly decides to for an experimental attempt.
        text_result = "Offline OCR not attempted by default due to reliability concerns."
        source_type = "offline_not_attempted"
        
    return text_result, source_type

if __name__ == '__main__':
    print("\n--- Testing OCR Module ---")
    test_ocr_path = 'ocr_test_image.jpg'
    if os.path.exists(test_ocr_path):
        img = cv2.imread(test_ocr_path)
        if img is not None:
            print("\nTest 1: Offline Tesseract (using psm 1 for general layout)")
            # This direct call is for testing Tesseract itself on a good image.
            text, src = extract_text_from_image_offline(img.copy(), psm_mode=1) 
            print(f"Text: {text} (Source: {src})")

            print("\nTest 2: Get OCR Text - Prefer Online (simulating internet)")
            class TempUtils: # Mock utils for this test
                def check_internet_connection(self): print("MockUtils: Simulating Internet ON for OCR test"); return True
            original_utils = utils
            utils = TempUtils()
            text_on, src_on = get_ocr_text(img.copy(), prefer_online=True)
            print(f"Text: {text_on} (Source: {src_on})")
            utils = original_utils # Restore original utils

        else: print(f"Could not read {test_ocr_path}")
    else: print(f"{test_ocr_path} not found for testing.")