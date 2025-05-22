# utils.py
import socket
# In utils.py
"""def check_internet_connection(host="8.8.8.8", port=53, timeout=3):
    print("DEBUG: Simulating NO internet connection.") # Add this for clarity
    return False # Force no internet"""

def check_internet_connection(host="8.8.8.8", port=53, timeout=3):
    
  #  Checks for internet connection by trying to connect to a known host.
  #  Host: 8.8.8.8 (Google DNS)
  #  Port: 53/tcp (DNS)
  #  Timeout: 3 seconds
  #  Returns True if connection is successful, False otherwise.
   
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        # print("Internet connection available.") # Optional: for debugging
        return False # Force no internet
    except socket.error as ex:
        # print(f"No internet connection: {ex}") # Optional: for debugging
        return False

if __name__ == '__main__':
    # Test the function
    print("Checking internet connection...")
    if check_internet_connection():
        print("Result: Internet connection is available.")
    else:
        print("Result: No internet connection.")