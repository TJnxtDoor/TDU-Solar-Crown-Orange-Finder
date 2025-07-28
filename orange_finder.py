import pyautogui
import cv2
import numpy as np
import time
import os 

# Configuration
OBS_WINDOW_TITLE = "OBS"  
SCAN_INTERVAL = 1
DEBUG_MODE = True  

# Orange detection
ORANGE_HSV_LOWER = np.array([5, 100, 100])
ORANGE_HSV_UPPER = np.array([20, 255, 255])
MIN_ORANGE_SIZE = 50
MAX_ORANGE_SIZE = 5000

# PAYLINE detection
TEMPLATE_FILE = 'payline_template.png'
PAYLINE_TEMPLATE = None
if os.path.exists(TEMPLATE_FILE):
    try:
        PAYLINE_TEMPLATE = cv2.imread(TEMPLATE_FILE, 0)
        if PAYLINE_TEMPLATE is None:
            print(f"Warning: Could not read template file {TEMPLATE_FILE}")
    except Exception as e:
        print(f"Error loading template: {e}")

def get_obs_window():
    try:
        windows = pyautogui.getWindowsWithTitle(OBS_WINDOW_TITLE)
        if windows:
            window = windows[0]
            window.activate()
            time.sleep(0.5)
            print(f"Found OBS window at: {window.left},{window.top} {window.width}x{window.height}")
            return (window.left, window.top, window.width, window.height)
        print(f"No window found with title containing: {OBS_WINDOW_TITLE}")
    except Exception as e:
        print(f"Window error: {e}")
    return None

def analyze_screen(region):
    try:
        # Capture screen
        screenshot = pyautogui.screenshot(region=region)
        img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        debug_img = img.copy()
        
        # Detect orange elements
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        orange_mask = cv2.inRange(hsv, ORANGE_HSV_LOWER, ORANGE_HSV_UPPER)
        
        # Use better contour detection method
        contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        orange_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if MIN_ORANGE_SIZE < area < MAX_ORANGE_SIZE:
                orange_count += 1
                if DEBUG_MODE:
                    x,y,w,h = cv2.boundingRect(cnt)
                    cv2.rectangle(debug_img, (x,y), (x+w,y+h), (0,255,0), 2)
        
        # Detect PAYLINE
        payline_count = 0
        if PAYLINE_TEMPLATE is not None:
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                res = cv2.matchTemplate(gray, PAYLINE_TEMPLATE, cv2.TM_CCOEFF_NORMED)
                threshold = 0.8
                loc = np.where(res >= threshold)
                payline_count = len(loc[0])
                if DEBUG_MODE:
                    for pt in zip(*loc[::-1]):
                        cv2.rectangle(debug_img, pt, 
                                    (pt[0]+PAYLINE_TEMPLATE.shape[1], 
                                     pt[1]+PAYLINE_TEMPLATE.shape[0]), 
                                    (255,0,0), 2)
            except Exception as e:
                print(f"Template matching error: {e}")
        
        if DEBUG_MODE:
            cv2.imshow('Debug View', debug_img)
            cv2.waitKey(1)
        
        return orange_count, payline_count
        
    except Exception as e:
        print(f"Detection error: {e}")
        return 0, 0

def main():
    print("OBS Detector - Press Ctrl+C to stop")
    
    if PAYLINE_TEMPLATE is None and os.path.exists(TEMPLATE_FILE):
        print(f"Warning: Could not load template file {TEMPLATE_FILE}")
    elif not os.path.exists(TEMPLATE_FILE):
        print(f"Warning: {TEMPLATE_FILE} not found. Only orange detection will work.")
    
    obs_region = get_obs_window()
    if not obs_region:
        print("OBS window not found!")
        return
    
    try:
        while True:
            orange, payline = analyze_screen(obs_region)
            print(f"\r🍊 Orange: {orange} | PAYLINE: {payline}", end="", flush=True)
            time.sleep(SCAN_INTERVAL)
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        if DEBUG_MODE and cv2.getWindowProperty('Debug View', cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()