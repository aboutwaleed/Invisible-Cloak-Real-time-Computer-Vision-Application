import cv2
import numpy as np
import time
import mediapipe as mp

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentor = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

def create_background(cap, num_frames=30):
    print("Capturing background. Please move out of the frame...")
    backgrounds = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            backgrounds.append(frame)
        else:
            print(f"Warning: Couldn't capture frame {i+1}/{num_frames}")
        time.sleep(0.1)
    if backgrounds:
        return np.median(backgrounds, axis=0).astype(np.uint8)
    else:
        raise ValueError("Failed to capture background.")

def create_red_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Clean the mask
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, np.ones((3,3), np.uint8), iterations=1)
    return red_mask

def apply_cloak_effect(frame, cloak_mask, background):
    inv_mask = cv2.bitwise_not(cloak_mask)
    fg = cv2.bitwise_and(frame, frame, mask=inv_mask)
    bg = cv2.bitwise_and(background, background, mask=cloak_mask)
    return cv2.add(fg, bg)

def main():
    print("Starting Red Invisible Cloak with AI (MediaPipe)...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not available.")
        return

    try:
        background = create_background(cap)
    except ValueError as e:
        print(f"Error: {e}")
        cap.release()
        return

    print("Background captured. Now starting main loop. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Frame not captured.")
            continue

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]

        # Get red color mask
        red_mask = create_red_mask(frame)

        # Get person mask using MediaPipe
        results = segmentor.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        person_mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255

        # Combine red + person mask
        cloak_mask = cv2.bitwise_and(red_mask, person_mask)

        # Apply cloak effect
        final_output = apply_cloak_effect(frame, cloak_mask, background)

        cv2.imshow('AI Red Invisible Cloak', final_output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
