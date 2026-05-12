import cv2
import numpy as np
from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer
from action_controller import ActionController
from smoother import Smoother
from config_loader import load_config

class GestureMouseControl:
    def __init__(self):
        # Load configuration
        self.config = load_config()
        
        # Initialize components
        self.hand_tracker = HandTracker()
        self.gesture_recognizer = GestureRecognizer(
            click_threshold=self.config['control']['click_threshold']
        )
        self.action_controller = ActionController()
        self.smoother = Smoother(buffer_size=self.config['smoothing']['buffer_size'])
        
        # Set action controller cooldown from config
        self.action_controller.cooldown = self.config['control']['cooldown']
        
        # Screen dimensions
        self.screen_width = self.config['screen']['width']
        self.screen_height = self.config['screen']['height']
        
        # Dead zone (central region where mouse doesn't move)
        self.dead_zone = self.config['control']['dead_zone']
        self.dead_zone_center = (self.screen_width // 2, self.screen_height // 2)
        
        # State tracking
        self.prev_gesture = None
        
    def map_coordinates(self, x, y, frame_width, frame_height):
        """
        Map camera coordinates to screen coordinates.
        Includes dead zone logic to prevent drift.
        """
        # Normalize coordinates to 0-1 range
        norm_x = x / frame_width
        norm_y = y / frame_height
        
        # Apply dead zone (central region where cursor stays still)
        if abs(norm_x - 0.5) < (self.dead_zone / self.screen_width):
            norm_x = 0.5
        if abs(norm_y - 0.5) < (self.dead_zone / self.screen_height):
            norm_y = 0.5
            
        # Map to screen coordinates
        screen_x = int(norm_x * self.screen_width)
        screen_y = int(norm_y * self.screen_height)
        
        return screen_x, screen_y
    
    def run(self):
        """Main application loop."""
        # Open webcam
        cap = cv2.VideoCapture(self.config['camera']['device_id'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])
        
        print("Gesture Mouse Control Started!")
        print("Controls:")
        print("  - Move index finger to move mouse")
        print("  - Pinch thumb and index finger to click")
        print("  - Raise middle finger to scroll up")
        print("  - Press 'q' to quit")
        print("-" * 50)
        
        while True:
            gesture = "NO HAND"
            success, frame = cap.read()
            if not success:
                print("Failed to read from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]
            
            # Detect hands and draw landmarks
            frame = self.hand_tracker.find_hands(frame, draw=True)
            
            # Get hand landmarks
            landmarks = self.hand_tracker.get_landmarks(frame)
            
            if landmarks:
                # Recognize gesture
                gesture, position = self.gesture_recognizer.recognize_gesture(
                    landmarks, self.config
                )
                
                if position:
                    # Map camera coordinates to screen coordinates
                    screen_x, screen_y = self.map_coordinates(
                        position[0], position[1], frame_width, frame_height
                    )
                    
                    # Apply smoothing
                    smooth_x, smooth_y = self.smoother.smooth(screen_x, screen_y)
                    
                    # Execute action based on gesture
                    if gesture == "MOVE":
                        self.action_controller.move_cursor(smooth_x, smooth_y)
                        
                    elif gesture == "CLICK":
                        # Show click on screen overlay
                        cv2.circle(frame, position, 10, (0, 255, 0), -1)
                        self.action_controller.left_click()
                        
                    elif gesture == "SCROLL_UP":
                        self.action_controller.scroll_up(
                            self.config['control']['scroll_amount']
                        )

                    elif gesture == "SCROLL_DOWN":
                        self.action_controller.scroll_down(
                            self.config['control']['scroll_amount']
                        )
                    
            else:
                # No hand detected, reset smoother
                self.smoother.reset()
            
            # Display gesture info on frame
            gesture_text = f"Gesture: {gesture if landmarks else 'NO HAND'}"  # fix below
            cv2.putText(frame, gesture_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw dead zone indicator on frame (optional)
            cv2.circle(frame, (frame_width//2, frame_height//2), 
                      self.dead_zone//2, (255, 0, 0), 1)
            
            # Show the frame
            cv2.imshow("Gesture Mouse Control", frame)
            
            # Quit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed.")

if __name__ == "__main__":
    app = GestureMouseControl()
    app.run()