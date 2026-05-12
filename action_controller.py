import pyautogui
import time

# Safety feature - moving mouse to corner won't crash the program
pyautogui.FAILSAFE = False

class ActionController:
    def __init__(self):
        self.last_action_time = 0
        self.cooldown = 0.4  # seconds between actions

    def _cooldown_passed(self):
        """Check if enough time has passed since the last action."""
        now = time.time()
        if now - self.last_action_time >= self.cooldown:
            self.last_action_time = now
            return True
        return False

    def move_cursor(self, x, y):
        """Move the mouse cursor to a screen position."""
        pyautogui.moveTo(x, y)

    def left_click(self):
        """Perform a left mouse click."""
        if self._cooldown_passed():
            pyautogui.click()
            print("Action: Left Click")

    def right_click(self):
        """Perform a right mouse click."""
        if self._cooldown_passed():
            pyautogui.rightClick()
            print("Action: Right Click")

    def scroll_up(self, amount=3):
        """Scroll up."""
        if self._cooldown_passed():
            pyautogui.scroll(amount)
            print("Action: Scroll Up")

    def scroll_down(self, amount=3):
        """Scroll down."""
        if self._cooldown_passed():
            pyautogui.scroll(-amount)
            print("Action: Scroll Down")

    def double_click(self):
        """Perform a double click."""
        if self._cooldown_passed():
            pyautogui.doubleClick()
            print("Action: Double Click")

    def take_screenshot(self):
        """Take a screenshot and save it."""
        if self._cooldown_passed():
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            pyautogui.screenshot(f"screenshot_{timestamp}.png")
            print(f"Action: Screenshot saved")