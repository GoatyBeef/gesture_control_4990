import json

def load_config(path="config.json"):
    """Load configuration from a JSON file."""
    with open(path, "r") as f:
        config = json.load(f)
    print("Config loaded successfully!")
    return config

if __name__ == "__main__":
    config = load_config()
    print(f"Screen resolution: {config['screen']['width']}x{config['screen']['height']}")
    print(f"Click threshold: {config['control']['click_threshold']}px")
    print(f"Smoothing buffer: {config['smoothing']['buffer_size']} frames")
    print(f"Index fingertip landmark: #{config['landmarks']['index_tip']}")