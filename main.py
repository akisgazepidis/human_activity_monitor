import sys
from utils import HumanDetector

def main():
    print("--- Initializing Human Detection System ---")
    try:
        # Initialize the detector
        # This will also initialize the ActionAnalyzer internally
        detector = HumanDetector()
        
        # Start the main loop
        detector.detect_human()
        
    except KeyboardInterrupt:
        print("\nStopping system... Goodbye!")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()