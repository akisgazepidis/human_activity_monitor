import cv2
import os
import time
import datetime
import threading
from ultralytics import YOLO
from google import genai
from config import (
    MODEL_VARIANT, CAMERA_INDEX, OUTPUT_DIR, 
    CONFIDENCE_THRESHOLD, COOLDOWN_SECONDS,
    GEMINI_API_KEY, MODEL_NAME
)


class HumanDetector:
    def __init__(self):
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        self.model = YOLO(MODEL_VARIANT)
        self.capture = cv2.VideoCapture(CAMERA_INDEX)
        self.analyzer = ActionAnalyzer() # Initialize the AI Analyzer
        
        self.video_writer = None
        self.last_detection_time = 0
        self.current_video_path = None

    def detect_human(self):
        print("Starting Human Detector... Press 'q' to quit.")
        try:
            while True:
                grabbed, frame = self.capture.read()
                if not grabbed: break

                results = self.model(frame, verbose=False, classes=[0])
                person_found = len(results[0].boxes) > 0 and results[0].boxes.conf.max() > CONFIDENCE_THRESHOLD

                current_time = time.time()
                
                if person_found:
                    self.last_detection_time = current_time
                    if self.video_writer is None:
                        # Start recording
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        self.current_video_path = os.path.join(OUTPUT_DIR, f"person_{timestamp}.mp4")
                        h, w, _ = frame.shape
                        self.video_writer = cv2.VideoWriter(self.current_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (w, h))
                        print(f"🔴 Recording started: {self.current_video_path}")

                if self.video_writer is not None:
                    self.video_writer.write(frame)
                    
                    # Stop recording if person is gone longer than COOLDOWN_SECONDS
                    if not person_found and (current_time - self.last_detection_time > COOLDOWN_SECONDS):
                        self.video_writer.release()
                        print(f"✅ Video saved. Sending to Gemini 1.5 Flash...")
                        
                        # TRIGGER THE ANALYSIS
                        self.analyzer.analyze_async(self.current_video_path)
                        
                        self.video_writer = None
                        self.current_video_path = None

                cv2.imshow("Human Detection Feed", results[0].plot())
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        finally:
            self.capture.release()
            if self.video_writer: self.video_writer.release()
            cv2.destroyAllWindows()


class ActionAnalyzer:
    def __init__(self):
        # We define the API key and the specific Model ID here
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model_id = MODEL_NAME 

        # # code with available models
        # for model in self.client.models.list().models:
        #     print(f"Model ID: {model.name}")

    def _process_analysis(self, video_path):
        """Uploads video to Gemini and saves the response."""
        try:
            print(f"🧠 [AI] Uploading {video_path} to Gemini...")
            
            # 1. Upload the video file
            video_file = self.client.files.upload(file=video_path)

            # 2. Wait for Google to process the video context
            while video_file.state == "PROCESSING":
                time.sleep(2)
                video_file = self.client.files.get(name=video_file.name)

            if video_file.state == "FAILED":
                print("❌ [AI] Video processing failed.")
                return

            # 3. Request the analysis
            prompt = (
                "Describe the action of the human in this video. "
                "Are they walking, falling, or acting suspiciously? "
                "Provide a concise summary."
            )

            print(f"✨ [AI] Analyzing action...")
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[video_file, prompt]
            )
            
            # 4. PRINT the response to the terminal
            analysis_result = response.text
            print("-" * 30)
            print(f"AI ANALYSIS FOR: {os.path.basename(video_path)}")
            print(analysis_result)
            print("-" * 30)

            # 5. SAVE the response to a .txt file next to the video
            report_path = video_path.replace(".mp4", ".txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(analysis_result)
            print(f"💾 Report saved to: {report_path}")
                
            # 6. Cleanup cloud storage
            self.client.files.delete(name=video_file.name)

        except Exception as e:
            print(f"❌ [AI] Analysis Error: {e}")

    def analyze_async(self, video_path):
        """Runs the analysis in the background so your camera doesn't lag."""
        thread = threading.Thread(target=self._process_analysis, args=(video_path,))
        thread.start()