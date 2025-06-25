import cv2
import numpy as np
import os
import time
from ultralytics import YOLO

# ==============================================================================
# NASA 10 Coding Principles (focusing on 5 core principles for robustness)
# These principles guide the design to ensure predictable performance and reliability,
# particularly important for real-time systems.
#
# 1. No dynamic memory allocation after initialization:
#    - Principle: All major data structures and buffers should be
#      pre-allocated or sized at system startup. This avoids runtime memory
#      management overhead and ensures predictable performance.
#    - Application Context: While Python's memory management is automatic,
#      our design aims to reuse objects and buffers where feasible, reducing
#      unnecessary allocations within the main processing loop.
#
# 2. No recursion:
#    - Principle: All algorithms and control flows must be implemented iteratively.
#    - Application Context: This ensures predictable stack usage and avoids
#      potential stack overflow issues, crucial for long-running processes.
#
# 3. All loops must have a fixed upper bound:
#    - Principle: Loops must iterate a fixed number of times or have clear,
#      measurable termination conditions to guarantee real-time predictability.
#    - Application Context: Loops processing detected objects or color channels
#      are explicitly bounded, even when processing variable numbers of detections.
#
# 4. No floating-point arithmetic if not explicitly justified:
#    - Principle: Floating-point operations should only be used when absolutely
#      necessary, due to their potential performance implications and precision
#      considerations.
#    - Application Context: Machine learning inference inherently involves
#      floating-point arithmetic. Precision is carefully managed, and performance
#      metrics (like FPS) naturally use floats for accuracy.
#
# 5. All variables must be explicitly declared and initialized:
#    - Principle: Every variable must be initialized with a meaningful default
#      value or through function arguments.
#    - Application Context: Python's dynamic typing is used, but variables are
#      always given initial values to ensure clear state and prevent undefined
#      behavior. Type hints are used to improve readability and maintainability.
# ==============================================================================

class VideoProcessor:
    """
    Manages the core operations for video input and object instance segmentation.
    This class handles loading the pre-trained YOLOv8-seg model and processing
    individual video frames to identify and segment objects in real-time.
    """
    def __init__(self, model_path: str, input_video_path: str):
        """
        Initializes the VideoProcessor with the paths to the segmentation model
        and the input video file.

        Args:
            model_path (str): The file path to the trained YOLOv8-seg model (e.g., 'yolov8n-seg.pt').
            input_video_path (str): The file path for the video to be processed.
        """
        self.model_path = model_path
        self.input_video_path = input_video_path
        self.model: YOLO = None # Initialize model as None; it will be loaded later
        self.cap: cv2.VideoCapture = None # Initialize video capture object as None
        print(f"VideoProcessor initialized for model: {model_path} and video: {input_video_path}")

    def load_model_and_video(self) -> bool:
        """
        Loads the YOLOv8-seg model into memory and attempts to open the specified
        input video file using OpenCV. This method ensures all necessary resources
        are ready before processing begins.

        Returns:
            bool: True if both the model is loaded and the video file is opened
                  successfully; False otherwise.
        """
        print("Attempting to load YOLOv8-seg model and open video file...")
        try:
            # Verify the input video file exists before proceeding.
            if not os.path.exists(self.input_video_path):
                print(f"Error: Input video file '{self.input_video_path}' not found. Please ensure it is in the correct directory.")
                return False

            # Load the YOLOv8 segmentation model. This can take a moment depending on model size.
            self.model = YOLO(self.model_path)
            # Open the video capture stream from the specified file.
            self.cap = cv2.VideoCapture(self.input_video_path)

            # Check if the video file was opened successfully.
            if not self.cap.isOpened():
                print(f"Error: Could not open the video file '{self.input_video_path}'. "
                      "Verify the file path and video codec compatibility.")
                return False

            print("YOLOv8-seg model loaded and video file successfully opened.")
            return True
        except Exception as e:
            # Catch any unexpected errors during loading to ensure robust startup.
            print(f"Fatal Error encountered during model or video loading: {e}")
            self.release_resources() # Ensure any partially opened resources are closed.
            return False

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies object instance segmentation to a single video frame using the
        loaded YOLOv8-seg model. It overlays detected objects with unique
        colored masks, bounding boxes, and confidence scores.

        Args:
            frame (np.ndarray): The raw input video frame (a NumPy array representing an image).

        Returns:
            np.ndarray: The modified frame with segmentation masks, bounding boxes,
                        and class labels visually rendered. Returns the original frame
                        if the model is not loaded or no detections are found.
        """
        # Ensure the model is loaded before attempting inference.
        if self.model is None:
            print("Error: Segmentation model is not loaded. Skipping frame processing.")
            return frame

        # Get the dimensions of the current frame for mask resizing.
        height, width = frame.shape[:2]
        # Perform inference using the YOLOv8 model. Verbose output is disabled for cleaner console.
        results = self.model(frame, verbose=False)

        # Create a copy of the frame to draw overlays on, preserving the original.
        output_frame = frame.copy()

        # Check if the model returned any detection results with masks.
        if results and results[0].masks is not None and len(results[0].masks.data) > 0:
            # Iterate through each detected instance to apply its mask and bounding box.
            # We explicitly use range(len(...)) to adhere to NASA Principle 3 (fixed upper bound loop).
            for i in range(len(results[0].masks.data)):
                mask_tensor = results[0].masks.data[i] # The raw mask tensor for the current instance.
                bounding_box = results[0].boxes[i] # The bounding box data for the current instance.

                # Convert the mask tensor to a NumPy array and move it to CPU if on GPU.
                mask_np = mask_tensor.cpu().numpy()
                # Resize the mask to match the original frame's dimensions for accurate overlay.
                mask_resized = cv2.resize(mask_np, (width, height), interpolation=cv2.INTER_NEAREST)
                # Convert the mask to a boolean array where True indicates the object's pixels.
                mask_boolean = mask_resized.astype(bool)

                # Generate a random, distinct color for each detected object instance.
                # This ensures individual objects are visually distinguishable (NASA Principle 5: explicit init).
                color = np.random.randint(0, 255, size=3, dtype=np.uint8)
                alpha_blend = 0.5 # Transparency level for the mask overlay.

                # Apply the colored mask to the output frame.
                # Iterate over each color channel (BGR) to blend the mask.
                # This loop has a fixed upper bound (3 for B, G, R), adhering to NASA Principle 3.
                for c in range(3):
                    # Blend the original pixel with the new color based on the mask and alpha.
                    output_frame[:, :, c] = np.where(
                        mask_boolean, # Condition: If the pixel is part of the mask
                        output_frame[:, :, c] * (1 - alpha_blend) + color[c] * alpha_blend, # Blended color
                        output_frame[:, :, c] # Original color if not part of the mask
                    )

                # Extract bounding box coordinates and convert them to integers.
                x1, y1, x2, y2 = map(int, bounding_box.xyxy[0])
                # Retrieve the class label name for the detected object.
                label = self.model.names[int(bounding_box.cls)]
                # Get the confidence score for the detection (NASA Principle 5: explicit init).
                confidence_score = bounding_box.conf[0]

                # Draw the bounding box around the detected object.
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Add the label and confidence score text near the bounding box.
                cv2.putText(output_frame, f"{label} {confidence_score:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return output_frame

    def release_resources(self):
        """
        Releases the video capture object. This is a crucial step to free up
        system resources and prevent errors when the application closes.
        """
        print("Releasing video capture resources...")
        if self.cap: # Check if the capture object was successfully initialized.
            self.cap.release()
        print("Video resources have been released.")


class SegmentationVisualizer:
    """
    Manages the display window for showing the real-time segmented video frames
    and relevant performance metrics.
    """
    def __init__(self, display_window_name: str):
        """
        Initializes the visualizer by creating an OpenCV display window.

        Args:
            display_window_name (str): The name to be assigned to the display window.
        """
        self.display_window_name = display_window_name
        # Create a resizable window to display the video feed.
        cv2.namedWindow(self.display_window_name, cv2.WINDOW_AUTOSIZE) # NASA Principle 5: Explicit initialization
        print(f"SegmentationVisualizer prepared for display window: '{display_window_name}'")

    def update_display(self, frame: np.ndarray, fps: float, elapsed_ms: float) -> bool:
        """
        Renders the processed video frame to the display window and overlays
        performance statistics like Frames Per Second (FPS) and processing time.

        Args:
            frame (np.ndarray): The video frame (with segmentation overlays) to display.
            fps (float): The current calculated frames per second for performance monitoring.
            elapsed_ms (float): The time taken to process the current frame, in milliseconds.

        Returns:
            bool: Returns False if the user presses the 'q' key (quitting the application),
                  otherwise returns True to continue processing.
        """
        # Overlay the calculated FPS onto the top-left corner of the frame.
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # Overlay the processing time per frame.
        cv2.putText(frame, f"Time: {elapsed_ms:.2f}ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the prepared frame in the named window.
        cv2.imshow(self.display_window_name, frame)
        # Wait for a key press for 1 millisecond. If 'q' is pressed, signal termination.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False # Signal that the application should terminate.
        return True # Signal that the application should continue.


class ApplicationCoordinator:
    """
    The central orchestrator for the Real-Time Object Instance Segmentation application.
    This class brings together the `VideoProcessor` and `SegmentationVisualizer`
    components to manage the main application loop, ensuring a smooth flow from
    video input to segmented output display. It adheres to the Dependency Inversion
    Principle (DIP) by depending on abstractions rather than concrete implementations.
    """
    def __init__(self, video_processor: VideoProcessor, segmentation_visualizer: SegmentationVisualizer):
        """
        Constructs the ApplicationCoordinator by injecting instances of its dependent
        modules. This allows for flexible and testable component management.

        Args:
            video_processor (VideoProcessor): An instance responsible for handling
                                              video input and segmentation inference.
            segmentation_visualizer (SegmentationVisualizer): An instance responsible
                                                              for displaying the results.
        """
        self.video_processor = video_processor
        self.segmentation_visualizer = segmentation_visualizer
        # Define a maximum number of processing iterations as an upper bound (NASA Principle 3).
        # This is set to a very large number for practical "infinite" video processing.
        self.MAX_PROCESSING_ITERATIONS = 5000000 
        print("ApplicationCoordinator initialized and ready to manage the workflow.")

    def execute_application_loop(self):
        """
        Executes the main real-time processing loop of the application.
        This loop continuously reads video frames, processes them for object
        segmentation, and updates the display. It also handles graceful
        shutdown mechanisms.
        """
        print("\n--- Starting Real-Time Object Instance Segmentation Application ---")

        # Initialize core components. If any initialization fails, the application
        # cannot proceed and will abort gracefully.
        if not self.video_processor.load_model_and_video():
            print("Critical failure: The segmentation model or video could not be loaded. Aborting application.")
            return

        processing_iteration_count = 0 # Initialize iteration counter (NASA Principle 5).
        try:
            # The main application loop. It continues as long as frames are available
            # and the maximum iteration limit (a safety bound) has not been reached.
            while processing_iteration_count < self.MAX_PROCESSING_ITERATIONS:
                # Record the start time of the current iteration for performance calculation.
                iteration_start_time_seconds = time.time()

                # Read a frame from the video stream.
                ret, frame = self.video_processor.cap.read()
                # If 'ret' is False, it means the end of the video stream has been reached
                # or an error occurred while reading the frame.
                if not ret:
                    print("End of video stream or failure to read frame. Exiting processing loop.")
                    break

                # 1. Process the current frame for object instance segmentation.
                # This method handles model inference and overlaying results.
                processed_frame = self.video_processor.process_frame(frame)

                # 2. Calculate performance metrics and update the display.
                iteration_end_time_seconds = time.time()
                # Calculate elapsed time in milliseconds (NASA Principle 4: justified float for timing).
                current_elapsed_time_ms = (iteration_end_time_seconds - iteration_start_time_seconds) * 1000
                # Calculate Frames Per Second (FPS). Avoid division by zero.
                current_fps = 1.0 / (iteration_end_time_seconds - iteration_start_time_seconds) \
                                 if (iteration_end_time_seconds - iteration_start_time_seconds) > 0 else 0.0 # NASA Principle 4: justified float

                # Update the display window with the processed frame and performance metrics.
                # This also checks for user input (e.g., 'q' key press) to terminate the app.
                if not self.segmentation_visualizer.update_display(processed_frame, current_fps, current_elapsed_time_ms):
                    print("Application terminated by user input from the visualizer window.")
                    break # Exit the main loop if the user quits.

                processing_iteration_count += 1 # Increment the iteration counter.

        except KeyboardInterrupt:
            # Handle manual termination via Ctrl+C.
            print("\nKeyboardInterrupt detected. Initiating graceful shutdown sequence.")
        except Exception as unhandled_exception:
            # Catch any unexpected exceptions during the main loop to prevent unhandled crashes.
            print(f"An unexpected critical error occurred during application execution: {unhandled_exception}")
            # In a production-critical system, detailed error logging and state recovery mechanisms
            # would be implemented here.
        finally:
            # This block ensures that critical resources are released regardless of how
            # the application loop terminates (normally, by user, or by exception).
            print("\n--- Application Execution Finished ---")
            self.video_processor.release_resources() # Release the video capture.
            cv2.destroyAllWindows() # Close all OpenCV display windows.
            print("All system resources released gracefully.")


# ==============================================================================
# Main Application Entry Point
# ==============================================================================
if __name__ == "__main__":
    # Define configuration constants for the application.
    # These paths should be set to where your video file and YOLO model are located.
    APP_INPUT_VIDEO_PATH = 'park.mp4' # Ensure 'park.mp4' is present in the same directory.
    APP_MODEL_FILE_PATH = 'yolov8n-seg.pt' # The pre-trained YOLOv8 segmentation model.
    APP_DISPLAY_WINDOW_NAME = 'Real-time Object Instance Segmentation' # The title for the display window.

    # Instantiate the modular components of the application.
    # This setup follows the Dependency Inversion Principle for better modularity.
    video_handler = VideoProcessor(APP_MODEL_FILE_PATH, APP_INPUT_VIDEO_PATH)
    visualizer_handler = SegmentationVisualizer(APP_DISPLAY_WINDOW_NAME)

    # Create the main application coordinator, which will manage the overall flow.
    app_coordinator = ApplicationCoordinator(
        video_handler,
        visualizer_handler
    )

    # Start the application's main processing loop.
    app_coordinator.execute_application_loop()
