import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from moviepy.editor import AudioFileClip, VideoClip, VideoFileClip, ImageSequenceClip


AUDIO_SAMPLE_RATE = 44100

def read_face_landmarks(csv_path):
    """
    Reads a CSV file containing detected face landmark coordinates and extracts columns with 2d face landmarks labled ' x_0' - ' y_0' up to ' x_67'
    see https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format for details.
    
    :param csv_path: Path to the CSV file.
    :return: A list of (x, y) tuples representing face center coordinates per frame.
    """
    
    try:
        df = pd.read_csv(csv_path)
        # Construct the list of expected landmark columns
        columns = [f" x_{i}" for i in range(68)] + [f" y_{i}" for i in range(68)]
        # Filter to only available columns
        available_columns = [col for col in columns if col in df.columns]
        landmarks_df = df[available_columns]
        return landmarks_df
    except Exception as e:
        print(f"Error reading face landmark CSV: {e}")
        return None


def read_video(file_path='bond_raw_1.mp4'):
    try:
        clip = VideoFileClip(file_path)
        return clip
    except Exception as e:
        print(f"Error reading video file: {e}")
        return None
        
        
def get_face(landmarks_df):
    """
    Computes a single bounding box that contains the face for the entire video.
    
    :param landmarks_df: A pandas DataFrame where each row represents the coordinates 
                          for a frame and each column represents a landmark's x or y value.
    :return: A tuple (x_min, x_max, y_min, y_max) defining the crop area.
    """
    # Ensure we're handling only the relevant x and y columns
    x_columns = [f" x_{i}" for i in range(68)]
    y_columns = [f" y_{i}" for i in range(68)]
    
    # Extract x and y values for all frames
    x_values = landmarks_df[x_columns].values.flatten()
    y_values = landmarks_df[y_columns].values.flatten()

    # Calculate the bounding box
    x_min, x_max = int(min(x_values)), int(max(x_values))
    y_min, y_max = int(min(y_values)), int(max(y_values))

    return x_min, x_max, y_min - 200, y_max
    
    
    
def get_mouth(landmarks_df):
    """
    Computes a single bounding box that contains the face for the entire video.
    
    :param landmarks_df: A pandas DataFrame where each row represents the coordinates 
                          for a frame and each column represents a landmark's x or y value.
    :return: A tuple (x_min, x_max, y_min, y_max) defining the crop area.
    """
    # Ensure we're handling only the relevant x and y columns
    x_columns = [f" x_{i}" for i in range(48 , 67)]
    y_columns = [f" y_{i}" for i in range(48 , 67)]
    
    # Extract x and y values for all frames
    x_values = landmarks_df[x_columns].values.flatten()
    y_values = landmarks_df[y_columns].values.flatten()

    # Calculate the bounding box
    x_min, x_max = int(min(x_values)), int(max(x_values))
    y_min, y_max = int(min(y_values)), int(max(y_values))

    return x_min, x_max, y_min, y_max

    
def crop_video(clip, bounding_box):
    """
    Crops the video using a fixed bounding box around the face.
    
    :param clip: A VideoFileClip object.
    :param bounding_box: A tuple (x_min, x_max, y_min, y_max) defining the crop area.
    :return: A cropped VideoFileClip.
    """
    x_min, x_max, y_min, y_max = bounding_box

    def crop_frame(get_frame, t):
        frame = get_frame(t)
        return frame[y_min:y_max, x_min:x_max]

    return clip.fl(crop_frame)
    
    
def video_to_numpy_array(clip):
    """
    Converts a VideoClip to a NumPy array of pixel values for each frame.
    
    :param clip: The VideoFileClip object.
    :return: A 4D NumPy array with shape (n_frames, height, width, channels).
    """
    frames = []
    duration = clip.duration
    fps = clip.fps

    # Iterate through all frames of the video
    for t in np.arange(0, duration, 1 / fps):
        frame = clip.get_frame(t)  # Get the frame at time t
        frames.append(frame)  # Store the frame

    # Convert the list of frames to a NumPy array
    frames_array = np.array(frames)
    
    return frames_array


def filter_spatial(array, high = True, low = False):
    frames, rows, cols, channels = array.shape
    low_cutoff = int(min(rows, cols) * (0.45))
    high_cutoff = int(min(rows, cols) * (0.01))
    filtered_array = np.zeros_like(array, dtype=np.float32)  # Initialize with the same shape as the input array

    # Applying Fourier transformation frame by frame
    for j in range(channels):
        for i in range(frames):
            # Fourier transform on each channel of each frame
            fourier = np.fft.fft2(array[i, :, :, j])
            # Shift zero frequency to the center
            f_transform_shifted = np.fft.fftshift(fourier)
            crow, ccol = rows // 2, cols // 2
            if(low):
            # Create a low-pass filter mask
                mask = np.zeros((rows, cols), dtype=np.uint8)
                mask[crow - low_cutoff:crow + low_cutoff, ccol - low_cutoff:ccol + low_cutoff] = 1

                # Apply the mask to filter out high frequencies
                transformed = f_transform_shifted * mask

            if(high):    
                # Create a circular mask that keeps high frequencies
                mask = np.ones((rows, cols), dtype=np.uint8)
                mask[crow - high_cutoff:crow + high_cutoff, ccol - high_cutoff:ccol + high_cutoff]  = 0
        
                # Step 4: Apply the mask to the Fourier transform
                filtered_transform = f_transform_shifted * mask
                transformed = f_transform_shifted * mask

            
            # Perform Inverse Fourier Transform
            f_ishift = np.fft.ifftshift(transformed)
            filtered_image = np.fft.ifft2(f_ishift)

            # Store the magnitude of the inverse Fourier transform in the corresponding place
            filtered_array[i, :, :, j] = np.abs(filtered_image)

    return filtered_array
    

def generate_white_noise_video(width=12, height=480, fps=30, duration=10):

    output_path = Path(output).absolute()

    if output_path.exists():
        raise SystemExit(f"File {output_path} already exists.")

    # creating a white noise frame, filtering high spatial frequencies
    def make_frame(t):
    
        # return  np.random.uniform(0, 256, (height, width))
        
        # generating array of random values
        random_array = np.random.randint(0, 256, (height, width, duration * fps), dtype=np.uint8)
        filtered_array = filter_spatial(random_array)
        return filtered_array
        
    video = VideoClip(make_frame, duration=duration)
    video = video.set_fps(fps)
    return video
    
    
def save_video(output, video, quiet=False):
    logger = "bar" if not quiet else None
    try:
        video.write_videofile(str(output),
                                    codec='libx264', audio_codec='aac', logger=logger)
    except KeyboardInterrupt:
        output_path.unlink(missing_ok=True)
        raise SystemExit("Interrupted by user.")



def main():
    parser = argparse.ArgumentParser(description="Generate a white noise video with audio.")
    parser.add_argument("output", type=str, help="Output video file path.")
    parser.add_argument("-w", "--width", type=int, default=640, help="Video width.")
    parser.add_argument("-t", "--height", type=int, default=480, help="Video height.")
    parser.add_argument("-f", "--fps", type=int, default=30, help="Video frames per second.")
    parser.add_argument("-d", "--duration", type=int, default=10, help="Video duration in seconds.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Give less output.")

    args = parser.parse_args()

    # generate_white_noise_video(args.output, args.width, args.height, args.fps, args.duration, args.quiet)
    # landmarks = get_landmark_df()
    # video = read_video()
    # print(landmarks)
    # cropped_video = crop_video(clip, landmarks[], box_size=(100, 100))
    # save_video(output, video)
    video_clip = read_video("test.mp4")
    if video_clip:
        coordinates = read_face_landmarks("test.csv")
        # cropping so that face never leaves frame
        face_coords = get_face(coordinates)
        cropped_clip = crop_video(video_clip, face_coords)
        #save_video(args.output+".mp4", cropped_clip)
        
        
        # cropping out video of just the face

        mouth_coords = get_mouth(coordinates)
        mouth_clip = crop_video(video_clip, mouth_coords)
        #save_video(args.output+'mouth.mp4', mouth_clip)
        mouth_array = video_to_numpy_array(mouth_clip)
        filtered_mouth = filter_spatial(mouth_array)
        filtered_mouth_clip = ImageSequenceClip(list(filtered_mouth), fps=30)
        save_video(args.output + 'filtered_mouth.mp4', filtered_mouth_clip)

if __name__ == "__main__":
    try:
        main()
    except SystemExit as sys_exit:
        print(sys_exit, file=sys.stderr)
        sys.exit(1)
