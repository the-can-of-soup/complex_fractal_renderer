# This program should not be run with PyPy, because PyPy does not support opencv-python.

from complex_analysis import show_progress_bar
import cv2, os, json, time

with open('video_finalizer_info.json', 'r') as f:
    finalizer_info: dict = json.loads(f.read())
os.remove('video_finalizer_info.json')

# these properties must match those in main.py
FPS = finalizer_info['frame_rate']
RESOLUTION = tuple(finalizer_info['resolution'])

def main():
    # import frames and create video file
    video: cv2.VideoWriter = cv2.VideoWriter(finalizer_info['output_filename'], 0, FPS, RESOLUTION)
    frame_filenames: list[str] = os.listdir('frames')
    frame_filenames.sort()
    num_frames = len(frame_filenames)
    print('Video initialized.')
    i = 0
    start_time: float = time.time()
    for frame_filename in frame_filenames:
        show_progress_bar('Writing frames...', i / num_frames, start_time=start_time)
        video.write(cv2.imread(os.path.join('frames', frame_filename)))
        i += 1
    show_progress_bar('Writing frames...', 1, True, start_time=start_time)
    print('Saving video...')
    cv2.destroyAllWindows()
    video.release()
    print('Done!')

if __name__ == '__main__':
    main()
