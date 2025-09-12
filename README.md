# Complex Fractal Renderer

A Python program for rendering video or still frames of fractals that use complex numbers, such as the Mandelbrot Set or Newton Fractals. 

## Requirements
* Tested on Windows
* Python 3.10+
* Several hours of patience if you are rendering animations

## Installation
1. Install Python or PyPy.
2. Install Pillow (`pip install pillow` or `pypy -m pip install pillow`).
3. Install OpenCV (`pip install opencv-python`).
4. Clone the repo.
5. Run `main.py` at least once. You can close the window after text appears in the terminal.

## Note for PyPy Users
Because `opencv-python` doesn't work with PyPy, don't try to install it for PyPy. The portion of the program that uses OpenCV will automatically be spawned as a separate process running with CPython. Instead, run `pip install opencv-python` to install it for your CPython installation.

## How to Use
1. Create a ruleset (fractal), or choose an existing one. If creating your own, make a copy of `template.py` in the `rulesets` folder, name it whatever you want, and modify the `Ruleset` class' contents.\
   Some built-in rulesets are the Mandelbrot Set and Newton Fractals.
2. If you are making a video, create an animation. This should be a JSON file in the `animation_rules` folder. See `animation_rules/template.json` for information about how to create an animation.
3. Run `main.py` and select your render options.
4. Finished renders will be saved in the `renders` folder as an AVI file (video) or PNG file (still frame).

## How to Convert Output Videos

### Step 1. Converting to MP4/MOV
I've had some issues using the output videos from this program in the past, due to their abnormal uncompressed encoding. Follow these instructions to convert to an MOV or MP4 file while retaining 100% quality. Unfortunately you need to have [VLC media player](https://images.videolan.org/vlc/) installed.

1. Open VLC and go to `Media` > `Convert / Save...`.
2. Under `File Selection`, click `Add...` and select your AVI file from the `renders` folder.
3. Click `Convert / Save`.
4. Under `Settings`, `Convert`, to the right of the `Profile` dropdown, click the button to the right of the red X to create a new profile.
5. Under `Encapsulation`, choose `MP4/MOV`.
6. Under `Video codec`, check `Video`, choose `H-264` as the codec, and set the bitrate to `Not used`.
7. At the top of the window, type something like `Lossless MP4/MOV` as the name of the profile and then click `Save`.
8. In the `Profile` dropdown, choose the profile you just created.
9. Under `Destination`, choose the filename to save your video as. You can save it as either MOV or MP4. Either will work.
10. Click `Start` and **wait for the progress bar to reach the end of the video. Do not touch anything until it reaches the end.**
11. Once the progress reaches the end, the progress bar should turn gray again. At this point you can close VLC.

### Step 2. Fixing start of video cut off

Personally at this point I experienced an issue where certain applications, when reading the video, would always cut off the first few seconds. If you are not experiencing this issue, you can skip this section and be done here.

#### Step 2a. Installing FFmpeg

You can skip this subsection if you already have FFmpeg installed. These instructions are for Windows.

1. Go to: https://github.com/BtbN/FFmpeg-Builds/releases
2. Click `ffmpeg-master-latest-win64-gpl-shared.zip` and save it to a known location.
3. Double-click the file in File Explorer to preview its contents. You should see a single folder called `ffmpeg-master-latest-win64-gpl-shared`. Copy this folder.
4. Go to your `C:/` directory and paste the folder.
5. Rename the folder to `ffmpeg`.
6. In the Start menu, search `Edit the system environment variables` and open that.
7. Click `Environment Variables...`.
8. Under `User variables for <username>`, select `Path` and click `Edit...`.
9. Click `New` and type `C:\ffmpeg\bin` in the textbox that appears.
10. Click `OK`, then `OK`, then `OK`.

#### Step 2b. Fixing the video

1. Open Command Prompt and run `ffmpeg --help` to ensure FFmpeg is installed. If you see `'ffmpeg' is not recognized as an internal or external command, operable program or batch file.` then FFmpeg wasn't correctly installed.
2. Change the current working directory to the `renders` folder. Do this with the command `cd "<PATH TO RENDERS FOLDER>"`, except with your path.
3. Run the following command: `ffmpeg -i "<INPUT FILENAME>" -c:v copy -an "<OUTPUT FILENAME>"` except with your input and output video filenames.

The output video created by this command should now work correctly in all applications.
