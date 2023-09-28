import numpy as np
import pandas as pd
from pathlib import Path
import cv2

#TODO 2023-09-27 19:54: - [ ] saving

def save_array_as_video(array, video_filename='output/videos/long_short_rel_entr_curves_frames.mp4', fps=30.0, isColor=False, debug_print=False, progress_print=True):
    """
    Save a 3D numpy array as a grayscale video.

    NOTE: .avi is MUCH faster than .mp4, by like 100x or more!
    
    Parameters:
    - array: numpy array of shape (timesteps, height, width)
    - output_filename: name of the output video file
    - fps: frames per second for the output video


    Usage:

        from pyphocorehelpers.plotting.video_output_helpers import save_array_as_video

    """
    gray_frames = cv2.normalize(array, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) # same size as array
    if debug_print:
        print(f'array.shape: {array.shape}')
        print(f'gray_frames.shape: {gray_frames.shape}')
    assert array.shape == gray_frames.shape, f"gray_frames should be the same size as the input array, just scaled to greyscale int8!"
    
    # Extract width and height from the array:
    n_frames = np.shape(array)[0]
    height = np.shape(array)[1]
    width = np.shape(array)[2]

    # initialize video writer
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter(str(video_filename), fourcc, fps, (width, height))

    # new frame after each addition of water

    for i in np.arange(n_frames):
        if progress_print:
            print(f'saving frame {i}/{n_frames}')
        gray = np.squeeze(gray_frames[i,:,:]) # single frame
        gray_3c = cv2.merge([gray, gray, gray])
        out.write(gray_3c)

    # close out the video writer
    out.release()
    if progress_print:
        print(f'done! video saved to {video_filename}')
    return Path(video_filename).resolve()

""" 

# def normalize_and_save_as_video(array, output_filename='output.mp4', fps=30.0):
#     array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)
#     return save_array_as_video(array, output_filename, fps)


# def colormap_and_save_as_video(array, output_filename='output.mp4', fps=30.0, colormap=cv2.COLORMAP_JET):
#     array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)
#     color_array = cv2.applyColorMap(array, colormap)
#     return save_array_as_video(color_array, output_filename, fps, isColor=True)
    

# Example usage
# array = np.random.randint(0, 256, (4123, 90, 117), dtype=np.uint8)

# Save the `long_short_rel_entr_curves_frames` array out to the file:
# video_out_path = save_array_as_video(active_relative_entropy_results_xr_dict['long_short_rel_entr_curves_frames'].to_numpy(), output_filename='output/videos/long_short_rel_entr_curves_frames.mp4')


# array = active_relative_entropy_results_xr_dict['long_short_rel_entr_curves_frames'].to_numpy()
# video_out_path = normalize_and_save_as_video(array=array, output_filename='output/videos/long_short_rel_entr_curves_frames.mp4')

"""