from typing import Optional
import numpy as np
import pandas as pd
from pathlib import Path
import cv2

import matplotlib.pyplot as plt # for export_array_as_image
from PIL import Image # for export_array_as_image


def save_array_as_image(img_data, desired_width: Optional[int] = 1024, desired_height: Optional[int] = None, colormap='viridis', skip_img_normalization:bool=False, out_path='output/numpy_array_as_image.png') -> (Image, Path):
    """ Exports a numpy array to file as a colormapped image
    
    # Usage:
    
        from pyphocorehelpers.plotting.media_output_helpers import save_array_as_image
    
        image, out_path = save_array_as_image(img_data, desired_height=100, desired_width=None, skip_img_normalization=True)
        image
                
    """
    # Assuming `your_array` is your numpy array
    # For the colormap, you can use any colormap from matplotlib. 
    # In this case, 'hot' is used.
    # Get the specified colormap
    colormap = plt.get_cmap(colormap)

    if skip_img_normalization:
        norm_array = img_data
    else:
        # Normalize your array to 0-1
        norm_array = (img_data - np.min(img_data)) / np.ptp(img_data)

    # Apply colormap
    image_array = colormap(norm_array)

    # Convert to PIL image and remove alpha channel
    image = Image.fromarray((image_array[:, :, :3] * 255).astype(np.uint8))

    if desired_width is not None:
        # Specify width
        assert desired_height is None, f"please don't provide both width and height, the other will be calculated automatically."
        # Calculate height to preserve aspect ratio
        desired_height = int(desired_width * image_array.shape[0] / image_array.shape[1])
    elif (desired_height is not None):
        # Specify height:
        assert desired_width is None, f"please don't provide both width and height, the other will be calculated automatically."
        # Calculate width to preserve aspect ratio
        desired_width = int(desired_height * image_array.shape[1] / image_array.shape[0])
    else:
        raise ValueError("you must specify width or height of the output image")

    # Resize image
    # image = image.resize((new_width, new_height), Image.LANCZOS)
    image = image.resize((desired_width, desired_height), Image.NEAREST)

    out_path = Path(out_path).resolve()
    # Save image to file
    image.save(out_path)

    return image, out_path




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

        from pyphocorehelpers.plotting.media_output_helpers import save_array_as_video
        
        video_out_path = save_array_as_video(array=active_relative_entropy_results['snapshot_occupancy_weighted_tuning_maps'], video_filename='output/videos/snapshot_occupancy_weighted_tuning_maps.avi', isColor=False)
        print(f'video_out_path: {video_out_path}')
        reveal_in_system_file_manager(video_out_path)
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