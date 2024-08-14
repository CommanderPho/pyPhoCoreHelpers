import os
import io
from typing import Optional, Union, List, Dict, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from glob import glob

import matplotlib.pyplot as plt # for export_array_as_image
from PIL import Image, ImageOps, ImageFilter # for export_array_as_image

from plotly.graph_objects import Figure as PlotlyFigure # required for `fig_to_clipboard`
from matplotlib.figure import FigureBase
# from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import copy_image_to_clipboard # required for `fig_to_clipboard`


def add_border(image: Image.Image, border_size: int = 5, border_color: tuple = (0, 0, 0)) -> Image.Image:
    return ImageOps.expand(image, border=border_size, fill=border_color)

def add_shadow(image: Image.Image, offset: int = 5, background_color: tuple = (0, 0, 0, 0), shadow_color: tuple = (0, 0, 0, 255)) -> Image.Image:
    total_width = image.width + offset
    total_height = image.height + offset

    shadow = Image.new('RGBA', (total_width, total_height), background_color)
    shadow.paste(image, (offset, offset))

    shadow_layer = Image.new('RGBA', (total_width, total_height), shadow_color)
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=offset))

    shadow = Image.alpha_composite(shadow_layer, shadow)

    return shadow



def get_array_as_image(img_data, desired_width: Optional[int] = None, desired_height: Optional[int] = None, colormap='viridis', skip_img_normalization:bool=False, export_grayscale:bool=False) -> Image.Image:
    """ Like `save_array_as_image` except it skips the saving to disk. Converts a numpy array to file as a colormapped image
    
    # Usage:
    
        from pyphocorehelpers.plotting.media_output_helpers import get_array_as_image
    
        image = get_array_as_image(img_data, desired_height=100, desired_width=None, skip_img_normalization=True)
        image
                
    """
    # Assuming `your_array` is your numpy array
    if export_grayscale:
        # Convert to grayscale (normalize if needed)
        if skip_img_normalization:
            norm_array = img_data
        else:
            norm_array = (img_data - np.min(img_data)) / np.ptp(img_data)

        # Scale to 0-255 and convert to uint8
        image = Image.fromarray((norm_array * 255).astype(np.uint8), mode='L')
    else:
        ## Color export mode!
        assert (colormap is None) or (colormap == 'viridis'), f"colormap should not be specified is export_grayscale=True"
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
        
    
    if ((desired_width is None) and (desired_height is None)):
        desired_width = 1024 # set a default arbitrarily

    if desired_width is not None:
        # Specify width
        assert desired_height is None, f"please don't provide both width and height, the other will be calculated automatically."
        # Calculate height to preserve aspect ratio
        desired_height = int(desired_width * norm_array.shape[0] / norm_array.shape[1])
    elif (desired_height is not None):
        # Specify height:
        assert desired_width is None, f"please don't provide both width and height, the other will be calculated automatically."
        # Calculate width to preserve aspect ratio
        desired_width = int(desired_height * norm_array.shape[1] / norm_array.shape[0])
    else:
        raise ValueError("you must specify width or height of the output image")

    # Resize image
    # image = image.resize((new_width, new_height), Image.LANCZOS)
    image = image.resize((desired_width, desired_height), Image.NEAREST)

    return image

def save_array_as_image(img_data, desired_width: Optional[int] = 1024, desired_height: Optional[int] = None, colormap='viridis', skip_img_normalization:bool=False, out_path='output/numpy_array_as_image.png', export_grayscale:bool=False) -> Tuple[Image.Image, Path]:
    """ Exports a numpy array to file as a colormapped image
    
    # Usage:
    
        from pyphocorehelpers.plotting.media_output_helpers import save_array_as_image
    
        image, out_path = save_array_as_image(img_data, desired_height=100, desired_width=None, skip_img_normalization=True)
        image
                
    """
    image: Image.Image = get_array_as_image(img_data=img_data, desired_width=desired_width, desired_height=desired_height, colormap=colormap, skip_img_normalization=skip_img_normalization, export_grayscale=export_grayscale)

    out_path = Path(out_path).resolve()
    # Save image to file
    image.save(out_path)

    return image, out_path


def get_array_as_image_stack(imgs: List[Image.Image], offset=10, single_image_alpha_level:float=0.5,
                            #   border_size: Optional[int] = 5, shadow_offset: Optional[int] = 10,
                              should_add_border: bool = False, border_size: int = 5, border_color: Tuple[int, int, int] = (0, 0, 0),
                              should_add_shadow: bool = False, shadow_offset: int = 10, shadow_color: Tuple[int, int, int, int] = (0, 0, 0, 255),
                              ) -> Image.Image:
   
    """ Handles 3D images
    Given a list of equally sized figures, how do I overlay them in a neat looking stack and produce an output graphic from that?
    I want them offset slightly from each other to make a visually appealing stack

    single_image_alpha_level = 0.5 - adjust this value to set the desired transparency level (0.0 to 1.0)
    offset = 10  # your desired offset

    2024-01-12 - works well

    Usage:
        from pyphocorehelpers.plotting.media_output_helpers import get_array_as_image_stack, save_array_as_image_stack

        # Let's assume you have a list of images
        images = ['image1.png', 'image2.png', 'image3.png']  # replace this with actual paths to your images
        output_img, output_path = render_image_stack(out_figs_paths, offset=55, single_image_alpha_level=0.85)

    """
    # Make a general alpha adjustment to the images
    if (single_image_alpha_level is None) or (single_image_alpha_level == 1.0):
        # Open the images
        pass

    else:
        print(f'WARNING: transparency mode is very slow! This took ~50sec for ~30 images')
        # only do this if transparency of layers is needed, as this is very slow (~50sec)
        imgs = [img.convert("RGBA") for img in imgs] # convert to RGBA explicitly, seems to be very slow.
        for i in range(len(imgs)):
            for x in range(imgs[i].width):
                for y in range(imgs[i].height):
                    r, g, b, a = imgs[i].getpixel((x, y))
                    imgs[i].putpixel((x, y), (r, g, b, int(a * single_image_alpha_level)))

    # Assume all images are the same size
    width, height = imgs[0].size

    # Create a new image with size larger than original ones, considering offsets
    output_width = width + offset * (len(imgs) - 1)
    output_height = height + offset * (len(imgs) - 1)

    output_img = Image.new('RGBA', (output_width, output_height))

    should_add_border = (should_add_border and (border_size is not None) and (border_size > 0))
    should_add_shadow = (should_add_shadow and (shadow_offset is not None) and (shadow_offset > 0))
    # Overlay images with offset
    #    for i, img in enumerate(imgs):
    #       output_img.paste(img, (i * offset, i * offset), img)

    if should_add_border:
        width += 2 * border_size
        height += 2 * border_size

    if should_add_shadow:
        width += shadow_offset
        height += shadow_offset

    output_width = width + offset * (len(imgs) - 1)
    output_height = height + offset * (len(imgs) - 1)

    output_img = Image.new('RGBA', (output_width, output_height))

    for i, img in enumerate(imgs):
        if add_border:
            img = add_border(img, border_size=border_size, border_color=border_color)
        if add_shadow:
            img = add_shadow(img, offset=shadow_offset, shadow_color=shadow_color)
        output_img.paste(img, (i * offset, i * offset), img)

    return output_img


# @function_attributes(short_name=None, tags=['image', 'stack', 'batch', 'file', 'stack'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-12 00:00', related_items=[])
def save_array_as_image_stack(images: List[Path], offset=10, single_image_alpha_level:float=0.5):
    """ 
    Given a list of equally sized figures, how do I overlay them in a neat looking stack and produce an output graphic from that?
    I want them offset slightly from each other to make a visually appealing stack


    single_image_alpha_level = 0.5 - adjust this value to set the desired transparency level (0.0 to 1.0)
    offset = 10  # your desired offset

    2024-01-12 - works well

    Usage:
        from pyphocorehelpers.plotting.media_output_helpers import save_array_as_image_stack

        # Let's assume you have a list of images
        images = ['image1.png', 'image2.png', 'image3.png']  # replace this with actual paths to your images
        output_img, output_path = render_image_stack(out_figs_paths, offset=55, single_image_alpha_level=0.85)

    """
    # Open the images
    imgs = [Image.open(i) for i in images]

    output_img = get_array_as_image_stack(imgs, offset=offset, single_image_alpha_level=single_image_alpha_level)

    output_path: Path = Path('output/stacked_images.png').resolve()
    output_img.save(output_path)
    return output_img, output_path




#TODO 2023-09-27 19:54: - [ ] saving

def save_array_as_video(array, video_filename='output/videos/long_short_rel_entr_curves_frames.mp4', fps=30.0, isColor=False, colormap=cv2.COLORMAP_VIRIDIS, skip_img_normalization=False, debug_print=False, progress_print=True):
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
    if skip_img_normalization:
        array = array
    else:
        # Normalize your array to 0-1
        array = (array - np.nanmin(array, axis=(1,2,), keepdims=True)) / np.ptp(array, axis=(1,2,), keepdims=True)
    
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
        if isColor and (colormap is not None):
            # NEW: apply colormap if provided and isColor is True
            color_array = cv2.applyColorMap(gray_3c, colormap)
            out.write(color_array)
        else:
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




def create_video_from_images(image_folder: str, output_video_file: str, seconds_per_frame: float, frame_size: tuple = None, codec: str = 'mp4v') -> Path:
    """ 
    Loads sequence of images from a folder and joins them into a video where each frame is a fixed duration (`seconds_per_frame`)
    
    # Usage:
        from pyphocorehelpers.plotting.media_output_helpers import create_video_from_images
        from pyphoplacecellanalysis.GUI.Napari.napari_helpers import napari_export_image_sequence
        ## Save images from napari to disk:
        desired_save_parent_path = Path('2024-08-08 - TransitionMatrix/PosteriorPredictions').resolve()
        imageseries_output_directory = napari_export_image_sequence(viewer=viewer, imageseries_output_directory=desired_save_parent_path, slider_axis_IDX=0)
        ## Build video from saved images:
        video_out_file = desired_save_parent_path.joinpath('output_video.mp4')
        create_video_from_images(image_folder=imageseries_output_directory, output_video_file=video_out_file, seconds_per_frame=0.2)

    """
    if not isinstance(image_folder, Path):
        image_folder = Path(image_folder).resolve()
    if not isinstance(output_video_file, Path):
        output_video_file = Path(output_video_file).resolve()
            

    images = sorted(glob(os.path.join(image_folder, '*.png')))  # Adjust the extension if necessary
    if not images:
        raise ValueError("No images found in the specified folder.")
    
    first_image = cv2.imread(images[0])
    height, width, _ = first_image.shape
    frame_size = frame_size or (width, height)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    fps = 1 / seconds_per_frame
    video_writer = cv2.VideoWriter(output_video_file.resolve().as_posix(), fourcc, fps, frame_size)
    
    for image_file in images:
        img = cv2.imread(image_file)
        if img.shape[1] != frame_size[0] or img.shape[0] != frame_size[1]:
            img = cv2.resize(img, frame_size)
        video_writer.write(img)
    
    video_writer.release()
    return output_video_file


def create_gif_from_images(image_folder: str, output_video_file: str, seconds_per_frame: float) -> Path:
    """ 
    Loads sequence of images from a folder and joins them into an animated GIF where each frame is a fixed duration (`seconds_per_frame`)
    
    # Usage:
        from pyphocorehelpers.plotting.media_output_helpers import create_gif_from_images
        from pyphoplacecellanalysis.GUI.Napari.napari_helpers import napari_export_image_sequence
        ## Save images from napari to disk:
        desired_save_parent_path = Path('2024-08-08 - TransitionMatrix/PosteriorPredictions').resolve()
        imageseries_output_directory = napari_export_image_sequence(viewer=viewer, imageseries_output_directory=desired_save_parent_path, slider_axis_IDX=0)
        ## Build animated .gif from saved images:
        gif_out_file = desired_save_parent_path.joinpath('output_video.gif')
        create_gif_from_images(image_folder=imageseries_output_directory, output_video_file=gif_out_file, seconds_per_frame=0.2)

    """
    from PIL import Image # create_gif_from_images
    
    if not isinstance(image_folder, Path):
        image_folder = Path(image_folder).resolve()
    if not isinstance(output_video_file, Path):
        output_video_file = Path(output_video_file).resolve()
        
    images = sorted(glob(os.path.join(image_folder, '*.png')))  # Adjust the extension if necessary
    if not images:
        raise ValueError("No images found in the specified folder.")
    
    frames = [Image.open(image) for image in images]
    duration = int(seconds_per_frame * 1000)  # Convert seconds to milliseconds
    
    frames[0].save(output_video_file, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)
    return output_video_file





# plotly.graph_objs._figure.Figure

# @function_attributes(short_name=None, tags=['clipboard', 'image', 'figure', 'export', 'matplotlib', 'plotly'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-05 13:59', related_items=[])
def fig_to_clipboard(a_fig: Union[PlotlyFigure, FigureBase], format="png", **kwargs):
    """ Any common figure type (matplotlib, Plotly, etc) to clipboard as image  _______________________________________________________________________________________________  
    
    from pyphocorehelpers.plotting.media_output_helpers import fig_to_clipboard
    
    fig_to_clipboard(fig)

    """
    from pyphocorehelpers.programming_helpers import copy_image_to_clipboard
    
    _fig_save_fn = None
    if isinstance(a_fig, FigureBase):
        # Matplotlib Figure:
        _fig_save_fn = a_fig.savefig
        if format == 'png':
            kwargs.setdefault('bbox_inches', 'tight') # crops off the empty white margins

    elif isinstance(a_fig, PlotlyFigure):
        # Plotly Figure:
        _fig_save_fn = a_fig.write_image
    else:
        raise NotImplementedError(f"type(a_fig): {type(a_fig)}, a_fig: {a_fig}")
    ## Perform the image generation to clipboard:
    with io.BytesIO() as buf:
        _fig_save_fn(buf, format=format, **kwargs)
        buf.seek(0)
        img = Image.open(buf)
        # Send the image to the clipboard
        copy_image_to_clipboard(img)
        # Close the buffer and figure to free resources
        buf.close()
            


