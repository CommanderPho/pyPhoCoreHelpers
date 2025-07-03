 
from __future__ import annotations # prevents having to specify types for typehinting as strings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    ## typehinting only imports here
    from pyphoplacecellanalysis.Pho2D.data_exporting import HeatmapExportKind

import os
import io
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import nptyping as ND
from nptyping import NDArray
import numpy as np
import pandas as pd
from pathlib import Path
# import cv2
from copy import deepcopy
from glob import glob

import matplotlib.pyplot as plt # for export_array_as_image
from PIL import Image, ImageOps, ImageFilter # for export_array_as_image
from PIL import ImageDraw, ImageFont

from plotly.graph_objects import Figure as PlotlyFigure # required for `fig_to_clipboard`
from matplotlib.figure import FigureBase
# from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import copy_image_to_clipboard # required for `fig_to_clipboard`

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.programming_helpers import copy_image_to_clipboard
from pyphocorehelpers.image_helpers import ImageHelpers
from pyphocorehelpers.assertion_helpers import Assert

# from pyphoplacecellanalysis.Pho2D.data_exporting import HeatmapExportKind


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


def img_data_to_greyscale(img_data: NDArray, min_val: Optional[float]=None, max_val: Optional[float]=None, should_invert: bool=False) -> NDArray[np.uint8]:
    """ rescales the img_data array to 0-255 
    
    if `should_invert == True`, higher values are darker (with max_val == 1.0)

    Usage 1:
        from pyphocorehelpers.plotting.media_output_helpers import img_data_to_greyscale
        
        norm_array = img_data_to_greyscale(img_data)

    Usage 2:
        from pyphocorehelpers.plotting.media_output_helpers import img_data_to_greyscale

        norm_array = img_data_to_greyscale(img_data, min_val = kwargs.pop('vmin', None), max_val = kwargs.pop('vmax', None), should_invert=True)

    """
    # norm_array = (img_data - np.min(img_data)) / np.ptp(img_data) ## Default

    # Normalize your array to 0-1 using nan-aware functions
    
    if min_val is None:
        min_val = np.nanmin(img_data)
    if max_val is None:
        max_val = np.nanmax(img_data)

    assert (min_val < max_val), f"min_val: {min_val}, min_val: {min_val}, img_data: {img_data}"
    ptp_val = max_val - min_val
    norm_array = (img_data - min_val) / ptp_val

    # Scale to 0-255 and convert to uint8
    if should_invert:
        # Invert the values: higher values become darker (0), lower values become lighter (255)
        return (255 - (norm_array * 255)).astype(np.uint8)
    else:
        # Original behavior: higher values become lighter (255), lower values become darker (0)
        return (norm_array * 255).astype(np.uint8)


@metadata_attributes(short_name=None, tags=['image','post-hoc','export', 'posterior'], input_requires=[], output_provides=[], uses=[], used_by=['ImagePostRenderFunctionSets'], creation_date='2025-04-01 00:00', related_items=[])
class ImageOperationsAndEffects:
    
    @classmethod
    def create_fn_builder(cls, a_fn, **static_kwargs):
        """  Creates a function builder that captures static parameters and allows dynamic parameters to be added later.
        
        
        a_fn = ImageOperationsAndEffects.create_fn_builder(ImageOperationsAndEffects.add_solid_border, border_color = (0, 0, 0, 255))
        
        """
        def _create_new_img_operation_function(*dynamic_args, **dynamic_kwargs):
            """Create a function that adds a specific label to an image."""
            final_kwargs = (deepcopy(static_kwargs) | dynamic_kwargs) ## override any static kwargs with the dynamic ones
            return lambda an_img: a_fn(an_img, *dynamic_args, **final_kwargs)
        
        return _create_new_img_operation_function


    # @classmethod
    # def add_simple_bottom_label(cls, image: Image.Image, label_text: str, padding: int = None, font_size: int = None,  
    #                     text_color: tuple = (0, 0, 0), background_color: tuple = (255, 255, 255, 255), 
    #                     with_text_outline: bool = False, relative_font_size: float = 0.10, 
    #                     relative_padding: float = 0.025) -> Image.Image:
    #     """Adds a horizontally centered label underneath the bottom of an image.
        
    #     Parameters:
    #     -----------
    #     image : Image.Image
    #         The PIL Image to add a label to
    #     label_text : str
    #         The text to display as the label
    #     padding : int, optional
    #         Vertical padding between image and label. If None, calculated from relative_padding.
    #     font_size : int, optional
    #         Font size for the label text. If None, calculated from relative_font_size.
    #     text_color : tuple, optional
    #         RGB color for the text, by default (0, 0, 0) (black)
    #     background_color : tuple, optional
    #         RGBA color for the label background, by default (255, 255, 255, 255) (white)
    #     with_border : bool, optional
    #         Whether to add a border around the text, by default True
    #     relative_font_size : float, optional
    #         Font size as a proportion of image height, by default 0.03 (3% of image height)
    #     relative_padding : float, optional
    #         Padding as a proportion of image height, by default 0.02 (2% of image height)
            
    #     Returns:
    #     --------
    #     Image.Image
    #         A new image with the label added below the original image
            
    #     Usage:
    #     ------
    #     from pyphocorehelpers.plotting.media_output_helpers import add_bottom_label
        
    #     # Create an image with a label that scales with image size
    #     labeled_image = add_bottom_label(original_image, "Time (seconds)", relative_font_size=0.04)
    #     labeled_image
    #     """
    #     # Calculate font size and padding based on image height if not provided
    #     img_height = image.height
        
    #     if font_size is None:
    #         font_size = max(int(img_height * relative_font_size), 20)  # Minimum font size of 8
        
    #     if padding is None:
    #         padding = max(int(img_height * relative_padding), 10)  # Minimum padding of 5
        
    #     # Try to load a nicer font if available, otherwise use default
    #     try:
    #         # Try to use a common font that should be available on most systems
    #         # get a font
    #         font = ImageHelpers.get_font('FreeMono.ttf', size=font_size)
    #         # font = ImageFont.truetype("Arial", font_size)
    #     except IOError:
    #         # Fall back to default font with specified size
    #         try:
    #             # For newer Pillow versions that support size in load_default
    #             font = ImageFont.load_default(size=font_size)
    #         except TypeError:
    #             # For older Pillow versions that don't support size parameter
    #             default_font = ImageFont.load_default()
    #             # Try to find a bitmap font of appropriate size as alternative
    #             try:
    #                 font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    #             except IOError:
    #                 # If all else fails, use the default font
    #                 font = default_font
    #                 print(f"Warning: Could not load font with specified size {font_size}. Text may appear smaller than expected.")


    #     # Create a temporary drawing context to measure text dimensions
    #     temp_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
    #     temp_draw = ImageDraw.Draw(temp_img)
        
    #     # Use getbbox() for newer Pillow versions, fallback to textsize()
    #     try:
    #         # For newer Pillow versions
    #         bbox = temp_draw.textbbox((0, 0), label_text, font=font)
    #         text_width = bbox[2] - bbox[0]
    #         text_height = bbox[3] - bbox[1]
    #     except AttributeError:
    #         # For older Pillow versions
    #         text_width, text_height = temp_draw.textsize(label_text, font=font)
        
    #     # Create a new image with space for the label
    #     new_width = image.width
    #     new_height = image.height + padding + text_height + padding
        
    #     # Create the new image with the background color
    #     if image.mode == 'RGBA':
    #         new_image = Image.new('RGBA', (new_width, new_height), background_color)
    #     else:
    #         # Convert background_color to RGB if the image is not RGBA
    #         new_image = Image.new(image.mode, (new_width, new_height), background_color[:3])
        
    #     # Paste the original image at the top
    #     new_image.paste(image, (0, 0))
        
    #     # Create a drawing context for the new image
    #     draw = ImageDraw.Draw(new_image)
        
    #     # Calculate the position to center the text horizontally
    #     text_x = (new_width - text_width) // 2
    #     text_y = image.height + padding
        
    #     # Draw the text with or without border
    #     if with_text_outline:
    #         # Calculate border thickness based on font size
    #         border_thickness = max(1, int(font_size * 0.05))  # 5% of font size, minimum 1px
            
    #         def draw_text_with_border(draw, x, y, text, font, fill, thickness=1):
    #             # Draw shadow/border (using black color)
    #             shadow_color = (0, 0, 0)
    #             for dx in range(-thickness, thickness + 1):
    #                 for dy in range(-thickness, thickness + 1):
    #                     if dx != 0 or dy != 0:  # Skip the center position
    #                         draw.text((x + dx, y + dy), text, font=font, fill=shadow_color)
    #             # Draw text itself
    #             draw.text((x, y), text, font=font, fill=fill)
                
    #         draw_text_with_border(draw, text_x, text_y, label_text, font, fill=text_color, thickness=border_thickness)
    #     else:
    #         # Draw text without border
    #         draw.text((text_x, text_y), label_text, font=font, fill=text_color)
        
    #     return new_image


    @classmethod
    def add_bottom_label(cls, image: Image.Image, label_text: str, padding: int = None, font_size: int = None,  
                        text_color: tuple = (255, 255, 255), background_color: tuple = (66, 66, 66, 255), 
                        text_outline_shadow_color=None,
                        relative_font_size: float = 0.06,
                        relative_padding: float = 0.025, fixed_label_region_height: Optional[int] = 520,
                        # font='OpenSansCondensed-LightItalic.ttf',
                        font='ndastroneer.ttf',
                        debug_print=False,
                        ) -> Image.Image:
        """Adds a vertically oriented label at the bottom of an image.
        
        
        """

        # Calculate font size and padding based on image height if not provided
        original_img_height: int = deepcopy(image.height)
        original_img_width: int = deepcopy(image.width) 
            
        if font_size is None:
            font_size = max(int(original_img_height * relative_font_size), 20)  # Minimum font size of 8
            if debug_print:
                print(f'computing font size with relative_font_size: {relative_font_size}: font_size: {font_size} |', end='\t')
        if padding is None:
            padding = max(int(original_img_height * relative_padding), 0)  # Minimum padding of 0
        
        # Try to load a nicer font if available, otherwise use default
        try:
            font = ImageHelpers.get_font(font, size=font_size, allow_caching=True) # 'FreeMono.ttf'
        except IOError:
            # Fall back to default font with specified size
            try:
                font = ImageFont.load_default(size=font_size)
            except TypeError:
                default_font = ImageFont.load_default()
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                except IOError:
                    font = default_font
                    print(f"Warning: Could not load font with specified size {font_size}. Text may appear smaller than expected.")


        # label_kwargs = dict(font=font, align='center', anchor="mm") ## WORKS!, seems to be aligned better for small images, worse for large ones
        # label_kwargs = dict(font=font, align='center', anchor="ms") ## works okay.. #TODO 2025-05-27 15:39: - [ ] Seems to cut-off labels (all of them) for some reason?
        label_kwargs = dict(font=font, align='center', anchor="mt") ## NOW working, but has too much space on the bottom (because I add padding somewhere)
        
        # Create a temporary drawing context to measure text dimensions
        _temp_empty_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        _temp_empty_draw = ImageDraw.Draw(_temp_empty_img)
        
        # Use getbbox() for newer Pillow versions, fallback to textsize()
        # try:
        #     bbox = _temp_empty_draw.textbbox((0, 0), label_text, **label_kwargs)
        #     text_width = int(np.ceil(bbox[2] - bbox[0]))
        #     text_height = int(np.ceil(bbox[3] - bbox[1]))
        # except AttributeError:
        #     text_width, text_height = _temp_empty_draw.textsize(label_text, font=font, spacing=0) # , direction=None
        required_text_width, required_text_height = _temp_empty_draw.textsize(label_text, font=font, spacing=0, direction='ltr')
        required_text_height = required_text_height # + padding
        required_text_width = required_text_width # + padding


        # For vertical text, we need to swap width and height
        rotated_text_width: int = deepcopy(required_text_height)  # After 90 degree rotation
        rotated_text_height: int = deepcopy(required_text_width)  # After 90 degree rotation
        
        if debug_print:
            print(f'rotated_text_width: {rotated_text_width}, rotated_text_height: {rotated_text_height}', end='\t')
            
        if fixed_label_region_height is not None:
            if ((padding + rotated_text_height) > fixed_label_region_height):
                raise ValueError(f'Needed spacing for the label exceeds the specfied fixed_label_region_height: {fixed_label_region_height}, required space (padding + rotated_text_height): {(padding + rotated_text_height)}, padding: {padding}, rotated_text_height: {rotated_text_height}.')
            active_total_label_region_height: int = fixed_label_region_height ## override with the `fixed_label_region_height`
        else:
            active_total_label_region_height: int = (padding + rotated_text_height)

        # Create a new image with space for the label
        new_width: int = deepcopy(original_img_width)
        new_height: int = int(original_img_height + active_total_label_region_height)

            
        # ==================================================================================================================================================================================================================================================================================== #
        # Create the `new_larger_image` with the text label at the bottom                                                                                                                                                                                                                      #
        # ==================================================================================================================================================================================================================================================================================== #
        # Create the new image with the background color
        if image.mode == 'RGBA':
            new_larger_image = Image.new('RGBA', (new_width, new_height), background_color)
        else:
            new_larger_image = Image.new(image.mode, (new_width, new_height), background_color[:3])
        
        if debug_print:
            print(f'new_larger_image.size(w: {new_width}, h: {new_height}', end='\t')
            

        # Paste the original image at the top
        new_larger_image.paste(image, (0, 0))
        

        # ==================================================================================================================================================================================================================================================================================== #
        # Create the temporary `_temp_label_image` which is to be rotated                                                                                                                                                                                                                      #
        # ==================================================================================================================================================================================================================================================================================== #
        # Create a transparent background for the text
        # _debug_red_color = (255, 0, 0, 90)
        _clear_color = (0, 0, 0, 0)
        _active_label_bg_color = _clear_color
        _temp_label_image = Image.new('RGBA', (required_text_width, required_text_height), _active_label_bg_color)
        _temp_draw_label = ImageDraw.Draw(_temp_label_image)
        
        _internal_temp_box_text_x: int = (required_text_width // 2)
        _internal_temp_box_text_y: int = 0 # (font_size // 2)        
        
        # print(f'text_width: {text_width}, text_height: {text_height}, _internal_temp_box_text_x: {_internal_temp_box_text_x}, _internal_temp_box_text_y: {_internal_temp_box_text_y}')

        # Draw the text
        if (text_outline_shadow_color is not None):
            # Calculate border thickness based on font size
            border_thickness = max(1, int(font_size * 0.05))  # 5% of font size, minimum 1px
            
            # Draw text with outline
            for dx in range(-border_thickness, border_thickness + 1):
                for dy in range(-border_thickness, border_thickness + 1):
                    if dx != 0 or dy != 0:  # Skip the center position
                        _temp_draw_label.text((dx, dy), label_text, fill=text_outline_shadow_color, **label_kwargs)
            
            # Draw the main text
            # draw_label_temp.text((0, 0), label_text, fill=text_color, **label_kwargs)
            _temp_draw_label.text((_internal_temp_box_text_x, _internal_temp_box_text_y), label_text, fill=text_color, **label_kwargs)
        else:
            # Draw text without an outline
            _temp_draw_label.text((_internal_temp_box_text_x, _internal_temp_box_text_y), label_text, fill=text_color, **label_kwargs) # , direction=''
        
        
        # Rotate the text 270 degrees (so it reads from bottom to top)
        _temp_label_image = _temp_label_image.rotate(270, expand=1)

        # Get the dimensions of the rotated text image
        # rotated_width, rotated_height = _temp_label_image.size

        # Calculate position to center the text horizontally
        # For 270 degree rotation, we need to center based on the height of the original text
        # because after rotation, the height becomes the width
        # text_x = (new_width - rotated_width) // 2 ## WORKS, centers the result: 
        text_x = 0 ## WORKS, centers the result: 
        text_y = original_img_height + padding  # Position at the bottom of the original image plus padding

        # Paste the rotated text at the bottom center of the image
        new_larger_image.paste(_temp_label_image, (text_x, text_y), _temp_label_image)

        if debug_print:
            print(f'done.', end='\n') ## terminate the line
            
        return new_larger_image


    def add_half_width_rectangle(image: Image.Image, side: str = 'left', 
                                color: tuple = (200, 200, 255, 255), background_color: tuple = (255, 255, 255, 255),
                                height_fraction: float = 0.1) -> Image.Image:
        """Adds a rectangle that fills half the width of the image.
        
        Parameters:
        -----------
        image : Image.Image
            The PIL Image to add the rectangle to
        side : str, optional
            Which side to place the rectangle, either 'left' or 'right', by default 'left'
        color : tuple, optional
            RGBA color for the rectangle, by default (200, 200, 255, 255) (light blue)
        vertical_position : str, optional
            Where to place the rectangle vertically, either 'top', 'middle', 'bottom', or 'full',
            by default 'bottom'
        height_fraction : float, optional
            What fraction of the image height the rectangle should occupy (when not 'full'),
            by default 0.1 (10% of image height)
            
        Returns:
        --------
        Image.Image
            A new image with the rectangle added
            
        Usage:
        ------
        from pyphocorehelpers.plotting.media_output_helpers import add_half_width_rectangle
        
        # Add a blue rectangle to the left half of the bottom of the image
        modified_image = add_half_width_rectangle(
            original_image, 
            side='left',
            color=(100, 100, 255, 200),  # Semi-transparent blue
            vertical_position='bottom',
            height_fraction=0.15  # 15% of image height
        )
        
        # Add a red rectangle covering the entire right half
        modified_image = add_half_width_rectangle(
            original_image, 
            side='right',
            color=(255, 100, 100, 255),  # Red
            vertical_position='full'
        )
        """
        # Validate parameters
        if side not in ['left', 'right']:
            raise ValueError("side must be either 'left' or 'right'")
        
        # Create a new image with space for the rectangle
        new_width = image.width

        # Get image dimensions
        width, height = image.size
        half_width = width // 2
        rect_height = int(height * height_fraction)
        new_height = image.height + rect_height
        
        # Create the new image with the background color
        if image.mode == 'RGBA':
            new_image = Image.new('RGBA', (new_width, new_height), background_color)
        else:
            # Convert background_color to RGB if the image is not RGBA
            new_image = Image.new(image.mode, (new_width, new_height), background_color[:3])
        
        # Paste the original image at the top
        new_image.paste(image, (0, 0))
        
        # Create a drawing context
        draw = ImageDraw.Draw(new_image)
        
        # Calculate rectangle coordinates
        half_width = new_width // 2
        rect_top = image.height
        rect_bottom = new_height
        
        # Draw the rectangle on the specified half
        if side == 'left':
            draw.rectangle([(0, rect_top), (half_width, rect_bottom)], fill=color)
        else:  # right
            draw.rectangle([(half_width, rect_top), (new_width, rect_bottom)], fill=color)
        
        return new_image


    def add_solid_border(image: Image.Image, border_width: int = 5,  border_color: tuple = (0, 0, 0, 255)) -> Image.Image:
        """Adds a solid border around an image by extending it on all sides.
        
        Parameters:
        -----------
        image : Image.Image
            The PIL Image to add a border to
        border_width : int, optional
            Width of the border in pixels, by default 5
        border_color : tuple, optional
            RGBA color for the border, by default (0, 0, 0, 255) (solid black)
            
        Returns:
        --------
        Image.Image
            A new image with the border added around the original image
            
        Usage:
        ------
        from pyphocorehelpers.plotting.media_output_helpers import add_solid_border
        
        # Add a 10-pixel red border around the image
        bordered_image = add_solid_border(
            original_image, 
            border_width=10,
            border_color=(255, 0, 0, 255)  # Red
        )
        
        # Add a default 5-pixel black border
        bordered_image = add_solid_border(original_image)
        """
        # Get original image dimensions
        original_width, original_height = image.size
        
        # Calculate new dimensions
        new_width = original_width + (2 * border_width)
        new_height = original_height + (2 * border_width)
        
        # Create a new image with the border color
        if image.mode == 'RGBA':
            new_image = Image.new('RGBA', (new_width, new_height), border_color)
        else:
            # Convert border_color to RGB if the image is not RGBA
            new_image = Image.new(image.mode, (new_width, new_height), border_color[:3])
        
        # Paste the original image in the center
        new_image.paste(image, (border_width, border_width))
        
        return new_image



@metadata_attributes(short_name=None, tags=['function'], input_requires=[], output_provides=[], uses=['ImageOperationsAndEffects'], used_by=[], creation_date='2025-05-30 04:49', related_items=['ImageOperationsAndEffects'])
class ImagePostRenderFunctionSets:
    """ Provides SETS of operations to be performed on images, specifically for a particular export type (greyscale heatmaps for individual 1D decoders like 'long_LR' vs. RAW_RGBA (pseudo2D)

    Usage:
        from pyphocorehelpers.plotting.media_output_helpers import ImagePostRenderFunctionSets, ImageOperationsAndEffects

        'raw_rgba': HeatmapExportConfig.init_for_export_kind(export_kind=HeatmapExportKind.RAW_RGBA, 
                                                            raw_RGBA_only_parameters = dict(spikes_df=deepcopy(get_proper_global_spikes_df(owning_pipeline_reference)), xbin=deepcopy(a_decoder.xbin), lower_bound_alpha=0.1, drop_below_threshold=1e-3, t_bin_size=time_bin_size, use_four_decoders_version=False), desired_height=desired_height, 
                                                            post_render_image_functions_builder_fn=ImagePostRenderFunctionSets._build_mergedColorDecoders_image_export_functions_dict),

    """
    @classmethod
    def _build_no_op_image_export_functions_dict(cls, a_decoder_decoded_epochs_result: DecodedFilterEpochsResult) -> List[Dict[str, Callable]]:
        """ empty/no-op 
        post_render_image_functions_dict_list: List[Dict[str, Callable]] = _build_image_export_functions_dict(a_decoder_decoded_epochs_result=a_decoder_decoded_epochs_result)

        """
        num_filter_epochs: int = a_decoder_decoded_epochs_result.num_filter_epochs
        # Build post-image-generation callback functions _____________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        post_render_image_functions_dict_list: List = [dict() for i in np.arange(num_filter_epochs)] ## empty dict
        return post_render_image_functions_dict_list


    @classmethod
    def _build_mergedColorDecoders_image_export_functions_dict(cls, a_decoder_decoded_epochs_result: DecodedFilterEpochsResult) -> List[Dict[str, Callable]]:
        """ 
        post_render_image_functions_dict_list: List[Dict[str, Callable]] = _build_image_export_functions_dict(a_decoder_decoded_epochs_result=a_decoder_decoded_epochs_result)

        """
        from neuropy.core.epoch import ensure_dataframe
        from pyphocorehelpers.assertion_helpers import Assert

        num_filter_epochs: int = a_decoder_decoded_epochs_result.num_filter_epochs
        active_filter_epochs: pd.DataFrame = ensure_dataframe(a_decoder_decoded_epochs_result.active_filter_epochs)

        Assert.require_columns(active_filter_epochs, required_columns=['maze_id'])
        is_epoch_pre_post_delta = active_filter_epochs['maze_id'].to_numpy()

        # Build post-image-generation callback functions _____________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #

        fixed_label_region_height: Optional[int] = 520

        # font_size = 144
        # font_size = 96
        # font_size = 72
        font_size = 48

        create_label_function = ImageOperationsAndEffects.create_fn_builder(ImageOperationsAndEffects.add_bottom_label, font_size=font_size, text_color=(255, 255, 255), background_color=(66, 66, 66), fixed_label_region_height=fixed_label_region_height)
        # create_half_width_rectangle_function = ImageOperationsAndEffects.create_fn_builder(ImageOperationsAndEffects.add_half_width_rectangle, height_fraction = 0.1)    
        create_solid_border_function = ImageOperationsAndEffects.create_fn_builder(ImageOperationsAndEffects.add_solid_border) # border_color = (0, 0, 0, 255)

        post_render_image_functions_dict_list: List = []

        for i in np.arange(num_filter_epochs):
            active_captured_single_epoch_result: SingleEpochDecodedResult = a_decoder_decoded_epochs_result.get_result_for_epoch(active_epoch_idx=i)

            # Prepare a multi-line, sideways label _______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
            complete_epoch_identifier_str = ''

            ## mode to use
            curr_epoch_info_dict = active_captured_single_epoch_result.epoch_info_tuple._asdict()
            active_epoch_id: int = curr_epoch_info_dict.get('label', None)
            if active_epoch_id is not None:
                active_epoch_id = int(active_epoch_id)
                # complete_epoch_identifier_str = f"{complete_epoch_identifier_str}lbl[{active_epoch_id:03d}]" # 2025-06-03 - 'p_x_given_n[067]'
                complete_epoch_identifier_str = f"{complete_epoch_identifier_str}L{active_epoch_id:03d}"
            else:
                print(f'falling back to plain epoch IDXs because label was not found!')
                active_epoch_data_IDX: int = active_captured_single_epoch_result.epoch_data_index
                if active_epoch_data_IDX is not None:
                    complete_epoch_identifier_str = f'{complete_epoch_identifier_str}IDX{active_epoch_data_IDX:03d}'

            ## OUTPUTS: complete_epoch_identifier_str
            is_post_delta: bool = (is_epoch_pre_post_delta[i] > 0)

            ## get pre/post delta label:
            earliest_t = active_captured_single_epoch_result.time_bin_edges[0]
            # earliest_t_ms = earliest_t * 1e-3
            earliest_t_str: str = "{:08.4f}".format(earliest_t)
            # earliest_t_str: str = f"{earliest_t:.4f}"

            # Create an image with a label
            # labeled_image = add_bottom_label(original_image, "Time (seconds)", font_size=14)
            # curr_x_axis_label_str: str = f'{earliest_t}'
            # if not is_post_delta:
            #      curr_x_axis_label_str = f'{curr_x_axis_label_str} (pre-delta)'
            # else:
            #     curr_x_axis_label_str = f'{curr_x_axis_label_str} (post-delta)'

            curr_x_axis_label_str: str = f''
            if not is_post_delta:
                #  curr_x_axis_label_str = f'PRE'
                    side = 'left'
                    epoch_rect_color = '#4169E1'

            else:
                # curr_x_axis_label_str = f'POST'
                side = 'right'
                epoch_rect_color = '#DC143C'

            # curr_x_axis_label_str = f"{curr_x_axis_label_str}[{i}]"
            # curr_x_axis_label_str = f"{curr_x_axis_label_str}\n{earliest_t_str}"


            if len(complete_epoch_identifier_str) > 0:
                curr_x_axis_label_str = f"{complete_epoch_identifier_str}: {earliest_t_str}" ## add separator if needed for time
            else:
                curr_x_axis_label_str = f"{earliest_t_str}" # // 2025-06-03 09:10 working

            # curr_post_render_image_functions_dict = {'add_bottom_label': (lambda an_img: add_bottom_label(an_img, curr_x_axis_label_str, font_size=8))}
            curr_post_render_image_functions_dict = {
                # 'add_bottom_label': create_label_function(curr_x_axis_label_str, font_size=font_size, text_color=(255, 255, 255), background_color=(66, 66, 66), text_outline_shadow_color=None, fixed_label_region_height=fixed_label_region_height, debug_print=False),
                'add_bottom_label': create_label_function(curr_x_axis_label_str, font_size=font_size, text_color=epoch_rect_color, background_color=(66, 66, 66), text_outline_shadow_color=None, fixed_label_region_height=fixed_label_region_height, debug_print=False),
                # 'create_solid_border_function': create_solid_border_function(border_width = 10, border_color = epoch_rect_color),
                # 'create_half_width_rectangle_function': create_half_width_rectangle_function(side, epoch_rect_color), ## create rect to indicate pre/post delta
                # 'create_half_width_rectangle_function': create_half_width_rectangle_function(side, epoch_rect_color),
            }
            post_render_image_functions_dict_list.append(curr_post_render_image_functions_dict)                        
        # END for i in np.arange(num_filter_epochs)


        return post_render_image_functions_dict_list



    
def get_array_as_image(img_data: NDArray[ND.Shape["IM_HEIGHT, IM_WIDTH, 4"], np.floating], desired_width: Optional[int] = None, desired_height: Optional[int] = None, export_kind: Optional[HeatmapExportKind] = None,
                        colormap='viridis', skip_img_normalization:bool=False, export_grayscale:bool=False,
                        include_value_labels: bool = False, allow_override_aspect_ratio:bool=False, flip_vertical_axis: bool = False, debug_print=False, **kwargs) -> Image.Image:
    """ Like `save_array_as_image` except it skips the saving to disk. Converts a numpy array to file as a colormapped image
    
    # Usage:
    
        from pyphocorehelpers.plotting.media_output_helpers import get_array_as_image
    
        image = get_array_as_image(img_data, desired_height=100, desired_width=None, skip_img_normalization=True)
        image
        
    # Usage 2:
    
        img_data = np.transpose(img_data, axes=(1, 0, 2)) # Convert to (H, W, 4)
        kwargs = {}
        desired_height = 400
        desired_width = None
        skip_img_normalization = True
        _out_img = get_array_as_image(img_data, desired_height=desired_height, desired_width=desired_width, skip_img_normalization=skip_img_normalization, export_kind=HeatmapExportKind.RAW_RGBA, **kwargs)
        _out_img

                
    """
    from pyphoplacecellanalysis.Pho2D.data_exporting import HeatmapExportKind

    if export_kind is None:
        if export_grayscale:
            export_kind = HeatmapExportKind.GREYSCALE
        else:
            export_kind = HeatmapExportKind.COLORMAPPED
    else:
        if debug_print:
            print(f'export_kind: {export_kind} explicitly provided, so ignoring export_grayscale: {export_grayscale}')



    if np.ndim(img_data) < 2:
        img_data = np.atleast_2d(img_data).T # (50, ) -> (50, 1)



    # Assuming `your_array` is your numpy array
    if export_kind.value == HeatmapExportKind.GREYSCALE.value:
    # if export_grayscale:
        # Convert to grayscale (normalize if needed)
        assert (colormap is None) or (colormap == 'viridis'), f"colormap should not be specified when export_grayscale=True" # (default 'viridis' is safely ignored)
        
        if skip_img_normalization:
            print(f'WARN: when `export_grayscale == True`, `skip_img_normalization == True` makes no sense and will be ignored.')
            
        norm_array = img_data_to_greyscale(img_data, min_val = kwargs.pop('vmin', None), max_val = kwargs.pop('vmax', None), should_invert=True)
        # Scale to 0-255 and convert to uint8
        image = Image.fromarray(norm_array, mode='L') # .shape: (59, 4, 67)
        

    elif export_kind.value == HeatmapExportKind.COLORMAPPED.value:
        ## Color export mode!
        assert (colormap is not None)
        # Get the specified colormap
        colormap = plt.get_cmap(colormap)

        if skip_img_normalization:
            norm_array = img_data
        else:
            # Normalize your array to 0-1 using nan-aware functions
            min_val = kwargs.pop('vmin', None)
            if min_val is None:
                min_val = np.nanmin(img_data)
                
            max_val = kwargs.pop('vmax', None)
            if max_val is None:
                max_val = np.nanmax(img_data)

            assert (min_val < max_val), f"min_val: {min_val}, min_val: {min_val}, img_data: {img_data}"
            ptp_val = max_val - min_val
            norm_array = (img_data - min_val) / ptp_val

        # Apply colormap
        image_array = colormap(norm_array)

        # Convert to PIL image and remove alpha channel
        image = Image.fromarray((image_array[:, :, :3] * 255).astype(np.uint8)) # TypeError: Cannot handle this data type: (1, 1, 3, 4), |u1  || 2025-06-04 Failing for 1D input arrts which end up as (1D, 4): IndexError: too many indices for array: array is 2-dimensional, but 3 were indexed
        

    elif export_kind.value == HeatmapExportKind.RAW_RGBA.value:
        ## Raw ready to use RGBA image is passed in:
        # Convert to PIL image and remove alpha channel
        image = Image.fromarray((img_data * 255).astype(np.uint8)) # TypeError: Cannot handle this data type: (1, 1, 3, 4), |u1
        assert skip_img_normalization == True, f"it does not make sense to re-normalize the RGBA image"
        norm_array = img_data
        

    else:
        raise NotImplementedError(f"export_kind: {export_kind}")    
    
    ## OUTPUT: image: Image

    # Optionally flip the image vertically
    if flip_vertical_axis:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        
    if ((desired_width is None) and (desired_height is None)):
        desired_width = 1024 # set a default arbitrarily

    if desired_width is not None:
        # Specify width
        assert ((desired_height is None) or allow_override_aspect_ratio), f"please don't provide both width and height, the other will be calculated automatically. If you meant to force this set `allow_override_aspect_ratio=True` to override."
        if (desired_height is None):
            # Calculate height to preserve aspect ratio
            desired_height = int(desired_width * norm_array.shape[0] / norm_array.shape[1])
        
    elif (desired_height is not None):
        # Specify height:
        assert ((desired_width is None) or allow_override_aspect_ratio), f"please don't provide both width and height, the other will be calculated automatically. If you meant to force this set `allow_override_aspect_ratio=True` to override."
        if (desired_width is None):
            # Calculate width to preserve aspect ratio
            desired_width = int(desired_height * norm_array.shape[1] / norm_array.shape[0])
    else:
        raise ValueError("you must specify width or height of the output image")

    # Resize image
    # image = image.resize((new_width, new_height), Image.LANCZOS)
    image = image.resize((int(desired_width), int(desired_height)), Image.NEAREST)

    if include_value_labels:
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        def draw_text_with_border(draw, x, y, text, font, fill):
            # Draw shadow/border (using black color)
            shadow_color = (0, 0, 0)
            draw.text((x - 1, y - 1), text, font=font, fill=shadow_color)
            draw.text((x + 1, y - 1), text, font=font, fill=shadow_color)
            draw.text((x - 1, y + 1), text, font=font, fill=shadow_color)
            draw.text((x + 1, y + 1), text, font=font, fill=shadow_color)
            # Draw text itself
            draw.text((x, y), text, font=font, fill=fill)

        # Iterate over pixels and annotate with their values
        for y in range(norm_array.shape[0]):
            for x in range(norm_array.shape[1]):
                value = norm_array[y, x]
                text = f'{value:.2f}'
                # text = "{:07.2f}".format(value) # f'{value:.2f}'
                text_x = x * desired_width // norm_array.shape[1]
                text_y = y * desired_height // norm_array.shape[0]

                # Draw the text with border
                draw_text_with_border(draw, text_x, text_y, text, font, fill=(255, 255, 255))

                # Rotate the text by 45 degrees (angle)
                text_image = Image.new('RGBA', (image.size[0], image.size[1]), (255, 255, 255, 0))
                text_draw = ImageDraw.Draw(text_image)
                draw_text_with_border(text_draw, text_x, text_y, text, font, fill=(255, 255, 255))
                # text_image = text_image.rotate(45, expand=1)
                image.paste(text_image, (0, 0), text_image)


    # ==================================================================================================================================================================================================================================================================================== #
    # Call the post-render functions, which do things like: Add bottom time label, adding colored border, etc                                                                                                                                                                                                                                                                #
    # ==================================================================================================================================================================================================================================================================================== #    
    post_render_image_functions = kwargs.pop('post_render_image_functions', {})
    for a_render_fn_name, a_render_fn in post_render_image_functions.items():
        if debug_print:
            print(f'\tperforming: {a_render_fn_name}')
        image = a_render_fn(image)

    return image


def save_array_as_image(img_data, desired_width: Optional[int] = 1024, desired_height: Optional[int] = None, export_kind: Optional[HeatmapExportKind] = None, out_path='output/numpy_array_as_image.png', colormap:str='viridis', skip_img_normalization:bool=False, export_grayscale:bool=False, include_value_labels: bool = False, allow_override_aspect_ratio:bool=False, flip_vertical_axis: bool = False, **kwargs) -> Tuple[Image.Image, Path]:
    """ Exports a numpy array to file as a colormapped image
    
    # Usage:
    
        from pyphocorehelpers.plotting.media_output_helpers import save_array_as_image
    
        image, out_path = save_array_as_image(img_data, desired_height=100, desired_width=None, skip_img_normalization=True)
        image
        
        
                
    """
    image: Image.Image = get_array_as_image(img_data=img_data, desired_width=desired_width, desired_height=desired_height, export_kind=export_kind, colormap=colormap, skip_img_normalization=skip_img_normalization, export_grayscale=export_grayscale, include_value_labels=include_value_labels, allow_override_aspect_ratio=allow_override_aspect_ratio, flip_vertical_axis=flip_vertical_axis, **kwargs)
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
        output_img, output_path = get_array_as_image_stack(out_figs_paths, offset=55, single_image_alpha_level=0.85)

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
    output_width = width + abs(offset) * (len(imgs) - 1)
    output_height = height + abs(offset) * (len(imgs) - 1)

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

    output_width = width + abs(offset) * (len(imgs) - 1)
    output_height = height + abs(offset) * (len(imgs) - 1)

    output_img = Image.new('RGBA', (output_width, output_height))

    for i, img in enumerate(imgs):
        if add_border:
            img = add_border(img, border_size=border_size, border_color=border_color)
        if add_shadow:
            img = add_shadow(img, offset=shadow_offset, shadow_color=shadow_color)
        output_img.paste(img, (i * offset, i * offset), img)

    return output_img



def vertical_image_stack(imgs: List[Image.Image], padding=10, v_overlap: int=0, separator_color=None) -> Image.Image:
    """ Builds a stack of images into a vertically concatenated image.
    offset = 10  # your desired offset

    Usage:
        from pyphocorehelpers.plotting.media_output_helpers import vertical_image_stack, horizontal_image_stack

        # Open the images
        _raster_imgs = [Image.open(i) for i in _out_rasters_save_paths]
        _out_vstack = vertical_image_stack(_raster_imgs, padding=5)
        _out_vstack
        
    """
    # Ensure all images are in RGBA mode
    imgs = [img.convert('RGBA') if img.mode != 'RGBA' else img for img in imgs]
    
    widths = np.array([img.size[0] for img in imgs])
    heights = np.array([img.size[1] for img in imgs])

    # Create a new image with size larger than original ones, considering offsets
    # output_height = (np.sum(heights) + (padding * (len(imgs) - 1))) - v_overlap
    output_height = np.sum(heights) - v_overlap
    if isinstance(padding, str) and padding.endswith('%'):
        # if it's a string like '1%', it specifies the desired width in terms of the total image height
        padding_percent = padding.strip('%')
        padding_percent = float(padding_percent)
        assert (padding <= 100.0) and (padding >= 0.0), f"padding: {padding} is invalid! Should be a percentage of the total image height like '1%'."
        padding = (padding_percent * output_height) / float(len(imgs) - 1) # padding in px

    output_total_padding_height: float = (padding * (len(imgs) - 1))
    output_height = output_height + output_total_padding_height
    output_width = np.max(widths)
    # print(f'output_width: {output_width}, output_height: {output_height}')
    output_img = Image.new('RGBA', (output_width, output_height))
    cum_height = 0
    for i, img in enumerate(imgs):
        curr_img_width, curr_img_height = img.size
        output_img.paste(img, (0, cum_height), img)
        # cum_height += (curr_img_height+padding) - v_overlap
        cum_height += curr_img_height - v_overlap ## add the current image height
        if (separator_color is not None) and (padding > 0):
            ## fill the between-image area with a separator_color
            _tmp_separator_img = Image.new('RGBA', (output_width, padding), separator_color)
            output_img.paste(_tmp_separator_img, (0, cum_height))
                    
        cum_height += padding

    return output_img


def horizontal_image_stack(imgs: List[Image.Image], padding=10, separator_color=None) -> Image.Image:
    """ Builds a stack of images into a horizontally concatenated image.
    offset = 10  # your desired offset

    Usage:
        from pyphocorehelpers.plotting.media_output_helpers import vertical_image_stack, horizontal_image_stack

        # Open the images
        _raster_imgs = [Image.open(i) for i in _out_rasters_save_paths]
        # _out_vstack = vertical_image_stack(_raster_imgs, padding=5)
        # _out_vstack
        _out_hstack = horizontal_image_stack(_raster_imgs, padding=5)
        _out_hstack

    """
    # Ensure all images are in RGBA mode
    imgs = [img.convert('RGBA') if img.mode != 'RGBA' else img for img in imgs]
    
    ## get the sizes of each image
    widths = np.array([img.size[0] for img in imgs])
    heights = np.array([img.size[1] for img in imgs])

    # Create a new image with size larger than original ones, considering offsets
    output_width = np.sum(widths)
    if isinstance(padding, str) and padding.endswith('%'):
        # if it's a string like '1%', it specifies the desired width in terms of the total image width
        padding_percent = padding.strip('%')
        padding_percent = float(padding_percent)
        assert (padding <= 100.0) and (padding >= 0.0), f"padding: {padding} is invalid! Should be a percentage of the total image width like '1%'."
        padding = (padding_percent * output_width) / float(len(imgs) - 1) # padding in px

    output_total_padding_width: float = (padding * (len(imgs) - 1))
    output_width = output_width + output_total_padding_width
    output_height = np.max(heights)
    
    # print(f'output_width: {output_width}, output_height: {output_height}')
    output_img = Image.new('RGBA', (output_width, output_height))
    cum_width = 0
    for i, img in enumerate(imgs):
        curr_img_width, curr_img_height = img.size
        output_img.paste(img, (cum_width, 0), img)
        # cum_height += curr_img_height
        cum_width += curr_img_width ## add the current image width
        if (separator_color is not None) and (padding > 0):
            ## fill the between-image area with a separator_color
            _tmp_separator_img = Image.new('RGBA', (padding, output_height), separator_color)
            output_img.paste(_tmp_separator_img, (cum_width, 0))
    
        cum_width += padding

    return output_img


def image_grid(imgs: List[List[Image.Image]], v_padding=5, h_padding=5) -> Image.Image:
    """ Builds a stack of images into a horizontally concatenated image.
    offset = 10  # your desired offset

    Usage:
        from pyphocorehelpers.plotting.media_output_helpers import vertical_image_stack, horizontal_image_stack, image_grid

        # Open the images
        _raster_imgs = [Image.open(i) for i in _out_rasters_save_paths]
        # _out_vstack = vertical_image_stack(_raster_imgs, padding=5)
        # _out_vstack
        _out_hstack = horizontal_image_stack(_raster_imgs, padding=5)
        _out_hstack

    """
    # Ensure all images are in RGBA mode
    imgs = [img.convert('RGBA') if img.mode != 'RGBA' else img for img in imgs]
    
    # Assume all images are the same size
    width, height = imgs[0].size
    
    widths = np.array([img.size[0] for img in imgs])
    heights = np.array([img.size[1] for img in imgs])

    # Create a new image with size larger than original ones, considering offsets
    output_width = np.sum(widths) + (padding * (len(imgs) - 1))
    output_height = np.max(heights)
    # print(f'output_width: {output_width}, output_height: {output_height}')
    output_img = Image.new('RGBA', (output_width, output_height))
    # cum_height = 0
    cum_width = 0
    for i, img in enumerate(imgs):
        curr_img_width, curr_img_height = img.size
        output_img.paste(img, (cum_width, 0), img)
        # cum_height += (curr_img_height+padding)
        cum_width += (curr_img_width+padding)

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
@function_attributes(short_name=None, tags=['cv2'], input_requires=[], output_provides=[], uses=['cv2'], used_by=['PosteriorExporting.save_posterior_to_video'], creation_date='2024-09-06 11:34', related_items=[])
def save_array_as_video(array, video_filename='output/videos/long_short_rel_entr_curves_frames.mp4', fps=30.0, isColor=False, colormap=None, skip_img_normalization=False, debug_print=False, progress_print=True):
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
    import cv2
    if colormap is None:
        colormap = cv2.COLORMAP_VIRIDIS
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

    ## Check the path exists first:
    video_filepath: Path = Path(video_filename).resolve()
    video_parent_path = video_filepath.parent
    # assert video_parent_path
    if (not video_parent_path.exists()):
        print(f'target output directory (video_parent_path: "{video_parent_path}") does not exist. Creating it.')
        video_parent_path.mkdir(exist_ok=True)

    # initialize video writer
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter(str(video_filename), fourcc, fps, (width, height))

    # new frame after each addition of water
    progress_print_every_n_frames: int = 15 # print progress only once every 15 frames so it doesn't spam the output log
    for i in np.arange(n_frames):
        if progress_print and (i % 15 == 0):
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


@function_attributes(short_name=None, tags=['cv2'], input_requires=[], output_provides=[], uses=['cv2'], used_by=[], creation_date='2024-09-06 11:33', related_items=[])
def colormap_and_save_as_video(array, video_filename='output.avi', fps=30.0, colormap=None):
    import cv2
    if colormap is None:
        colormap = cv2.COLORMAP_VIRIDIS
    # array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)
    # color_array = cv2.applyColorMap(array, colormap)
    return save_array_as_video(array, video_filename=video_filename, fps=fps, isColor=True, colormap=colormap)
    
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



@function_attributes(short_name=None, tags=['cv2'], input_requires=[], output_provides=[], uses=['cv2'], used_by=[], creation_date='2024-09-06 11:33', related_items=[])
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
    import cv2

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
            

def figure_to_pil_image(a_fig: Union[PlotlyFigure, FigureBase], format="png", **kwargs) -> Optional[Image.Image]:
    """ Convert a Matplotlib Figure to a PIL Image.

    Parameters:
        fig (matplotlib.figure.Figure): The Matplotlib figure to convert.

    Returns:
        PIL.Image.Image: The resulting PIL Image.
        
    Usage:
        from pyphocorehelpers.plotting.media_output_helpers import figure_to_pil_image
    
        fig_img = figure_to_pil_image(a_fig=fig)
    """
    _fig_save_fn = None
    img = None
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
        # Optionally, convert the image to RGB (if not already in that mode)
        if img.mode != 'RGB':
            img = img.convert('RGB')    
        # Close the buffer and figure to free resources
        buf.close()
    
    return img






@function_attributes(short_name=None, tags=['colormap', 'grayscale', 'image'], input_requires=[], output_provides=[], uses=[], used_by=['blend_images'], creation_date='2024-08-21 00:00', related_items=[])
def apply_colormap(image: np.ndarray, color: tuple) -> np.ndarray:
    colored_image = np.zeros((*image.shape, 3), dtype=np.float32)
    for i in range(3):
        colored_image[..., i] = image * color[i]
    return colored_image

@function_attributes(short_name=None, tags=['image'], input_requires=[], output_provides=[], uses=['apply_colormap'], used_by=[], creation_date='2024-08-21 00:00', related_items=[])
def blend_images(images: list, cmap=None) -> np.ndarray:
    """ Tries to pre-combine images to produce an output image of the same size

    # 'coolwarm'
    images = [a_seq_mat.todense().T for i, a_seq_mat in enumerate(sequence_frames_sparse)]
    blended_image = blend_images(images)
    # blended_image = blend_images(images, cmap='coolwarm')
    blended_image

    # blended_image = Image.fromarray(blended_image, mode="RGB")
    # # blended_image = get_array_as_image(blended_image, desired_height=100, desired_width=None, skip_img_normalization=True)
    # blended_image

    """
    from matplotlib.colors import Normalize
    
    if cmap is None:
        # Non-colormap mode:
        # Ensure images are in the same shape
        combined_image = np.zeros_like(images[0], dtype=np.float32)

        for img in images:
            combined_image += img.astype(np.float32)

    else:
        # colormap mode
        # Define a colormap (blue to red)
        cmap = plt.get_cmap(cmap)
        norm = Normalize(vmin=0, vmax=(len(images) - 1))

        combined_image = np.zeros((*images[0].shape, 3), dtype=np.float32)

        for i, img in enumerate(images):
            color = cmap(norm(i))[:3]  # Get RGB color from colormap
            colored_image = apply_colormap(img, color)
            combined_image += colored_image

    combined_image = np.clip(combined_image, 0, 255)  # Ensure pixel values are within valid range
    return combined_image.astype(np.uint8)



from pypdf import PdfReader, PdfWriter, Transformation ## for PDFHelpers
from pathlib import Path


@metadata_attributes(short_name=None, tags=['PDF', 'export', 'filesystem', 'concatenate', 'combine'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-06-19 11:48', related_items=[])
class PDFHelpers:
    """ Helpers for Concatenating single-figure PDFs and other things
    
    from pyphocorehelpers.plotting.media_output_helpers import PDFHelpers
    
    """

    @classmethod
    def concatenate_pdfs_horizontally(cls, pdf_paths: List[Union[str, Path]], output_path: Union[str, Path], padding: float = 0):
        """
        Horizontally concatenate multiple single-page PDFs.

        Parameters:
        -----------
        pdf_paths : List[Union[str, Path]]
            List of paths to PDF files to concatenate horizontally
        output_path : Union[str, Path]
            Path for the output concatenated PDF
        padding : float, optional
            Spacing between PDFs in points, by default 0

        Returns:
        --------
        Path
            Path to the created concatenated PDF

        Usage:
        ------
        from pyphocorehelpers.plotting.media_output_helpers import PDFHelpers

        # Concatenate multiple PDFs horizontally
        pdf_files = [pdf1_path, pdf2_path, pdf3_path]
        output_file = PDFHelpers.concatenate_pdfs_horizontally(
            pdf_paths=pdf_files,
            output_path="combined_output.pdf",
            padding=10
        )

        # Concatenate just two PDFs (backwards compatible)
        output_file = PDFHelpers.concatenate_pdfs_horizontally(
            pdf_paths=[left_pdf_path, right_pdf_path],
            output_path="combined_output.pdf"
        )
        """
        if not pdf_paths:
            raise ValueError("pdf_paths cannot be empty")

        if len(pdf_paths) == 1:
            # If only one PDF, just copy it
            import shutil
            shutil.copy2(pdf_paths[0], output_path)
            return Path(output_path)

        # Convert paths to Path objects
        pdf_paths = [Path(p) for p in pdf_paths]
        output_path = Path(output_path)

        # Read all PDFs and get their pages
        readers = [PdfReader(pdf_path) for pdf_path in pdf_paths]
        pages = [reader.pages[0] for reader in readers]  # Assume single-page PDFs

        # Get dimensions of all pages
        widths = [float(page.mediabox.width) for page in pages]
        heights = [float(page.mediabox.height) for page in pages]

        # Calculate new page dimensions
        total_padding = padding * (len(pages) - 1) if len(pages) > 1 else 0
        new_width = sum(widths) + total_padding
        new_height = max(heights)

        # Create new PDF with combined page
        writer = PdfWriter()
        new_page = writer.add_blank_page(width=new_width, height=new_height)

        # Add pages horizontally with padding
        current_x_offset = 0
        for i, (page, width) in enumerate(zip(pages, widths)):
            if current_x_offset > 0:
                # Create transformation to translate the page
                transformation = Transformation().translate(tx=current_x_offset, ty=0)
                new_page.merge_transformed_page(page, transformation)
            else:
                # First page at origin
                new_page.merge_page(page)

            # Update offset for next page
            current_x_offset += width + padding

        # Save the result
        with open(output_path, 'wb') as output_file:
            writer.write(output_file)

        print(f"Successfully concatenated {len(pdf_paths)} PDFs to: {output_path}")
        return output_path


    @classmethod
    def concatenate_pdfs_vertically(cls, pdf_paths: List[Union[str, Path]], output_path: Union[str, Path], padding: float = 0):
        """
        Vertically concatenate multiple single-page PDFs.

        Parameters:
        -----------
        pdf_paths : List[Union[str, Path]]
            List of paths to PDF files to concatenate vertically
        output_path : Union[str, Path]
            Path for the output concatenated PDF
        padding : float, optional
            Spacing between PDFs in points, by default 0

        Returns:
        --------
        Path
            Path to the created concatenated PDF
        """
        if not pdf_paths:
            raise ValueError("pdf_paths cannot be empty")

        if len(pdf_paths) == 1:
            import shutil
            shutil.copy2(pdf_paths[0], output_path)
            return Path(output_path)

        # Convert paths to Path objects
        pdf_paths = [Path(p) for p in pdf_paths]
        output_path = Path(output_path)

        # Read all PDFs and get their pages
        readers = [PdfReader(pdf_path) for pdf_path in pdf_paths]
        pages = [reader.pages[0] for reader in readers]

        # Get dimensions of all pages
        widths = [float(page.mediabox.width) for page in pages]
        heights = [float(page.mediabox.height) for page in pages]

        # Calculate new page dimensions
        total_padding = padding * (len(pages) - 1) if len(pages) > 1 else 0
        new_width = max(widths)
        new_height = sum(heights) + total_padding

        # Create new PDF with combined page
        writer = PdfWriter()
        new_page = writer.add_blank_page(width=new_width, height=new_height)

        # Add pages vertically with padding (from top to bottom)
        current_y_offset = new_height  # Start from top
        for i, (page, height) in enumerate(zip(pages, heights)):
            current_y_offset -= height  # Move down by page height

            if current_y_offset != (new_height - height):  # Not the first page
                # Create transformation to translate the page
                transformation = Transformation().translate(tx=0, ty=current_y_offset)
                new_page.merge_transformed_page(page, transformation)
            else:
                # First page
                transformation = Transformation().translate(tx=0, ty=current_y_offset)
                new_page.merge_transformed_page(page, transformation)

            # Update offset for next page (subtract padding)
            current_y_offset -= padding

        # Save the result
        with open(output_path, 'wb') as output_file:
            writer.write(output_file)

        print(f"Successfully concatenated {len(pdf_paths)} PDFs vertically to: {output_path}")
        return output_path

    @function_attributes(short_name=None, tags=['pdf','scale', 'resize'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-07-03 16:58', related_items=[])
    @classmethod
    def scale_pdf(cls, input_path: str, output_path: str, scale_factor: float = 0.5):
        """ Scale all aspects of a vector PDF uniformly by `scale_factor`.
        
        Parameters:
            input_path (str): Path to input PDF file.
            output_path (str): Path to save scaled output PDF.
            scale_factor (float): Scaling factor (e.g., 0.5 for 50% size).
            
            
        Usage:
        
            from pyphocorehelpers.plotting.media_output_helpers import PDFHelpers

            # Scale input.pdf down to 50% size, scaling around the center
            PDFHelpers.scale_pdf(input_path=_fig2_final_combined_output_path, output_path=_fig2_final_combined_output_path.with_stem(f"scaled_output"), scale_factor=0.5, center=False)

        """
        from pypdf import PdfReader, PdfWriter, Transformation ## for PDFHelpers
        from pypdf.generic import RectangleObject
        reader = PdfReader(input_path)
        writer = PdfWriter()

        for page in reader.pages:
            # Original size
            w = float(page.mediabox.width)
            h = float(page.mediabox.height)
            transform = Transformation().scale(scale_factor)

            # Apply transformation
            page.add_transformation(transform)

            # Compute new bounding box
            new_w = w * scale_factor
            new_h = h * scale_factor

            # Create new rectangle
            new_box = RectangleObject([0, 0, new_w, new_h])

            # Update MediaBox and CropBox
            page.mediabox = new_box
            page.cropbox = new_box

            writer.add_page(page)

        # Write output
        with open(output_path, "wb") as f_out:
            writer.write(f_out)
            