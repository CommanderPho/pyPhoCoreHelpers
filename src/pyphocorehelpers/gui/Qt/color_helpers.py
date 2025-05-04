 from __future__ import annotations # prevents having to specify types for typehinting as strings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    ## typehinting only imports here
    from matplotlib.colors import LinearSegmentedColormap


from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from nptyping import NDArray
from copy import deepcopy
import numpy as np
import pandas as pd
from neuropy.utils.mixins.enum_helpers import StringLiteralComparableEnum
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
import pyphoplacecellanalysis.External.pyqtgraph as pg
from qtpy import QtGui # for QColor
from qtpy.QtGui import QColor, QBrush, QPen



def debug_print_color(color: QColor):
    if color.alphaF() == 1.0:
        color_hex_format = QColor.HexRgb
    else:
        color_hex_format = QColor.HexArgb
    print(f'rgbaF: {color.getRgbF()}, HexARgb: {color.name(color_hex_format)}')
    

def build_adjusted_color(color: QColor, hue_shift:float=0.0, saturation_scale:float=1.0, value_scale:float=1.0, alpha_scale: float=1.0, wants_return_as_hex_string:bool=False, wants_hex_string_include_alpha: bool=True):
    """ Builds a copy of the color QColor with optionally modified HSV properties
    Example:
        from pyphocorehelpers.gui.Qt.color_helpers import build_adjusted_color
    
        debug_print_color(curr_color)
        curr_color_copy = build_adjusted_color(curr_color, hue_shift=0.0, saturation_scale=0.35, value_scale=1.0)
        debug_print_color(curr_color_copy)

    """
    if isinstance(color, str):
        color = QtGui.QColor(color) ## convert to QColor if needed
    
    curr_color_copy = color.convertTo(QColor.Hsv) # makes a copy of color
    # curr_color_copy.setHsv(curr_color_copy.hue(),curr_color_copy.saturation(), curr_color_copy.value())
    # np.clip(v, 0.0, 1.0) ensures the values are between 0.0 and 1.0
    curr_color_copy.setHsvF(np.clip((curr_color_copy.hueF() + hue_shift), 0.0, 1.0),
                            np.clip((saturation_scale*curr_color_copy.saturationF()), 0.0, 1.0),
                            np.clip((value_scale * curr_color_copy.valueF()), 0.0, 1.0))
    curr_color_copy.setAlphaF(np.clip((alpha_scale*curr_color_copy.alphaF()), 0.0, 1.0))
    # curr_color_copy.setAlphaF(color.alphaF())
    assert curr_color_copy.isValid(), "Constructed color is invalid!"
    
    if not wants_return_as_hex_string:
        # return QColor
        return curr_color_copy
    else:
        ## convert to a hex string to return
        return ColorFormatConverter.qColor_to_hexstring(curr_color_copy, include_alpha=wants_hex_string_include_alpha)


# @function_attributes(short_name=None, tags=['color', 'HSV', 'conversion'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-29 13:53', related_items=[])
def calculate_hsv_shift(colorA: Union[str, QColor], colorB: Union[str, QColor], debug_print=False) -> Dict[str, float]:
    """ Computes the HSV shift/scale factors between two colors
    from pyphocorehelpers.gui.Qt.color_helpers import calculate_hsv_shift
            
    NOTE: outputs are suitable for direct input into `build_adjusted_color(...)`

    Usage:    
    
        from pyphocorehelpers.gui.Qt.color_helpers import calculate_hsv_shift
        
        hsv_diff = calculate_hsv_shift(colorA='#1f02c2' , colorB='#13007f', debug_print=True) # hsvB - hsvA
        hsv_diff # {'hue_shift': -0.00022222222222223476, 'saturation_scale': 1.0104226090442343, 'value_scale': 0.654639175257732, 'alpha_scale': 1.0}
                
    """
    if isinstance(colorA, str):
        colorA = QtGui.QColor(colorA)
    if isinstance(colorB, str):
        colorB = QtGui.QColor(colorB)
        
    if debug_print:
        debug_print_color(colorA)
        debug_print_color(colorB)
    
    hsvA = np.array(colorA.getHsvF()) # (0.6918333333333333, 0.9896849011978333, 0.7607843137254902, 1.0)
    hsvB = np.array(colorB.getHsvF()) # (0.6918333333333333, 0.9896849011978333, 0.7607843137254902, 1.0)
    assert len(hsvA) == 4
    assert len(hsvB) == 4
    if debug_print:
        print(f'hsvA: {hsvA}\nhsvB: {hsvB}')
    # hsv_diff: NDArray = (hsvB - hsvA)
    
    # saturation_diff = max(hsvB[1], hsvB[0])
    
    hsv_diff = np.array([(hsvB[0] - hsvA[0]), np.nan_to_num((hsvB[1] / hsvA[1]), nan=1.0), np.nan_to_num((hsvB[2] / hsvA[2]), nan=1.0), np.nan_to_num((hsvB[3] / hsvA[3]), nan=1.0)])  
    
    assert len(hsv_diff) == 4
    return dict(zip(['hue_shift', 'saturation_scale', 'value_scale', 'alpha_scale'], hsv_diff)) # dict(hue_shift=0.0, saturation_scale=1.0, value_scale=1.0, alpha_scale=1.0)

    


def adjust_saturation(rgb, saturation_factor: float):
    """ adjusts the rgb colors by the saturation_factor by converting to HSV space.
    
    """
    import matplotlib.colors as mcolors
    import colorsys
    # Convert RGB to HSV
    hsv = mcolors.rgb_to_hsv(rgb)

    if np.ndim(hsv) < 3:
        # Multiply the saturation by the saturation factor
        hsv[:, 1] *= saturation_factor
        
        # Clip the saturation value to stay between 0 and 1
        hsv[:, 1] = np.clip(hsv[:, 1], 0, 1)
        
    else: 
        # Multiply the saturation by the saturation factor
        hsv[:, :, 1] *= saturation_factor
        # Clip the saturation value to stay between 0 and 1
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 1)
    
    # Convert back to RGB
    return mcolors.hsv_to_rgb(hsv)



@metadata_attributes(short_name=None, tags=['colormap', 'color', 'static'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-08-30 19:55', related_items=[])
class ColormapHelpers:
    """ 
    from pyphocorehelpers.gui.Qt.color_helpers import ColormapHelpers
        
    ColormapHelpers.
    """
    # Create a function to modify the colormap's alpha channel
    @classmethod
    def create_transparent_colormap(cls, cmap_name: Optional[str]=None, color_literal_name: Optional[str]=None, lower_bound_alpha=0.1) -> NDArray:
        """ 
        Usage:
            additional_cmap_names = dict(zip(TrackTemplates.get_decoder_names(), ['red', 'purple', 'green', 'orange'])) # {'long_LR': 'red', 'long_RL': 'purple', 'short_LR': 'green', 'short_RL': 'orange'}

            long_epoch_config = long_short_display_config_manager.long_epoch_config.as_pyqtgraph_kwargs()
            short_epoch_config = long_short_display_config_manager.short_epoch_config.as_pyqtgraph_kwargs()

            color_dict = {'long_LR': long_epoch_config['brush'].color(), 'long_RL': apply_LR_to_RL_adjustment(long_epoch_config['brush'].color()),
                            'short_LR': short_epoch_config['brush'].color(), 'short_RL': apply_LR_to_RL_adjustment(short_epoch_config['brush'].color())}
            additional_cmap_names = {k: ColorFormatConverter.qColor_to_hexstring(v) for k, v in color_dict.items()}

            additional_cmaps = {k: ColormapHelpers.create_transparent_colormap(color_literal_name=v, lower_bound_alpha=0.1) for k, v in additional_cmap_names.items()}
        
        """
        from pyphoplacecellanalysis.External.pyqtgraph.colormap import ColorMap
        
        # Get the base colormap
        assert (cmap_name is not None) or (color_literal_name is not None)
        if color_literal_name is not None:
            assert cmap_name is None
            cmap = pg.ColorMap(np.array([0.0, 1.0]), np.array([pg.mkColor(color_literal_name).getRgb()[:3] + (0,), pg.mkColor(color_literal_name).getRgb()[:3] + (255,)], dtype=np.ubyte))
        else:
            assert cmap_name is not None
            cmap = pg.colormap.get(cmap_name, source='matplotlib')

        # Create a lookup table with the desired number of points (default 256)
        lut = cmap.getLookupTable(alpha=True, mode=ColorMap.BYTE)
        
        # `ColorMap.BYTE` (0 to 255), `ColorMap.FLOAT` (0.0 to 1.0) or `ColorMap.QColor`.
        
        # Modify the alpha values
        alpha_channel = lut[:, 3]  # Extract the alpha channel (4th column)
        alpha_channel = np.linspace(lower_bound_alpha, 1, len(alpha_channel))  # Linear alpha gradient from lower_bound_alpha to 1
        lut[:, 3] = (alpha_channel * 255).astype(np.uint8)  # Convert to 0-255 range
        
        return lut
        
    @classmethod
    def desaturate_colormap(cls, cmap, desaturation_factor: float):
        """
        Desaturate a colormap by a given factor.

        Parameters:
        - cmap: A Matplotlib colormap instance.
        - desaturation_factor: A float between 0 and 1, with 0 being fully desaturated (greyscale)
        and 1 being fully saturated (original colormap colors).

        Returns:
        - new_cmap: A new Matplotlib colormap instance with desaturated colors.

        Usage:
            # Load the existing 'viridis' colormap
            viridis = plt.cm.get_cmap('viridis')
            # Create a desaturated version of 'viridis'
            desaturation_factors = np.linspace(start=1.0, stop=0.0, num=6)
            desaturated_viridis = [ColormapHelpers.desaturate_colormap(viridis, a_desaturation_factor) for a_desaturation_factor in desaturation_factors]
            for a_cmap in desaturated_viridis:
                display(a_cmap)

                
        """
        import matplotlib.pyplot as plt
        # Get the colormap colors and the number of entries in the colormap
        cmap_colors = cmap(np.arange(cmap.N))
        
        # Convert RGBA to RGB
        cmap_colors_rgb = cmap_colors[:, :3]
        
        # Create an array of the same shape filled with luminance values
        # The luminance of a color is a weighted average of the R, G, and B values
        # These weights are based on how the human eye perceives color intensity
        luminance = np.dot(cmap_colors_rgb, [0.299, 0.587, 0.114]).reshape(-1, 1)
        
        # Create a grayscale version of the colormap
        grayscale_cmap = np.hstack([luminance, luminance, luminance])
        
        # Blend the original colormap with the grayscale version
        blended_cmap = desaturation_factor * cmap_colors_rgb + (1 - desaturation_factor) * grayscale_cmap
        
        # Add the alpha channel back and create a new colormap
        new_cmap_colors = np.hstack([blended_cmap, cmap_colors[:, 3:]])
        new_cmap = plt.matplotlib.colors.ListedColormap(new_cmap_colors)
        
        return new_cmap


    @classmethod
    def make_saturating_red_cmap(cls, time: float, N_colors:int=256, min_alpha: float=0.0, max_alpha: float=0.82, debug_print:bool=False) -> LinearSegmentedColormap:
        """ time is between 0.0 and 1.0 

        Usage: Test Example:
            from pyphocorehelpers.gui.Qt.color_helpers import ColormapHelpers

            n_time_bins = 5
            cmaps = [ColormapHelpers.make_saturating_red_cmap(float(i) / float(n_time_bins - 1)) for i in np.arange(n_time_bins)]
            for cmap in cmaps:
                cmap
                
        Usage:
            # Example usage
            # You would replace this with your actual data and timesteps
            data = np.random.rand(10, 10)  # Sample data
            n_timesteps = 5  # Number of timesteps

            # Plot data with increasing red for each timestep
            fig, axs = plt.subplots(1, n_timesteps, figsize=(15, 3))
            for i in range(n_timesteps):
                time = i / (n_timesteps - 1)  # Normalize time to be between 0 and 1
                # cmap = make_timestep_cmap(time)
                cmap = make_red_cmap(time)
                axs[i].imshow(data, cmap=cmap)
                axs[i].set_title(f'Timestep {i+1}')
            plt.show()

        """
        from matplotlib.colors import LinearSegmentedColormap

        colors = np.array([(0, 0, 0), (1, 0, 0)]) # np.shape(colors): (2, 3)
        if debug_print:
            print(f'np.shape(colors): {np.shape(colors)}')
        # Apply a saturation change
        saturation_factor = float(time) # 0.5  # Increase saturation by 1.5 times
        adjusted_colors = adjust_saturation(colors, saturation_factor)
        if debug_print:
            print(f'np.shape(adjusted_colors): {np.shape(adjusted_colors)}')
        adjusted_colors = adjusted_colors.tolist()
        ## Set the alpha of the first color to 0.0 and of the final color to 0.82
        adjusted_colors = [[*v, max_alpha] for v in adjusted_colors]
        adjusted_colors[0][-1] = min_alpha

        # n_bins = [2]  # Discretizes the interpolation into bins
        return LinearSegmentedColormap.from_list('CustomMap', adjusted_colors, N=N_colors)


    # Convert to LinearSegmentedColormap
    @classmethod
    def colormap_to_linear_segmented(cls, cmap, n_samples=256) -> LinearSegmentedColormap:
        """
        Converts a Colormap to a LinearSegmentedColormap.

        Args:
            cmap (Colormap): The original colormap to convert.
            n_samples (int): Number of samples to take from the original colormap.

        Returns:
            LinearSegmentedColormap: The converted colormap.
        """
        from matplotlib.colors import LinearSegmentedColormap
        from pyphoplacecellanalysis.External.pyqtgraph.colormap import ColorMap
        if isinstance(cmap, (LinearSegmentedColormap,)):
            return deepcopy(cmap) # already the correct type
        else:
            ## needs convert                          
            colors = cmap(np.linspace(0, 1, n_samples))  # Sample the original colormap
            return LinearSegmentedColormap.from_list(f"{cmap.name}_linear", colors)

    @classmethod
    def mpl_to_pg_colormap(cls, mpl_cmap_name: Union[str, Any], resolution=256) -> pg.ColorMap:
        """
        Converts a Matplotlib colormap to a PyQtGraph ColorMap.
        
        Args:
            mpl_cmap_name (str): Name of the Matplotlib colormap.
            resolution (int): Number of discrete color steps (default is 256).
        
        Returns:
            pg.ColorMap: The equivalent PyQtGraph ColorMap.
        """
        import matplotlib.pyplot as plt
        mpl_cmap = plt.get_cmap(mpl_cmap_name)
        positions = np.linspace(0, 1, resolution)
        colors = [mpl_cmap(i) for i in positions]
        colors_rgb = [tuple(int(c * 255) for c in color[:3]) for color in colors]
        return pg.ColorMap(positions, colors_rgb)
                
            

    @classmethod
    def create_colormap_transparent_below_value(cls, mycmap: Union[str, Any], low_value_cuttoff:float=0.2, below_low_value_cuttoff_alpha_value: float=0.0, resampled_num_colors:int=7):
        """ Modifies the provided colormap by settings the opacity/alpha of all values below `low_value_cuttoff` (where values always go [0.0, 1.0]) to the value `below_low_value_cuttoff_alpha_value`
        Usage:
            additional_cmap_names = dict(zip(TrackTemplates.get_decoder_names(), ['red', 'purple', 'green', 'orange'])) # {'long_LR': 'red', 'long_RL': 'purple', 'short_LR': 'green', 'short_RL': 'orange'}

            long_epoch_config = long_short_display_config_manager.long_epoch_config.as_pyqtgraph_kwargs()
            short_epoch_config = long_short_display_config_manager.short_epoch_config.as_pyqtgraph_kwargs()

            color_dict = {'long_LR': long_epoch_config['brush'].color(), 'long_RL': apply_LR_to_RL_adjustment(long_epoch_config['brush'].color()),
                            'short_LR': short_epoch_config['brush'].color(), 'short_RL': apply_LR_to_RL_adjustment(short_epoch_config['brush'].color())}
            additional_cmap_names = {k: ColorFormatConverter.qColor_to_hexstring(v) for k, v in color_dict.items()}

            additional_cmaps = {k: ColormapHelpers.create_transparent_colormap(color_literal_name=v, lower_bound_alpha=0.1) for k, v in additional_cmap_names.items()}
        
        """
        from matplotlib.colors import LinearSegmentedColormap
        from pyphoplacecellanalysis.External.pyqtgraph.colormap import ColorMap
        
        if isinstance(mycmap, str):
            mycmap = pg.colormap.get(mycmap, source='matplotlib')

        # original_n_colors: int = mycmap.N
        # print(f'original_n_colors: {original_n_colors}')

        # Get colors by sampling the colormap
        # resampled_num_colors: int = 7  # Number of colors to extract
        if resampled_num_colors is None:
            resampled_num_colors: int = original_n_colors  # Number of colors to extract

        ## convert to LinearSegmented if needed:
        mycmap = cls.colormap_to_linear_segmented(cmap=mycmap, n_samples=resampled_num_colors)        
        assert isinstance(mycmap, (LinearSegmentedColormap, )), f"type(mycmap): {type(mycmap)}" 
        _resampled_cmap = mycmap.resampled(resampled_num_colors)


        sampled_color_reference_arr = np.array([(float(i) / float(resampled_num_colors - 1)) for i in range(resampled_num_colors)]) ## array ranging between 0.0 and 1.0
        sampled_color_reference_idxs = np.arange(len(sampled_color_reference_arr))
        sampled_colors = np.array([list(_resampled_cmap(i / (resampled_num_colors - 1))) for i in range(resampled_num_colors)])
        # sampled_colors.shape # (num_colors, 4)
        
        is_value_below_cutoff = (sampled_color_reference_arr < low_value_cuttoff)
        
        # sampled_color_reference_arr[is_value_below_cutoff] ## values
        below_cuttoff_indicies = sampled_color_reference_idxs[is_value_below_cutoff]
        
        # sampled_colors[below_cuttoff_indicies][-1] = 0.0 # set alpha

        # sampled_colors[is_value_below_cutoff][-1] = below_low_value_cuttoff_alpha_value # set alpha

        for idx in below_cuttoff_indicies:
            sampled_colors[idx][-1] = below_low_value_cuttoff_alpha_value # set alpha  

        # sampled_colors[0][-1] = 0.0 # set alpha
        # sampled_colors

        # Rebuild the colormap
        reconstructed_cmap = LinearSegmentedColormap.from_list(f"reconstructed_{_resampled_cmap.name}", sampled_colors)

        return reconstructed_cmap
                

@metadata_attributes(short_name=None, tags=['color', 'dataseries', 'series', 'helper'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-21 13:50', related_items=['UnitColoringMode'])
class ColorFormatConverter:
     
    @classmethod
    def _hexArgb_to_hexRGBA(cls, hex_Argb_str:str) -> str:
        """ converts a hexArgb string such as one output by `pen.color().name(QtGui.QColor.HexArgb)` to a regular hex_RGBA string like would be used for matplotlib.
        
        '#0b0049ff'
        
        QColor.HexArgb: '#AARRGGBB' A “#” character followed by four two-digit hexadecimal numbers (i.e. #AARRGGBB).
        Output Format (HexRGBA): '#RRGGBBAA'
        
        Usage:
            from pyphocorehelpers.gui.Qt.color_helpers import ColorFormatConverter
            from pyphocorehelpers.gui.Qt.color_helpers import hexArgb_to_hexRGBA
            pen=pg.mkPen('#0b0049')
            hex_Argb_str:str = pen.color().name(QtGui.QColor.HexArgb) # '#ff0b0049'
            hex_RGBA_str = hexArgb_to_hexRGBA(hex_Argb_str)
            hex_RGBA_str # '#0b0049ff'

        """
        hex_rgb_str_part = hex_Argb_str[3:] # get the rgb characters
        hex_alpha_str_part: str = hex_Argb_str[1:3] # get the "alpha" components
        hex_RGBA_str: str = f"#{hex_rgb_str_part}{hex_alpha_str_part}"
        return hex_RGBA_str

    @classmethod
    def qColor_to_hexstring(cls, qcolor: QtGui.QColor, include_alpha:bool=True, use_HexArgb_instead_of_HexRGBA:bool=False) -> str:
        """ converts a QColor to a hex string 
        
        include_alpha: if True, returns a hex string containing the alpha values
        use_HexArgb_instead_of_HexRGBA:bool; default False, don't use typically.
            If False results in a string like '#80ff000
        
        
        Notes on getting hex colors:
            getting the name of a QColor with .name(QtGui.QColor.HexRgb) results in a string like '#ff0000'
            getting the name of a QColor with .name(QtGui.QColor.HexArgb) results in a string like '#80ff0000'

        Usage:
            from pyphocorehelpers.gui.Qt.color_helpers import ColorFormatConverter
            ColorFormatConverter.qColor_to_hexstring(aQColor)
        """
        if not include_alpha:
            return qcolor.name(QtGui.QColor.HexRgb)
        else:
            hex_Argb_str = qcolor.name(QtGui.QColor.HexArgb)
            if use_HexArgb_instead_of_HexRGBA:
                return hex_Argb_str
            else:
                return cls._hexArgb_to_hexRGBA(hex_Argb_str)

    # ==================================================================================================================== #
    # Color NDArray Conversions                                                                                             #
    # ==================================================================================================================== #
    @classmethod
    def auto_detect_color_NDArray_is_255_array_format(cls, colors_ndarray: np.ndarray) -> bool:
        """ tries to auto-detect the format of the color NDArray in terms of whether it contains 0.0-1.0 or 0.0-255.0 values. 
        returns True if it is 255_array_format, and False otherwise
        """
        return (not np.all(colors_ndarray <= 1.0)) # all are less than 1.0 implies that it NOT a 255_format_array



    @classmethod
    def Colors_NDArray_Convert_to_255_array(cls, colors_ndarray: np.ndarray) -> np.ndarray:
        """ takes an [4, nCell] np.array of (0.0 - 255.0) values for the color and converts it to a 0.0-1.0 array of the same shape.
        Reciprocal: Colors_NDArray_Convert_to_zero_to_one_array
        """
        converted_colors_ndarray = deepcopy(colors_ndarray)
        converted_colors_ndarray[0:2, :] *= 255 # [1.0, 0.0, 0.0, 1.0]
        return converted_colors_ndarray
    
    @classmethod
    def Colors_NDArray_Convert_to_zero_to_one_array(cls, colors_ndarray: np.ndarray) -> np.ndarray:
        """ takes an [4, nCell] np.array of 0.0-1.0 values for the color and converts it to a (0.0 - 255.0) array of the same shape.
        Reciprocal: Colors_NDArray_Convert_to_255_array
        """
        converted_colors_ndarray = deepcopy(colors_ndarray).astype(float)
        colors_shape = np.shape(converted_colors_ndarray)
        n_colors: int = colors_shape[0]
        assert n_colors in [3, 4], f"n_colors must be either 3 (RGB) or 4 (RGBA) but instead it is {n_colors}. Is the array transposed? colors_shape: {colors_shape}"
        if n_colors == 3:
            color_idx_range = np.arange(3) # 0:2, RGB
        elif n_colors == 4:
            color_idx_range = np.arange(4) # 0:3, RGBA
        else:
            raise NotImplementedError(f'n_colors: {n_colors}')
        
        print(f'color_idx_range: {color_idx_range}')
        converted_colors_ndarray[color_idx_range, :] /= 255
        # converted_colors_ndarray[0:2, :] /= 255 # UFuncTypeError: Cannot cast ufunc 'divide' output from dtype('float64') to dtype('uint8') with casting rule 'same_kind'
        return converted_colors_ndarray

    @classmethod
    def qColorsList_to_NDarray(cls, qcolors_list, is_255_array:bool) -> np.ndarray:
        """ takes a list[QColor] and returns a [4, nCell] np.array with the color for each in the list 
        
        is_255_array: bool - if False, all RGB color values are (0.0 - 1.0), else they are (0.0 - 255.0)
        I was having issues with this list being in the range 0.0-1.0 instead of 0-255.
        
        Note: Matplotlib requires zero_to_one_array format
        
        Extracted on 2024-08-30 from `pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers.DataSeriesColorHelpers`

        """

        # allocate new neuron_colors array:
        n_cells = len(qcolors_list)
        neuron_colors = np.zeros((4, n_cells))
        for i, curr_qcolor in enumerate(qcolors_list):
            curr_color = curr_qcolor.getRgbF() # (1.0, 0.0, 0.0, 0.5019607843137255)
            neuron_colors[:, i] = curr_color[:]
        if is_255_array:
            neuron_colors = cls.Colors_NDArray_Convert_to_255_array(neuron_colors) 
        return neuron_colors
    

    @classmethod
    def colors_NDarray_to_qColorsList(cls, colors_ndarray: np.ndarray, is_255_array:Optional[bool]=None) -> list:
        """ Takes a [4, nCell] np.array and returns a list[QColor] with the color for each cell in the array
        
        is_255_array: bool - if False, all RGB color values are in range (0.0 - 1.0), else they are in range (0.0 - 255.0)
        
        Note: Matplotlib requires zero_to_one_array format
        
        Extracted on 2024-08-30 from `pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers.DataSeriesColorHelpers`
        """
        if is_255_array is None:
            is_255_array = cls.auto_detect_color_NDArray_is_255_array_format(colors_ndarray)

        if is_255_array:
            colors_ndarray = cls.Colors_NDArray_Convert_to_zero_to_one_array(colors_ndarray)

        n_cells = colors_ndarray.shape[1]
        qcolors_list = []
        for i in range(n_cells):
            curr_color = QColor.fromRgbF(*colors_ndarray[:, i])
            qcolors_list.append(curr_color)
            
        return qcolors_list


    @classmethod
    def convert_pen_brush_to_matplot_kwargs(cls, pen, brush) -> Dict:
        """ converts a pyqtgraph (pen: QPen, brush: QBrush) combination into matplotlib kwargs dict 
        Usage:
            from pyphocorehelpers.gui.Qt.color_helpers import convert_pen_brush_to_matplot_kwargs

            matplotlib_rect_kwargs = convert_pen_brush_to_matplot_kwargs(pen, brush)
            matplotlib_rect_kwargs

        """
        return dict(linewidth=pen.widthF(), edgecolor=cls._hexArgb_to_hexRGBA(pen.color().name(QtGui.QColor.HexArgb)), facecolor=cls._hexArgb_to_hexRGBA(brush.color().name(QtGui.QColor.HexArgb)))


