from typing import Dict, List, Optional,  OrderedDict, Union
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
    

def build_adjusted_color(color: QColor, hue_shift=0.0, saturation_scale=1.0, value_scale=1.0):
    """ Builds a copy of the color QColor with optionally modified HSV properties
    Example:
    	from pyphocorehelpers.gui.Qt.color_helpers import build_adjusted_color
    
        debug_print_color(curr_color)
        curr_color_copy = build_adjusted_color(curr_color, hue_shift=0.0, saturation_scale=0.35, value_scale=1.0)
        debug_print_color(curr_color_copy)

    """
    curr_color_copy = color.convertTo(QColor.Hsv) # makes a copy of color
    # curr_color_copy.setHsv(curr_color_copy.hue(),curr_color_copy.saturation(), curr_color_copy.value())
    # np.clip(v, 0.0, 1.0) ensures the values are between 0.0 and 1.0
    curr_color_copy.setHsvF(np.clip((curr_color_copy.hueF() + hue_shift), 0.0, 1.0),
                            np.clip((saturation_scale*curr_color_copy.saturationF()), 0.0, 1.0),
                            np.clip((value_scale * curr_color_copy.valueF()), 0.0, 1.0))
    curr_color_copy.setAlphaF(color.alphaF())
    assert curr_color_copy.isValid(), "Constructed color is invalid!"
    return curr_color_copy


@metadata_attributes(short_name=None, tags=['color', 'dataseries', 'series', 'helper'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-21 13:50', related_items=['UnitColoringMode'])
class ColorFormatConverter:
     
    @classmethod
    def _hexArgb_to_hexRGBA(cls, hex_Argb_str:str) -> str:
        """ converts a hexArgb string such as one output by `pen.color().name(QtGui.QColor.HexArgb)` to a regular hex_RGBA string like would be used for matplotlib.
        
        '#0b0049ff'
        
        QColor.HexArgb: '#AARRGGBB' A “#” character followed by four two-digit hexadecimal numbers (i.e. #AARRGGBB).
        Output Format (HexRGBA): '#RRGGBBAA'
        
        Usage:
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
        converted_colors_ndarray = deepcopy(colors_ndarray)
        converted_colors_ndarray[0:2, :] /= 255
        return converted_colors_ndarray
    


    @classmethod
    def convert_pen_brush_to_matplot_kwargs(cls, pen, brush) -> Dict:
        """ converts a pyqtgraph (pen: QPen, brush: QBrush) combination into matplotlib kwargs dict 
        Usage:
            from pyphocorehelpers.gui.Qt.color_helpers import convert_pen_brush_to_matplot_kwargs

            matplotlib_rect_kwargs = convert_pen_brush_to_matplot_kwargs(pen, brush)
            matplotlib_rect_kwargs

        """
        return dict(linewidth=pen.widthF(), edgecolor=cls._hexArgb_to_hexRGBA(pen.color().name(QtGui.QColor.HexArgb)), facecolor=cls._hexArgb_to_hexRGBA(brush.color().name(QtGui.QColor.HexArgb)))


