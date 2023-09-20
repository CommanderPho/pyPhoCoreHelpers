from typing import Dict, List, Optional
from qtpy import QtGui # for QColor
from qtpy.QtGui import QColor, QBrush, QPen

import numpy as np



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


def hexArgb_to_hexRGBA(hex_Argb_str:str) -> str:
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


def convert_pen_brush_to_matplot_kwargs(pen, brush) -> Dict:
	""" converts a pyqtgraph (pen: QPen, brush: QBrush) combination into matplotlib kwargs dict 
     Usage:
        from pyphocorehelpers.gui.Qt.color_helpers import convert_pen_brush_to_matplot_kwargs

        matplotlib_rect_kwargs = convert_pen_brush_to_matplot_kwargs(pen, brush)
        matplotlib_rect_kwargs

     """
	return dict(linewidth=pen.widthF(), edgecolor=hexArgb_to_hexRGBA(pen.color().name(QtGui.QColor.HexArgb)), facecolor=hexArgb_to_hexRGBA(brush.color().name(QtGui.QColor.HexArgb)))


