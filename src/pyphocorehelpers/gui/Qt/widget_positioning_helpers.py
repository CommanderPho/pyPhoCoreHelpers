#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QDesktopWidget


class WidgetPositioningHelpers:
    """ Helpers for positioning and aligning Qt windows and widgets """

    @classmethod
    def move_widget_to_top_left_corner(cls, widget, screen_index:int=None, debug_print:bool=False):
        """ Moves a widget's top_left_corner to the top_left_corner of the screen.
        Specifying a screen_index does not work. Only works if screen_index=None
  
        Examples:
              WidgetPositioningHelpers.move_widget_to_top_left_corner(spike_raster_plt_3d, screen_index=None, debug_print=True)
  
        """
        if screen_index is None:
            # Global Desktop Widget:		
            desktopWidget = QDesktopWidget()
            desktopRect = desktopWidget.availableGeometry() # PyQt5.QtCore.QRect(0, 0, 3570, 2120)
            
        else:
            # return cls.move_widget_to_top_left_corner_of_screen(widget=widget, screen_index=screen_index, debug_print=debug_print)
            raise NotImplementedError
            widget = widget.app.desktop()
            assert screen_index <= (widget.screenCount()-1), f"specified screen_index screen_index: {screen_index} is invalid. widget.screenCount(): {widget.screenCount()}"
            desktopRect = widget.screenGeometry(screen_index) # get screen geometry of secondary screen. PyQt5.QtCore.QRect(-3440, 0, 3440, 1440)

        if debug_print:
            print(f'desktopRect: {desktopRect}') # desktopRect: PyQt5.QtCore.QRect(0, 0, 3570, 2120)
        currWindowRect = widget.frameGeometry()
        if debug_print:
            print(f'pre-update: currWindowRect: {currWindowRect}')
        currWindowRect.moveTopLeft(desktopRect.topLeft()) # move topLeft point of currWindowRect
        if debug_print:
            print(f'post-update: currWindowRect: {currWindowRect}')
        
        if debug_print:
            print(f'secondary_screen_rect: {desktopRect}') # secondary_screen_rect.topLeft() # PyQt5.QtCore.QPoint(-3440, 0)
            print(f'widget: {widget}')
        
        # Move to top-left corner of secondary screen:
        widget.move(currWindowRect.topLeft())  
  
        
    @classmethod
    def align_3d_and_2d_windows(cls, spike_raster_plt_3d, spike_raster_plt_2d, debug_print=False):
        """ align the two separate windows (for the 3D Raster Plot and the 2D Raster plot that controls it)
            Usage:
                WidgetPositioningHelpers.align_3d_and_2d_windows(spike_raster_plt_3d, spike_raster_plt_2d)
        """
        # _move_widget_to_top_left_corner(spike_raster_plt_3d, screen_index=1, debug_print=debug_print) # move to secondary screen's top-left corner.
        geom = spike_raster_plt_3d.window().geometry() # get the QTCore PyRect object
        if debug_print:
            print(f'geom.bottom: {geom.bottom()}') # geom.bottom: 941
        x,y,dx,dy = geom.getRect()
        if debug_print:
            print(f'geom: {geom}') # geom: PyQt5.QtCore.QRect(-3354, 176, 1920, 900)
        # after moving window down on screen: geom: PyQt5.QtCore.QRect(-3362, 478, 1920, 900)
        # THEREFORE larger y values are down
        desired_2d_window_y = y + dy # place directly below 3D window
        # desired_2d_window_y = y
        if debug_print:
            print(f'y: {y}, dy: {dy}, desired_2d_window_y: {desired_2d_window_y}')
        spike_raster_plt_2d.window().setGeometry(x, desired_2d_window_y, dx, int(dy*0.5))
  
  
    @classmethod
    def align_window_edges(cls, main_window, secondary_window, relative_position = 'below', resize_to_main=(1.0, None), debug_print=False):
        """ align the two separate windows (with main_window being the one that's stationary and secondary_window being the one adjusted to sit relative to it).
        
            relative_position: str? - 'above', 'below', or None
            resize_to_main: (percent_of_main_width?, percent_of_main_height?): specifying None for either value will prevent resize in that dimension

            Usage:
                WidgetPositioningHelpers.align_3d_and_2d_windows(spike_raster_plt_3d, spike_raster_plt_2d)
        """
        # _move_widget_to_top_left_corner(spike_raster_plt_3d, screen_index=1, debug_print=debug_print) # move to secondary screen's top-left corner.
        main_win_geom = main_window.window().geometry() # get the QTCore PyRect object
        if debug_print:
            print(f'geom.bottom: {main_win_geom.bottom()}') # geom.bottom: 941
        main_x, main_y, main_dx, main_dy = main_win_geom.getRect() # Note: dx & dy refer to width and height
        if debug_print:
            print(f'geom: {main_win_geom}') # geom: PyQt5.QtCore.QRect(-3354, 176, 1920, 900)
        # after moving window down on screen: geom: PyQt5.QtCore.QRect(-3362, 478, 1920, 900)
        # THEREFORE larger y values are down
        second_win_geom = secondary_window.window().geometry()
        secondary_x, secondary_y, secondary_dx, secondary_dy = second_win_geom.getRect() # Note: dx & dy refer to width and height
        
        ## Resizing:
        desired_percent_of_main_width, desired_percent_of_main_height = resize_to_main
        if desired_percent_of_main_width is not None:
            desired_secondary_dx = int(main_dx * desired_percent_of_main_width)
        else:
            # Unaltered width:
            desired_secondary_dx = secondary_dx
        if desired_percent_of_main_height is not None:
            desired_secondary_dy = int(main_dy * desired_percent_of_main_height)
        else:
            # Unaltered height:
            desired_secondary_dy = secondary_dy
            
        ## Repositioning:
        if relative_position is None:
            desired_secondary_window_x = secondary_x
            desired_secondary_window_y = secondary_y
        elif relative_position == 'below':
            desired_secondary_window_x = main_x
            desired_secondary_window_y = main_y + main_dy # place directly below 3D window
        elif relative_position == 'above':
            desired_secondary_window_x = main_x
            desired_secondary_window_y = main_y - desired_secondary_dy # subtract the secondary window's height from the top of the primary window
            # TODO: make sure they both fit on the screen together.
        else:
            raise NotImplementedError

        if debug_print:
            print(f'y: {main_y}, dy: {main_dy}, desired_2d_window_y: {desired_secondary_window_y}')
        secondary_window.window().setGeometry(desired_secondary_window_x, desired_secondary_window_y, desired_secondary_dx, desired_secondary_dy)
  
  
  