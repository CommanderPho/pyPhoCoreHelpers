# SyncedTimelineWindowLink
from enum import Enum
# from pyqtgraph.Qt import QtCore
import numpy as np
import pandas as pd


# Perform Initial (one-time) update from source -> controlled:
def connect_additional_controlled_spike_raster_plotter(spike_raster_plt_2d, controlled_spike_raster_plt):
    """ Connect an additional plotter to a source that's driving the update of the data-window:
    
    Requirements:
        source_spike_raster_plt:
            .spikes_window.active_time_window
            .window_scrolled
        
        controlled_spike_raster_plt:
            .spikes_window.update_window_start_end(float, float)
        
    Usage:
        
        spike_raster_plt_3d, spike_raster_plt_2d, spike_3d_to_2d_window_connection = build_spike_3d_raster_with_2d_controls(curr_spikes_df)
        spike_raster_plt_3d_vedo = Spike3DRaster_Vedo(curr_spikes_df, window_duration=15.0, window_start_time=30.0, neuron_colors=None, neuron_sort_order=None)
        extra_vedo_sync_connection = connect_additional_controlled_spike_raster_plotter(spike_raster_plt_2d, spike_raster_plt_3d_vedo)
    
    """
    controlled_spike_raster_plt.spikes_window.update_window_start_end(spike_raster_plt_2d.spikes_window.active_time_window[0], spike_raster_plt_2d.spikes_window.active_time_window[1])
    # Connect to update self when video window playback position changes
    sync_connection = spike_raster_plt_2d.window_scrolled.connect(controlled_spike_raster_plt.spikes_window.update_window_start_end)
    return sync_connection



def connect_additional_controlled_plotter(source_spike_raster_plt, controlled_plt):
    """ allow the window to control InteractivePlaceCellDataExplorer (ipspikesDataExplorer) objects;
    source_spike_raster_plt: the spike raster plotter to connect to as the source
    controlled_plt: should be a InteractivePlaceCellDataExplorer object (ipspikesDataExplorer), but can be any function with a valid update_window_start_end @QtCore.Slot(float, float) slot.
    
    Requirements:
        source_spike_raster_plt:
            .spikes_window.active_time_window
            .window_scrolled
        
        controlled_plt:
            .disable_ui_window_updating_controls()
            .update_window_start_end(float, float)
    
    
    Usage:
    
        from pyphoplacecellanalysis.GUI.Qt.SpikeRasterWindows.Spike3DRasterWindowWidget import Spike3DRasterWindowWidget

        # Build the controlled ipspikesDataExplorer:
        display_output = dict()
        pActiveSpikesBehaviorPlotter = None
        display_output = display_output | curr_active_pipeline.display(DefaultDisplayFunctions._display_3d_interactive_spike_and_behavior_browser, active_config_name, extant_plotter=display_output.get('pActiveSpikesBehaviorPlotter', None)) # Works now!
        ipspikesDataExplorer = display_output['ipspikesDataExplorer']
        display_output['pActiveSpikesBehaviorPlotter'] = display_output.pop('plotter') # rename the key from the generic "plotter" to "pActiveSpikesBehaviorPlotter" to avoid collisions with others
        pActiveSpikesBehaviorPlotter = display_output['pActiveSpikesBehaviorPlotter']

        # Build the contolling raster window:
        spike_raster_window = Spike3DRasterWindowWidget(curr_spikes_df)
        # Call this function to connect them:
        extra_interactive_spike_behavior_browser_sync_connection = connect_additional_controlled_plotter(spike_raster_window.spike_raster_plt_2d, ipspikesDataExplorer)
    
    """
    # Perform Initial (one-time) update from source -> controlled:
    controlled_plt.disable_ui_window_updating_controls() # disable the GUI for manual updates.
    controlled_plt.update_window_start_end(source_spike_raster_plt.spikes_window.active_time_window[0], source_spike_raster_plt.spikes_window.active_time_window[1])
    # Connect to update self when video window playback position changes
    sync_connection = source_spike_raster_plt.window_scrolled.connect(controlled_plt.update_window_start_end)
    return sync_connection


# class SyncedTimelineWindowLink_SyncOption(Enum):
#         Bidirectional = 1 # Keep both synced
#         VideoToTimeline = 2 #  Set timeline time from video
#         TimelineToVideo = 3  # Set video time from timeline



# class SyncedTimelineWindowLink(QtCore.QObject):


    
#     def __init__(self, spike_raster_plt_2d, controlled_spike_raster_plt, parent=None, sync_option=SyncedTimelineWindowLink_SyncOption.Bidirectional):
#         """ 
#             spike_raster_plt_2d: must be a spike_raster_plt_2d
#             controlled_spike_raster_plt: should be for example a spike_raster_plt_3d. Must implement spike_raster_plt_3d.spikes_window.update_window_start_end
#         """
#         super(SyncedTimelineWindowLink, self).__init__(parent=parent)
    
#         # TODO: Currently the self.sync_option does nothing. Only video window -> timeline window is currently fully implemented
#         self.sync_option = sync_option

#         # Perform Initial (one-time) update from source -> controlled:
#         controlled_spike_raster_plt.spikes_window.update_window_start_end(spike_raster_plt_2d.spikes_window.active_time_window[0], spike_raster_plt_2d.spikes_window.active_time_window[1])
  
#         # Connect to update self when video window playback position changes
#         self._sync_connection = spike_raster_plt_2d.window_scrolled.connect(controlled_spike_raster_plt.spikes_window.update_window_start_end)
        
#         # On destroy of either object, remove the connection:
        
#         self._sync_connection


#     # @QtCore.pyqtSlot(float, float, float)
#     # def on_window_duration_changed(self, start_t, end_t, duration):
#     #     """ changes self.half_render_window_duration """
#     #     print(f'SpikeRasterBase.on_window_duration_changed(start_t: {start_t}, end_t: {end_t}, duration: {duration})')


#     # @QtCore.pyqtSlot(float, float)
#     # def on_window_changed(self, start_t, end_t):
#     #     # called when the window is updated
#     #     if self.enable_debug_print:
#     #         print(f'SpikeRasterBase.on_window_changed(start_t: {start_t}, end_t: {end_t})')
#     #     profiler = pg.debug.Profiler(disabled=True, delayed=True)
#     #     self._update_plots()
#     #     profiler('Finished calling _update_plots()')
    
