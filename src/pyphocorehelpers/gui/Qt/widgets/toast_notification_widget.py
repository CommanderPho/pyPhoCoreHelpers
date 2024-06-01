import sys
from typing import Optional
from qtpy import QtCore, QtWidgets
from qtpy.QtWidgets import QWidget, QLabel, QApplication, QVBoxLayout, QPushButton, QStyle
from qtpy.QtCore import QTimer, Qt

class ToastWidget(QWidget):
    """
    Displays a message for a short time only in its parent window

    
    from pyphocorehelpers.gui.Qt.widgets.toast_notification_widget import ToastWidget, ToastShowingWidgetMixin

    
    """
    def __init__(self, message, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        self.label = QLabel(message)
        self.label.setAlignment(Qt.AlignCenter)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(True)
        
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.adjustSize()
        self.setFixedSize(self.size())
        
        self.durationMs = 2000
        QTimer.singleShot(self.durationMs, self.close)

    def show_message(self, message, durationMs: Optional[int]=None):
        if durationMs is not None:
            self.durationMs = durationMs
            
        self.label.setText(message)
        self.adjustSize()
        self.setFixedSize(self.size())

        # Position the toast in the bottom-right corner of the parent
        if self.parent():
            parent_size = self.parent().size()
            self.move(
                self.parent().frameGeometry().topLeft().x() + parent_size.width() - self.width() - 10,
                self.parent().frameGeometry().topLeft().y() + parent_size.height() - self.height() - 10
            )

        QTimer.singleShot(self.durationMs, self.close)
        self.show()




class ToastShowingWidgetMixin:
    """ implementors contain a toast widget
    
    """
    @property
    def toast_widget(self) -> ToastWidget:
        """The toast_widget property."""
        return self.toast
    
    def _init_ToastShowingWidgetMixin(self):
        self.toast = ToastWidget('Hello, PyQt5!', self)


    def show_toast_message(self, message: str, durationMs:int=2000):
        """ shows a temporary toast message
        """
        self.toast_widget.show_message(message, durationMs=int(durationMs))

    @classmethod
    def conform(cls, obj):
        """ makes the object conform to this mixin by adding its properties. 
        Usage:
            from pyphoplacecellanalysis.General.Pipeline.Stages.Computation import PipelineWithComputedPipelineStageMixin, ComputedPipelineStage
            from pyphoplacecellanalysis.General.Pipeline.Stages.Display import PipelineWithDisplayPipelineStageMixin, PipelineWithDisplaySavingMixin
            from pyphoplacecellanalysis.General.Pipeline.Stages.Filtering import FilteredPipelineMixin
            from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import PipelineWithInputStage, PipelineWithLoadableStage
            from pyphoplacecellanalysis.General.Pipeline.Stages.BaseNeuropyPipelineStage import PipelineStage
            from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import NeuropyPipeline

            ToastShowingWidgetMixin.conform(curr_active_pipeline)

        """
        def conform_to_implementing_method(func):
            """ captures 'obj', 'cls'"""
            setattr(type(obj), func.__name__, func)
            
        conform_to_implementing_method(cls._init_ToastShowingWidgetMixin)
        conform_to_implementing_method(cls.show_toast_message)
        # conform_to_implementing_method(cls.write_figure_to_daily_programmatic_session_output_path)
        # conform_to_implementing_method(cls.write_figure_to_output_path)
        # setattr(type(cls.toast_widget), 'toast_widget', cls.toast_widget)
        obj._init_ToastShowingWidgetMixin() 





class ExampleApp(ToastShowingWidgetMixin, QWidget):
    """ Example app containing a toast widget """
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 200, 100)
        self.setWindowTitle('Toast Example')
        # Create a button that when clicked, will display the toast
        self.button = QPushButton('Show Toast', self)
        self.button.clicked.connect(self.show_toast)  # Connect the button signal to the slot

        # Layout to place the button in the application
        layout = QVBoxLayout()  
        layout.addWidget(self.button)
        self.setLayout(layout)

        self._init_ToastShowingWidgetMixin()
        self.show()
        
        # Show the toast
        # QTimer.singleShot(1000, self.show_toast)  # Show toast after 1 second

    def show_toast(self):
        message = 'This is a toast message!'
        self.toast_widget.show_message(message)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ExampleApp()
    sys.exit(app.exec_())

