from pathlib import Path
import ipywidgets as widgets
from IPython.display import display
from PIL import Image as PILImage
from io import BytesIO
from typing import Dict
from datetime import datetime
from PIL import Image, ImageDraw, UnidentifiedImageError
import ipywidgets as widgets
from datetime import datetime



# Load PDF and convert each page to a PIL Image


def load_pdf_as_images(pdf_path: str, poppler_path: Path=None, last_page: int=None):
    # Convert PDF to list of images (one per page)
    from pdf2image import convert_from_path
    if poppler_path is None:
        poppler_path = Path(r"C:\Users\pho\lib\poppler-24.07.0\Library\bin").resolve()
    images = convert_from_path(pdf_path, poppler_path=poppler_path, last_page=last_page)
    return images

def create_placeholder_image(text, width, height):
    img = Image.new('RGB', (width, height), color='gray')
    d = ImageDraw.Draw(img)
    d.text((10, height // 2 - 10), text, fill=(255, 255, 255))
    return img

class ImageNavigator:
    """ 
    from pyphocorehelpers.gui.Jupyter.JupyterImageNavigatorWidget import ImageNavigator
    
    
    Example 1:
        from pyphocorehelpers.gui.Jupyter.JupyterImageNavigatorWidget import ImageNavigator

        # Create 6 example ImageNavigator instances with different placeholder images and sizes
        image_dict_1 = {
            datetime(2023, 4, 11): create_placeholder_image("Image 1-1", 200, 200),
            datetime(2023, 5, 12): create_placeholder_image("Image 1-2", 200, 200)
        }

        image_dict_2 = {
            datetime(2023, 6, 13): create_placeholder_image("Image 2-1", 250, 150),
            datetime(2023, 7, 14): create_placeholder_image("Image 2-2", 250, 150)
        }

        image_dict_3 = {
            datetime(2023, 8, 15): create_placeholder_image("Image 3-1", 300, 300),
            datetime(2023, 9, 16): create_placeholder_image("Image 3-2", 300, 300)
        }

        image_dict_4 = {
            datetime(2023, 10, 17): create_placeholder_image("Image 4-1", 150, 250),
            datetime(2023, 11, 18): create_placeholder_image("Image 4-2", 150, 250)
        }

        image_dict_5 = {
            datetime(2023, 12, 19): create_placeholder_image("Image 5-1", 350, 200),
            datetime(2024, 1, 20): create_placeholder_image("Image 5-2", 350, 200)
        }

        image_dict_6 = {
            datetime(2024, 2, 21): create_placeholder_image("Image 6-1", 200, 350),
            datetime(2024, 3, 22): create_placeholder_image("Image 6-2", 200, 350)
        }

        # Instantiate the ImageNavigator objects with different sizes
        navigator1 = ImageNavigator(image_dict_1, "Navigator 1")
        navigator2 = ImageNavigator(image_dict_2, "Navigator 2")
        navigator3 = ImageNavigator(image_dict_3, "Navigator 3")
        navigator4 = ImageNavigator(image_dict_4, "Navigator 4")
        navigator5 = ImageNavigator(image_dict_5, "Navigator 5")
        navigator6 = ImageNavigator(image_dict_6, "Navigator 6")

        # Adjust the layout of each navigator to fill its container based on its image size
        navigator1.vbox.layout = widgets.Layout(width='auto', height='auto')
        navigator2.vbox.layout = widgets.Layout(width='auto', height='auto')
        navigator3.vbox.layout = widgets.Layout(width='auto', height='auto')
        navigator4.vbox.layout = widgets.Layout(width='auto', height='auto')
        navigator5.vbox.layout = widgets.Layout(width='auto', height='auto')
        navigator6.vbox.layout = widgets.Layout(width='auto', height='auto')

        # Collect their VBox elements for display
        navigator_widgets = [
            navigator1.vbox,
            navigator2.vbox,
            navigator3.vbox,
            navigator4.vbox,
            navigator5.vbox,
            navigator6.vbox
        ]

        # Define a GridBox layout with 3 columns and a grid gap, allowing the navigators to expand
        grid = widgets.GridBox(
            navigator_widgets,
            layout=widgets.Layout(grid_template_columns="repeat(3, 1fr)", grid_gap="10px")
        )

        # Display the grid
        display(grid)


    """
    def __init__(self, image_dict: Dict[datetime, PILImage.Image], image_title: str):
        assert len(image_dict) > 0
        self.image_dict = image_dict
        self.image_title = image_title
        self.keys = list(image_dict.keys())
        self.current_index = 0
        self.total_images = len(self.keys)
        
        # Widgets
        self.title_label = widgets.Label(value=self.image_title)
        self.image_display = widgets.Image()
        self.date_label = widgets.Label(value=str(self.keys[self.current_index]))
        self.counter_label = widgets.Label(value=f"Image {self.current_index + 1}/{self.total_images}")
        
        self.left_button = widgets.Button(description="←", layout=widgets.Layout(width='50px'))
        self.right_button = widgets.Button(description="→", layout=widgets.Layout(width='50px'))

        # Set up event listeners for the buttons
        self.left_button.on_click(self.on_left_click)
        self.right_button.on_click(self.on_right_click)

        # Initial display setup
        self.update_image()

        # Layout
        self.controls = widgets.HBox([self.counter_label, self.left_button, self.date_label, self.right_button])
        self.vbox = widgets.VBox([self.title_label, self.image_display, self.controls])

    def update_image(self):
        """Updates the image display and the date label based on the current index."""
        current_key = self.keys[self.current_index]
        img = self.image_dict[current_key]
        try:
            if isinstance(img, (Path, str)):
                ## load the image
                if img.name.endswith('.pdf'):
                    img = load_pdf_as_images(img, poppler_path=Path(r"C:\Users\pho\lib\poppler-24.07.0\Library\bin").resolve(), last_page=1)[0] # get the first page only
                else:
                    img = Image.open(img.as_posix())
                self.image_dict[current_key] = img
                
        except (UnidentifiedImageError, BaseException) as e:
            img = None
            print(f'err: {e}, skipping.') 
            img = create_placeholder_image("<No Image>", width=100, height=100)
            # raise e

        # Convert PIL image to bytes for display in ipywidgets.Image
        with BytesIO() as output:
            img.save(output, format="PNG")
            img_data = output.getvalue()

        # Update the image widget and date label
        self.image_display.value = img_data
        self.date_label.value = str(current_key)
        self.counter_label.value = f"Image {self.current_index + 1}/{self.total_images}"

    def on_left_click(self, _):
        """Handle left button click: go to the previous image."""
        self.current_index = (self.current_index - 1) % len(self.keys)
        self.update_image()

    def on_right_click(self, _):
        """Handle right button click: go to the next image."""
        self.current_index = (self.current_index + 1) % len(self.keys)
        self.update_image()

    def display(self):
        """Display the widget."""
        display(self.vbox)

class ContextSidebar:
    """ 
    Usage:
        from pyphocorehelpers.gui.Jupyter.JupyterImageNavigatorWidget import ContextSidebar
    
        # Example usage with some example contexts
        context_dict = {
            'Context 1': 'Description or content for Context 1',
            'Context 2': 'Description or content for Context 2',
            'Context 3': 'Description or content for Context 3',
        }

        sidebar = ContextSidebar(context_dict)
        sidebar.display()

    
    """
    def __init__(self, context_dict):
        self.context_dict = context_dict
        self.context_tabs = list(context_dict.keys())

        # Create a sidebar with a vertical tab layout
        self.tab_widget = widgets.Tab()

        # Set the tab titles and content
        self.tab_widget.children = tuple(self.context_dict.values()) # [self.create_tab_content(context) for context in self.context_tabs]
        for i, context in enumerate(self.context_tabs):
            self.tab_widget.set_title(i, context)

        # Layout the sidebar vertically
        self.sidebar = widgets.VBox([self.tab_widget], layout=widgets.Layout(width='auto'))

    def create_tab_content(self, context):
        # Create content for each tab (you can customize this)
        label = widgets.Label(value=f"Currently viewing context: {context}")
        return widgets.VBox([label])

    def display(self):
        """Display the sidebar widget."""
        display(self.sidebar)
        

def build_context_images_navigator_widget(curr_context_images_dict, curr_context_desc_str: str, max_num_widget_debug: int = 5):
    """ Builds a single tab's contents (a widget consisting of a title and a `GridBox` of images

    Usage:
    
        from pyphocorehelpers.gui.Jupyter.JupyterImageNavigatorWidget import build_context_images_navigator_widget
        
        ## INPUTS: _final_out_dict_dict: Dict[ContextDescStr, Dict[ImageNameStr, Dict[datetime, Path]]]
        context_tabs_dict = {curr_context_desc_str:build_context_images_navigator_widget(curr_context_images_dict, curr_context_desc_str=curr_context_desc_str, max_num_widget_debug=2) for curr_context_desc_str, curr_context_images_dict in list(_final_out_dict_dict.items())}
        sidebar = ContextSidebar(context_tabs_dict)
        sidebar.display()


    """
    label_layout = widgets.Layout(width='auto',height='auto')
    # Collect their VBox elements for display
    navigators = []

    for an_image_name, an_image_history_dict in curr_context_images_dict.items():
        if len(navigators) < max_num_widget_debug:
            if len(an_image_history_dict) > 0:
                navigator1 = ImageNavigator(an_image_history_dict, an_image_name)
                navigator1.vbox.layout = widgets.Layout(width='auto', height='auto')
                navigators.append(navigator1)


    navigator_widgets = [nav.vbox for nav in navigators]
    # Define a GridBox layout with 3 columns and a grid gap, allowing the navigators to expand
    # title_label = widgets.Label(value=f"Currently viewing context: {curr_context_desc_str}", layout=label_layout, **label_style_kwargs)
    
    title_label_widget = widgets.HBox([widgets.Label(value=f"Currently viewing context: "),
                                       widgets.Button(description=f"{curr_context_desc_str}", button_style='danger', layout=label_layout),
    ])

    grid = widgets.GridBox(
        navigator_widgets,
        layout=widgets.Layout(grid_template_columns="repeat(3, 1fr)", grid_gap="10px")
    )
    return widgets.VBox([title_label_widget, grid])


