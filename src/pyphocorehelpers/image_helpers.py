from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
import PIL, os, glob
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from math import ceil, floor
from pyphocorehelpers.assertion_helpers import Assert
import importlib.resources as resources


# PATH = r"C:\Users\path\to\images"

def build_icon(baseIconPath, overlayIconPath, outputIconPath=None, should_defer_show:bool=False) -> Image:
    """ 2022-10-04 - This builds a simple icon with an overlay icon. Not quite working because the overlay icon is rendering with a white background.
    
    from pyphocorehelpers.image_helpers import build_icon
    
    
    """
    MAX_IMG_SIZE = 256
    if outputIconPath is None:
        outputIconPath = 'new_im.png'

    base_img = Image.open(baseIconPath)
    img_width, img_height = base_img.size
    print(f'img_width: {img_width}, img_height: {img_height}')
    # find largest dimension:
    if img_width > MAX_IMG_SIZE:
        # clipping on widget
        scaling_factor = float(MAX_IMG_SIZE) / float(img_width) # get the scaling factored needed to bring the img_width down to MAX_IMG_SIZE
        print(f'scaling_factor: {scaling_factor}')
        frame_width = min(img_width, MAX_IMG_SIZE) # clip frame_width to max
        frame_height = ceil(img_height * scaling_factor) # wait this would mess up the aspect ratio if the image wasn't square and larger than 256 yeah?
    else:
        print(f'TODO: pretending everything is okay if the width is not too big.')
        frame_width = min(img_width, MAX_IMG_SIZE)
        frame_height = min(img_height, MAX_IMG_SIZE) # wait this would mess up the aspect ratio if the image wasn't square and larger than 256 yeah?

    base_img.thumbnail((frame_width, frame_height)) # ensures it is the same height as the base image, as it has the whitespace built in the image

    print(f'frame_width: {frame_width}, frame_height: {frame_height}')
    new_im = Image.new('RGB', (frame_width, frame_height))
    new_im.paste(base_img) # add the base image to the new icon

    overlay_img = Image.open(overlayIconPath)
    #Here I resize my opened image, so it is no bigger than 100,100

    overlay_img.thumbnail((frame_width, frame_height)) # ensures it is the same height as the base image, as it has the whitespace built in the image
    #Iterate through a 4 by 4 grid with 100 spacing, to place my image
    # y_cord = (j//images_per_row)*scaled_img_height
    # new_im.paste(im, (i,y_cord))
    new_im.paste(overlay_img)
    if not should_defer_show:
        new_im.show()
    new_im.save(outputIconPath, "PNG")
    # Close the loaded images:
    base_img.close()
    overlay_img.close()

    return new_im


def build_icon_example_grid(icons_path=Path(r"C:\Users\path\to\images"), should_save:bool=False, should_defer_show:bool=False) -> Image:

    frame_width = 1920
    images_per_row = 5
    padding = 2

    os.chdir(icons_path)

    images = glob.glob("*.png")
    images = images[:30]                #get the first 30 images

    img_width, img_height = Image.open(images[0]).size
    sf = (frame_width-(images_per_row-1)*padding)/(images_per_row*img_width)       #scaling factor
    scaled_img_width = ceil(img_width*sf)                   #s
    scaled_img_height = ceil(img_height*sf)

    number_of_rows = ceil(len(images)/images_per_row)
    frame_height = ceil(sf*img_height*number_of_rows) 

    new_im = Image.new('RGB', (frame_width, frame_height))

    i,j=0,0
    for num, im in enumerate(images):
        if num%images_per_row==0:
            i=0
        im = Image.open(im)
        #Here I resize my opened image, so it is no bigger than 100,100
        im.thumbnail((scaled_img_width,scaled_img_height))
        #Iterate through a 4 by 4 grid with 100 spacing, to place my image
        y_cord = (j//images_per_row)*scaled_img_height
        new_im.paste(im, (i,y_cord))
        print(i, y_cord)
        i=(i+scaled_img_width)+padding
        j+=1

    if not should_defer_show:
        new_im.show()
        
    if should_save:
        new_im.save("out.jpg", "JPEG", quality=80, optimize=True, progressive=True)
        
    return new_im



class ImageHelpers:
    """ 
    from pyphocorehelpers.image_helpers import ImageHelpers
    
    """
    @classmethod
    def get_font_path(cls, *args) -> Path:
        """ 
        from pyphocorehelpers.image_helpers import ImageHelpers
        
        
        fonts_folder_path: Path = ImageHelpers.get_font_path()
        Assert.path_exists(fonts_folder_path)

        a_font_path: Path = ImageHelpers.get_font_path('FreeMono.ttf')
        Assert.path_exists(a_font_path)

        
        """
        fonts_folder_path = resources.files('pyphocorehelpers.Resources').joinpath('fonts')
        Assert.path_exists(fonts_folder_path)
        final_font_path: Path = fonts_folder_path
        for a_path_part in args:
            final_font_path = final_font_path.joinpath(a_path_part)
        Assert.path_exists(final_font_path)  
        return final_font_path
    

    @classmethod
    def get_font(cls, *args, size:int=40) -> ImageFont:
        """ gets the actual font

        Usage:         
            from pyphocorehelpers.image_helpers import ImageHelpers
            # get a font
            fnt = ImageHelpers.get_font('FreeMono.ttf', size=88)
            fnt
            ## OUTPUTS: a_font_path

        """
        a_font_path = cls.get_font_path(*args)
        # get a font
        return ImageFont.truetype(a_font_path.as_posix(), size)

    

    @classmethod
    def empty_image(cls, width: int=800, height: int=600, background_color = (255, 255, 255, 0)) -> PIL.Image.Image:
        """ Creates an empty/blank image with specified dimensions and the optional background_color
        Usage:
            from pyphocorehelpers.image_helpers import ImageHelpers
            
            empty_image = ImageHelpers.empty_image(width=800, height=200)
            empty_image

        """
        # Create a new empty image
        # Parameters: mode (RGB, RGBA, etc.), size (width, height), color (default is black)

        if len(background_color) > 3:
            # Create a solid background image
            img_type: str = 'RGB'
        else:
            # Create a transparent image (with alpha channel)
            assert len(background_color) == 4, f"length of background_color should be 3 or 4, but it was: {background_color}"
            img_type: str = 'RGBA'
            
        
        return Image.new(img_type, (width, height), background_color)


    @classmethod
    def load_png_images_pathlib(cls, directory_path: Path, debug_print:bool=False) -> Dict:
        """ For the specified directory, loads (non-recurrsively) all the .png images present in the folder as PIL.Image objects
        
        Expects images with names like: 'p_x_given_n[5].png'
        
        
        # Example usage:
            a_path = flat_img_out_paths[0].joinpath('raw_rgba').resolve()
            Assert.path_exists(a_path)
            print(f'a_path: {a_path}')
            # parent_output_folder = Path('output/array_to_images').resolve()
            images_dict = load_png_images_pathlib(a_path)

            # Print the loaded images
            print(f"Loaded {len(images_dict)} PNG images:")
            # for name, img in images_dict.items():
            #     print(f"{name}: {img.format}, {img.size}, {img.mode}")

        """
        
        # Sort the images by their numeric index
        def extract_index(key):
            # Extract the number between '[' and ']'
            import re
            match = re.search(r'\[(\d+)\]', key)
            if match:
                return int(match.group(1))
            return 0


        # ==================================================================================================================================================================================================================================================================================== #
        # begin function body                                                                                                                                                                                                                                                                  #
        # ==================================================================================================================================================================================================================================================================================== #

        # Convert to Path object if it's a string
        directory = Path(directory_path)
        
        # Get all PNG files in the directory
        png_files = list(directory.glob("*.png"))
        
        # Load each image as a PIL Image object
        images = {}
        for file_path in png_files:
            try:
                img = Image.open(file_path)
                # Use the filename (without extension) as the key
                filename = file_path.stem
                images[filename] = img
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        # Get the sorted keys
        sorted_keys = sorted(images.keys(), key=extract_index)

        # Create a dictionary with sorted images
        sorted_images_dict = {key: images[key] for key in sorted_keys}

        if debug_print:
            # Display the sorted order
            for key in sorted_keys:
                print(f"{key}: {images[key].size}")

        # Return the sorted dictionary
        return sorted_images_dict
        

    @classmethod
    def build_img_html_tag(cls, img):
        """ used for building Obsidian Canvas nodes 
        
        Usage:
        
            html_img_tag = build_img_html_tag(img)
            create_text_node(html_img_tag, script_data["x"], script_data["y"]+script_data["height"]+120, 400, 400)

        """
        # Convert the image to bytes
        img_byte_array = io.BytesIO()
        img.save(img_byte_array, format=img.format)
        img_byte_array = img_byte_array.getvalue()

        # Convert the bytes to base64
        base64_str = base64.b64encode(img_byte_array).decode('utf-8')

        # Create HTML image tag
        html_img_tag = f'<img src="data:image/png;base64,{base64_str}" alt="Image">'
        return html_img_tag





# def _main():
#     # selectedBaseIconFilename = 'timeline-svgrepo-com.svg' # doesn't work without conversion because it's an SVG
#     selectedBaseIconFilename = 'heat-map-icon-21.jpg'

#     # 'noise-control-off-remove'
#     selectedOverlayIconFilename = r'png\1x\noise-control-off-delete.png'
#     # selectedOverlayIconFilename = r'svg\noise-control-off-add.svg'

#     baseIconPath = Path(r'C:\Users\pho\repos\VSCode Extensions\vscode-favorites\icons\favorites.png').resolve()
#     # baseIconPath = BaseIconParentPath.joinpath(selectedBaseIconFilename)
#     overlayIconPath = OverlayIconsParentPath.joinpath(selectedOverlayIconFilename)

#     new_icon = build_icon(baseIconPath, overlayIconPath, outputIconPath='new_test_overlay_icon.png')
#     return new_icon
#     # return build_icon()



# if __name__ == '__main__':
#     BaseIconParentPath = Path(r'C:\Users\pho\repos\Spike3DWorkEnv\Spike3D\EXTERNAL\Design\Icons\Potential')
# 	OverlayIconsParentPath = Path(r'C:\Users\pho\repos\Spike3DWorkEnv\Spike3D\EXTERNAL\Design\Icons\Potential\Overlays')
#     _main()
