from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
from pathlib import Path
import shutil

from pyjsoncanvas import (
    Canvas,
    TextNode,
    FileNode,
    # LinkNode,
    # GroupNode,
    # GroupNodeBackgroundStyle,
    Edge,
    Color,
)

import shutil ## for copying images
from pyphocorehelpers.image_helpers import ImageHelpers


# @metadata_attributes(short_name=None, tags=['canvas', 'obsidian'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-05-21 15:17', related_items=[])
class ObsidianCanvasHelper:
    """ Helps with Obsidian Canvases

    Usage:    
		from pyphocorehelpers.Filesystem.obsidian_canvas_helpers import ObsidianCanvasHelper
		
		canvas_url = Path(r"D:/PhoGlobalObsidian2022/🌐🧠 Working Memory/Pho-Kamran Paper 2024/2025-05-15 - Pho Sorted Events.canvas")
		write_modified_canvas_path: Path = canvas_url.with_name(f'_programmatic_test.canvas')
		loaded_canvas = ObsidianCanvasHelper.load(canvas_url=canvas_url)

		# image_folder_path: Path = Path(r'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/collected_outputs/2025-05-21/gor01_two_2006-6-07_16-40-19_normal_computed_[1, 2]_5.0/ripple/psuedo2D_nan_filled/raw_rgba').resolve()
		image_folder_path: Path = Path(r'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/collected_outputs/2025-05-21/gor01_two_2006-6-07_16-40-19_normal_computed_[1, 2]_5.0/laps/psuedo2D_nan_filled/raw_rgba').resolve()

		target_canvas, _write_status = ObsidianCanvasHelper.add_images_to_canvas(image_folder_path=image_folder_path, image_glob="p_x_given_n*.png", target_canvas=None, write_modified_canvas_path=write_modified_canvas_path, override_write_mode='w')
		# target_canvas, _write_status



    """
    
    @classmethod
    def load(cls, canvas_url: Path) -> Canvas:
        # Load JSON from a string
        with open(canvas_url, 'r', encoding='utf-8') as f:
            # canvas_json = json.load(f)
            canvas_json = f.read()
            
        # Create Canvas from the loaded JSON
        loaded_canvas = Canvas.from_json(json_str=canvas_json)
        return loaded_canvas
    
    @classmethod
    def save(cls, canvas: Canvas, canvas_url: Path, write_mode:str='w'):
        """ Save Canvas back to file 
        
        _status_code
        
        """
        ## INPUTS: active_canvas
        # Save the canvas as JSON
        json_str = canvas.to_json()
        with open(canvas_url, mode=write_mode, encoding='utf-8') as f:
            # canvas_json = json.load(f)
            _status_code = f.write(json_str)
            if _status_code:
                print(f'write to canvas: {_status_code}')
        return _status_code      


    @classmethod 
    def add_images_to_canvas(cls, image_folder_path: Path, image_glob: str = "*.png", target_canvas: Optional[Canvas]=None, write_modified_canvas_path: Path=None, vault_relative_image_dir_filepath: str = 'z__META\__IMAGES', x_padding: int = 2, override_write_mode='x', debug_print = False, canvas_image_node_scale: float=0.2):
        """ 
        Adds the images to the canvas
        
        """
        def _subfn_get_img_obsidian_global_unique_name(an_img_path: Path) -> str:    
            """ from my personal export path conventions, builds a globally unique image name (which has to be the case for a canvas)"""
            
            img_path_name_parts = an_img_path.parts
            date_part_index = next((i for i, part in enumerate(img_path_name_parts) if re.match(r'^\d{4}-\d{2}-\d{2}', part)), None)
            session_part_index: int = date_part_index + 1
            img_out_context_parts: List[str] = img_path_name_parts[session_part_index:] # ('gor01_two_2006-6-07_16-40-19_normal_computed_[1, 2]_5.0', 'ripple', 'psuedo2D_nan_filled', 'raw_rgba', 'p_x_given_n[9].png')
            return '-'.join(img_out_context_parts) # 'gor01_two_2006-6-07_16-40-19_normal_computed_[1, 2]_5.0-ripple-psuedo2D_nan_filled-raw_rgba-p_x_given_n[9].png'


        # ==================================================================================================================================================================================================================================================================================== #
        # BEGIN FUNCTION BODY                                                                                                                                                                                                                                                                  #
        # ==================================================================================================================================================================================================================================================================================== #
        images_dict: Dict = ImageHelpers.load_png_images_pathlib(image_folder_path, image_glob=image_glob)
        # Print the loaded images
        print(f"Loaded {len(images_dict)} PNG images from '{image_folder_path}'.")

        ## INPUTS: loaded_canvas
        # Create a new canvas
        if target_canvas is None:
            print(f'creating new Canvas as none was provided!')
            write_mode = 'x' # set write mode to create only so it doesn't overwrite an existing cnvas
            target_canvas = Canvas(nodes=[], edges=[])
        else:
            write_mode = 'w' # overwrite
            
        if override_write_mode is not None:
            write_mode = override_write_mode

        ## INPUTS: active_canvas, debug_print
        obsidian_vault_root_path = Path(r'D:\PhoGlobalObsidian2022')
        
        obsidian_canvas_image_link_path: Path = obsidian_vault_root_path.joinpath(vault_relative_image_dir_filepath) #  Path(r'D:\PhoGlobalObsidian2022\z__META\__IMAGES')
        # image path is like "collected_outputs\2025-05-21\gor01_two_2006-6-07_16-40-19_normal_computed_[1, 2]_5.0\ripple\psuedo2D_nan_filled\raw_rgba\p_x_given_n[7].png"

        initial_x = 0
        initial_y = 0
        max_num_to_add: int = 1000
        n_added: int = 0
        
        for i, (img_name, an_img) in enumerate(images_dict.items()):
            if i < max_num_to_add:
                an_img_path: Path = image_folder_path.joinpath(f'{img_name}.png')
                assert an_img_path.exists(), f"an_img_path: {an_img_path} does not exist"        
                global_unique_image_filename: str = f"{_subfn_get_img_obsidian_global_unique_name(an_img_path=an_img_path)}"
                an_img_vault_filepath = obsidian_canvas_image_link_path.joinpath(global_unique_image_filename)
                # assert not vault_image_filepath.exists()
                # Copy the file
                shutil.copy2(an_img_path, an_img_vault_filepath)
                if debug_print:
                    print(f'copying image from: "{an_img_path}" to vault_image_filepath: "{an_img_vault_filepath}"...')

                # an_img_height, an_img_width = an_img.size
                an_img_width, an_img_height = an_img.size
                
                if canvas_image_node_scale is not None:
                    an_img_width = int(round(canvas_image_node_scale * float(an_img_width)))
                    an_img_height = int(round(canvas_image_node_scale * float(an_img_height)))
                # node_url_str: str = vault_relative_image_dir_filepath
                node_url_str: str = an_img_vault_filepath.relative_to(obsidian_vault_root_path).as_posix()
                file_node = FileNode(x=initial_x, y=initial_y, width=an_img_width, height=an_img_height, file=node_url_str)
                target_canvas.add_node(file_node)
                initial_x = initial_x + an_img_width + x_padding
                n_added = n_added + 1
                
            else:
                # print(f'skipping because max_num_to_add: {max_num_to_add}')
                pass
            
        # END for i, (img_name, an_i...
        print(f'added {n_added} images to canvas.')
            
        # save the canvas back to a file _____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        if write_modified_canvas_path is not None:
            # write_modified_canvas_path: Path = write_modified_canvas_path.with_name(f'_programmatic_test.canvas')
            _write_status = cls.save(canvas=target_canvas, canvas_url=write_modified_canvas_path, write_mode=write_mode)
        else:
            print(f'no write_modified_canvas_path provided, skipping write')
            _write_status = None

        return target_canvas, _write_status
