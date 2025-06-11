import asyncio
from time import time
from threading import Timer # used in `throttle(...)`
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from enum import Enum
import ipywidgets as widgets
from IPython.display import display


def create_dry_run_execution_widget(execution_function, execute_button_label="Execute for Real", dry_run_result_label="Dry Run Results:", 
                                   result_display_height="300px", layout_width="100%"):
    """Creates a widget that displays dry run results and provides a button to execute for real.

    Parameters:
    -----------
    execution_function : callable
        A function that takes is_dryrun as parameter and returns results to display
    execute_button_label : str
        Label for the execution button
    dry_run_result_label : str
        Label for the dry run results section
    result_display_height : str
        Height of the result display area
    layout_width : str
        Width of the overall widget

    Returns:
    --------
    widgets.VBox
        The interactive widget container

    Example:
    --------
    def run_file_operations(is_dryrun=True):
        # Your code here that uses is_dryrun parameter
        result = copy_dict, moved_files_dict_files = AcrossSessionHelpers._copy_exported_files_from_session_folder_to_collected_outputs(
            BATCH_DATE_TO_USE=BATCH_DATE_TO_USE, 
            cuttoff_date=cuttoff_date, 
            target_dir=collected_outputs_directory,
            custom_file_globs_dict={'h5': '*.h5'}, 
            is_dry_run=is_dryrun)
        return result

    widget = create_dry_run_execution_widget(run_file_operations)
    display(widget)
    """
    import ipywidgets as widgets
    from IPython.display import display

    # Run the function in dry run mode first
    dry_run_result = execution_function(is_dryrun=True)

    # Create output widget to display the dry run results
    output_area = widgets.Output(layout=widgets.Layout(height=result_display_height, width=layout_width, 
                                                      border='1px solid #ddd', overflow_y='auto'))
    with output_area:
        display(dry_run_result)

    # Create status area for execution results
    status_area = widgets.Output(layout=widgets.Layout(width=layout_width))

    # Create the execution button
    execute_button = widgets.Button(
        description=execute_button_label,
        button_style='danger',  # Use danger to indicate this is a real operation
        tooltip='Execute the operation for real (not a dry run)',
        icon='check'
    )

    def on_execute_button_clicked(_):
        # Clear previous status
        status_area.clear_output()

        # Disable button while executing
        execute_button.disabled = True
        execute_button.description = "Executing..."

        try:
            with status_area:
                print("Executing operation...")
                real_result = execution_function(is_dryrun=False)
                print("Operation completed successfully!")
                display(real_result)
        except Exception as e:
            with status_area:
                print(f"Error during execution: {str(e)}")
        finally:
            # Re-enable button
            execute_button.disabled = False
            execute_button.description = execute_button_label

    execute_button.on_click(on_execute_button_clicked)

    # Create labels
    dry_run_header = widgets.HTML(f"<h3>{dry_run_result_label}</h3>")
    execution_header = widgets.HTML("<h3>Execution Results:</h3>")

    # Assemble the widget
    return widgets.VBox([
        dry_run_header,
        output_area,
        execute_button,
        execution_header,
        status_area
    ])


class DryRunExecutionContext:
    """A context manager that captures code execution with is_dryrun=True and provides 
    a button to re-execute with is_dryrun=False.

    Usage:
    ------
    with DryRunExecutionContext() as ctx:
        # Your code here that uses ctx.is_dryrun
        copy_dict, moved_files_dict_files = AcrossSessionHelpers._copy_exported_files_from_session_folder_to_collected_outputs(
            BATCH_DATE_TO_USE=BATCH_DATE_TO_USE, 
            cuttoff_date=cuttoff_date, 
            target_dir=collected_outputs_directory,
            custom_file_globs_dict={'h5': '*.h5'}, 
            is_dry_run=ctx.is_dryrun)

        # Store your results to be displayed
        ctx.results = (copy_dict, moved_files_dict_files)
    """

    def __init__(self, execute_button_label="Execute for Real", 
                 dry_run_result_label="Dry Run Results:", 
                 result_display_height="300px", 
                 layout_width="100%"):
        """Initialize the context manager.

        Parameters:
        -----------
        execute_button_label : str
            Label for the execution button
        dry_run_result_label : str
            Label for the dry run results section
        result_display_height : str
            Height of the result display area
        layout_width : str
            Width of the overall widget
        """
        import ipywidgets as widgets
        from IPython.display import display

        self.is_dryrun = True
        self.results = None
        self.execute_button_label = execute_button_label
        self.dry_run_result_label = dry_run_result_label
        self.result_display_height = result_display_height
        self.layout_width = layout_width

    def __enter__(self):
        """Enter the context with is_dryrun=True."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and create the widget to display results and execute button."""
        import ipywidgets as widgets
        from IPython.display import display

        if exc_type is not None:
            # Exception occurred, return False to allow exception to propagate
            return False

        # Store the dry run results
        dry_run_results = self.results

        # Create output widget to display the dry run results
        output_area = widgets.Output(layout=widgets.Layout(
            height=self.result_display_height, 
            width=self.layout_width, 
            border='1px solid #ddd', 
            overflow_y='auto'
        ))

        with output_area:
            display(dry_run_results)

        # Create status area for execution results
        status_area = widgets.Output(layout=widgets.Layout(width=self.layout_width))

        # Create the execution function that captures the original code
        def execute_for_real():
            # Store the original results
            original_results = self.results

            # Set is_dryrun to False for real execution
            self.is_dryrun = False
            self.results = None

            # Re-run the code by re-entering the context
            # This is a placeholder as we can't actually re-run the context
            # Instead, we'll provide instructions to the user
            return "You need to re-run the cell with is_dryrun=False to execute for real."

        # Create the execution button
        execute_button = widgets.Button(
            description=self.execute_button_label,
            button_style='danger',  # Use danger to indicate this is a real operation
            tooltip='Execute the operation for real (not a dry run)',
            icon='check'
        )

        def on_execute_button_clicked(_):
            # Show instructions to set is_dryrun=False and re-run the cell
            status_area.clear_output()
            with status_area:
                print("To execute for real:")
                print("1. Locate the 'is_dryrun = True' line in your code")
                print("2. Change it to 'is_dryrun = False'")
                print("3. Re-run the cell")

        execute_button.on_click(on_execute_button_clicked)

        # Create labels
        dry_run_header = widgets.HTML(f"<h3>{self.dry_run_result_label}</h3>")
        execution_header = widgets.HTML("<h3>To Execute For Real:</h3>")

        # Assemble the widget
        widget = widgets.VBox([
            dry_run_header,
            output_area,
            execute_button,
            execution_header,
            status_area
        ])

        # Display the widget
        display(widget)

        # Return True to suppress any exceptions
        return True


class InteractiveDryRunContext:
    """A context manager that allows toggling between dry run and real execution 
    with interactive widgets.

    Usage:
    ------
    with InteractiveDryRunContext() as ctx:
        # Your code here that uses ctx.is_dryrun
        copy_dict, moved_files_dict_files = AcrossSessionHelpers._copy_exported_files_from_session_folder_to_collected_outputs(
            BATCH_DATE_TO_USE=BATCH_DATE_TO_USE, 
            cuttoff_date=cuttoff_date, 
            target_dir=collected_outputs_directory,
            custom_file_globs_dict={'h5': '*.h5'}, 
            is_dry_run=ctx.is_dryrun)

        # Optionally display or return results
        display(copy_dict)
    """

    def __init__(self, initial_dry_run=True):
        """Initialize the context manager.

        Parameters:
        -----------
        initial_dry_run : bool
            Initial state of the dry run toggle
        """
        import ipywidgets as widgets
        from IPython.display import display

        self.is_dryrun = initial_dry_run

        # Create the toggle widget
        self.toggle = widgets.ToggleButton(
            value=initial_dry_run,
            description='Dry Run Mode' if initial_dry_run else 'Real Execution Mode',
            button_style='info' if initial_dry_run else 'danger',
            tooltip='Toggle between dry run and real execution',
            icon='check' if not initial_dry_run else 'ban'
        )

        def on_toggle_change(change):
            self.is_dryrun = change['new']
            self.toggle.description = 'Dry Run Mode' if self.is_dryrun else 'Real Execution Mode'
            self.toggle.button_style = 'info' if self.is_dryrun else 'danger'
            self.toggle.icon = 'ban' if self.is_dryrun else 'check'

        self.toggle.observe(on_toggle_change, names='value')

        # Display the toggle widget
        display(self.toggle)

    def __enter__(self):
        """Enter the context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        # Just pass through any exceptions
        return False


