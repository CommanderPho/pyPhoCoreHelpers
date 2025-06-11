import asyncio
from time import time
from threading import Timer # used in `throttle(...)`
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from enum import Enum
import ipywidgets as widgets
from IPython.display import display

class DryRunExecutionWidget:
    """A widget that executes a function in dry-run mode first and provides a button 
    to execute it for real.

    Usage:
    ------
    # Define the function that takes is_dryrun parameter
    def my_operation(is_dryrun=True):
        copy_dict, moved_files_dict_files = AcrossSessionHelpers._copy_exported_files_from_session_folder_to_collected_outputs(
            BATCH_DATE_TO_USE=BATCH_DATE_TO_USE, 
            cuttoff_date=cuttoff_date, 
            target_dir=collected_outputs_directory, 
            custom_file_globs_dict={'h5': '*.h5'}, 
            is_dry_run=is_dryrun
        )
        return (copy_dict, moved_files_dict_files)

    # Create and use the widget
    widget = DryRunExecutionWidget(my_operation)
    """

    def __init__(self, operation_function, execute_button_label="Execute for Real (Non-dry run)"):
        """Initialize the widget.

        Parameters:
        -----------
        operation_function : callable
            A function that takes an is_dryrun parameter and returns results
        execute_button_label : str
            Label for the execute button
        """
        import ipywidgets as widgets
        from IPython.display import display

        self.operation_function = operation_function

        # Create the execute button
        self.execute_button = widgets.Button(
            description=execute_button_label,
            button_style='danger',
            tooltip='Execute the operation for real (not a dry run)',
            icon='exclamation-triangle',
            layout=widgets.Layout(width='250px')
        )

        # Create status display
        self.status = widgets.HTML(
            value="<b>Status:</b> Previewing in dry run mode",
            layout=widgets.Layout(margin='10px 0')
        )

        # Container for preview results
        self.preview_area = widgets.Output(
            layout=widgets.Layout(
                border='1px solid #ddd',
                padding='10px',
                margin='10px 0',
                max_height='300px',
                overflow_y='auto'
            )
        )

        # Container for real execution results
        self.execution_area = widgets.Output(
            layout=widgets.Layout(
                border='1px solid #ddd',
                padding='10px',
                margin='10px 0',
                max_height='300px',
                overflow_y='auto'
            )
        )

        def on_execute_button_clicked(_):
            self._execute_for_real()

        self.execute_button.on_click(on_execute_button_clicked)

        # Create header labels
        self.preview_header = widgets.HTML("<h4>Dry Run Preview:</h4>")
        self.execution_header = widgets.HTML("<h4>Real Execution Results:</h4>")

        # Create the widget UI
        self.widget = widgets.VBox([
            self.status,
            self.preview_header,
            self.preview_area,
            self.execute_button,
            self.execution_header,
            self.execution_area
        ])

        # Execute in dry run mode immediately
        self._execute_dry_run()

        # Display the widget
        display(self.widget)

    def _execute_dry_run(self):
        """Execute the operation in dry run mode."""
        try:
            self.status.value = "<b>Status:</b> Running dry run preview..."

            # Clear previous preview
            self.preview_area.clear_output()

            # Execute the function with is_dryrun=True
            with self.preview_area:
                from IPython.display import display
                print("Dry run preview (no actual changes made):")
                results = self.operation_function(is_dryrun=True)
                display(results)

            self.status.value = "<b>Status:</b> Dry run preview completed"

        except Exception as e:
            self.status.value = f"<b>Status:</b> Error during dry run: {str(e)}"
            with self.preview_area:
                import traceback
                print(f"Error during dry run: {str(e)}")
                traceback.print_exc()

    def _execute_for_real(self):
        """Execute the operation for real (non-dry run)."""
        try:
            self.status.value = "<b>Status:</b> Executing for real..."
            self.execute_button.disabled = True

            # Clear previous execution results
            self.execution_area.clear_output()

            # Execute the function with is_dryrun=False
            with self.execution_area:
                from IPython.display import display
                print("Executing operation for real:")
                results = self.operation_function(is_dryrun=False)
                print("Real execution completed.")
                display(results)

            self.status.value = "<b>Status:</b> Real execution completed"

        except Exception as e:
            self.status.value = f"<b>Status:</b> Error during real execution: {str(e)}"
            with self.execution_area:
                import traceback
                print(f"Error during real execution: {str(e)}")
                traceback.print_exc()
        finally:
            self.execute_button.disabled = False
