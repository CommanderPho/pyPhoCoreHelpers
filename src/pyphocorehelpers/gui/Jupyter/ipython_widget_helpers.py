import asyncio
from time import time

def throttle(wait):
    """ Decorator that prevents a function from being called more than once every wait period. 
    https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20Events.html#throttling
    
    from pyphocorehelpers.gui.Jupyter.ipython_widget_helpers import throttle
    
    
    """
    def decorator(fn):
        time_of_last_call = 0
        scheduled, timer = False, None
        new_args, new_kwargs = None, None
        def throttled(*args, **kwargs):
            nonlocal new_args, new_kwargs, time_of_last_call, scheduled, timer
            def call_it():
                nonlocal new_args, new_kwargs, time_of_last_call, scheduled, timer
                time_of_last_call = time()
                fn(*new_args, **new_kwargs)
                scheduled = False
            time_since_last_call = time() - time_of_last_call
            new_args, new_kwargs = args, kwargs
            if not scheduled:
                scheduled = True
                new_wait = max(0, wait - time_since_last_call)
                timer = Timer(new_wait, call_it)
                timer.start()
        return throttled
    return decorator


""" 
from pyphocorehelpers.gui.Jupyter.ipython_widget_helpers import throttle

slider = widgets.IntSlider()
text = widgets.IntText()

@throttle(0.2)
def value_changed(change):
    text.value = change.new
slider.observe(value_changed, 'value')

widgets.VBox([slider, text])


"""