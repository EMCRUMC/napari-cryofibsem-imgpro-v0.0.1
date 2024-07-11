import cv2
import numpy as np
from napari.layers import Image
import concurrent.futures
from magicgui import magic_factory
from napari_plugin_engine import napari_hook_implementation


def process_slice(slice_data, central_tendency, average_intensity, datatype):

    slice_adjusted = None
    if central_tendency == "median":
        current_intensity = np.median(slice_data)
        ratio = average_intensity / current_intensity

        if datatype == "uint16":
            slice_adjusted = np.clip(slice_data * ratio, a_min=0, a_max=65535).astype(np.uint16)
        elif datatype == "uint8":
            slice_adjusted = np.clip(slice_data * ratio, a_min=0, a_max=255).astype(np.uint8)

    elif central_tendency == "mean":
        current_intensity = np.mean(slice_data)
        ratio = average_intensity / current_intensity

        if datatype == "uint16":
            slice_adjusted = np.clip(slice_data * ratio, a_min=0, a_max=65535).astype(np.uint16)
        elif datatype == "uint8":
            slice_adjusted = np.clip(slice_data * ratio, a_min=0, a_max=255).astype(np.uint8)

    return slice_adjusted


@magic_factory(
    call_button="Correct Brightness Variation",
    stack={"label": "Input Stack"},
    central_tendency={"label": "Central Tendency", "choices": ["median", "mean"]},
    pixel_value_adjust={"label": "Average Intensity Adjustment", "widget_type": "SpinBox", "min": -10000, "max": 10000}
)
def brightness(
        stack: Image,
        central_tendency: str = "median",
        pixel_value_adjust: int = 0
) -> Image:
    """
    This widget corrects the global brightness variations across the slices of a stack.
    It simply obtains the average pixel intensity value of all the slices, calculates a
    ratio between the average to the current slice, and adjusts the pixel intensity values
    by multiplying to the ratio.

    Parameters
    ----------
    Stack : "Image"
        Stack to be processed

    Central Tendency : str
        Chosen measure of central tendency

    Average Intensity Adjustment : int
        Value that may be added or subtracted to the average intensity value

    Returns
    -------
        napari Image layer containing the decurtained image

    """
    if stack is None:  # Handles null cases
        print("Please select a stack.")
        return

    if len(stack.data.shape) > 2:
        datatype = stack.data.dtype
        stack = stack.data
        processed_slices = []
        slice_order = []  # To keep track of slice order

        average_intensity = None
        if central_tendency == "median":
            median_intensities = np.median(stack, axis=(1, 2))
            average_intensity = np.median(median_intensities)
        elif central_tendency == "mean":
            mean_intensities = np.mean(stack, axis=(1, 2))
            average_intensity = np.mean(mean_intensities)
        average_intensity += pixel_value_adjust

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_slice = {
                executor.submit(process_slice, stack[slice_idx], central_tendency, average_intensity, datatype):
                    slice_idx for slice_idx in range(stack.shape[0])}
            for future in concurrent.futures.as_completed(future_to_slice):
                slice_idx = future_to_slice[future]
                slice_order.append(slice_idx)
                processed_slices.append(future.result())

        # Sort processed slices based on original order
        processed_slices = [x for _, x in sorted(zip(slice_order, processed_slices))]
        processed_stack = np.stack(processed_slices)

    else:
        processed_stack = None
        print("Please select a stack, not a single image.")

    stack_name = f"Corrected_stack_{central_tendency}_{pixel_value_adjust}"

    # Returns the processed stack with the parameters in the name
    return Image(processed_stack, name=stack_name)


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return brightness
