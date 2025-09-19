def calculate_layer_compositions(initial_data, dilution_percentage):
    """
    calculating layer-wise compositions with dilution effects from previous layers.

    parameters:
        initial_data (np.ndarray): Path to input CSV file containing initial compositions
        dilution_percentage (float): How much of the previous layer affects the current layer (0-100)

    Returns:
        np.ndarray: Output array
    """
    # Calculate dilution factor (as a decimal)
    dilution_factor = dilution_percentage / 100.0

    output_data = initial_data * (1 - dilution_factor)
    output_data[..., 0, :] = initial_data[..., 0, :]

    for i in range(1, initial_data.shape[-2]):
        previous_layer_comp = output_data[..., i - 1, :]
        current_layer_comp = output_data[..., i, :]

        # Apply dilution, always positive
        diluted_portion = previous_layer_comp * dilution_factor

        output_data[..., i, :] = current_layer_comp + diluted_portion

    return output_data
