def convert_pixel_distance_to_meters(pixel_distance, reference_height_in_meters, reference_height_in_pixels):
    """
    Converts a distance in pixels to meters.

    Parameters:
    - pixel_distance (float): Distance measured in pixels.
    - reference_height_in_meters (float): Known reference height in meters.
    - reference_height_in_pixels (float): Corresponding height in pixels.

    Returns:
    - float: Converted distance in meters.
    """
    return (pixel_distance * reference_height_in_meters) / reference_height_in_pixels

def convert_meters_to_pixel_distance(meters, reference_height_in_meters, reference_height_in_pixels):
    """
    Converts a distance in meters to pixels.

    Parameters:
    - meters (float): Distance measured in meters.
    - reference_height_in_meters (float): Known reference height in meters.
    - reference_height_in_pixels (float): Corresponding height in pixels.

    Returns:
    - float: Converted distance in pixels.
    """
    return (meters * reference_height_in_pixels) / reference_height_in_meters