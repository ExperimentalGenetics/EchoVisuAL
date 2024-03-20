import numpy as np
from pydicom import dcmread
from pydicom.errors import InvalidDicomError
from skimage.draw import polygon


def add_padding(image, padding='constant'):
    """
    Add padding for given image above and below
    :param padding: edge|constant as a padding method
    :param image: array with image data
    :return: padded image
    """
    padding_height = (1024 - image.shape[0]) // 2
    padding_width = (1024 - image.shape[1]) // 2
    if padding == 'edge':
        return np.pad(image, ((padding_height, padding_height), (padding_width, padding_width), (0, 0)), mode='edge')
    else:
        return np.pad(image, ((padding_height, padding_height), (padding_width, padding_width), (0, 0)),
                      mode='constant', constant_values=0)


def read_frame_from_dicom_for_training(path, frame_number, x0, y0, x1, y1, padding='constant'):
    """
    Read a dicom frame for training and apply optionally augmentation and padding
    :param padding:
    :param path: file_path
    :param frame_number: numer of the frame
    :param x0: x0 coordinate - needed for cropping
    :param y0: y0 coordinate - needed for cropping
    :param x1: x1 coordinate - needed for cropping
    :param y1: y1 coordinate - needed for cropping
    :return: frame
    """
    # Convert Eager-Tensor
    path = path.numpy().decode("utf-8")

    frame = read_frame_from_dicom(path, frame_number, x0, y0, x1, y1)
    # Add padding
    frame = add_padding(frame, padding)
    return frame


def read_frame_from_dicom(path, frame_number, x0, y0, x1, y1):
    """
    Read a dicom frame and crop it with provided coordinates
    :param path: file_path
    :param frame_number: numer of the frame
    :param x0: x0 coordinate - needed for cropping
    :param y0: y0 coordinate - needed for cropping
    :param x1: x1 coordinate - needed for cropping
    :param y1: y1 coordinate - needed for cropping
    :return: frame
    """
    try:
        file = dcmread(path)
    except InvalidDicomError:
        raise InvalidDicomError

    pixel_data = file.pixel_array

    if file[0x0028, 0x0008].value == 1:
        cropped_pixel_data = pixel_data[y0:y1, x0:x1, :]
    else:
        cropped_pixel_data = pixel_data[:, y0:y1, x0:x1, :]
        cropped_pixel_data = cropped_pixel_data[frame_number - 1, :, :]
    return cropped_pixel_data


def create_binary_mask(x_coords, y_coords, height, width):
    """
    Creates a binary mask if coordinates are given, otherwise creates a mask with only background pixels
    :param x_coords: x coordinates of polygon shape
    :param y_coords: y coordinates of polygon shape
    :param height: image height
    :param width: image width
    :return: mask with given height and width
    """
    # Create an empty binary_mask
    binary_mask = np.zeros((height, width, 1), dtype=int)

    # Generate the polygon binary_mask
    if x_coords != '-1':
        xcoords = np.fromstring(x_coords, dtype=int, sep=',')
        ycoords = np.fromstring(y_coords, dtype=int, sep=',')
        rr, cc = polygon(ycoords, xcoords, shape=(height, width))

        binary_mask[rr, cc] = 1
    # Add padding
    binary_mask = add_padding(binary_mask)
    return binary_mask


def create_binary_mask_for_training(x_coords, y_coords, x0, y0, x1, y1):
    """
    Creates a binary mask if coordinates are given, otherwise creates a mask with only background pixels
    :param x_coords: x coordinates of polygon shape
    :param y_coords: y coordinates of polygon shape
    :param x0: x0 coordinate of cutting frame
    :param y0: y0 coordinate of cutting frame
    :param x1: x1 coordinate of cutting frame
    :param y1: y1 coordinate of cutting frame
    :return: mask with given height and width
    """

    # Calculate image shapes
    height = y1 - y0
    width = x1 - x0
    # Convert Eager-Tensor
    x_coords_conv = x_coords.numpy().decode("utf-8")
    y_coords_conv = y_coords.numpy().decode("utf-8")

    return create_binary_mask(x_coords=x_coords_conv, y_coords=y_coords_conv, height=height, width=width)
