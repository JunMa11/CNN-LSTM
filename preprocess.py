import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image
import torch.nn.functional as F

intensityproperties = {
            "max": 1929.0,
            "mean": 74.16419982910156,
            "median": 77.99217987060547,
            "min": -406.9988098144531,
            "percentile_00_5": -55.999996185302734,
            "percentile_99_5": 180.0,
            "std": 44.36494064331055
        }

def fast_resize_segmentation(segmentation, new_shape, mode="nearest"):
    '''
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    '''
    tpe = segmentation.dtype

    if isinstance(segmentation, torch.Tensor):
        assert len(segmentation.shape[2:]) == len(new_shape), f"segmentation.shape = {segmentation.shape}, new_shape = {new_shape}"
    else:
        assert len(segmentation.shape[1:]) == len(new_shape), f"segmentation.shape = {segmentation.shape}, new_shape = {new_shape}"
        segmentation = torch.from_numpy(segmentation).unsqueeze(0).float()
    #if order == 0:
        #return resize(segmentation.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False).astype(tpe)
    if mode == "nearest":
        seg_torch = torch.nn.functional.interpolate(segmentation, new_shape, mode=mode)
        reshaped = seg_torch
    else:
        #reshaped = np.zeros(new_shape, dtype=segmentation.dtype)
        unique_labels = torch.unique(segmentation)
        seg_torch = segmentation
        reshaped = torch.zeros([*seg_torch.shape[:2], *new_shape], dtype=seg_torch.dtype, device=seg_torch.device)
        for i, c in enumerate(unique_labels):
            #mask = segmentation == c
            #reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            mask = seg_torch == c
            reshaped_multihot = torch.nn.functional.interpolate(mask.float(), new_shape, mode=mode, align_corners=False)
            reshaped[reshaped_multihot >= 0.5] = c

    return reshaped


def fast_resample_data_or_seg_to_shape(data,
                                  new_shape,
                                  is_seg: bool = False,
                                  order: int = 3, order_z: int = 0,
                                    ):

    device = torch.device("cpu")
    order_to_mode_map = {
        0: "nearest",
        1: "trilinear" if new_shape[0] > 1 else "bilinear",
        2: "trilinear" if new_shape[0] > 1 else "bilinear",
        3: "trilinear" if new_shape[0] > 1 else "bicubic",
        4: "trilinear" if new_shape[0] > 1 else "bicubic",
        5: "trilinear" if new_shape[0] > 1 else "bicubic",
    }
    
    if is_seg:
        #print(f"seg.shape: {data.shape}")
        resize_fn = fast_resize_segmentation
        kwargs = {
            "mode": order_to_mode_map[order]
        }
    else:
        #print(f"data.shape: {data.shape}")
        resize_fn = torch.nn.functional.interpolate
        kwargs = {
            'mode': order_to_mode_map[order],
            'align_corners': False
        }
    dtype_data = data.dtype
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        if not isinstance(data, torch.Tensor):
            torch_data = torch.from_numpy(data).float()
            #torch_data = torch.as_tensor(data.get())
        else:
            torch_data = data.float()
        if new_shape[0] == 1:
            torch_data = torch_data.transpose(1, 0)
            new_shape = new_shape[1:]
        else:
            torch_data = torch_data.unsqueeze(0)
        
        torch_data = resize_fn(torch_data.to(device), tuple(new_shape), **kwargs)

        if new_shape[0] == 1:
            torch_data = torch_data.transpose(1, 0)
        else:
            torch_data = torch_data.squeeze(0)

        # if use_gpu:
        #     torch_data = torch_data.cpu()
        reshaped_final_data = torch_data
        # if isinstance(data, np.ndarray):
        #     reshaped_final_data = torch_data.numpy().astype(dtype_data)
        # else:
        #     reshaped_final_data = torch_data.to(dtype_data)
        
        #print(f"Reshaped data from {shape} to {new_shape}")
        #print(f"reshaped_final_data shape: {reshaped_final_data.shape}")
        assert reshaped_final_data.ndim == 4, f"reshaped_final_data.shape = {reshaped_final_data.shape}"
        return reshaped_final_data
    else:
        print("no resampling necessary")
        return data

def resample_img(itk_image, out_spacing=[
                0.7333984375,
                0.7333984375,
                2.0,
            ], is_label=False, out_size = [], out_origin = [], out_direction= []):
    original_spacing = itk_image.GetSpacing()
    #print(original_spacing)
    original_size = itk_image.GetSize()
    

    if not out_size:
        out_size = [ int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
    out_size[2] = 84
    new_shape = [out_size[2], out_size[1], out_size[0]]
    image_array = sitk.GetArrayFromImage(itk_image)
    #print(image_array.shape, new_shape, original_size)
    resample_img = fast_resample_data_or_seg_to_shape(image_array[None], new_shape, is_label)
    # set up resampler

    return resample_img

def resample_cmr(itk_image, is_label=False, out_origin = [], out_direction= []):
    original_spacing = itk_image.GetSpacing()
    out_spacing = original_spacing
    #print(original_spacing)
    original_size = itk_image.GetSize()
    
    out_size = list(original_size)
    image_array = sitk.GetArrayFromImage(itk_image)
    #print(out_size)
    if original_size[2] < 25:
        out_size[2] = 25
        new_shape = [out_size[2], out_size[1], out_size[0]]
        #print(image_array.shape, new_shape, original_size)
        resample_img = fast_resample_data_or_seg_to_shape(image_array[None], new_shape, is_label)[0]
        
    elif original_size[2] > 25:
        resample_img = image_array[:25, :, :]
    else:
        resample_img = image_array
    #print(itk_image.GetSize())

    return resample_img

def normalize(image):
    '''
    CT normalization
    '''
    mean_intensity = intensityproperties['mean']
    std_intensity = intensityproperties['std']
    lower_bound = intensityproperties['percentile_00_5']
    upper_bound = intensityproperties['percentile_99_5']

    image = torch.clamp(image, lower_bound, upper_bound)
    image -= mean_intensity
    image /= max(std_intensity, 1e-8)
    return image

def z_normalize(image):
    mean = image.mean()
    std = image.std()
    image -= mean
    image /= (max(std, 1e-8))
    return image



def pad_3d(image_array: np.ndarray) -> np.ndarray:
    """
    Pads each slice of a 3D medical image array (from SimpleITK) to make it square and resizes it to the target size.
    
    :param image_array: Input NumPy array (assumed to be a 3D medical image with shape [depth, height, width])
    :param target_size: Target size (width and height will be equal for each slice)
    :param fill_color: Value for padding (default: 0)
    :return: Processed 3D NumPy array with shape [depth, target_size, target_size]
    """
    # Get original dimensions
    D, H, W = image_array.shape
    
    # Determine padding to make H == W (square)
    pad_h = max(0, W - H)
    pad_w = max(0, H - W)
    
    # Pad equally on both sides
    padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    if isinstance(image_array, np.ndarray):
        image_array = torch.from_numpy(image_array)
    padded_image = F.pad(image_array, padding, mode='constant', value=0)
    return padded_image




