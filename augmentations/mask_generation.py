import cv2
import numpy as np


def extract_masked_region(image_path, mask_path, output_path):
    """
    Extracts a region from an image based on a mask, creating a PNG with a transparent background.

    Args:
        image_path: Path to the input image (any format).
        mask_path: Path to the mask image (any format, non-black pixels are the mask).
        output_path: Path to save the extracted region as a PNG with transparency.
    """
    try:
        # Load images
        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale

        if img is None or mask is None:
            raise FileNotFoundError("Image or mask file not found.")
        if img.shape[:2] != mask.shape[:2]:
            raise ValueError("Image and mask dimensions must match.")

        # Threshold the mask: non-black pixels become white, black pixels become black
        _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)


        # Apply mask to the image
        masked_img = cv2.bitwise_and(img, img, mask=thresh)

        # Convert to RGBA to add alpha channel.
        masked_img_rgba = cv2.cvtColor(masked_img, cv2.COLOR_BGR2BGRA)
        masked_img_rgba[:, :, 3] = thresh  #Set the alpha channel.

        cv2.imwrite(output_path, masked_img_rgba)
        print(f"Masked region saved to {output_path}")

    except cv2.error as e:
        print(f"OpenCV error: {e}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")



def overlay_image_with_interpolation(background_img, foreground_img, alpha=0.5, x_offset=0, y_offset=0, blur_radius=5):
    """
    Overlays a foreground image onto a background image with interpolation for a smoother blend.

    Args:
        background_img: Background image (NumPy array).
        foreground_img: Foreground image (NumPy array), must have alpha channel (RGBA).
        alpha: Transparency of foreground (0.0 fully transparent, 1.0 fully opaque).
        x_offset: Horizontal offset.
        y_offset: Vertical offset.
        blur_radius: Radius for Gaussian blur (higher values = smoother blend).

    Returns:
        Overlaid image (NumPy array) or None if error.
    """
    try:
        rows, cols, channels = foreground_img.shape
        if channels != 4:
            raise ValueError("Foreground image must have an alpha channel (RGBA).")

        roi = background_img[y_offset:rows + y_offset, x_offset:cols + x_offset]
        if roi.shape[:2] != foreground_img.shape[:2]:
            raise ValueError("Foreground image size is incompatible with background ROI.")

        alpha_mask = foreground_img[:, :, 3] / 255.0
        alpha_mask = alpha * alpha_mask

        #Interpolate the alpha mask for smoother edges.
        blurred_alpha_mask = cv2.GaussianBlur(alpha_mask,(blur_radius,blur_radius),0)

        # Blend only the foreground with alpha, background remains opaque.
        for c in range(0, 3):
            roi[:, :, c] = blurred_alpha_mask * foreground_img[:, :, c] + (1 - blurred_alpha_mask) * roi[:, :, c]

        background_img[y_offset:rows + y_offset, x_offset:cols + x_offset] = roi
        return background_img

    except Exception as e:
        print(f"Error overlaying images: {e}")
        return None
