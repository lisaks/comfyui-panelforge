import torch
import numpy as np
from PIL import Image, ImageDraw
import io

class FrameNode:
    """
    FrameNode: The basic building block for a comic panel
    
    This node processes an individual comic frame and handles
    its border, padding, and other style settings.
    """
    
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        """
        Define the input types for the frame node.
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "border_width": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 20,
                    "step": 1,
                    "display": "slider"
                }),
                "padding": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "display": "slider"
                }),
                "border_color": (["black", "white"],),
            },
            "optional": {
                "caption": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "FRAME_DATA")
    RETURN_NAMES = ("preview", "frame_data")
    FUNCTION = "process_frame"
    CATEGORY = "Pixstri/Comics"
    
    def process_frame(self, image, border_width, padding, border_color, caption=None):
        """
        Process a comic frame with borders and padding.
        
        Parameters:
        -----------
        image : torch.Tensor
            The input image tensor
        border_width : int
            Width of the frame border in pixels
        padding : int
            Internal padding between the border and the image
        border_color : str
            Color of the frame border ("black" or "white")
        caption : str, optional
            Text caption for the frame
            
        Returns:
        --------
        tuple(torch.Tensor, dict)
            The processed image with frame and the frame data
        """
        # Convert tensor to PIL image for processing
        # Take the first batch item if there are multiple
        if len(image.shape) == 4:
            pil_image = self._tensor_to_pil(image[0])
        else:
            pil_image = self._tensor_to_pil(image)
        
        # Get the dimensions
        width, height = pil_image.size
        
        # Create a new image with border and padding
        total_border = border_width + padding
        new_width = width + (total_border * 2)
        new_height = height + (total_border * 2)
        
        # Create a new image with the border color
        if border_color == "black":
            frame_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))
        else:
            frame_image = Image.new("RGB", (new_width, new_height), (255, 255, 255))
        
        # If padding > 0, create an inner padding area
        if padding > 0:
            inner_width = width + (padding * 2)
            inner_height = height + (padding * 2)
            inner_x = border_width
            inner_y = border_width
            
            # Create inner padding area (white or light gray)
            if border_color == "black":
                inner_color = (240, 240, 240)  # Light gray for black borders
            else:
                inner_color = (240, 240, 240)  # Light gray for white borders too
            
            # Draw the inner padding area
            draw = ImageDraw.Draw(frame_image)
            draw.rectangle(
                [(inner_x, inner_y), 
                 (inner_x + inner_width - 1, inner_y + inner_height - 1)],
                fill=inner_color
            )
        
        # Paste the original image onto the frame
        frame_image.paste(pil_image, (total_border, total_border))
        
        # Add caption if provided
        if caption and caption.strip():
            draw = ImageDraw.Draw(frame_image)
            # Simple caption at the bottom
            # In a real implementation, you'd want more sophisticated text rendering
            caption_y = new_height - border_width - 15  # Position above the border
            draw.text((total_border, caption_y), caption.strip(), fill=(0, 0, 0))
        
        # Convert back to tensor
        result_tensor = self._pil_to_tensor(frame_image)
        
        # Create frame data dictionary
        frame_data = {
            "width": new_width,
            "height": new_height,
            "image": result_tensor,
            "border_width": border_width,
            "padding": padding,
            "caption": caption if caption else ""
        }
        
        return (result_tensor, frame_data)
    
    def _tensor_to_pil(self, tensor):
        """Convert a PyTorch tensor to a PIL Image."""
        # Handle different tensor formats
        if not torch.is_tensor(tensor):
            return tensor  # Already a PIL image or something else
        
        # Clone tensor to avoid modifying the original
        tensor = tensor.clone()
        
        # Standardize tensor format - handle different shapes
        if len(tensor.shape) == 4:  # BCHW or BHWC or B111C
            if tensor.shape[1] == 3:  # BCHW format
                tensor = tensor[0].permute(1, 2, 0)  # Convert to HWC
            elif tensor.shape[3] == 3:  # BHWC format
                tensor = tensor[0]  # Just remove batch dim
            elif tensor.shape[1] == 1 and tensor.shape[2] == 1:  # B111C format?
                tensor = tensor[0, 0]  # Extract the image
        elif len(tensor.shape) == 3:  # CHW or HWC
            if tensor.shape[0] == 3:  # CHW format
                tensor = tensor.permute(1, 2, 0)  # Convert to HWC
        
        # Convert to numpy with proper scaling
        if tensor.dtype == torch.uint8:
            img_np = tensor.cpu().numpy()
        else:
            img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
        
        # Create PIL image - handle grayscale vs RGB
        if img_np.shape[-1] == 1:
            img_pil = Image.fromarray(img_np[:, :, 0], 'L')
        else:
            img_pil = Image.fromarray(img_np)
            
        return img_pil
    
    def _pil_to_tensor(self, pil_image):
        """Convert a PIL Image to a PyTorch tensor in BCHW format"""
        # Convert PIL to numpy array
        img_np = np.array(pil_image)
        
        # Keep as uint8 for better compatibility
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8)
        
        # Convert to tensor and keep as uint8
        img_tensor = torch.from_numpy(img_np)
        
        # Return in BCHW format
        if len(img_tensor.shape) == 2:  # Grayscale
            return img_tensor.unsqueeze(0).unsqueeze(0)
        else:  # RGB
            # Move channels to correct position and add batch dimension
            return img_tensor.permute(2, 0, 1).unsqueeze(0)


class RowNode:
    """
    RowNode: Combines multiple frames horizontally into a row
    
    This node organizes multiple comic frames into a horizontal row,
    with configurable spacing and alignment.
    """
    
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        """
        Define the input types for the row node.
        """
        return {
            "required": {
                "spacing": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
                "alignment": (["top", "center", "bottom"],),
            },
            "optional": {
                "frame1": ("FRAME_DATA",),
                "frame2": ("FRAME_DATA",),
                "frame3": ("FRAME_DATA",),
                "frame4": ("FRAME_DATA",),
                "frame5": ("FRAME_DATA",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "ROW_DATA")
    RETURN_NAMES = ("preview", "row_data")
    FUNCTION = "combine_frames"
    CATEGORY = "Pixstri/Comics"
    
    def combine_frames(self, spacing, alignment, frame1=None, frame2=None, frame3=None, frame4=None, frame5=None):
        """
        Combine multiple frames into a horizontal row.
        
        Parameters:
        -----------
        spacing : int
            Spacing between frames in pixels
        alignment : str
            Vertical alignment of frames ("top", "center", "bottom")
        frame1-frame5 : dict, optional
            Frame data dictionaries from FrameNode
            
        Returns:
        --------
        tuple(torch.Tensor, dict)
            The combined row image and the row data
        """
        # Collect all provided frames (filter out None values)
        frames = [f for f in [frame1, frame2, frame3, frame4, frame5] if f is not None]
        
        if not frames:
            # Return empty image and data if no frames
            empty_img = Image.new("RGB", (400, 200), (240, 240, 240))
            draw = ImageDraw.Draw(empty_img)
            draw.text((150, 90), "No frames provided", fill=(0, 0, 0))
            empty_tensor = self._pil_to_tensor(empty_img)
            return (empty_tensor, {"width": 400, "height": 200, "frames": []})
        
        # Calculate the total width and maximum height
        total_width = sum(frame["width"] for frame in frames) + spacing * (len(frames) - 1)
        max_height = max(frame["height"] for frame in frames)
        
        # Create a new image for the row
        row_image = Image.new("RGB", (total_width, max_height), (255, 255, 255))
        
        # Position to place the next frame
        x_offset = 0
        
        # Track frame positions for row data
        frame_positions = []
        
        # Place each frame on the row
        for frame in frames:
            frame_tensor = frame["image"]
            frame_pil = self._tensor_to_pil(frame_tensor)
            frame_width = frame["width"]
            frame_height = frame["height"]
            
            # Calculate Y position based on alignment
            if alignment == "top":
                y_offset = 0
            elif alignment == "bottom":
                y_offset = max_height - frame_height
            else:  # center
                y_offset = (max_height - frame_height) // 2
            
            # Paste the frame onto the row
            row_image.paste(frame_pil, (x_offset, y_offset))
            
            # Record the frame position
            frame_positions.append({
                "x": x_offset,
                "y": y_offset,
                "width": frame_width,
                "height": frame_height
            })
            
            # Update the x offset
            x_offset += frame_width + spacing
            
        # Convert to tensor for preview
        result_tensor = self._pil_to_tensor(row_image)
        
        # Create row data dictionary
        row_data = {
            "width": total_width,
            "height": max_height,
            "image": result_tensor,
            "frames": frame_positions,
            "spacing": spacing,
            "alignment": alignment
        }
        
        return (result_tensor, row_data)
    
    def _tensor_to_pil(self, tensor):
        """Convert a PyTorch tensor to a PIL Image"""
        if not torch.is_tensor(tensor):
            return tensor
        
        # Clone tensor to CPU
        tensor = tensor.cpu().clone()
        
        # Handle BCHW format (which is what FrameNode outputs)
        if len(tensor.shape) == 4:  # BCHW format
            tensor = tensor[0]  # Remove batch dimension
            if tensor.shape[0] == 3 or tensor.shape[0] == 1:  # CHW format
                tensor = tensor.permute(1, 2, 0)  # CHW to HWC
        
        # Convert to numpy array
        if tensor.dtype == torch.uint8:
            img_np = tensor.numpy()
        else:
            img_np = (tensor.numpy() * 255).astype(np.uint8)
        
        # Create PIL image
        if len(img_np.shape) == 2:
            return Image.fromarray(img_np, 'L')
        elif img_np.shape[-1] == 1:
            return Image.fromarray(img_np[:, :, 0], 'L')
        else:
            return Image.fromarray(img_np)
    
    def _pil_to_tensor(self, pil_image):
        """Convert a PIL Image to a PyTorch tensor in BCHW format for ComfyUI compatibility."""
        # Convert PIL to numpy
        img_np = np.array(pil_image).astype(np.float32) / 255.0
        
        # Ensure proper dimensions for RGB images
        if len(img_np.shape) == 2:  # Grayscale
            img_np = np.expand_dims(img_np, axis=2)  # Add channel dimension
        
        # Convert HWC to BCHW format (batch, channels, height, width)
        img_np = img_np.transpose(2, 0, 1)  # HWC to CHW
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # Add batch dimension
        
        return img_tensor


class PageNode:
    """
    PageNode: Combines multiple rows vertically into a complete comic page
    
    This node stacks multiple rows to create a full comic page layout,
    with configurable spacing and alignment.
    """
    
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        """
        Define the input types for the page node.
        """
        return {
            "required": {
                "spacing": ("INT", {
                    "default": 15,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
                "alignment": (["left", "center", "right"],),
                "page_color": (["white", "off-white", "light-gray"],),
                "page_width": ("INT", {
                    "default": 1000,
                    "min": 500,
                    "max": 3000,
                    "step": 10,
                    "display": "slider"
                }),
                "page_margin": ("INT", {
                    "default": 30,
                    "min": 0,
                    "max": 100,
                    "step": 5,
                    "display": "slider"
                }),
            },
            "optional": {
                "row1": ("ROW_DATA",),
                "row2": ("ROW_DATA",),
                "row3": ("ROW_DATA",),
                "row4": ("ROW_DATA",),
                "row5": ("ROW_DATA",),
                "title": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("page_image",)
    FUNCTION = "create_page"
    CATEGORY = "Pixstri/Comics"
    
    def create_page(self, spacing, alignment, page_color, page_width, page_margin, 
                    row1=None, row2=None, row3=None, row4=None, row5=None, title=None):
        """
        Create a comic page by combining multiple rows.
        
        Parameters:
        -----------
        spacing : int
            Vertical spacing between rows in pixels
        alignment : str
            Horizontal alignment of rows ("left", "center", "right")
        page_color : str
            Background color of the page
        page_width : int
            Width of the page in pixels
        page_margin : int
            Margin around the page content in pixels
        row1-row5 : dict, optional
            Row data dictionaries from RowNode
        title : str, optional
            Title for the comic page
            
        Returns:
        --------
        torch.Tensor
            The complete comic page image
        """
        # Collect all provided rows (filter out None values)
        rows = [r for r in [row1, row2, row3, row4, row5] if r is not None]
        
        if not rows:
            # Return empty page if no rows
            empty_img = Image.new("RGB", (page_width, page_width * 1.4), (240, 240, 240))
            draw = ImageDraw.Draw(empty_img)
            draw.text((page_width//2 - 80, page_width//2), "Empty Comic Page", fill=(0, 0, 0))
            empty_tensor = self._pil_to_tensor(empty_img)
            return (empty_tensor,)
        
        # Set page background color
        if page_color == "white":
            bg_color = (255, 255, 255)
        elif page_color == "off-white":
            bg_color = (252, 252, 245)
        else:  # light-gray
            bg_color = (240, 240, 240)
        
        # Calculate heights and determine max width
        total_height = sum(row["height"] for row in rows) + spacing * (len(rows) - 1) + (page_margin * 2)
        max_row_width = max(row["width"] for row in rows)
        
        # Determine page width (use specified width or fit content)
        content_width = max_row_width + (page_margin * 2)
        if content_width > page_width:
            page_width = content_width
        
        # Add extra space for title if provided
        title_height = 0
        if title and title.strip():
            title_height = 50  # Space for title
            total_height += title_height
        
        # Create the page canvas
        page_image = Image.new("RGB", (page_width, total_height), bg_color)
        draw = ImageDraw.Draw(page_image)
        
        # Add title if provided
        y_offset = page_margin
        if title and title.strip():
            # Draw title text
            draw.text((page_width//2 - len(title)*4, y_offset), title.strip(), fill=(0, 0, 0))
            y_offset += title_height
        
        # Place each row on the page
        for row in rows:
            row_tensor = row["image"]
            row_pil = self._tensor_to_pil(row_tensor)
            row_width = row["width"]
            row_height = row["height"]
            
            # Calculate X position based on alignment
            if alignment == "left":
                x_offset = page_margin
            elif alignment == "right":
                x_offset = page_width - row_width - page_margin
            else:  # center
                x_offset = (page_width - row_width) // 2
            
            # Paste the row onto the page
            page_image.paste(row_pil, (x_offset, y_offset))
            
            # Update the y offset
            y_offset += row_height + spacing
        
        # Convert to tensor
        result_tensor = self._pil_to_tensor(page_image)
        
        return (result_tensor,)
    
    def _tensor_to_pil(self, tensor):
        """Convert a PyTorch tensor to a PIL Image."""
        if not torch.is_tensor(tensor):
            return tensor  # Already a PIL image or something else
        
        # Clone and move to CPU
        tensor = tensor.cpu().clone()
        
        # Handle uint8 tensors by converting to float
        if tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0
        
        # Handle B111C format specifically - this is the problematic format
        if len(tensor.shape) == 4:
            if tensor.shape[1] == 1 and tensor.shape[2] == 1:  # B111C format
                tensor = tensor.squeeze(1).squeeze(1)  # Remove singleton dimensions
            elif tensor.shape[3] == 3:  # BHWC format
                tensor = tensor.permute(0, 3, 1, 2)  # BHWC to BCHW
        
        # Ensure we're in BCHW format and remove batch dimension
        if len(tensor.shape) == 4:
            tensor = tensor[0]  # Remove batch dimension to get CHW
        
        # Convert CHW to HWC for PIL
        if len(tensor.shape) == 3 and (tensor.shape[0] == 3 or tensor.shape[0] == 1):
            tensor = tensor.permute(1, 2, 0)  # CHW to HWC
        
        # Convert to numpy with proper scaling
        img_np = (tensor.numpy() * 255).astype(np.uint8)
        
        # Create PIL image - handle grayscale vs RGB
        if len(img_np.shape) == 2:
            img_pil = Image.fromarray(img_np, 'L')
        elif img_np.shape[-1] == 1:
            img_pil = Image.fromarray(img_np[:, :, 0], 'L')
        else:
            img_pil = Image.fromarray(img_np)
            
        return img_pil
    
    def _pil_to_tensor(self, pil_image):
        """Convert a PIL Image to a PyTorch tensor in BCHW format"""
        # Convert PIL to numpy array
        img_np = np.array(pil_image)
        
        # Keep as uint8 for better compatibility
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8)
        
        # Convert to tensor and keep as uint8
        img_tensor = torch.from_numpy(img_np)
        
        # Return in BCHW format
        if len(img_tensor.shape) == 2:  # Grayscale
            return img_tensor.unsqueeze(0).unsqueeze(0)
        else:  # RGB
            # Move channels to correct position and add batch dimension
            return img_tensor.permute(2, 0, 1).unsqueeze(0)


# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "FrameNode": FrameNode,
    "RowNode": RowNode,
    "PageNode": PageNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "FrameNode": "Comic Frame",
    "RowNode": "Comic Row",
    "PageNode": "Comic Page"
}
