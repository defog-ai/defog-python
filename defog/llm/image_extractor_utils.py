"""
Image utilities for downloading and processing images for Claude API.
"""

import asyncio
import base64
import io
import aiohttp
from typing import Optional, Dict, Any, Tuple
import logging
from PIL import Image

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image downloading, processing, and encoding for Claude API."""

    # Claude API image limits
    MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB
    MAX_IMAGE_DIMENSION = 8192  # Max width or height
    SUPPORTED_FORMATS = {"image/jpeg", "image/png", "image/gif", "image/webp"}

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": "Claude-Image-Processor/1.0"},
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def download_image(self, url: str) -> Tuple[bytes, str]:
        """
        Download image from URL.

        Args:
            url: Image URL to download

        Returns:
            Tuple of (image content as bytes, content type)

        Raises:
            ValueError: If URL is invalid or download fails
            aiohttp.ClientError: For network-related errors
        """
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL: {url}")

        if not self.session:
            raise RuntimeError("ImageProcessor must be used as async context manager")

        try:
            async with self.session.get(url) as response:
                response.raise_for_status()

                # Check content type
                content_type = response.headers.get("content-type", "").lower()
                
                # Extract base content type (remove parameters like charset)
                content_type = content_type.split(";")[0].strip()
                
                if content_type not in self.SUPPORTED_FORMATS:
                    # Try to infer from URL extension
                    if url.lower().endswith((".jpg", ".jpeg")):
                        content_type = "image/jpeg"
                    elif url.lower().endswith(".png"):
                        content_type = "image/png"
                    elif url.lower().endswith(".gif"):
                        content_type = "image/gif"
                    elif url.lower().endswith(".webp"):
                        content_type = "image/webp"
                    else:
                        raise ValueError(
                            f"Unsupported image format: {content_type}. "
                            f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
                        )

                # Read content
                content = await response.read()

                if len(content) == 0:
                    raise ValueError("Downloaded image is empty")

                logger.info(f"Downloaded image: {len(content)} bytes from {url}")
                return content, content_type

        except aiohttp.ClientError as e:
            raise aiohttp.ClientError(f"Failed to download image from {url}: {e}")

    def process_image(self, image_data: bytes, content_type: str) -> Dict[str, Any]:
        """
        Process image to ensure it meets Claude API requirements.

        Args:
            image_data: Image content as bytes
            content_type: MIME type of the image

        Returns:
            Dictionary with processed image data and metadata
        """
        try:
            # Open image with PIL
            img = Image.open(io.BytesIO(image_data))
            
            # Get original dimensions
            original_width, original_height = img.size
            original_format = img.format
            
            # Check if we need to resize
            needs_resize = (
                original_width > self.MAX_IMAGE_DIMENSION
                or original_height > self.MAX_IMAGE_DIMENSION
            )
            
            # Check if we need to compress
            needs_compress = len(image_data) > self.MAX_IMAGE_SIZE_BYTES
            
            processed_data = image_data
            processed_width = original_width
            processed_height = original_height
            
            if needs_resize or needs_compress:
                # Calculate new dimensions if needed
                if needs_resize:
                    ratio = min(
                        self.MAX_IMAGE_DIMENSION / original_width,
                        self.MAX_IMAGE_DIMENSION / original_height,
                    )
                    new_width = int(original_width * ratio)
                    new_height = int(original_height * ratio)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    processed_width = new_width
                    processed_height = new_height
                    logger.info(
                        f"Resized image from {original_width}x{original_height} "
                        f"to {new_width}x{new_height}"
                    )
                
                # Convert to appropriate format and compress if needed
                output = io.BytesIO()
                
                # Handle different formats
                if content_type == "image/png" and img.mode in ("RGBA", "LA", "PA"):
                    # Keep PNG for images with transparency
                    img.save(output, format="PNG", optimize=True)
                    final_content_type = "image/png"
                elif content_type == "image/gif" and getattr(img, "is_animated", False):
                    # Keep GIF for animated images
                    img.save(output, format="GIF", save_all=True)
                    final_content_type = "image/gif"
                else:
                    # Convert to JPEG for better compression
                    if img.mode in ("RGBA", "LA", "PA"):
                        # Convert transparency to white background
                        background = Image.new("RGB", img.size, (255, 255, 255))
                        if img.mode == "RGBA":
                            background.paste(img, mask=img.split()[3])
                        else:
                            background.paste(img)
                        img = background
                    elif img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")
                    
                    # Start with high quality and reduce if needed
                    quality = 95
                    while quality >= 70:
                        output.seek(0)
                        output.truncate()
                        img.save(output, format="JPEG", quality=quality, optimize=True)
                        if output.tell() <= self.MAX_IMAGE_SIZE_BYTES:
                            break
                        quality -= 5
                    
                    final_content_type = "image/jpeg"
                
                processed_data = output.getvalue()
                
                logger.info(
                    f"Processed image: {len(image_data)} bytes -> {len(processed_data)} bytes"
                )
            else:
                final_content_type = content_type
            
            return {
                "success": True,
                "data": processed_data,
                "media_type": final_content_type,
                "width": processed_width,
                "height": processed_height,
                "original_width": original_width,
                "original_height": original_height,
                "format": original_format,
                "size_bytes": len(processed_data),
                "original_size_bytes": len(image_data),
                "was_resized": needs_resize,
                "was_compressed": len(processed_data) < len(image_data),
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
            }

    async def process_image_for_api(self, url: str) -> Dict[str, Any]:
        """
        Download and process image for Claude API submission.

        Args:
            url: Image URL to process

        Returns:
            Dictionary with base64 encoded image and metadata
        """
        try:
            # Download image
            image_data, content_type = await self.download_image(url)
            
            # Process image
            result = self.process_image(image_data, content_type)
            
            if not result["success"]:
                return result
            
            # Encode to base64
            base64_data = base64.b64encode(result["data"]).decode("utf-8")
            
            return {
                "success": True,
                "base64_data": base64_data,
                "media_type": result["media_type"],
                "width": result["width"],
                "height": result["height"],
                "format": result["format"],
                "size_bytes": result["size_bytes"],
                "original_size_bytes": result["original_size_bytes"],
                "was_resized": result["was_resized"],
                "was_compressed": result["was_compressed"],
            }
            
        except Exception as e:
            logger.error(f"Error processing image for API: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
            }

    def encode_local_image(self, file_path: str) -> Dict[str, Any]:
        """
        Encode a local image file for Claude API.

        Args:
            file_path: Path to local image file

        Returns:
            Dictionary with base64 encoded image and metadata
        """
        try:
            # Read file
            with open(file_path, "rb") as f:
                image_data = f.read()
            
            # Determine content type from file extension
            file_lower = file_path.lower()
            if file_lower.endswith((".jpg", ".jpeg")):
                content_type = "image/jpeg"
            elif file_lower.endswith(".png"):
                content_type = "image/png"
            elif file_lower.endswith(".gif"):
                content_type = "image/gif"
            elif file_lower.endswith(".webp"):
                content_type = "image/webp"
            else:
                # Try to detect from file content
                img = Image.open(file_path)
                format_map = {
                    "JPEG": "image/jpeg",
                    "PNG": "image/png",
                    "GIF": "image/gif",
                    "WEBP": "image/webp",
                }
                content_type = format_map.get(img.format, "image/jpeg")
            
            # Process image
            result = self.process_image(image_data, content_type)
            
            if not result["success"]:
                return result
            
            # Encode to base64
            base64_data = base64.b64encode(result["data"]).decode("utf-8")
            
            return {
                "success": True,
                "base64_data": base64_data,
                "media_type": result["media_type"],
                "width": result["width"],
                "height": result["height"],
                "format": result["format"],
                "size_bytes": result["size_bytes"],
                "original_size_bytes": result["original_size_bytes"],
                "was_resized": result["was_resized"],
                "was_compressed": result["was_compressed"],
                "file_path": file_path,
            }
            
        except Exception as e:
            logger.error(f"Error encoding local image: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
            }