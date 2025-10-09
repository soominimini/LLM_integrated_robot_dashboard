#!/usr/bin/env python3

# Copyright (c) 2024 LuxAI S.A.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import torch
from diffusers import DiffusionPipeline
import uuid
from datetime import datetime
from typing import Optional

class ImageGenerator:
    """
    Generate images using Stable Diffusion for story scenes
    """
    
    def __init__(self):
        """Initialize the Stable Diffusion pipeline"""
        try:
            # Check if CUDA is available
            device = "cuda"
            torch_dtype = torch.float16
            
            # Load the Stable Diffusion model
            self.pipe = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", 
                torch_dtype=torch_dtype, 
                use_safetensors=True,
                 variant="fp16")

            
            self.pipe = self.pipe.to(device)
            self.device = device
            self._available = True
            
            print(f"Stable Diffusion loaded successfully on {device}")
            
        except Exception as e:
            print(f"Warning: Could not initialize Stable Diffusion: {e}")
            self.pipe = None
            self._available = False
    
    def generate_image(self, prompt: str, output_dir: str = "generated_images", filename_prefix: str = None) -> Optional[str]:
        """
        Generate an image from a text prompt
        
        Args:
            prompt: Text description for the image
            output_dir: Directory to save the generated image
            filename_prefix: Optional prefix for the filename
            
        Returns:
            str: Path to the generated image file, or None if failed
        """
        if not self._available or not self.pipe:
            return None
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            if filename_prefix:
                filename = f"{filename_prefix}_{timestamp}_{unique_id}.png"
            else:
                filename = f"story_scene_{timestamp}_{unique_id}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Generate the image
            print(f"Generating image for prompt: {prompt[:100]}...")
            
            try:
                # Generate image (SDXL returns StableDiffusionXLPipelineOutput)
                result = self.pipe(prompt=prompt)
                images = None
                if hasattr(result, 'images'):
                    images = result.images
                elif isinstance(result, (list, tuple)):
                    images = result[0] if result else None
                
                if images is None:
                    print(f"No images returned by pipeline: {type(result)}")
                    return None
                
                # If a single PIL image is returned
                if hasattr(images, 'save'):
                    image = images
                else:
                    # Assume list-like of PIL images
                    if not images:
                        print("No images generated (empty list)")
                        return None
                    image = images[0]
                
                # Save the image
                image.save(filepath)
                print(f"Image saved to: {filepath}")
                return filepath
            except Exception as e:
                print(f"Error in pipeline call: {e}")
                return None
            
        except Exception as e:
            print(f"Error generating image: {e}")
            return None

    '''
        def generate_image(self, prompt: str, output_dir: str = "generated_images", filename_prefix: str = None, prev_img) -> Optional[str]:
        """
        Generate an image from a text prompt
        
        Args:
            prompt: Text description for the image
            output_dir: Directory to save the generated image
            filename_prefix: Optional prefix for the filename
            
        Returns:
            str: Path to the generated image file, or None if failed
        """
        if not self._available or not self.pipe:
            return None
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            if filename_prefix:
                filename = f"{filename_prefix}_{timestamp}_{unique_id}.png"
            else:
                filename = f"story_scene_{timestamp}_{unique_id}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Generate the image
            print(f"Generating image for prompt: {prompt[:100]}...")
            
            try:
                # Generate image (SDXL returns StableDiffusionXLPipelineOutput)
                if prev_img != None:
                prev_img_converted = Image.open(prev_img).convert("RGB")
                    result = img2img(prompt=prompt, image=prev_img_converted, strength=0.6, guidance_scale = 7.5)
                else:
                    result = self.pipe(prompt=prompt)
                images = None
                if hasattr(result, 'images'):
                    images = result.images
                elif isinstance(result, (list, tuple)):
                    images = result[0] if result else None
                
                if images is None:
                    print(f"No images returned by pipeline: {type(result)}")
                    return None
                
                # If a single PIL image is returned
                if hasattr(images, 'save'):
                    image = images
                else:
                    # Assume list-like of PIL images
                    if not images:
                        print("No images generated (empty list)")
                        return None
                    image = images[0]
                
                # Save the image
                image.save(filepath)
                print(f"Image saved to: {filepath}")
                return filepath
            except Exception as e:
                print(f"Error in pipeline call: {e}")
                return None
            
        except Exception as e:
            print(f"Error generating image: {e}")
            return None
    '''


    def generate_story_scene_image(self, sentence: str, story_context: str = "", output_dir: str = "generated_images", filename_prefix: str = None) -> Optional[str]:
        """
        Generate an image for a story scene based on a sentence
        
        Args:
            sentence: The sentence describing the scene
            story_context: Additional context from the story
            output_dir: Directory to save the generated image
            filename_prefix: Optional prefix for the filename
            
        Returns:
            str: Path to the generated image file, or None if failed
        """
        if not sentence.strip():
            return None
        
        # Create a better prompt for image generation
        # Clean up the sentence and add context
        clean_sentence = sentence.strip()
        
        # Add story context if available
        if story_context:
            prompt = f"{clean_sentence} {story_context} Children's book water colour illustration style, colorful, detailed, safe for children"
        else:
            prompt = f"{clean_sentence} Children's book water colour illustration style, colorful, detailed, safe for children"
        
        # Remove any markdown formatting
        prompt = prompt.replace("**", "").replace("*", "")
        
        return self.generate_image(prompt, output_dir, filename_prefix)
    
    def is_available(self) -> bool:
        """
        Check if image generation is available
        
        Returns:
            bool: True if image generation is available, False otherwise
        """
        return self._available 