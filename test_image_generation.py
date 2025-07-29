#!/usr/bin/env python3

import torch
from diffusers import StableDiffusionPipeline
import os

def test_stable_diffusion():
    """Test Stable Diffusion image generation"""
    
    try:
        print("Testing Stable Diffusion...")
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16
            print("Using CUDA")
        else:
            device = "cpu"
            torch_dtype = torch.float32
            print("Using CPU")
        
        # Load the pipeline
        print("Loading pipeline...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch_dtype, 
            use_safetensors=True
        )
        
        pipe = pipe.to(device)
        print("Pipeline loaded successfully")
        
        # Test with a simple prompt
        prompt = "A cute cat playing in a garden"
        print(f"Generating image for: {prompt}")
        
        # Generate image
        result = pipe(
            prompt=prompt, 
            num_inference_steps=10,
            added_cond_kwargs={}  # Required for SDXL
        )
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        
        if result and isinstance(result, tuple) and len(result) > 0:
            images = result[0]
            print(f"Images type: {type(images)}")
            print(f"Images: {images}")
            
            if images and len(images) > 0:
                image = images[0]
                print(f"Image type: {type(image)}")
                
                # Save the image
                output_path = "test_image.png"
                image.save(output_path)
                print(f"Image saved to: {output_path}")
                return True
            else:
                print("No images in result")
                return False
        else:
            print("Invalid result")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_stable_diffusion()
    print(f"Test {'passed' if success else 'failed'}") 