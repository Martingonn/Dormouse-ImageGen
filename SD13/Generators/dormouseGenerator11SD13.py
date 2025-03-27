from diffusers import StableDiffusionPipeline
import torch

# Ensure you have a GPU with enough VRAM for better performance
# If not, you can use CPU, but it will be slower
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to(device)

# Disable the safety checker

pipe.safety_checker = None

def generate_images(prompt, num_images=5, output_path="generated_images"):
    # Create the output directory if it doesn't exist
    import os
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Generate multiple images
    for i in range(num_images):
        # Generate the image
        image = pipe(prompt).images[0]
        
        # Save the image with a unique filename
        filename = f"image_{i+1}.png"
        image.save(os.path.join(output_path, filename))
        print(f"Generated image {i+1} of {num_images}")

if __name__ == "__main__":
    prompt = str(input("Input image prompt: "))
    num_images = int(input("Input number of images: "))
    output_path = str(input("Input image folder name: "))
    generate_images(prompt, num_images, output_path)
