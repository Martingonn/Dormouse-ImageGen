from diffusers import StableDiffusionPipeline
import torch

# Ensure you have a GPU with enough VRAM for better performance
# If not, you can use CPU, but it will be slower
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to(device)

def generate_image(prompt, output_path="generated_image.png"):
    # Generate the image
    image = pipe(prompt).images[0]
    
    # Save the image
    image.save(output_path)

if __name__ == "__main__":
    prompt = str(input("Input image prompt: "))
    generate_image(prompt)
