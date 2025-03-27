from diffusers import StableDiffusionPipeline
import torch
from huggingface_hub import notebook_login

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_id):
    """Load model with auth check"""
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
            use_auth_token=True  
        )
    except Exception as e:
        print(f"\nAuthentication failed for {model_id}: {str(e)}")
        print("Attempting public download...")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
            use_auth_token=False  
        )
    
    pipe.to(device)
    pipe.safety_checker = None
    return pipe

def generate_images(pipe, prompt, num_images=5, output_path="generated_images"):
    """Generate and save images"""
    import os
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i in range(num_images):
        image = pipe(prompt).images[0]
        filename = f"image_{i+1}.png"
        image.save(os.path.join(output_path, filename))
        print(f"Generated image {i+1} of {num_images}")

if __name__ == "__main__":
    print("Available models:")
    print("1. CompVis/stable-diffusion-v1-4 (Public)")
    print("2. CompVis/stable-diffusion-v1-5 (Public)")
    print("3. CompVis/stable-diffusion-v1-inference (Public)")
    print("4. Custom model (enter Hugging Face ID)")
    print("5. Load local model (enter folder path)")
    
    choice = input("Enter model number or custom ID: ")
    
    if choice == "5":
        model_path = input("Enter local model folder path: ")
        pipe = load_model(model_path)
    elif choice == "4":
        model_id = input("Enter Hugging Face model ID: ")
        pipe = load_model(model_id)
    else:
        model_map = {
            "1": "CompVis/stable-diffusion-v1-4",
            "2": "CompVis/stable-diffusion-v1-5",
            "3": "CompVis/stable-diffusion-v1-inference"
        }
        model_id = model_map.get(choice, "CompVis/stable-diffusion-v1-4")
        pipe = load_model(model_id)
    
    prompt = input("Input image prompt: ")
    num_images = int(input("Input number of images: "))
    output_path = input("Input image folder name: ")
    
    generate_images(pipe, prompt, num_images, output_path)
