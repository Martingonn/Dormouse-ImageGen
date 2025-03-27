from diffusers import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import os

# Function to read prompts from file
def read_prompts_from_file(file_path):
    prompts = []
    with open(file_path, 'r') as file:
        for line in file:
            prompts.append(line.strip())
    return prompts

# Function to load image paths and prompts
def load_image_paths_and_prompts(image_dir, prompt_file_path):
    image_paths = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith('.jpg') or file.endswith('.png')]
    print(f"Found {len(image_paths)} images.")
    
    prompts = read_prompts_from_file(prompt_file_path)
    print(f"Found {len(prompts)} prompts.")
    
    if len(image_paths) != len(prompts):
        raise ValueError("Number of images and prompts must match.")
    
    return image_paths, prompts

# Define a custom dataset class
class MyDataset(Dataset):
    def __init__(self, image_paths, prompts):
        self.image_paths = image_paths
        self.prompts = prompts
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # Define a transformation to resize images to 512x512
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),  # Resize to 512x512
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize([0.5], [0.5])  # Normalize pixel values
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.image_paths[idx]))
        prompt = self.prompts[idx]
        inputs = self.tokenizer(prompt, return_tensors="pt")
        return {"image": image, "input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask}

# Load dataset
image_dir = str(input("Input path to training image folder: "))
prompt_file_path = str(input("Input image description .txt file path: "))
where_model_file_path = str(input("Input saving path for model training to save: "))

image_paths, prompts = load_image_paths_and_prompts(image_dir, prompt_file_path)

dataset = MyDataset(image_paths, prompts)
batch_size = 4
data_loader = DataLoader(dataset, batch_size=batch_size)

# Load the model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")  # Use GPU if available

# Freeze some layers to reduce training time and focus on fine-tuning
for name, param in pipe.text_encoder.named_parameters():
    param.requires_grad = False  # Freeze text encoder

# Make more layers of the UNet trainable
unet_layers = list(pipe.unet.named_parameters())
trainable_layers = unet_layers[-10:]  # Adjust this number based on how many layers you want to train

for name, param in pipe.unet.named_parameters():
    param.requires_grad = False

for name, param in trainable_layers:
    param.requires_grad = True

# Define the optimizer and scheduler
params_to_train = []
for name, param in pipe.text_encoder.named_parameters():
    if param.requires_grad:
        params_to_train.append(param)
for name, param in pipe.unet.named_parameters():
    if param.requires_grad:
        params_to_train.append(param)

if not params_to_train:
    print("No trainable parameters found. Please adjust the model configuration.")
else:
    optimizer = torch.optim.Adam(params_to_train, lr=1e-4)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=len(data_loader),  # Add warm-up phase
        num_training_steps=len(data_loader) * 5,  
    )

    # Train the model
    for epoch in range(5):
        for batch in data_loader:
            try:
                # Zero the gradients
                optimizer.zero_grad()

                # Generate images
                inputs = pipe(batch["input_ids"].to("cuda"), attention_mask=batch["attention_mask"].to("cuda")).images

                # Calculate loss (this is a simplified example; you might need a more complex loss function)
                loss = torch.mean((inputs - batch["image"].to("cuda")) ** 2)

                # Backward pass
                loss.backward()

                # Update parameters
                optimizer.step()

                # Update scheduler
                scheduler.step()

                print(f"Epoch {epoch+1}, Loss: {loss.item()}")
            except Exception as e:
                print(f"Error occurred: {e}")

        # Save the model after each epoch
        save_path = f"fine_tuned_model_epoch_{epoch+1}"
        pipe.save_pretrained(save_path)
        print(f"Model saved to {save_path}")

    # Save the final model
    final_save_path = where_model_file_path
    pipe.save_pretrained(final_save_path)
    print(f"Final model saved to {final_save_path}")
