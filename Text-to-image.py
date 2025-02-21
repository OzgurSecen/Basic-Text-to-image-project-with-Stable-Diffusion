import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import matplotlib.pyplot as plt

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

def generate_image(prompt):
    with torch.autocast("cuda"):
        image = pipe(prompt).images[0]

    plt.imshow(image)
    plt.axis('off')
    plt.show()

    image.save("generated_image.png")
    print("Image saved as 'generated_image.png'")

user_prompt = input("Enter your prompt: ")

generate_image(user_prompt)