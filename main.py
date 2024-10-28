import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
from datetime import datetime
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VGG Model with selected layers for style transfer
class VGGFeatures(nn.Module):
    def __init__(self, layers=['0', '5', '10', '19', '28']):
        super(VGGFeatures, self).__init__()
        self.selected_layers = layers
        self.model = models.vgg19(pretrained=True).features[:29].to(device).eval()
        for param in self.model.parameters():
            param.requires_grad = False  # Freeze model parameters

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.model):
            x = layer(x)
            if str(i) in self.selected_layers:
                features.append(x)
        return features

# Load and preprocess image
def load_image(image_path, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    image = transform(Image.open(image_path)).unsqueeze(0).to(device)
    return image

# Compute Gram matrix for style loss
def compute_gram_matrix(tensor):
    _, channels, height, width = tensor.size()
    tensor = tensor.view(channels, height * width)
    gram_matrix = torch.mm(tensor, tensor.t())
    return gram_matrix / (channels * height * width)


def train(original_img, style_img, model, config):
    generated_img = original_img.clone().requires_grad_(True).to(device)
    optimizer = optim.Adam([generated_img], lr=config["learning_rate"])
    
    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"])

    for step in range(config["total_steps"]):
        generated_features = model(generated_img)
        original_features = model(original_img)
        style_features = model(style_img)

        style_loss, content_loss = 0, 0
        for gen_feat, orig_feat, style_feat in zip(generated_features, original_features, style_features):
            content_loss += torch.mean((gen_feat - orig_feat) ** 2)
            G = compute_gram_matrix(gen_feat)
            A = compute_gram_matrix(style_feat)
            style_loss += torch.mean((G - A) ** 2)

        total_loss = config["alpha"] * content_loss + config["beta"] * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % config["save_interval"] == 0:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_path = os.path.join(config["output_dir"], f"generated_{timestamp}.png")
            save_image(generated_img, output_path)
            print(f"Step [{step}/{config['total_steps']}], Total Loss: {total_loss.item():.4f}, Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PyTorch Style Transfer")
    parser.add_argument("--content_image", type=str, required=True, help="Path to the content image.")
    parser.add_argument("--style_image", type=str, required=True, help="Path to the style image.")
    parser.add_argument("--output_dir", type=str, default="output_images", help="Directory to save output images.")
    parser.add_argument("--img_size", type=int, default=256, help="Image size for processing.")
    parser.add_argument("--total_steps", type=int, default=6000, help="Total number of steps for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimization.")
    parser.add_argument("--alpha", type=float, default=1, help="Weight for content loss.")
    parser.add_argument("--beta", type=float, default=0.01, help="Weight for style loss.")
    parser.add_argument("--save_interval", type=int, default=200, help="Interval to save generated images.")
    
    args = parser.parse_args()

    # Configuration dictionary
    config = {
        "total_steps": args.total_steps,
        "learning_rate": args.learning_rate,
        "alpha": args.alpha,
        "beta": args.beta,
        "img_size": args.img_size,
        "save_interval": args.save_interval,
        "output_dir": args.output_dir
    }

    # Load images
    original_img = load_image(args.content_image, config["img_size"])
    style_img = load_image(args.style_image, config["img_size"])

    # Load VGG model
    model = VGGFeatures().to(device)

    # Train and generate style transfer image
    train(original_img, style_img, model, config)

if __name__ == "__main__":
    main()
