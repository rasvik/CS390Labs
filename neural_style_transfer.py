import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision as tv

class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.grab_layer_set = {0, 5, 10, 19, 28}
        self.model = tv.models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        grab_features = []
        for index in range(len(self.model)):
            x = self.model[index](x)
            if index in self.grab_layer_set:
                grab_features.append(x)
        return grab_features


def preprocess(image_name, image_size):
    image = Image.open(image_name)
    image = image.convert('RGB')
    image = preprocess_helper(image).unsqueeze(0)
    return image.to(device)


def gram_matrix(matrices, index, channel, height, width):
    return matrices[index].view(channel, height * width).mm(matrices[index].view(channel, height * width).t())


def compute_residuals_content(gen, orig):
    return 0.5 * torch.sum((gen - orig) ** 2)


def compute_residuals_style(dims, gen, orig):
    return (0.25 * torch.sum((gen - orig) ** 2)) / (dims.shape[1] ** 2 * (dims.shape[2] * dims.shape[3]) ** 2)


device = torch.device("cuda")
image_size = 500

preprocess_helper = tv.transforms.Compose([tv.transforms.Resize((image_size, image_size)), tv.transforms.ToTensor()])

original_img = preprocess("drive/MyDrive/swissvalley.jpeg", image_size)
generated_img = preprocess("drive/MyDrive/swissvalley.jpeg", image_size)
style_img = preprocess("drive/MyDrive/starrynight.jpeg", image_size)
generated_img.requires_grad = True

model = VGG().to(device)
model.train(False)

epochs = 1000
learning_rate = 0.005
alpha = 1
beta = 10000000000
optimizer = optim.Adam([generated_img], lr=learning_rate)

for step in range(epochs):
    generated_features = model(generated_img)
    original_features = model(original_img)
    style_features = model(style_img)

    style_loss = content_loss = 0
    for index in range(len(generated_features)):
        channels, height, width = generated_features[index].shape[1], generated_features[index].shape[2], \
                                  generated_features[index].shape[3]
        content_loss += compute_residuals_content(generated_features[index], original_features[index])
        gram_generated = gram_matrix(generated_features, index, channels, height, width)
        gram_style = gram_matrix(style_features, index, channels, height, width)
        style_loss += compute_residuals_style(generated_features[index], gram_generated, gram_style)

    total_loss = alpha * content_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

tv.utils.save_image(generated_img, "drive/MyDrive/generated.png")
