import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, spectrogram
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import mne
from mne.datasets.sleep_physionet import age

#setting up some basic config stuff
OUT_DIR = "output_eeg_art"
os.makedirs(OUT_DIR, exist_ok=True)

FS = 100  # this is the sampling frequency of the EEG data
BAND_RANGES = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}


# getting EEG data from MNE's Sleep-EDF dataset
print("Downloading EEG data")
subject_data = age.fetch_data(subjects=[0], recording=[1])
raw_fname = subject_data[0][0]
raw = mne.io.read_raw_edf(raw_fname, preload=True)

# picking the EEG channel
try:
    raw.pick_channels(["EEG Fpz-Cz"])
except Exception:
    picks = [ch for ch in raw.ch_names if 'EEG' in ch or 'Fpz' in ch or 'Cz' in ch]
    raw.pick_channels([picks[0]] if picks else [raw.ch_names[0]])

raw.resample(FS)
data = raw.get_data()[0]
print(f"Loaded EEG shape: {data.shape}, sampling rate: {FS} Hz")

# filtering function
def bandpass(data, fs, low, high, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data)

# converting the EEG signal into band-specific spectrogram images
def eeg_to_band_images(signal, fs, bands=BAND_RANGES):
    band_imgs = {}
    for band, (low, high) in bands.items():
        filtered = bandpass(signal, fs, low, high)
        f, t, Sxx = spectrogram(filtered, fs=fs, nperseg=256, noverlap=128)
        band_imgs[band] = np.log1p(Sxx)
    return band_imgs

def save_composite_image(bands_dict, path):
    def norm(x):
        x = x - np.min(x)
        if np.max(x) > 0:
            x = x / np.max(x)
        return x
    R = norm(bands_dict["alpha"])
    G = norm(bands_dict["beta"])
    B = norm(bands_dict["gamma"])
    rgb = np.stack([R, G, B], axis=-1)
    plt.figure(figsize=(3, 3), dpi=150)
    plt.axis("off")
    plt.imshow(rgb, aspect="auto")
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()

# generating the super cool EEG spectrogram image
print("Generating EEG image")
segment = data[0:FS * 30] 
bands = eeg_to_band_images(segment, FS)
eeg_img_path = os.path.join(OUT_DIR, "eeg_content.png")
save_composite_image(bands, eeg_img_path)
print(f"EEG image saved at: {eeg_img_path}")

# NST Implementation using PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# resizing the images to 224x224 and normalizing
def load_image_resized(path, size=(224,224)):
    img = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])
    return transform(img).unsqueeze(0).to(device)

def save_image(tensor, path):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(tensor.device)
    std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(tensor.device)
    
    img = tensor.clone().detach()
    img = img * std + mean

    if img.dim() == 4 and img.size(0) == 1:
        img = img.squeeze(0)
    
    img = torch.clamp(img, 0, 1)
    transforms.ToPILImage()(img).save(path)

# Loss modules for content and style
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

def gram_matrix(tensor):
    b,c,h,w = tensor.size()
    features = tensor.view(b*c, h*w)
    G = torch.mm(features, features.t())
    return G.div(b*c*h*w)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()
    def forward(self, input):
        G = gram_matrix(input)
        if G.shape != self.target.shape:
            s1 = G.shape[0]
            s2 = self.target.shape[0]
            m = min(s1, s2)
            Gc = G[:m, :m]
            Tc = self.target[:m, :m]
            self.loss = nn.functional.mse_loss(Gc, Tc)
        else:
            self.loss = nn.functional.mse_loss(G, self.target)
        return input

def get_style_model_and_losses(cnn, content_img, style_img):
    cnn = cnn.features.to(device).eval()
    content_layers = ["conv_4"]
    style_layers = ["conv_1","conv_2","conv_3","conv_4","conv_5"]
    model = nn.Sequential()
    content_losses = []
    style_losses = []
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        else:
            name = str(layer)
        model.add_module(name, layer)
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss", content_loss)
            content_losses.append(content_loss)
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss", style_loss)
            style_losses.append(style_loss)
    return model, style_losses, content_losses

def run_style_transfer(cnn, content_img, style_img, input_img, steps=300, style_weight=1e6, content_weight=1):
    model, style_losses, content_losses = get_style_model_and_losses(cnn, content_img, style_img)
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    run = [0]
    while run[0] <= steps:
        def closure():
            input_img.data.clamp_(0,1)
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            loss = style_weight*style_score + content_weight*content_score
            loss.backward(retain_graph=True)
            run[0] += 1
            if run[0]%50==0:
                print(f"Step {run[0]} Style: {style_score.item():.2f} Content: {content_score.item():.2f}")
            return loss
        optimizer.step(closure)
    input_img.data.clamp_(0,1)
    return input_img

# running the NST process
content_img = load_image_resized(eeg_img_path)

style_img_path = "style.jpg"
if not os.path.exists(style_img_path):
    rnd = np.random.rand(224,224,3)
    plt.imsave(style_img_path, rnd)

style_img = load_image_resized(style_img_path)
input_img = content_img.clone()
cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
output = run_style_transfer(cnn, content_img, style_img, input_img, steps=200)

out_path = os.path.join(OUT_DIR, "stylized_eeg_art.png")
save_image(output, out_path)
print("Image ready")

