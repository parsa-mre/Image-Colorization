import torch
# from torchvision.utils import save_image
from skimage import color
import os
import imageio
import numpy as np
import matplotlib.pyplot as plt


def show_res(x_o, y_o, y_pred_o):

    x = torch.permute(x_o, [1, 2, 0])
    y = torch.permute(y_o, [1, 2, 0])
    y_pred = torch.permute(y_pred_o, [1, 2, 0])

    axes = []
    fig = plt.figure()
    fig.set_dpi(120)
    fig.set_size_inches(15, 4)

    y = y.reshape(256, 256, 2)

    axes.append(fig.add_subplot(1, 6, 1))
    axes[-1].imshow(y[:, :, 0], cmap='gray')
    axes[-1].set_title('y1')
    axes.append(fig.add_subplot(1, 6, 2))
    axes[-1].imshow(y[:, :, 1], cmap='gray')
    axes[-1].set_title('y2')

    x = x.reshape(1, 256, 256, 1)
    x *= 255

    y = y.reshape(1, 256, 256, 2)
    y *= 128

    lab_original = torch.cat([x, y], dim=3).cpu().numpy()
    lab_original = lab_original.reshape(256, 256, 3)
    rgb_original = color.lab2rgb(lab_original)

    axes.append(fig.add_subplot(1, 6, 3))
    axes[-1].imshow(rgb_original)
    axes[-1].set_title('original')

    axes.append(fig.add_subplot(1, 6, 4))
    axes[-1].imshow(y_pred[:, :, 0], cmap='gray')
    axes[-1].set_title('y1_pred')
    axes.append(fig.add_subplot(1, 6, 5))
    axes[-1].imshow(y_pred[:, :, 1], cmap='gray')
    axes[-1].set_title('y2_pred')

    y_pred = y_pred.reshape(1, 256, 256, 2)
    y_pred *= 128

    lab_pred = torch.cat([x, y_pred], dim=3).cpu().numpy()
    lab_pred = lab_pred.reshape(256, 256, 3)
    rgb_pred = color.lab2rgb(lab_pred)

    axes.append(fig.add_subplot(1, 6, 6))
    axes[-1].imshow(rgb_pred)
    axes[-1].set_title('pred')


def evaluate_notebook(gen, val_loader, epoch, folder, device):
    print("evaluating...")
    gen.eval()
    for idx, (x, y) in enumerate(val_loader):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():

            y_fake = gen(x)
            show_res(x, y, y_fake)

    gen.train()


def evaluate(gen, val_loader, epoch, folder, device):
    print("evaluating...")
    gen.eval()
    for idx, (x, y) in enumerate(val_loader):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():

            y_fake = gen(x)

            y_fake = y_fake * 128
            x_scaled = x * 255

            lab = torch.cat([x_scaled, y_fake], dim=1).permute(
                0, 2, 3, 1).cpu().numpy()
            lab.reshape(256, 256, 3)
            rgb = color.lab2rgb(lab)

            rgb = rgb.reshape(256, 256, 3)
            x = x.cpu().numpy().reshape(256, 256)

            y0 = y_fake[:, 0, :, :].reshape(256, 256, 1).cpu().numpy() + 128
            y1 = y_fake[:, 1, :, :].reshape(256, 256, 1).cpu().numpy() + 128

            imageio.imwrite(os.path.join(folder, f"./G{idx}E{epoch}.png"), rgb)
            imageio.imwrite(os.path.join(folder, f"./y0{idx}E{epoch}.png"), y0)
            imageio.imwrite(os.path.join(folder, f"./y1{idx}E{epoch}.png"), y1)

    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
