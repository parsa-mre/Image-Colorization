import torch
import torch.optim as optim
import torch.nn as nn
import tqdm
import config
from discriminator import Discriminator
from generator import Generator
import data


def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler, epoch):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        if epoch % 3 != 2:
            # Train Discriminator
            with torch.cuda.amp.autocast():
                y_fake = gen(x)
                D_real = disc(x, y)
                D_real_loss = bce(D_real, torch.ones_like(D_real))
                D_fake = disc(x, y_fake.detach())
                D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss) / 4

            disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)

            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA / 2
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0 and epoch % 3 != 2:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )


def train_with_epoch(epochs):

    disc = Discriminator().to(config.DEVICE)
    gen = Generator().to(config.DEVICE)

    opt_disc = optim.Adam(
        disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(
        gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    train_dataset = data.FlowerDataset(config.TRAIN_DATA_DIR)
    train_loader = data.DataLoader(
        train_dataset, config.BATCH_SIZE, shuffle=True)

    # val_dataset = data.FlowerDataset(config.VAL_DATA_DIR, train=False)
    # val_loader = data.DataLoader(val_dataset, shuffle=False)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    for epoch in range(epochs):
        train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler, epoch
        )

        # evaluate_notebook(gen, val_loader, epoch, IMAGE_FOLDER, DEVICE)
