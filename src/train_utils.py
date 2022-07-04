import os
import torch
import wandb
import numpy as np
from tqdm import tqdm


def forward(model, x, y, criterion, device):
    x = x.to(device)
    y = y.to(device)
    prd = model(x)
    loss = criterion(prd, y.squeeze(1))
    return loss, prd


def evaluate(model, dataloader, device, criterion):
    losses = []
    model.eval()
    with torch.no_grad():
        val_iter = iter(dataloader)
        for i in range(len(dataloader)):
            img, _, resized_mask = next(val_iter)
            loss, _ = forward(model, img, resized_mask, criterion, device)
            losses.append(loss.item())
    return np.mean(losses)


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    device,
    optimizer,
    scheduler,
    cfg,
    criterion,
):

    losses_train, losses_train_mean = [], []
    losses_val = []
    loss_val = None
    best_val_loss = 1e6

    tr_it = iter(train_dataloader)
    progress_bar = tqdm(range(cfg["n_iter"]))

    torch.cuda.empty_cache()

    if cfg["wandb_logging"]:
        wandb.watch(model)
    for i in progress_bar:
        try:
            img, _, resized_mask = next(tr_it)
        except StopIteration:
            tr_it = iter(train_dataloader)
            img, _, resized_mask = next(tr_it)

        model.train()
        torch.set_grad_enabled(True)

        loss, _ = forward(model, img, resized_mask, criterion, device)

        optimizer.zero_grad()
        loss.backward()

        if cfg["clip_grad"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        losses_train.append(loss.item())
        losses_train_mean.append(np.mean(losses_train[-1:-10:-1]))
        progress_bar.set_description(
            f"loss: {loss.item():.5f}, avg loss: {np.mean(losses_train):.5f}"
        )
        if cfg["wandb_logging"]:
            wandb.log({"train_loss": loss.item(), "iter": i})

        if i % cfg["n_iter_val"] == 0:
            loss_val = evaluate(model, val_dataloader, device, criterion)
            losses_val.append(loss_val)
            progress_bar.set_description(f"val_loss: {loss_val:.5f}")
            if cfg["wandb_logging"]:
                wandb.log({"val_loss": loss_val, "iter": i})

        if scheduler:
            if scheduler.__class__.__name__ == "ReduceLROnPlateau" and loss_val:
                scheduler.step(loss_val)
            else:
                scheduler.step()

        if cfg["save_best_val"] and loss_val < best_val_loss:
            best_val_loss = loss_val
            checkpoint_path = cfg["checkpoint_path"]
            torch.save(
                model.state_dict(),
                f"{checkpoint_path}/{model.__class__.__name__}_{loss_val:.3f}.pth",
            )
            if cfg["wandb_logging"]:
                wandb.save(os.path.join(wandb.run.dir, "model.h5"))

    if cfg["wandb_logging"]:
        wandb.finish()
