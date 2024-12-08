import torch
import torch.amp
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
from training.losses import ssim_loss
from utils.visualization import display_images_grid, log_losses, AsyncTensorBoardLogger, denormalize
import random

def train_model(model, train_loader, val_loader, test_loader, optimizer, scheduler, scaler, num_epochs, clip_value, color_mode, interval=5):
    visualization_enable = True
    tensorboard_enable = False
    cross_validation_enable = True
    
    train_losses = []
    val_losses = []
    test_losses = []
    grad_norms = []
    
    # stałe obrazy do wizualizacji podczas treningu
    fixed_train_image = next(iter(train_loader))[0].unsqueeze(0).to(model.device)
    fixed_test_image = next(iter(test_loader))[0].unsqueeze(0).to(model.device)

    
    # inicjalizacja asynchronicznego loggera
    if (tensorboard_enable):
        logger = AsyncTensorBoardLogger(log_dir=f'runs/{model.__class__.__name__}_async_1')

    # rozmiar okna SSIM może być dynamiczny, zależny od epoki, eksperyment
    ssim_kernel = 11
    
    for epoch in range(num_epochs):
        # ________________________________________________________________________________________________________________ TRAIN
        model.train()
        epoch_losses = []
        epoch_grad_norms = []
        
        # dynamiczny rozmiar okna ssim
        # if epoch > .5 * num_epochs:
        #     ssim_kernel = 9
        # elif epoch > .8 * num_epochs:
        #     ssim_kernel = 5

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_data in pbar:
                # przeniesienie danych na device
                batch_data = batch_data.to(model.device)

                # trening na batchu z metody modelu (BASE tam szukać funkcji straty)
                losses, grad_norm = model.train_step(batch_data, optimizer, scaler, ssim_kernel, grad_clip_value=clip_value)

                # dodanie wartości strat do słownika (dla AE stochastycznego, dla deterministycznego wartości wpisane = 0)
                epoch_losses.append(losses)
                epoch_grad_norms.append(grad_norm)

                # logowanie prędkości przetwarzania, działa średnio, ale spektakularnie
                images_per_second = pbar.format_dict['rate'] * batch_data.size(0) if pbar.format_dict['rate'] else 0
                pbar.set_postfix_str(f'{images_per_second:.2f} img/s, lr: {scheduler._last_lr}')

        # straty dla epoki są uśrednione, zbiór treningowy
        train_losses.append((sum(epoch_losses) / len(epoch_losses)).item())
        avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms)
        grad_norms.append(avg_grad_norm)
        # ________________________________________________________________________________________________________________ TRAIN END
        
        # ________________________________________________________________________________________________________________ EVAL
        if (cross_validation_enable):
            model.eval()
            # zbiór walidacyjny
            val_loss = model.evaluate(val_loader, ssim_kernel)
            val_losses.append(val_loss)
            
            # zbiór testowy
            test_loss = model.evaluate(test_loader, ssim_kernel)
            test_losses.append(test_loss)

        if (tensorboard_enable):
            # logowanie metryk do TensorBoard tylko raz na epokę
            logger.log_scalar('Loss/epoch/train', train_losses[-1], epoch)
            if (cross_validation_enable):
                logger.log_scalar('Loss/epoch/val', val_loss, epoch)
                logger.log_scalar('Loss/epoch/train', test_loss, epoch)
            logger.log_scalar('GradNorm/train_epoch', avg_grad_norm, epoch)

            # logowanie histogramów wag i gradientów co 'interval' epok
            if (epoch + 1) % interval == 0:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        logger.log_histogram(f'Weights/{name}', param, epoch)
                        if param.grad is not None:
                            logger.log_histogram(f'Gradients/{name}', param.grad, epoch)

            # logowanie obrazów rekonstrukcji do TensorBoard co 'interval' epok
            # if (epoch + 1) % interval == 0:
            #     with torch.no_grad():
            #         # użyj ostatniego batcha z epoki do logowania obrazów
            #         input_images = batch_data.cpu()
            #         reconstructed_images = model(batch_data).cpu()

            #         # odwrócenie normalizacji ...

            #         # sprawdzenie kształtu tensorów
            #         if input_images.dim() == 3:
            #             # jeśli brak kanału, dodaj go
            #             input_images = input_images.unsqueeze(1)
            #         if reconstructed_images.dim() == 3:
            #             reconstructed_images = reconstructed_images.unsqueeze(1)
            #         elif reconstructed_images.dim() == 5:
            #             # usuń nadmiarowy wymiar
            #             reconstructed_images = reconstructed_images.squeeze(2)
            #         if input_images.dim() == 5:
            #             input_images = input_images.squeeze(2)

            #         # logowanie tylko pierwszych 4 obrazów z batcha
            #         num_images = min(4, input_images.size(0))
            #         writer = logger.writer
            #         writer.add_images('Input Images', input_images[:num_images], epoch)
            #         writer.add_images('Reconstructed Images', reconstructed_images[:num_images], epoch)

        # dla schedulera reduce on plateau
        # krok scheduler na podstawie straty walidacyjnej (czy treningowej??????)
        # if (cross_validation_enable):
        #     scheduler.step(val_losses[-1])
        # else:
        #     scheduler.step(train_losses[-1])
        # dla schedulera harmonicznego 
        scheduler.step()
    
        # ________________________________________________________________________________________________________________ EVAL END
        
        # wizualizacja co określoną liczbę epok
        if visualization_enable :#and (epoch + 1) % interval == 0:
            # losowy obraz treningowy
            random_batch = random.choice(list(train_loader))
            batch_size = random_batch.size(0)
            random_index = random.randint(0, batch_size - 1)
            random_train_image = random_batch[random_index].unsqueeze(0).to(model.device)

            # losowy obraz testowy
            random_batch = random.choice(list(test_loader))
            batch_size = random_batch.size(0)
            random_index = random.randint(0, batch_size - 1)
            random_test_image = random_batch[random_index].unsqueeze(0).to(model.device)
            
            with torch.no_grad():
                fixed_train_recon = model(fixed_train_image)
                fixed_test_recon = model(fixed_test_image)
                random_train_recon = model(random_train_image)
                random_test_recon = model(random_test_image)
                 
            image_tuples = [
                (random_train_image.squeeze(0).cpu(), random_train_recon.squeeze(0).cpu()),
                (fixed_train_image.squeeze(0).cpu(), fixed_train_recon.squeeze(0).cpu()),
                (random_test_image.squeeze(0).cpu(), random_test_recon.squeeze(0).cpu()),
                (fixed_test_image.squeeze(0).cpu(), fixed_test_recon.squeeze(0).cpu())
            ]
            
            # wizualizacja
            if (epoch + 1) % interval == 0:
                display_images_grid(
                                epoch=epoch,
                                image_pairs=image_tuples,
                                train_losses=train_losses,
                                val_losses=val_losses,
                                test_losses=test_losses,
                                grad_norms=grad_norms,
                                num_epochs=num_epochs,
                                max_loss=1.05 * max(max(train_losses), max(val_losses), max(test_losses)),
                                min_loss=0.95 * min(min(train_losses), min(val_losses), min(test_losses)),
                                max_grad_norm= 1.05 * max(grad_norms),
                                color_mode=color_mode
                                )

    # zamknij logger
    if tensorboard_enable and logger:
        logger.close()

    return train_losses, val_losses, test_losses, grad_norms
    
def train_model1(model, data_loader, optimizer, scheduler, scaler, num_epochs, clip_value, color_mode, interval=5):
    train_losses = {"loss": []}
    grad_norms = []
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    plt.ion()  # tr ineraktywny, może powodować problemy

    # rozmiar okna ssim jest dynamiczny, zalezy od epoki, eksperyment
    ssim_kernel = 9


    for epoch in range(num_epochs):
        model.train()
        epoch_losses = {"loss": []}
        epoch_grad_norms = []
        
        # dynamiczne okno ssim, wersja eksperymentalna
        # if epoch > .7 * num_epochs:
        #     ssim_kernel = 3
        # elif epoch > .5 * num_epochs:
        #     ssim_kernel = 5
        # elif epoch > .3 * num_epochs:
        #     ssim_kernel = 7
        
        with tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_data in pbar:
                # przeniesienie danych na device
                batch_data = batch_data.to(model.device)
                
                # trening na batchu z metody modelu (BASE tam szukać funkcji straty)
                losses, grad_norm = model.train_step(batch_data, optimizer, scaler, ssim_kernel, clip_value)

                # dodanie wartości strat do słownika (dla AE stochastycznego, dla deterministycznego wartości wpisane = 0)
                for key in epoch_losses:
                    epoch_losses[key].append(losses[key])
                epoch_grad_norms.append(grad_norm)
                
                # logowanie prędkości przetwarzania, działa średnio, ale spektakularnie
                images_per_second = pbar.format_dict['rate'] * batch_data.size(0) if pbar.format_dict['rate'] else 0
                pbar.set_postfix_str(f'{images_per_second:.2f} img/s')

        # straty dla epoki są uśrednione
        for key in train_losses:
            train_losses[key].append(sum(epoch_losses[key]) / len(epoch_losses[key]))
        grad_norms.append(sum(epoch_grad_norms) / len(epoch_grad_norms))

        # log strat dla epoki
        if interval > 0 and epoch % interval == 0:
            log_losses(epoch, epoch_losses, epoch_grad_norms, scheduler.get_last_lr()[0])

        # wizualizacja strat
        model.log_and_visualize(fig, axs, epoch, epoch_losses, batch_data, color_mode)
        
        scheduler.step(sum(epoch_losses["loss"]) / len(epoch_losses["loss"]))

    plt.ioff()  # wyłączenie trybu interaktywnego
    plt.show()
    
    return train_losses, grad_norms