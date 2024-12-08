import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.CustomDataset import CustomImageDataset
from datasets.transforms import get_transform
from models.res_cae import initialize_model as initialize_res_cae
from models.utils import save_model
from training.trainer import train_model
from utils.file_utils import create_directory_if_not_exists
from utils.visualization import plot_losses
from torchsummary import summary
import matplotlib.pyplot as plt

# słownik z typami modeli oraz ścieżkami do ich konfiguracji
model_configs = {
    "ResCAE": {
        "initialize_model": initialize_res_cae,
        "config_path": './configs/res_cae_config.json'
    },
}

def main(model_type):
    if model_type not in model_configs:
        raise ValueError(f"Unsupported model type: {model_type}")
    # jeśli True, wydrukuje architekturę
    print_summary = False
    
    # wybór akceleratora, w przyszłosci można implementować trening multi gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # selektor zbioru (konkretny przypadek)
    img_set = 1
    # obrazy sa transformowane do rozmiaru: (H x W)
    input_shape = (512, 512)
    batch_size = 4
    num_workers = 6
    
    # RGB lub L- zmiana powinna wiązać się z dostosowaniem liczby kanałów wejściowych i wyjściowych
    color_mode = "L"
    
    num_epochs = 196
    learning_rate = 0.001

    # gradient clipping
    clip_value = 1
    
    # iterwał wyświetlania rekonstrukcji 
    interval = 1
    
    
    config = model_configs[model_type]
    config_path = config['config_path']
    model = config['initialize_model'](config_path, input_shape=input_shape, device=device).to(device)
    

    if print_summary:
        summary(model, (lambda shape, mode: (3,) + shape if mode == "RGB" else (1,) + shape)(input_shape, color_mode))
    else:

        train_img_dir = f'./data/train/set{img_set}'
        val_img_dir = f'./data/validation/set{img_set}'
        test_img_dir = f'./data/test/set{img_set}'
        transform = get_transform(color_mode, input_shape)
        
        train_dataset = CustomImageDataset([train_img_dir], transform=transform, color_mode=color_mode)
        val_dataset = CustomImageDataset([val_img_dir], transform=transform, color_mode=color_mode)
        test_dataset = CustomImageDataset([test_img_dir], transform=transform, color_mode=color_mode)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, multiprocessing_context='spawn', persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, multiprocessing_context='spawn', persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, multiprocessing_context='spawn', persistent_workers=True)


        # PARAMETR optimizer
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001, amsgrad= True, foreach=None)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30, threshold=.05 , threshold_mode='abs', min_lr=1e-6,)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=5, eta_min=learning_rate/1000)
        scaler = torch.amp.GradScaler()

        losses, val_losses, test_losses, grad_norms = train_model(model, train_loader, val_loader, test_loader, optimizer, scheduler, scaler, num_epochs, clip_value, color_mode, interval)
        
        plot_losses(losses, val_losses, test_losses, grad_norms)

        # zapis checkpointu, można wyłączyć
        # checkpointy są nadpisywane po typie
        save_dir = "./saved_models/"
        create_directory_if_not_exists(save_dir)
        save_dir = save_dir + model_type
        save_model(model, optimizer, num_epochs, save_dir)
        
        #________________________________________________________________________________________
        # po treningu zapisuje wykresy strat i norm gradientów
        dpi = 400

        # wykres strat treningowych, walidacyjnych i testowych
        plt.figure(figsize=(10, 6))
        plt.plot(losses, label="Losses", color="blue")
        plt.plot(val_losses, label="Validation Losses", color="orange")
        plt.plot(test_losses, label="Test Losses", color="green")

        plt.title("Training, Validation, and Test Losses")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig("losses_plot.png", dpi=dpi)
        plt.close()

        # wykres norm gradientów
        plt.figure(figsize=(10, 6))
        plt.plot(grad_norms, label="Gradient Norms", color="red")

        plt.title("Gradient Norms")
        plt.xlabel("Epochs")
        plt.ylabel("Gradient Norms")
        plt.legend()
        plt.grid(True)
        plt.savefig("gradient_norms_plot.png", dpi=dpi)
        plt.close()
        #________________________________________________________________________________________

        

    

if __name__ == "__main__":
    model_type = "ResCAE"
    print(f"Training model: {model_type}")
    main(model_type)

