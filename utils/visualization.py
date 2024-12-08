import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import threading
import queue
from typing import List, Tuple
from torch.utils.tensorboard import SummaryWriter
from io import BytesIO

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean, device=tensor.device).reshape(1, len(mean), 1, 1)
    std = torch.tensor(std, device=tensor.device).reshape(1, len(std), 1, 1)
    return tensor * std + mean

def show_reconstructions(fig, axs, model, data, epoch, color_mode):
    model.eval()
    with torch.no_grad():
        recon = model(data)
    recon = recon.cpu()
    data = data.cpu()

    # można dostosować mean i std dla konkretnego zestawu danych
    if color_mode == "RGB":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        mean = [0.5]
        std = [0.5]

    # recon = denormalize(recon, mean, std)
    # data = denormalize(data, mean, std)

    recon = torch.clamp(recon, 0, 1)
    data = torch.clamp(data, 0, 1)

    recon = recon.numpy()
    data = data.numpy()

    axs[0].clear()
    axs[1].clear()

    if color_mode == "RGB":
        axs[0].imshow(data[0].transpose(1, 2, 0))
        axs[1].imshow(recon[0].transpose(1, 2, 0))
    else:
        axs[0].imshow(data[0].transpose(1, 2, 0).squeeze(), cmap='gray')
        axs[1].imshow(recon[0].transpose(1, 2, 0).squeeze(), cmap='gray')

    axs[0].set_title('Original')
    axs[0].axis('off')
    axs[1].set_title('Reconstructed')
    axs[1].axis('off')

    fig.suptitle(f'Epoch {epoch + 1}')
    plt.draw()
    plt.pause(0.01)

def plot_losses(train_losses, val_losses, test_losses, grad_norms):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(16, 6))
    
    # wykres strat treningowych, walidacyjnych i testowych
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epoka')
    plt.ylabel('Strata')
    plt.title('Straty')
    plt.legend()
    
    # wykres norm gradientów
    plt.subplot(1, 2, 2)
    plt.plot(epochs, grad_norms, label='Norma Gradientu')
    plt.xlabel('Epoka')
    plt.ylabel('Norma')
    plt.title('Norma Gradientu')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def display_images_grid(
    epoch: int,
    image_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    train_losses: List[float],
    val_losses: List[float],
    test_losses: List[float],
    grad_norms: List[float],
    num_epochs: int,
    max_loss: float = 1.0,
    min_loss: float = 0.0,
    max_grad_norm: float = 1.0,
    color_mode: str = 'RGB'
) -> None:
    """
    wyświetla obrazy oryginalne i zrekonstruowane w strukturze 2x5 przy użyciu OpenCV,
    z odpowiednimi podpisami do każdego obrazu oraz dynamicznie aktualizowanymi wykresami strat
    i norm gradientów, zintegrowanymi w jednym oknie.
    """
    rows, cols = 2, 5
    grid_images = []
    
    titles = [
        "Treningowy Losowy - Oryginal", "Treningowy- Oryginal",
        "Testowy Losowy - Oryginal", "Testowy - Oryginal",
        "Wykres Strat",
        "Treningowy losowy - Rekonstrukcja", "Treningowy - Rekonstrukcja",
        "Testowy losowy - Rekonstrukcja", "Testowy- Rekonstrukcja",
        "Normy gradientów"
    ]

    if len(image_pairs) != 4:
        raise ValueError(f"Expected 4 image pairs, but got {len(image_pairs)}")


    # przetwarzanie każdego obrazu oryginalnego i jego rekonstrukcji
    for idx, (orig_image, recon_image) in enumerate(image_pairs):
        orig_image = orig_image.cpu().numpy()
        recon_image = recon_image.cpu().numpy()

        if color_mode == 'RGB':
            orig_image = np.transpose(orig_image, (1, 2, 0))  # (C, H, W) -> (H, W, C)
            recon_image = np.transpose(recon_image, (1, 2, 0))
            orig_image = cv2.cvtColor((orig_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            recon_image = cv2.cvtColor((recon_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        else:
            orig_image = (orig_image.squeeze() * 255).astype(np.uint8)
            recon_image = (recon_image.squeeze() * 255).astype(np.uint8)
            # konwersja do 3-kanałowego obrazu, aby umożliwić dodanie tekstu
            orig_image = cv2.cvtColor(orig_image, cv2.COLOR_GRAY2BGR)
            recon_image = cv2.cvtColor(recon_image, cv2.COLOR_GRAY2BGR)

        # Dodanie podpisów do obrazów
        cv2.putText(orig_image, titles[idx], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(recon_image, titles[idx + 5], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Dodanie do listy obrazów
        grid_images.append(orig_image)
        grid_images.append(recon_image)

    # generowanie wykresu strat jako obrazu
    plt.figure(figsize=(4, 3))
    epochs_range = range(1, epoch + 2)
    plt.plot(epochs_range, train_losses, label="Strata treningowa", color="blue")
    if val_losses:
        plt.plot(epochs_range, val_losses, label="Strata walidacyjna", color="orange")
    if test_losses:
        plt.plot(epochs_range, test_losses, label="Strata testowa", color="green")
    plt.xlim([1, num_epochs])
    plt.ylim([min_loss, max_loss])
    plt.xlabel("Epoka")
    plt.ylabel("Strata")
    plt.title("Wykres strat")
    plt.legend()
    plt.grid(True)
    # zapisywanie wykresu do obrazu w pamięci
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    loss_plot = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    plt.close()
    loss_plot = cv2.imdecode(loss_plot, cv2.IMREAD_COLOR)
    cv2.putText(loss_plot, titles[4], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # generowanie wykresu norm gradientów jako obrazu
    plt.figure(figsize=(4, 3))
    plt.plot(epochs_range, grad_norms, label="Norma Gradientu", color="purple")
    plt.xlim([1, num_epochs])
    plt.ylim([0, max_grad_norm])
    plt.xlabel("Epoka")
    plt.ylabel("Norma")
    plt.title("Wykres Norm Gradientów")
    plt.legend()
    plt.grid(True)
    # zapisywanie wykresu do obrazu w pamięci
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    grad_plot = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    plt.close()
    grad_plot = cv2.imdecode(grad_plot, cv2.IMREAD_COLOR)
    cv2.putText(grad_plot, titles[9], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    grid_images.insert(4, loss_plot)
    grid_images.append(grad_plot)

    if len(grid_images) != rows * cols:
        raise ValueError(f"Expected {rows * cols} images, but got {len(grid_images)}")

    # ustalenie maksymalnej wysokości i szerokości obrazów w siatce
    max_height = max(img.shape[0] for img in grid_images)
    max_width = max(img.shape[1] for img in grid_images)

    # zmiana rozmiaru obrazów do jednakowych wymiarów
    resized_images = []
    for img in grid_images:
        resized_img = cv2.resize(img, (max_width, max_height))
        resized_images.append(resized_img)

    # łączenie obrazów w strukturę 2x5
    row_images = []
    for i in range(rows):
        row = resized_images[i * cols:(i + 1) * cols]
        row_images.append(np.hstack(row))
    images_grid = np.vstack(row_images)

    # wyświetlanie obrazu siatki w jednym oknie OpenCV
    window_name = 'Reconstruction and Plots'
    cv2.imshow(window_name, images_grid)
    cv2.setWindowTitle(window_name, f'Epoch {epoch + 1}')

    # wydłużenie opóźnienia, aby dać czas na odświeżenie, można próbować skrócić
    cv2.waitKey(100)

class AsyncTensorBoardLogger:
    def __init__(self, log_dir='runs', max_queue_size=1000):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._process_queue)
        self.thread.start()
    
    def _process_queue(self):
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                func, args, kwargs = self.queue.get(timeout=0.1)
                func(*args, **kwargs)
                self.queue.task_done()
            except queue.Empty:
                continue
    
    def log_scalar(self, tag, value, step):
        try:
            self.queue.put_nowait((self.writer.add_scalar, (tag, value, step), {}))
        except queue.Full:
            pass  # można dodać obsługę przepełnienia
    
    def log_histogram(self, tag, values, step):
        try:
            self.queue.put_nowait((self.writer.add_histogram, (tag, values, step), {}))
        except queue.Full:
            pass
    
    def log_images(self, tag, imgs, step):
        try:
            self.queue.put_nowait((self.writer.add_images, (tag, imgs, step), {}))
        except queue.Full:
            pass
    
    def close(self):
        self.stop_event.set()
        self.thread.join()
        self.writer.close()
        
def log_losses(epoch, epoch_losses, grad_norms, last_lr):
    """
    loguje średnie straty dla każdej epoki.
    dodatkowo również normy gradientów
    """
    avg_loss = sum(epoch_losses["loss"]) / len(epoch_losses["loss"])
    avg_grad_norm = sum(grad_norms) / len(grad_norms)
    
    print(f"Epoch [{epoch+1}], Loss: {avg_loss:.4f}, Grad Norm: {avg_grad_norm:.4f}, Last LR:  {last_lr:.6f}")
    