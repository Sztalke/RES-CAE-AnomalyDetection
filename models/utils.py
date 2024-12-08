import torch

def save_model(model, optimizer, epoch, filepath, ):
    '''
    zapisuje typ słownikowy oraz cały model.
    '''
    model_filepath = str(filepath + "_model")
    state_filepath = str(filepath + "_state")
    # zapisz cały obiekt modelu
    torch.save(model, model_filepath)
    
    # zapisz stan optymalizatora i epokę
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(state, state_filepath)

def load_model_and_state(filepath):
    model_filepath = str(filepath + "_model")
    state_filepath = str(filepath + "_state")
    
    # wczytaj cały obiekt modelu
    model = torch.load(model_filepath)
    
    # wczytaj stan optymalizatora i epokę
    state = torch.load(state_filepath)
    model.load_state_dict(state['model_state_dict'])
    
    optimizer = torch.optim.Adam(model.parameters())  # trzeba pamiętać o optymizerze !! ___________________________________________ NOTATKA
    optimizer.load_state_dict(state['optimizer_state_dict'])
    
    epoch = state['epoch']
    
    return model, optimizer, epoch
