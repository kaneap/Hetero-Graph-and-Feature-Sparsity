from datetime import datetime
import os
import torch

def save_model(model, directory=None, file_name=None):
    if file_name is None:
        file_name = f'model_{datetime.now()}.pt'
    else:
        file_name = file_name + '.pt'
    
    if directory is None:
        directory = 'graph-learning/saved_models'
    
    path = os.path.join(directory, file_name)

    torch.save(model.state_dict(), path)


    