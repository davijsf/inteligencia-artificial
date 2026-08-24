import torch, numpy as np;

if __name__ == "__main__": 
    print("torch: ", torch.__version__, "| numpy: ", np.__version__)
    print("GPU disponível: ", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Usando device:", device)