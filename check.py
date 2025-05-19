import torch
print("PyTorch CUDA version:", torch.version.cuda)
print("Is CUDA available?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("no GPU available")