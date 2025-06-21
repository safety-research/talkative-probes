import torchdata
print(f"torchdata version: {torchdata.__version__}")

try:
    from torchdata.stateful_dataloader import StatefulDataLoader
    print("StatefulDataLoader is available")
except ImportError:
    print("StatefulDataLoader is NOT available")

try:
    from torchdata.dataloader2 import DataLoader2
    print("DataLoader2 is still available")
except ImportError:
    print("DataLoader2 is NOT available")