import torch
from Configs.trainConf import configs
from Nets.model import Model


if __name__ == "__main__":
    model = Model(configs.net)
    model.eval()
    x = torch.randn(1, 3, 320, 320)
    try:
        torch.export.export(f=model, args=(x,))
        print("[JIT] torch.export successed.")
        exit(0)
    except Exception as e:
        print("[JIT] torch.export failed.")
        raise e
