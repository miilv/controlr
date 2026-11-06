"""Initialize a separate vocabulary by selecting existing classifier rows; source checkpoint is never modified."""
import argparse
import json


def select(yolo, names):
    import torch

    old = [yolo.model.names[i] for i in range(len(yolo.model.names))]
    indices = [old.index(k) for k in names]
    head = yolo.model.model[-1]
    for seq in head.cv3:
        conv = seq[-1]
        new = torch.nn.Conv2d(conv.in_channels, len(names), 1).to(conv.weight.device, conv.weight.dtype)
        with torch.no_grad():
            new.weight.copy_(conv.weight[indices])
            new.bias.copy_(conv.bias[indices])
        seq[-1] = new
    head.nc = len(names)
    head.no = len(names) + head.reg_max * 4
    yolo.model.nc = len(names)
    yolo.model.yaml['nc'] = len(names)
    yolo.model.names = dict(enumerate(names))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--classes', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    from ultralytics import YOLO

    model = YOLO(a.model)
    select(model, json.load(open(a.classes))['names'])
    model.save(a.out)


if __name__ == '__main__':
    main()
