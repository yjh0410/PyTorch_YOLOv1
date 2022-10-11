from .yolo import myYOLO


def build_yolo(args, device, train_size, num_classes=20, trainable=False):
    model = myYOLO(device=device,
                   input_size=train_size, 
                   num_classes=num_classes,
                   trainable=trainable)

    return model
