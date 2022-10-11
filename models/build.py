from .yolo import myYOLO


def build_yolo(args, device, train_size, num_classes=20, trainable=False):
    print('Let us train yolo on the %s dataset ......' % (args.dataset))
    model = myYOLO(device=device,
                   input_size=train_size, 
                   num_classes=num_classes,
                   trainable=trainable)

    return model
