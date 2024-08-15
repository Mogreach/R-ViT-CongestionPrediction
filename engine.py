import argparse
from model.factory import create_segmenter
from optim.factory import create_optimizer

def build_model(arg,cfg):

    model_cfg = cfg["model"][arg.backbone]
    dataset_cfg = cfg["dataset"][arg.dataset]
    im_size=arg.im_size
    crop_size = arg.crop_size
    window_size = arg.window_size
    window_stride = arg.window_stride

    if "mask_transformer" in arg.decoder:
        decoder_cfg = cfg["decoder"]["mask_transformer"]
    else:
        decoder_cfg = cfg["decoder"][arg.decoder]

    # model config
    if not im_size:
        im_size = dataset_cfg["im_size"]
    if not crop_size:
        crop_size = dataset_cfg.get("crop_size", im_size)
    if not window_size:
        window_size = dataset_cfg.get("window_size", im_size)
    if not window_stride:
        window_stride = dataset_cfg.get("window_stride", im_size)

    model_cfg["image_size"] = (crop_size, crop_size)
    model_cfg["backbone"] =arg.backbone
    model_cfg["dropout"] = arg.dropout
    model_cfg["drop_path_rate"] = arg.drop_path
    decoder_cfg["name"] = arg.decoder
    model_cfg["decoder"] = decoder_cfg
    model_cfg["channels"] = arg.in_channels


    net_kwargs=model_cfg
    net_kwargs["n_cls"] = arg.n_cls
    model = create_segmenter(net_kwargs)
    model.to("cuda")

    return model

def build_optimizer(arg,data_len,model):
    optimizer_kwargs = dict(
        opt=arg.optimizer,
        lr=arg.lr,
        weight_decay=arg.weight_decay,
        momentum=0.9,
        clip_grad=None,
        sched=arg.scheduler,
        epochs=arg.num_epochs,
        min_lr=1e-5,
        poly_power=0.9,
        poly_step_size=1,
    )
    optimizer_kwargs["iter_max"] = data_len * optimizer_kwargs["epochs"]
    optimizer_kwargs["iter_warmup"] = 0.0

    opt_args = argparse.Namespace()
    opt_vars = vars(opt_args)
    for k, v in optimizer_kwargs.items():
        opt_vars[k] = v

    optimizer = create_optimizer(opt_args, model)
    return optimizer

# def build_dataset(arg,cfg):
#     dataset_cfg = cfg["dataset"][arg.dataset]
#
#     # dataset config
#     world_batch_size = dataset_cfg["batch_size"]
#     num_epochs = dataset_cfg["epochs"]
#     lr = dataset_cfg["learning_rate"]
#     if arg.batch_size:
#         world_batch_size = arg.batch_size
#     if arg.epochs:
#         num_epochs = arg.epochs
#     if arg.learning_rate:
#         lr = arg.learning_rate
#     if arg.eval_freq is None:
#         eval_freq = dataset_cfg.get("eval_freq", 1)
#
#     dataset_kwargs = dict(
#         dataset=arg.dataset,
#         image_size=arg.im_size,
#         crop_size=arg.crop_size,
#         batch_size=arg.batch_size,
#         normalization=arg.model_cfg["normalization"],
#         split="train",
#         num_workers=10,
#     )
#     train_loader=build_dataset(arg_dict)

