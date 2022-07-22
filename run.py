import json
import sys
import os
import yaml
import argparse
import datetime
import os.path as osp
import importlib
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from utils.utils import Unbuffered


def main(config):
    # mdkir save dir
    if config['train']:
        time_str = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
        exp_ckpt_dir = osp.join(config["logging_params"]['ckpt']['save_dir'], time_str)
    else:
        exp_ckpt_dir = osp.join(config["logging_params"]['ckpt']['save_dir'], 'test')
    exp_logger_dir = os.path.join(exp_ckpt_dir, "log")
    os.makedirs(exp_ckpt_dir, exist_ok=True)
    os.makedirs(exp_logger_dir, exist_ok=True)

    # save print info to txt
    log_file = os.path.join(exp_logger_dir, "log.txt")
    sys.stdout = Unbuffered(sys.stdout, log_file)
    print(f"shell command:{sys.argv}")
    # print(json.dumps(config, indent=4))

    # Create model and experiment instance
    model_version = config['model_params']['model_name']
    full_path = "{}.{}".format("models", model_version)
    if importlib.util.find_spec(full_path):
        model = getattr(importlib.import_module(full_path), model_version)
    else:
        raise ValueError('Unknown class {}'.format(model_version))
    model = model(**config['model_params'])
      
    if config["task"] == "sceneflow":
        from scene_flow import SceneFlowModel
        experiment = SceneFlowModel(model, config)
    else:
        raise NotImplementedError

    if 'pre_trained_weights_checkpoint' in config['exp_params'].keys():
        print(f"Loading pre-trained model: {config['exp_params']['pre_trained_weights_checkpoint']}")
        checkpoint = torch.load(config['exp_params']['pre_trained_weights_checkpoint'], map_location=lambda storage, loc: storage)
        experiment.load_state_dict(checkpoint['state_dict'])


    # Create  Logger
    if config["logging_params"]["log"]:
        tensorboard_logger = TensorBoardLogger(save_dir=exp_logger_dir, name="default")

    # Create a trainer instance
    # use trainer_params to set num_nodes and gpus
    monitor = config["logging_params"]["ckpt"]["monitor"]
    ckpt_callback = ModelCheckpoint(
                                    dirpath=exp_ckpt_dir,
                                    monitor=monitor,
                                    filename="{epoch:03d}-{" + monitor + ":.4f}",
                                    save_last=True,
                                    save_top_k=config["logging_params"]["ckpt"]["top_k"],
                                    save_on_train_epoch_end=True,
                                    mode="min",
                                    )
    
    trainer = Trainer(logger=tensorboard_logger if config["logging_params"]["log"] else None,
                      callbacks=[ckpt_callback],
                      **config['trainer_params'])

    if config['train']:
        print('Start Training!')
        trainer.fit(experiment, ckpt_path=config['trainer_params'].get("resume_from_checkpoint", None))
    else:
        print('Start Testing')
        trainer.test(experiment, ckpt_path=config['trainer_params']["resume_from_checkpoint"])


if __name__ == '__main__':
    # Load config file from input arguments
    parser = argparse.ArgumentParser(description='Generic runner')
    parser.add_argument('--config', '-c',
                        dest='filename',
                        metavar='FILE',
                        help='Path to .yaml config file for the experiment',
                        default='configs/test/flowstep3d_self.yaml')
    parser.add_argument('--prefix', '-p',
                        default='default')
    parser.add_argument("--seed", default=12)
    parser.add_argument("--pre_ckpt", type=str, default=None)
    parser.add_argument("--test_ckpt", type=str, default=None)
    parser.add_argument("--gpus", type=int, default=None, help="the number of gpus")
    parser.add_argument("--batch_size", type=int, default=None)

    args = parser.parse_args()
    with open(args.filename, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    
    pl.utilities.seed.seed_everything(seed=args.seed)

    # Run
    config["logging_params"]["ckpt"]["save_dir"] = os.path.join(config["logging_params"]["ckpt"]["save_dir"], args.prefix)
    if args.pre_ckpt is not None:
        config["exp_params"]["pre_trained_weights_checkpoint"] = args.pre_ckpt
    if args.test_ckpt is not None:
        config["trainer_params"]["resume_from_checkpoint"] = args.test_ckpt
    if args.gpus is not None:
        config["trainer_params"]["gpus"] = args.gpus
    if args.batch_size is not None:
        config["exp_params"]["batch_size"] = args.batch_size
    main(config)
    sys.stdout.close()
