from pathlib import Path
import logging
import pytorch_lightning as pl
import hydra
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from cross_view_transformer.common import setup_config, setup_experiment, load_backbone
from cross_view_transformer.callbacks.visualization_callback import VisualizationCallback
from cross_view_transformer.tabular_logger import TabularLogger


log = logging.getLogger(__name__)
CONFIG_PATH = Path.cwd() / 'config'
CONFIG_NAME = 'config.yaml'


def maybe_resume_training(experiment):
    save_dir = Path(experiment.save_dir).resolve()
    checkpoints = list(save_dir.glob(
        f'**/{experiment.uuid}/checkpoints/*.ckpt'))

    log.info(f'Searching {save_dir}.')

    if not checkpoints:
        return None

    log.info(f'Found {checkpoints[-1]}.')

    return checkpoints[-1]


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    setup_config(cfg)

    pl.seed_everything(cfg.experiment.seed, workers=True)

    Path(cfg.experiment.save_dir).mkdir(exist_ok=True, parents=False)

    # Create and load model/data
    model_module, data_module, viz_fn = setup_experiment(cfg)


    # Optionally load model
    ckpt_path = maybe_resume_training(cfg.experiment)
    # ckpt_path = '/media/chz/My Passport/GKT-main/segmentation/outputs/2024-07-10/06-51-37/checkpoints/model_epoch=18.ckpt'
    if ckpt_path is not None:
        model_module.backbone = load_backbone(ckpt_path)

    # Loggers and callbacks
    logger = pl.loggers.TensorBoardLogger(save_dir=cfg.experiment.save_dir)
    tab_logger = TabularLogger(save_dir=cfg.experiment.save_dir)
    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(
                        # filename='model',
                        # every_n_train_steps=cfg.experiment.checkpoint_interval,
                        filename = 'model_{epoch}',  # 包含epoch和val_loss的文件名，便于区分
                        dirpath = 'checkpoints/',  # 保存检查点的目录
                        monitor = 'val/metrics/iou@0.40',  # 监控验证集的loss（或其他你选择的指标）
                        mode = 'max',  # 如果是loss，则保存最小时的模型；如果是accuracy，则可能是'max'
                        save_top_k = 1,  # 只保存最好的一个模型（在这个场景下，由于每轮都保存，所以k=1是合适的）
                        verbose = True,  # 打印保存信息
                        every_n_epochs = 1, # 每1个epoch保存一次
    ),
        VisualizationCallback(viz_fn, cfg.experiment.log_image_interval),
    ]

    # Train
    trainer = pl.Trainer(logger=[logger, tab_logger],
                         callbacks=callbacks,
                         enable_progress_bar=True,
                         # strategy=DDPStrategy(find_unused_parameters=False),
                         **cfg.trainer)
    trainer.fit(model_module, datamodule=data_module, ckpt_path=ckpt_path)
    trainer.validate(model_module, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()