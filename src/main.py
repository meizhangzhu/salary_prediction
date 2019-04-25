from __future__ import absolute_import
import json
import random
import code
import time
import sys
import math
import argparse

import torch
import numpy as np

from data_source import DataSource
sys.path.append("..")
from model import Model
from helpers import StatisticsReporter, metric_is_improving

def str2bool(v):
    return v.lower() in ('true', '1', "True")

norm_stats = {
    "yearsExperience": {"mean": 12.0, "min": 0, "max": 24.0},
    "milesFromMetropolis": {"mean": 49.53, "min": 0, "max": 99.0},
    "y": {"mean": 116.06, "min": 0, "max": 301},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model - numbers
    parser.add_argument("--encode_type", type=str, default="embedding")
    parser.add_argument("--embedding_dim", type=int, default=50)
    parser.add_argument("--hidden1_dim", type=int, default=100)
    parser.add_argument("--hidden2_dim", type=int, default=50)

    # training
    parser.add_argument("--seed", type=int, default=42, help="random initialization seed")
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout probability")
    parser.add_argument("--l2_penalty", type=float, default=0.0000, help="l2 penalty")
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer")
    parser.add_argument("--init_lr", type=float, default=0.001, help="init learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-7, help="init learning rate")
    parser.add_argument("--lr_decay_rate", type=float, default=0.8)
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="gradient clipping")
    parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=100, help="batch size for traing")
    parser.add_argument("--eval_batch_size", type=int, default=100, help="batch size for test")
    parser.add_argument("--standardize_x", type=str2bool, default=False, help="standardize X")
    parser.add_argument("--standardize_y", type=str2bool, default=False, help="standardize Y")

    # management
    parser.add_argument("--n_check_loss", type=int, default=100, help="check loss after n batches")
    parser.add_argument("--n_validate", type=int, default=1000, help="validate after n batches")
    parser.add_argument("--model_path", help="path to model")
    parser.add_argument("--enable_log", type=str2bool, default=False)
    parser.add_argument("--save_model", type=str2bool, default=False)
    config = parser.parse_args()

    # set random seeds
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # data loaders & number reporters
    trn_reporter = StatisticsReporter()
    dev_reporter = StatisticsReporter()
    train_data_source = DataSource("train", norm_stats, standardize_x=config.standardize_x, standardize_y=config.standardize_y)
    dev_data_source = DataSource("dev", norm_stats, y_scaler=train_data_source.y_scaler, standardize_x=config.standardize_x, standardize_y=config.standardize_y)
    test_data_source = DataSource("test", norm_stats, y_scaler=train_data_source.y_scaler, standardize_x=config.standardize_x, standardize_y=config.standardize_y)

    LOG_FILE_NAME = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

    def mlog(s):
        if config.enable_log:
            with open("../log/{}.log".format(LOG_FILE_NAME), "a+") as log_f:
                log_f.write(s+"\n")
        print(s)

    model = Model(config, norm_stats, y_scaler=train_data_source.y_scaler)
    mlog(str(model))
    if torch.cuda.is_available():
        mlog("----- Using GPU -----")
        model = model.cuda()

    if config.model_path:
        model.load_model(config.model_path)
        mlog("----- Model loaded -----")
        mlog("model path: {}".format(config.model_path))

    start_time = time.time()
    mlog("----- Hyper-parameters -----")
    for k, v in sorted(dict(config.__dict__).items()):
        mlog("{}: {}".format(k, v))

    if not config.model_path:
        n_step = 0
        loss_history = []
        lr = config.init_lr
        for epoch in range(1, config.n_epochs+1):
            if lr <= config.min_lr:
                break

            # Train
            n_batch = 0
            train_data_source.epoch_init()
            while True:
                batch_data = train_data_source.next(config.batch_size)
                if batch_data is None:
                    break

                model.train()
                model.train_step(batch_data, lr, n_step)
                trn_reporter.update_data(model.ret_statistics)
                n_step += 1
                n_batch += 1

                # Session result output
                if n_step > 0 and n_step % config.n_check_loss == 0:
                    display_str = "{:.2f}s Epoch {} batch {} - ".format(time.time()-start_time, epoch, n_batch)
                    display_str += trn_reporter.to_string()
                    mlog(display_str)
                    trn_reporter.clear()

                if n_step > 0 and n_step % config.n_validate == 0:
                    model.eval()

                    # Dev
                    display_str = "<Train> learning rate: {}".format(lr)
                    mlog(display_str)

                    dev_data_source.epoch_init(shuffle=False)
                    while True:
                        batch_data = dev_data_source.next(config.eval_batch_size)
                        if batch_data is None:
                            break

                        model.evaluate_step(batch_data)
                        dev_reporter.update_data(model.ret_statistics)

                    display_str = "\n<Dev> (recog) - {:.3f}s - ".format(time.time()-start_time)
                    display_str += dev_reporter.to_string()
                    display_str += "\n"
                    mlog(display_str)

                    # Save model if it has lower dev ppl
                    if config.save_model:
                        torch.save(model.state_dict(), "../data/model/{}.model.pt".format(LOG_FILE_NAME))
                        mlog("model saved to data/model/{}.model.pt".format(LOG_FILE_NAME))
                        if torch.cuda.is_available():
                            model = model.cuda()

                    loss_history.append(dev_reporter.get_value("loss"))
                    dev_reporter.clear()

                    # Learning rate decay every epoch
                    if not metric_is_improving(loss_history):
                        #model.set_optimizer()
                        lr = lr*config.lr_decay_rate
                        #lr = max(lr, config.min_lr)
    else:
        test_data_source.epoch_init()
        predictions = []
        while True:
            batch_data = test_data_source.next(config.eval_batch_size)
            if batch_data is None:
                break

            model.eval()
            Y_pred = model.evaluate_step(batch_data)
            Y_pred = Y_pred.tolist()

            predictions += Y_pred

        with open("../data/test_prediction.txt", "w+") as f:
            for v in predictions:
                f.write("{}\n".format(v))

