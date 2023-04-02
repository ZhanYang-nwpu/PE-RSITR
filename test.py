import os

import nltk
nltk.download('punkt')
import torch
import argparse
import tensorboard_logger as tb_logger
import logging
import random
import torch.backends.cudnn as cudnn
import numpy as np

import data
import engine
import utils
from layers import PE_RSITR as models
from vocab import deserialize_vocab
import clip

def parser_options():
    # Hyper Parameters setting
    parser = argparse.ArgumentParser('Transformer-based RSCTIR framework', add_help=False)
    # dataset
    parser.add_argument('--CLIP', default='ViT-B-32.pt', type=str, help='{ViT-B-32.pt, ViT-B-16.pt}')
    parser.add_argument('--datatype', default='rsitmd', type=str, help='{rsicd, rsitmd, sydney, ucm}')
    parser.add_argument('--data_path', default='/data/', type=str, help='path to datasets')
    parser.add_argument('--image_path', default='/data/', type=str, help='')
    parser.add_argument("--vocab_path", default='vocab/', type=str,help="The vocab path.")
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--batch_size_val', default=64, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--workers', default=0, type=int,
                        help='Number of data loader workers.')

    parser.add_argument('--resume', default="", type=str, help='path to best checkpoint (default: none)')

    # logs:
    parser.add_argument('--eval_step', default=1, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--print_freq', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--logger_name', default='logs/',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--ckpt_save_path', default='checkpoint/',
                        help='Path to save the model.')
    parser.add_argument('--dim_embed', type=int, default=512, metavar='N',
                        help='how many dimensions in embedding (default: 512)')

    opt = parser.parse_args()

    opt.data_path = opt.data_path + opt.datatype + '_precomp/'
    opt.image_path = opt.image_path + opt.datatype + '_precomp/' + opt.datatype + '_images/'
    opt.vocab_path = opt.vocab_path + opt.datatype + '_splits_vocab.json'

    opt.timestep = '2023-01-01T59-59-59'
    return opt


def main(opt):
    # make vocab
    vocab = deserialize_vocab(opt.vocab_path)

    # load CLIP
    CLIPmodel, preprocess = clip.load("/layers/" + opt.CLIP)
    opt.preprocess = preprocess
    opt.CLIPmodel = CLIPmodel

    # Create dataset, model, criterion and optimizer
    test_loader = data.get_test_loader(vocab, opt)
    model = models.factory(opt,
                           cuda=True,
                           data_parallel=False)

    # optionally resume from a checkpoint
    opt.resume = opt.ckpt_save_path + opt.datatype + opt.timestep + '/model_best.pth.tar'
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])

            # Eiters is used to show logs as the continuation of another
            # train
            model.Eiters = checkpoint['Eiters']

            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            rsum, all_scores = engine.validate(opt, test_loader, model)
            print(all_scores)
            utils.log_to_txt(
                contexts="Test: " + all_scores,
                filename=opt.logger_name + opt.datatype + "_validate.txt"
            )
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))


if __name__ == '__main__':
    opt = parser_options()

    # fix random seed
    seed = 2
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)

    # make logger
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    opt.logger_name = opt.logger_name + opt.datatype + opt.timestep +'/'
    tb_logger.configure(opt.logger_name, flush_secs=5)
    print("=========================================")

    # run experiment
    main(opt)
