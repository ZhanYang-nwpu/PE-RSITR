import os
import nltk
nltk.download('punkt')

import random
import torch
import argparse
import tensorboard_logger as tb_logger
import logging
from datetime import datetime
import torch.backends.cudnn as cudnn
import numpy as np

import data
import engine
import utils
from layers import PE_RSITR as models
from vocab import deserialize_vocab
import loss as loss_test
import clip

def parser_options():
    # Hyper Parameters setting
    parser = argparse.ArgumentParser('Parameter-Efficient Remote Sensing Image-Text Retrieval framework', add_help=False)
    # dataset
    parser.add_argument('--CLIP', default='ViT-B-32.pt', type=str, help='{ViT-B-32.pt, ViT-B-16.pt}')
    parser.add_argument('--datatype', default='rsitmd', type=str, help='{rsicd, rsitmd, sydney, ucm}')
    parser.add_argument('--data_path', default='/data/', type=str, help='path to datasets')
    parser.add_argument('--image_path', default='/data/', type=str, help='')
    parser.add_argument("--vocab_path", default='vocab/', type=str,help="The vocab path.")
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--batch_size_val', default=64, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--workers', default=0, type=int,
                        help='Number of data loader workers.')

    # optim
    parser.add_argument('--epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--lr', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_decay_param', default=0.7, type=float, help='')
    parser.add_argument('--lr_update_epoch', default=20, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--grad_clip', default=0., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--max_violation', default=True, action='store_true',
                        help='Use max instead of sum in the rank loss.')
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

    return opt

def main(opt):
    # make ckpt save dir
    if not os.path.exists(opt.ckpt_save_path):
        os.makedirs(opt.ckpt_save_path)

    # make vocab
    vocab = deserialize_vocab(opt.vocab_path)

    # load CLIP
    CLIPmodel, preprocess = clip.load("/layers/" + opt.CLIP)
    opt.preprocess = preprocess
    del CLIPmodel

    # Create dataset, model, criterion and optimizer
    train_loader, val_loader = data.get_loaders(vocab, opt)
   
    model = models.factory(opt,
                           cuda=True, 
                           data_parallel=False)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=opt.lr)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
         
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
   
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            rsum, all_scores =  engine.validate(opt,val_loader, model)
            print(all_scores)
        else:
            start_epoch = 0
            print("=> no checkpoint found at '{}'".format(opt.resume))
    else:
        start_epoch = 0

    # Train the Model
    best_rsum = 0
    best_score = ""

    # criterion
    Cross_Modal_loss = loss_test.ContrastiveLoss(opt=opt, margin=opt.margin, max_violation=opt.max_violation)
    Intra_Modal_loss = loss_test.ContrastiveLoss(opt=opt, margin=0.2, max_violation=opt.max_violation)

    for epoch in range(start_epoch, opt.epochs):
        utils.adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        engine.train(train_loader, model, optimizer, epoch, Cross_Modal_loss, Intra_Modal_loss, opt=opt)

        # evaluate on validation set
        if epoch % opt.eval_step == 0:
            rsum, all_scores = engine.validate(opt, val_loader, model)

            is_best = rsum > best_rsum
            if is_best:
                best_score = all_scores
            best_rsum = max(rsum, best_rsum)

            # save ckpt
            utils.save_checkpoint(
                {
                'epoch': epoch + 1,
                'arch': 'baseline',
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'options': opt,
                'Eiters': model.Eiters,
            },
                is_best,
                filename='checkpoint_{}.pth.tar'.format(epoch),
                prefix=opt.ckpt_save_path
            )

            print("Now  score:")
            print(all_scores)
            print("Best score:")
            print(best_score)

            utils.log_to_txt(
                contexts= "Epoch:{} ".format(epoch+1) + all_scores,
                filename=opt.logger_name + opt.datatype+ "_result.txt"
            )

            utils.log_to_txt(
                contexts= "Best:   " + best_score,
                filename=opt.logger_name+ opt.datatype +"_result.txt"
            )

def generate_random_samples(options):
    # load all anns
    caps = utils.load_from_txt(options.data_path+'train_caps.txt')
    fnames = utils.load_from_txt(options.data_path+'train_filename.txt')

    # merge
    assert len(caps) // 5 == len(fnames)
    all_infos = []
    for img_id in range(len(fnames)):
        cap_id = [img_id * 5 ,(img_id+1) * 5]
        all_infos.append([caps[cap_id[0]:cap_id[1]], fnames[img_id]])

    # shuffle
    random.shuffle(all_infos)

    # split_trainval
    percent = 0.8
    train_infos = all_infos[:int(len(all_infos)*percent)]
    val_infos = all_infos[int(len(all_infos)*percent):]

    # save to txt
    train_caps = []
    train_fnames = []
    for item in train_infos:
        for cap in item[0]:
            train_caps.append(cap)
        train_fnames.append(item[1])
    utils.log_to_txt(train_caps, options.data_path+'train_caps_verify.txt',mode='w')
    utils.log_to_txt(train_fnames, options.data_path+'train_filename_verify.txt',mode='w')

    val_caps = []
    val_fnames = []
    for item in val_infos:
        for cap in item[0]:
            val_caps.append(cap)
            val_fnames.append(item[1])
    utils.log_to_txt(val_caps, options.data_path+'val_caps_verify.txt',mode='w')
    utils.log_to_txt(val_fnames, options.data_path+'val_filename_verify.txt',mode='w')

    print("Generate random samples to {} complete.".format(options.data_path))


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
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    opt.logger_name = opt.logger_name + opt.datatype + TIMESTAMP
    opt.ckpt_save_path = opt.ckpt_save_path + opt.datatype +TIMESTAMP
    tb_logger.configure(opt.logger_name, flush_secs=5)

    f = open(opt.logger_name + opt.datatype+ "_train.txt", 'w')
    f.write(opt.__str__())
    f.close()
    f = open(opt.logger_name + opt.datatype + "_validate.txt", 'w')
    f.close()
    f = open(opt.logger_name + opt.datatype + "_result.txt", 'w')
    f.close()
    print("=========================================")

    # generate random train and val samples
    generate_random_samples(opt)

    # run experiment
    main(opt)
