import time
import numpy as np
import numpy
import torch
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
import logging
from scipy.spatial.distance import cdist
import loss as loss_test
import utils


def train(train_loader, model, optimizer, epoch, cross_loss, intra_loss,opt={}):
    # extract value
    grad_clip = opt.grad_clip
    print_freq = opt.print_freq

    # switch to train mode
    model.train()
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    train_logger = utils.LogCollector()

    end = time.time()
    params = list(model.parameters())
    for i, train_data in enumerate(train_loader):
        images, captions, lengths, caption_out, ids, word_id, word_mask, labelID= train_data

        # measure data loading time
        data_time.update(time.time() - end)
        model.logger = train_logger

        input_visual = Variable(images)
        input_text = Variable(captions)
        if torch.cuda.is_available():
            input_visual = input_visual.cuda()
            input_text = input_text.cuda()

        visual_emb, text_emb, visual_augme, text_augme = model(input_visual, input_text)

        torch.cuda.synchronize()
        # compute loss
        loss = 0.
        crossmodal_loss = cross_loss(visual_emb,text_emb)
        loss += crossmodal_loss
        intra_v_loss = intra_loss(visual_emb, visual_augme)
        intra_t_loss = intra_loss(text_emb, text_augme)
        loss += intra_v_loss
        loss += intra_t_loss

        if grad_clip > 0:
            clip_grad_norm(params, grad_clip)

        train_logger.update('Cross-Modal_loss', crossmodal_loss.cpu().data.numpy())
        train_logger.update('Intra-v_loss', intra_v_loss.cpu().data.numpy())
        train_logger.update('Intra-t_loss', intra_t_loss.cpu().data.numpy())

        optimizer.zero_grad()
        loss.backward()
        torch.cuda.synchronize()
        optimizer.step()
        torch.cuda.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
         
          logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f}\t'
                '{elog}\t'
                .format(epoch, i, len(train_loader),
                        batch_time=batch_time,
                        elog=str(train_logger)))

          utils.log_to_txt(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f}\t'
                '{elog}\t'
                    .format(epoch, i, len(train_loader),
                            batch_time=batch_time,
                            elog=str(train_logger)),
                opt.logger_name+  opt.datatype +"_train.txt"
            )

def encode_data(model, data_loader, opt, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = utils.AverageMeter()
    val_logger = utils.LogCollector()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    trip_loss = loss_test.ContrastiveLoss(opt=opt, margin=opt.margin, max_violation=opt.max_violation)

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None
    cap_lens = None
    attns = None
    # make sure val logger is used
    model.logger = val_logger
    for i, batch_data in enumerate(data_loader):
        images, captions, lengths, caption_out, ids, word_id, word_mask ,labelID= batch_data

        input_visual = Variable(images)
        input_text = Variable(captions)
        if torch.cuda.is_available():
            input_visual = input_visual.cuda()
            input_text = input_text.cuda()

        # compute the embeddings
        img_emb, cap_emb, l_img_emb, l_cap_emb = model(input_visual, input_text)

        if img_embs is None:
            if img_emb.dim() == 3:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
            else:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))
            cap_lens = [0] * len(data_loader.dataset)
        # cache embeddings
        ids = np.array(batch_data[-4])
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()

        # measure accuracy and record loss
        loss = 0.
        Trip_loss = trip_loss(img_emb, cap_emb)
        loss += Trip_loss

        val_logger.update('Trip_loss', Trip_loss.cpu().data.numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger)))

            utils.log_to_txt(
                'Test: [{0}/{1}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                .format(
                    i, len(data_loader), batch_time=batch_time,
                    e_log=str(model.logger)),
                filename=  opt.logger_name + opt.datatype + "_validate.txt"
            )

    return img_embs, cap_embs, cap_lens, attns

def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs, cap_lens, attns= encode_data(model, val_loader, opt, opt.print_freq, logging.info)

    img_embs = numpy.array([img_embs[i] for i in range(0, len(img_embs), 5)])

    start = time.time()
    sims = 1-cdist(img_embs, cap_embs, metric='cosine')
    end = time.time()
    print("calculate similarity time:", end-start)

    # caption retrieval
    (r1i, r5i, r10i, medri, meanri), (ranks, top1, top2, top3, top4, top5) = utils.i2t(img_embs, cap_embs, cap_lens, sims)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))
    # image retrieval
    (r1t, r5t, r10t, medrt, meanrt), (ranks, top1, top2, top3, top4, top5) = utils.t2i(
        img_embs, cap_embs, cap_lens, sims)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1t, r5t, r10t, medrt, meanrt))
    # sum of recalls to be used for early stopping
    currscore = (r1t + r5t + r10t + r1i + r5i + r10i) / 6.0

    all_score = "r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1t:{} r5t:{} r10t:{} medrt:{} meanrt:{}\n sum:{}\n ------\n".format(
        r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore
    )

    return currscore, all_score

