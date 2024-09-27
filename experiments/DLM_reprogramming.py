import os
import datasets
from transformers import AutoTokenizer
import argparse
import torch.nn as nn
import torch
import timm
import torch.optim as optim
from tensorboardX import SummaryWriter
import os
import data_utils
import json
import numpy as np
import pprint
import reprogramming_model
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
# from algorithm.flm_calculation import generate_label_mapping_by_frequency
# from algorithm.flm_calculation import _generate_label_mapping_by_frequency
# from algorithm.label_mapping import label_mapping_base
from functools import partial
import random
import sys
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import logging
sys.path.append(r'/data/rqiu/multimodal_rerprogramming/algorithm')
from  flm_calculation import *


train_hps = {
    #default:100
    'num_epochs' : 20,
    'max_iterations' :200 , # overridden by args
    'lr': 0.001, # overridden by args
    'batch_size' : 16,
    'validate_every' : 200, # validates on small subset of val set
    'label_reduction' : 'mean' # overridden by args
}

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def unnormalize_image(tensor, mean, std):
    """
    tensor: Normalized image of shape (nc, h, w)
    """
    mean = torch.tensor(mean)[:,None,None].to(device)
    std = torch.tensor(std)[:,None,None].to(device)
    return tensor * std + mean

def normalize_image(tensor, mean, std):
    """
    tensor: Unnormalized image of shape (nc, h, w)
    """
    mean = torch.tensor(mean)[:,None,None].to(device)
    std = torch.tensor(std)[:,None,None].to(device)
    return (tensor - mean) / std

def get_mapped_logits(logits, class_mapping, multi_label_remapper):
    """
    logits : Tensor of shape (batch_size, 1000) # imagenet class logits
    class_mapping: class_mapping[i] = list of image net labels for text class i
    reduction : max or mean
    """
    if multi_label_remapper is None:
        #print("Here in old remapper")
        reduction = train_hps['label_reduction']
        mapped_logits = []
        for class_no in range(len(class_mapping)):
            if reduction == "max":
                class_logits, _ = torch.max(logits[:,class_mapping[class_no]], dim = 1) # batch size
            elif reduction == "mean":
                class_logits = torch.mean(logits[:,class_mapping[class_no]], dim = 1) # batch size
            else:
                raise NotImplentedException()

            mapped_logits.append(class_logits)
        return torch.stack(mapped_logits, dim = 1)
    else:
        orig_prob_scores = nn.Softmax(dim=-1)(logits)
        mapped_logits = multi_label_remapper(orig_prob_scores)
        return mapped_logits
#m_per_class: 10
def create_label_mapping(n_classes, m_per_class, image_net_labels = None):
    """
    n_classes: No. of classes in text dataset
    m_per_class: Number of imagenet labels to be mapped to each text class
    """
    if image_net_labels is None:
        image_net_labels = range(1000)

    class_mapping = [[] for i in range(n_classes)]

    idx = 0
    for _m in range(m_per_class):
        for _class_no in range(n_classes):
            class_mapping[_class_no].append(image_net_labels[idx])
            idx += 1
    return class_mapping

def get_imagenet_label_list(vision_model, base_image, img_size):
    if base_image is None:
        torch.manual_seed(random.randint(1,100))#default = 42
        base_image = 2 * torch.rand(3, img_size, img_size).to(device) - 1.0

    logits = vision_model(base_image[None])[0]
    label_sort = torch.argsort(logits)
    label_list = label_sort.detach().cpu().numpy().tolist()

    return label_list


def save_checkpoint(model, learning_rate,last_lr, acc, image_net_labels, class_mapping, filepath):
    print("Saving model state at {}".format(filepath))
    torch.save({'state_dict': model.state_dict(),
                'acc' : acc,
                'image_net_labels' : image_net_labels,
                'class_mapping' : class_mapping,
                'learning_rate': learning_rate,
               'last_lr':last_lr},filepath)

def load_checkpoint(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    iteration = checkpoint_dict['iteration']

    image_net_labels = None
    if 'image_net_labels' in checkpoint_dict:
        image_net_labels = checkpoint_dict['image_net_labels']

    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, iteration, image_net_labels


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #imdb
    # p.add_argument('--text_dataset', type=str, default='imdb')
    p.add_argument('--text_dataset', type=str, default='Vulnerability_Dataset')
    p.add_argument('--nums', type=int)
    p.add_argument('--logdir', type=str, default="./data2/paarth/ReprogrammingTransformers/ReprogrammingModels")
    p.add_argument('--cache_dir', type=str, default = "./data2/paarth/HuggingFaceDatasets")
    p.add_argument('--img_patch_size', type=int, default = 16)
    p.add_argument('--img_size', type=int, default = 384)
    p.add_argument('--vision_model', type=str, default = 'vit_base_patch16_384')
    # p.add_argument('--vision_model', type=str, default='resnet50')
    # p.add_argument('--vision_model', type=str, default='inception_v3')
    # p.add_argument('--vision_model', type=str, default='resnet18')
    # p.add_argument('--base_image_path', type=str, default = None)
    p.add_argument('--base_image_path', type=str, default=None)
    p.add_argument('--label_reduction', type=str, default = 'max')
    p.add_argument('--pert_alpha', type=float, default = 0.2)
    p.add_argument('--lr', type=float, default = 0.01) #default 0.001 0.01best
    p.add_argument('--resume_training', type=int, default = 0)
    p.add_argument('--resume_model_ckpt_path', type=str, default = None)
    p.add_argument('--m_per_class', type=int, default = 1)
    p.add_argument('--pretrained_vm', type=int, default = 1)
    p.add_argument('--max_validation_batches', type=int, default = 100)
    # p.add_argument('--max_iterations', type=int, default = 100000)
    p.add_argument('--max_iterations', type=int,default = 1000)
    p.add_argument('--exp_name_extension', type=str, default = "")
    p.add_argument('--reg_alpha', type=float, default = 1e-4)#defalut 1e-4
    p.add_argument('--n_training', type=int, default = None)
    p.add_argument('--use_char_tokenizer', type=int, default = 0)
    p.add_argument('--reduced_labels', type=int, default = None)
    p.add_argument('--batch_size', type=int, default = 16)#default 20
    p.add_argument('--use_sign_method', type=int, default = 0)
    p.add_argument('--mapping-interval', type=int, default=1)
    p.add_argument('--sample', type=int, default=100)
    args = p.parse_args()

    train_hps['lr'] = args.lr
    train_hps['batch_size'] = args.batch_size
    train_hps['label_reduction'] = args.label_reduction
    train_hps['max_iterations'] = args.max_iterations

    seednum=random.randint(1,100)
    samples=args.sample
    print(f"args.nums == {args.nums} seed={seednum} samples={samples}")
    logging.basicConfig(filename=f'./test/devign0.01_sample_{samples}_{args.vision_model}_{str(args.nums)}_seed_{seednum}log', level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
    logging.basicConfig(level=logging.INFO)
    dataset_configs = data_utils.text_dataset_configs

    assert args.text_dataset in dataset_configs
    text_dataset_config = dataset_configs[args.text_dataset]

    subset = text_dataset_config['subset']
    val_split = text_dataset_config['val_split']
    text_key = text_dataset_config['sentence_mapping']
    data_files = text_dataset_config['data_files']

    dataset_name = args.text_dataset if data_files is None else 'json'

    train_split = "train"
    if args.n_training is not None:
        train_split = "train[0:{}]".format(args.n_training)



    # if args.use_char_tokenizer == 1:
    #     tokenizer = data_utils.CharacterLevelTokenizer()
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')




    # import Tokenizer
    # from  transformers import PreTrainedTokenizerFast
    # tokenize = tokenizers.Tokenizer.from_pretrained("distilbert-base-uncased")

    # tokenize.enable_truncation(max_length=512)
    # tokenize.save("./data/config.json")
    # tokenize.save_model('./data')
    #
    # tokenizer = Tokenizer.from_file('./data/config.json')
    # tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

    # from transformers import BertTokenizerFast
    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

    # para: path:str
    # from Pretrain_tokenize import Tokenize
    # tokenizer = Tokenize('./data2/test.csv')

    from data_tokenize import Tokenize
    tokenizer = Tokenize()

    # train_dataset_raw = datasets.load_dataset(dataset_name, subset, data_files=data_files, split=train_split, cache_dir = args.cache_dir)
    # train_dataset = train_dataset_raw.map(lambda e: tokenizer(e[text_key], truncation=True, padding='max_length'), batched=True)
    # train_dataset = train_dataset.map(lambda e: data_utils.label_mapper(e, args.text_dataset), batched=True)
    # train_dataset.set_format(type='torch', columns=['input_ids', 'label'])

    #train_dataset_raw.add_column('name',val) https://www.likecs.com/ask-173775.html
    # train_dataset_raw = datasets.load_dataset(path='csv',data_files='../dataset/vul/train.csv',split='train')
    # train_dataset_raw = datasets.load_dataset(path='csv',data_files='../dataset/vuldeepecker/vuldeepecker_undersampling.csv',split='train')
    #train_dataset_raw = datasets.load_dataset(path='csv',data_files='../dataset/our/our_train.csv',split='train')
    train_dataset_raw = datasets.load_dataset(path='csv', data_files='../dataset/devign/devign_train.csv', split='train')
    # train_dataset_raw = datasets.load_dataset(path='csv', data_files='../dataset/reveal/reveal_train.csv', split='train')
    #train_dataset_raw = datasets.load_dataset(path='csv', data_files='../dataset/d2a/d2a_train.csv', split='train')

    # train_dataset_raw = train_dataset_raw.shuffle(seed=seednum).select(range(samples))
    # train_dataset_raw = datasets.load_dataset(path='csv', data_files='../dataset/reveal/reveal_train.csv', split='train')
    train_dataset = train_dataset_raw.map(lambda e: tokenizer.tokenize(e[text_key]), batched=False)
    # train_dataset = train_dataset_raw.map(lambda e: tokenizer.tokenize(e[text_key]))


    train_dataset = train_dataset.map(lambda e: data_utils.label_mapper(e, args.text_dataset), batched=False)
    # train_dataset.set_format(type='torch', columns=['type_ids','value_ids','label'])
    train_dataset.set_format(type='torch', columns=['type_ids','label'])



    # val_dataset_raw = datasets.load_dataset(dataset_name, subset, data_files=data_files, split=val_split, cache_dir = args.cache_dir)
    # val_dataset = val_dataset_raw.map(lambda e: tokenizer(e[text_key], truncation=True, padding='max_length'), batched=True)
    # val_dataset = val_dataset.map(lambda e: data_utils.label_mapper(e, args.text_dataset), batched=True)
    # val_dataset.set_format(type='torch', columns=['input_ids', 'label'])
    # val_dataset_raw =  datasets.load_dataset(path='csv',data_files='../dataset/vul/val.csv',split='train')
    #val_dataset_raw = datasets.load_dataset(path='csv', data_files='../dataset/our/our_val.csv', split='train')
    val_dataset_raw = datasets.load_dataset(path='csv', data_files='../dataset/devign/devign_val.csv', split='train')
    # val_dataset_raw =  datasets.load_dataset(path='csv',data_files='../dataset/vuldeepecker/val.csv',split='train')
    # val_dataset_raw =  datasets.load_dataset(path='csv',data_files='../dataset/d2a/d2a_dev.csv',split='train')
    # val_dataset_raw = datasets.load_dataset(path='csv', data_files='../dataset/reveal/reveal_val.csv',split='train')
    val_dataset = val_dataset_raw.map(lambda e: tokenizer.tokenize(e[text_key]), batched=False)
    val_dataset = val_dataset.map(lambda e: data_utils.label_mapper(e, args.text_dataset), batched=False)
    # val_dataset.set_format(type='torch', columns=['type_ids','value_ids','label'])
    val_dataset.set_format(type='torch', columns=['type_ids','label'])


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_hps['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=train_hps['batch_size'], shuffle=True)

    print("Pretrained VM", args.pretrained_vm==1)
    vision_model = timm.create_model(args.vision_model, pretrained=args.pretrained_vm==1)

    for parameter in vision_model.parameters():
        # https://discuss.pytorch.org/t/best-practice-for-freezing-layers/58156
        parameter.requires_grad = False # to avoid gradient accumulation.
    vision_model.eval()
    vision_model.to(device)
    print("Vision model Frozen!")
    vocab_size = tokenizer.get_vocab()

    total_param = sum(p.numel() for p in vision_model.parameters())
    trainable_param = sum(p.numel() for p in vision_model.parameters() if p.requires_grad)
    print("模型总共参数: %d/训练参数: %d" % (total_param, trainable_param))

    img_mean = data_utils.image_model_configs[args.vision_model]['mean']
    img_std = data_utils.image_model_configs[args.vision_model]['std']

    n_classes = text_dataset_config['num_labels']
    

    reprogrammer = reprogramming_model.ReprogrammingFuntion(vocab_size, args.img_patch_size,
        args.img_size,
        img_path = args.base_image_path, alpha = args.pert_alpha, img_mean = img_mean, img_std = img_std)
    reprogrammer.to(device)

    image_net_labels = get_imagenet_label_list(vision_model, reprogrammer.base_image, args.img_size)

    # image_net_labels = generate_label_mapping_by_frequency(vision_model, reprogrammer, train_loader,args.m_per_class)
    # print("Imagenet Label Ordering..", image_net_labels[:])


    loss_criterion = nn.CrossEntropyLoss()

    #https://github.com/yatengLG/Focal-Loss-Pytorch/issues/2
    # loss_criterion = focal_loss(gamma=2,alpha=[4,1],num_classes=2)
    base_image_name = None
    if args.base_image_path is not None:
        base_image_name = args.base_image_path.split("/")[-1].split(".")[0]
    exp_name = "ds_{}_lr_{}_bimg_{}_vm_{}_alpha_{}_m_label_{}_{}".format(
        args.text_dataset, train_hps['lr'], base_image_name, 
        args.vision_model, args.pert_alpha, args.m_per_class, args.label_reduction
    )
    exp_name = "{}_{}".format(exp_name, args.exp_name_extension)
    logdir = os.path.join(args.logdir, exp_name)
    ckptdir = os.path.join(logdir, "CKPTS")
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if not os.path.exists(ckptdir):
        os.makedirs(ckptdir)



    iter_no = 0
    best_acc = 0.0
    best_iter_no = 0
    best_model_path = None
    prev_best_eval_iter = None

    iter_cnt = 0
    if args.resume_training == 1:
        if args.resume_model_ckpt_path is None:
            resume_model_path = os.path.join(ckptdir, "model.p")
        else:
            resume_model_path = args.resume_model_ckpt_path
        print("Resuming from ckpt", resume_model_path)
        if not os.path.exists(resume_model_path):
            raise Exception("model not found")
        reprogrammer, iter_no, image_net_labels = load_checkpoint(resume_model_path, reprogrammer)

    multi_label_remapper = None
    num_imagenet_labels = 1000
    if args.reduced_labels is not None:
        num_imagenet_labels = args.reduced_labels

    # optimizer = optim.Adam(reprogrammer.parameters(), lr=train_hps['lr'])

    optimizer = torch.optim.Adam(reprogrammer.parameters(), lr=train_hps['lr'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=10)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    # if n_classes < num_imagenet_labels:
    #     class_mapping = create_label_mapping(n_classes, args.m_per_class, image_net_labels)
    #     print("Class Mapping")
    #     pprint.pprint(class_mapping)
    #     optimizer = optim.Adam(reprogrammer.parameters(), lr=train_hps['lr'])
    #     # optimizer = optim.SGD(reprogrammer.parameters(), lr=train_hps['lr'])
    #     print('mapping')
    # else:
    #     print("Using Multi Label Remapper!")
    #     multi_label_remapper = reprogramming_model.MultiLabelRemapper(num_imagenet_labels, n_classes)
    #     multi_label_remapper.to(device)
    #     params = list(reprogrammer.parameters()) + list(multi_label_remapper.parameters())
    #     optimizer = optim.Adam(params, lr=train_hps['lr'])
    for epoch in range(train_hps['num_epochs']):
        image_net_labels = generate_label_mapping_by_frequency(vision_model, reprogrammer, train_loader,args.m_per_class)
        class_mapping = create_label_mapping(n_classes, args.m_per_class, image_net_labels)
        print()
        print(class_mapping)
        # train
        reprogrammer.train()
        total_num = 0
        loss_sum = 0
        acc = 0

        tp_num = 0
        fp_num = 0
        tn_num = 0
        fn_num = 0
        all_labels = []
        all_prediction = []
        # pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epo {epoch} Training Lr {train_hps['lr']}",ncols=100)
        # pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epo {epoch} Training Lr {scheduler.get_last_lr()}",ncols=200)
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epo {epoch} Training Lr: {scheduler.get_last_lr()[0]:.5f}", ncols=200)
        for bidx, batch in enumerate(pbar):
            sentence = batch['type_ids'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            programmed_img, pert_norm = reprogrammer(sentence)
            logits = vision_model(programmed_img)

            #Double sequence, the effect is not good, [Var:VarValue]
            # type_seq = batch['type_ids'].to(device)
            # value_seq = batch['value_ids'].to(device)
            # programmed_img_type, pert_norm_type = reprogrammer(type_seq)
            # programmed_img_value, pert_norm_value = reprogrammer(value_seq)
            # pert_norm = max(pert_norm_type,pert_norm_value)
            # labels = batch['label'].to(device)
            # logits_type = vision_model(programmed_img_type)
            # logits_value= vision_model(programmed_img_value)
            # logits,_ = torch.max(torch.stack([logits_type,logits_value]),dim=0)

            # add softmax, commit this 7.2
            # mapped_logits = get_mapped_logits(logits, class_mapping, multi_label_remapper)
            # loss = loss_criterion(mapped_logits, labels)

            mapped_logits = get_mapped_logits(logits, class_mapping, multi_label_remapper)
            loss = loss_criterion(mapped_logits, labels)

            reg_loss = pert_norm
            loss_total = loss + args.reg_alpha * reg_loss
            loss_total.backward()
            optimizer.step()

            total_num += labels.size(0)
            loss_sum += loss_total.item() * mapped_logits.size(0)
            prediction = torch.argmax(mapped_logits, dim=1)

            tp = torch.logical_and(prediction == 1, labels == 1).sum().item()
            fp = torch.logical_and(prediction == 1, labels == 0).sum().item()
            tn = torch.logical_and(prediction == 0, labels == 0).sum().item()
            fn = torch.logical_and(prediction == 0, labels == 1).sum().item()
            tp_num += tp
            fp_num += fp
            tn_num += tn
            fn_num += fn
            acc = (tp_num + tn_num) / (tp_num + fp_num + tn_num + fn_num)
            if tp_num + fp_num == 0:
                Precision = 0
            else:
                Precision= tp_num / (tp_num + fp_num)
            if tp_num + fn_num == 0:
                Recall = 0
            else:
                Recall = tp_num / (tp_num + fn_num)
            if Precision + Recall == 0:
                F1 = 0
            else:
                F1 = 2 * Precision * Recall / (Precision + Recall)
            loss_train = loss_sum / total_num

            # all_labels += labels.data.cpu()
            # all_prediction += prediction.data.cpu()
            # acc = accuracy_score(all_labels,all_prediction)
            # F1 = f1_score(all_labels,all_prediction)
            # Recall = recall_score(all_labels,all_prediction)
            # Precision = precision_score(all_labels,all_prediction)


            # print('')
            # print('----------')
            # print(prediction)
            # print(labels)
            # print('***********')
            # print(torch.sum(prediction == labels))
            # print('----------')
            # print(torch.sum(all_labels == all_prediction))
            # correct = torch.sum(prediction == labels)
            # true_num += correct.item()
            # acc = 100 * true_num / total_num
            # pbar.set_postfix_str(f"Training Acc: {100 * acc:.2f}% F1Score:{100 * f1score:.2f}%  PrecisionScore: {100 * precisionscore:.2f}% RecallScore: {100 * recallscore:.2f}% ")
            pbar.set_postfix_str(f"Training Acc: {100 * acc:.2f}% F1Score:{100 * F1:.2f}%"
                                 f" PrecisionScore: {100 * Precision:.2f}% RecallScore: {100 * Recall:.2f}% loss:{loss_train:.2f}")
        #logging.info(f'Epo {epoch} Training Lr: {scheduler.get_last_lr()[0]:.5f} Loss={loss_train:.4f}, Training Acc: {100 * acc:.2f}% F1Score:{100 * F1:.2f}% PrecisionScore: {100 * Precision:.2f}% RecallScore: {100 * Recall:.2f}% loss:{loss_train:.2f}')
        print(f"tp_num = {tp_num} tn_num = {tn_num} fp_num = {fp_num} fn_num = {fn_num}")

        #val
        all_labels = []
        all_prediction = []
        reprogrammer.eval()
        total_num = 0
        loss_sum = 0

        tp_num = 0
        fp_num = 0
        tn_num = 0
        fn_num = 0

        pbar = tqdm(val_loader, total=len(val_loader), desc=f"Epo {epoch} Valing Lr: {scheduler.get_last_lr()[0]:.5f} ",ncols=200)
        for bidx, batch in enumerate(pbar):
            sentence = batch['type_ids'].to(device)
            labels = batch['label'].to(device)
            programmed_img, pert_norm = reprogrammer(sentence)
            logits = vision_model(programmed_img)



            # type_seq = batch['type_ids'].to(device)
            # value_seq = batch['value_ids'].to(device)
            # programmed_img_type, pert_norm_type = reprogrammer(type_seq)
            # programmed_img_value, pert_norm_value = reprogrammer(value_seq)
            # pert_norm = max(pert_norm_type,pert_norm_value)
            # labels = batch['label'].to(device)
            #
            # logits_type = vision_model(programmed_img_type)
            # logits_value= vision_model(programmed_img_value)
            # logits,_ = torch.max(torch.stack([logits_type,logits_value]),dim=0)

            mapped_logits = get_mapped_logits(logits, class_mapping, multi_label_remapper)
            loss = loss_criterion(mapped_logits, labels)

            reg_loss = pert_norm
            loss_total = loss + args.reg_alpha * reg_loss

            total_num += labels.size(0)
            loss_sum += loss_total.item() * mapped_logits.size(0)
            prediction = torch.argmax(mapped_logits, dim=1)

            tp = torch.logical_and(prediction == 1, labels == 1).sum().item()
            fp = torch.logical_and(prediction == 1, labels == 0).sum().item()
            tn = torch.logical_and(prediction == 0, labels == 0).sum().item()
            fn = torch.logical_and(prediction == 0, labels == 1).sum().item()

            tp_num += tp
            fp_num += fp
            tn_num += tn
            fn_num += fn
            if tp_num + fp_num == 0:
                Precision = 0
            else:
                Precision = tp_num / (tp_num + fp_num)
            if tp_num + fn_num == 0:
                Recall = 0
            else:
                Recall = tp_num / (tp_num + fn_num)
            if Precision + Recall == 0:
                F1 = 0
            else:
                F1 = 2 * Precision * Recall / (Precision + Recall)
            acc = (tp_num + tn_num) / (tp_num + fp_num + tn_num + fn_num)

            loss_val = loss_sum / total_num
            # all_labels += labels.data.cpu()
            # all_prediction += prediction.data.cpu()
            # acc = accuracy_score(all_labels,all_prediction)
            # F1 = f1_score(all_labels,all_prediction)
            # Recall = recall_score(all_labels,all_prediction)
            # Precision = precision_score(all_labels,all_prediction)
            pbar.set_postfix_str(f"Valing Acc: {100 * acc:.2f}% F1Score: {100 * F1:.2f}% PrecisionScore: {100 * Precision:.2f}% RecallScore:{100 * Recall:.2f}% loss_val:{loss_val:.2f}")
        print(f"tp_num = {tp_num} tn_num = {tn_num} fp_num = {fp_num} fn_num = {fn_num}")
        logging.info(f'Epo {epoch} Valing Lr: {scheduler.get_last_lr()[0]:.5f} Loss={loss_val:.4f}, Valing Acc: {100 * acc:.2f}% F1Score:{100 * F1:.2f}% PrecisionScore: {100 * Precision:.2f}% RecallScore: {100 * Recall:.2f}% loss:{loss_train:.2f}')
        model_path = os.path.join(ckptdir, "model.p")
        save_checkpoint(reprogrammer, train_hps['lr'],scheduler.get_last_lr()[0], acc, image_net_labels, class_mapping, model_path)
        if acc >= best_acc:
            best_model_path = os.path.join(ckptdir, "model_best.p")
            save_checkpoint(reprogrammer, train_hps['lr'],scheduler.get_last_lr()[0], acc, image_net_labels, class_mapping, best_model_path)
            best_acc = acc
            logging.info(
                f'Best Acc at Epo {epoch} Valing Lr: {scheduler.get_last_lr()[0]:.5f} Loss={loss_val:.4f}, Valing Acc: {100 * acc:.2f}% F1Score:{100 * F1:.2f}% PrecisionScore: {100 * Precision:.2f}% RecallScore: {100 * Recall:.2f}% loss:{loss_train:.2f}')
        #Adjust learning rate
        scheduler.step()







if __name__ == '__main__':

    main()
