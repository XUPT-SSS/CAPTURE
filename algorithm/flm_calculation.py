import torch
from torch.nn.functional import one_hot
from tqdm import tqdm
import numpy as np
def get_dist_matrix(fx, y):
    fx = one_hot(torch.argmax(fx, dim = -1), num_classes=fx.size(-1))
    dist_matrix = [fx[y==i].sum(0).unsqueeze(1) for i in range(len(y.unique()))]
    # dist_matrix = [fx[y==i].sum(0).unsqueeze(1) for i in range(1000)]
    dist_matrix = torch.cat(dist_matrix, dim=1)
    return dist_matrix

def predictive_distribution_based_multi_label_mapping(dist_matrix, mlm_num: int):
    assert mlm_num * dist_matrix.size(1) <= dist_matrix.size(0), "source label number not enough for mapping"
    mapping_matrix = torch.zeros_like(dist_matrix, dtype=int)
    dist_matrix_flat = dist_matrix.flatten() # same memory
    for _ in range(mlm_num * dist_matrix.size(1)):
        loc = dist_matrix_flat.argmax().item()
        loc = [loc // dist_matrix.size(1), loc % dist_matrix.size(1)]
        mapping_matrix[loc[0], loc[1]] = 1
        dist_matrix[loc[0]] = -1
        if mapping_matrix[:, loc[1]].sum() == mlm_num:
            dist_matrix[:, loc[1]] = -1
    return mapping_matrix


def generate_label_mapping_by_frequency(network, visual_prompt, data_loader, mapping_num = 1 ):
    device = next(visual_prompt.parameters()).device
    if hasattr(network, "eval"):
        network.eval()
    fx0s = []
    ys = []
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Frequency Label Mapping", ncols=100)

    for bidx, batch in enumerate(pbar):
        x = batch['type_ids'].to(device)
        y = batch['label'].to(device)
        with torch.no_grad():
            x, pert_norm = visual_prompt(x)
            fx0 = network(x)
        fx0s.append(fx0)
        ys.append(y)

    fx0s = torch.cat(fx0s).cpu().float()
    ys = torch.cat(ys).cpu().int()
    if ys.size(0) != fx0s.size(0):
        assert fx0s.size(0) % ys.size(0) == 0
        ys = ys.repeat(int(fx0s.size(0) / ys.size(0)))
    dist_matrix = get_dist_matrix(fx0s, ys)
    pairs = torch.nonzero(predictive_distribution_based_multi_label_mapping(dist_matrix, mapping_num))
    mapping_sequence = pairs[:, 0][torch.sort(pairs[:, 1]).indices.tolist()]
    return mapping_sequence


def bak_generate_label_mapping_by_frequency(network, visual_prompt,reprogrammer, data_loader, mapping_num = 1):
    device = next(visual_prompt.parameters()).device
    if hasattr(network, "eval"):
        network.eval()

    fx0s = []
    ys = []
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Frequency Label Mapping", ncols=100)
    for bidx, batch in enumerate(pbar):
        x = batch['type_ids'].to(device)
        y = batch['label'].to(device)
        programmed_img,pert_norm = reprogrammer(x)
        with torch.no_grad():
            x = visual_prompt(programmed_img)
            fx0 = network(x)
        fx0s.append(fx0)
        ys.append(y)

    fx0s = torch.cat(fx0s).cpu().float()
    ys = torch.cat(ys).cpu().int()
    if ys.size(0) != fx0s.size(0):
        assert fx0s.size(0) % ys.size(0) == 0
        ys = ys.repeat(int(fx0s.size(0) / ys.size(0)))
    dist_matrix = get_dist_matrix(fx0s, ys)
    pairs = torch.nonzero(predictive_distribution_based_multi_label_mapping(dist_matrix, mapping_num))
    mapping_sequence = pairs[:, 0][torch.sort(pairs[:, 1]).indices.tolist()]
    return mapping_sequence

# def bak_generate_label_mapping_by_frequency(vision_model, visual_prompt,reprogrammer, data_loader, mapping_num = 1):
#     device = next(visual_prompt.parameters()).device
#     if hasattr(vision_model, "eval"):
#         vision_model.eval()
#
#     lable_list = [0 for i in range(1000)]
#     pbar = tqdm(data_loader, total=len(data_loader), desc=f"Frequency Label Mapping", ncols=100)
#     for bidx, batch in enumerate(pbar):
#         sentence = batch['type_ids'].to(device)
#         # labels = batch['label'].to(device)#[选择预测概率最高的label]:修改为预测正确最多的label不可行，随机生成的图像没有实际label
#         with torch.no_grad():
#             programmed_img, pert_norm = reprogrammer(sentence)
#             logits = vision_model(programmed_img)
#             label_sort = torch.argsort(logits, descending=True)[:, 0]
#             for idx in label_sort:
#                 lable_list[idx] += 1
#     label_list = [i[0] for i in sorted(enumerate(lable_list), key=lambda x: x[1], reverse=True)]
#     return label_list



def _generate_label_mapping_by_frequency(vision_model, reprogrammer, data_loader, mapping_num = 1):
    device = next(reprogrammer.parameters()).device
    if hasattr(vision_model, "eval"):
        vision_model.eval()
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Frequency Label Mapping", ncols=100)
    lable_list = [0 for i in range(1000)]
    for bidx, batch in enumerate(pbar):
        sentence = batch['type_ids'].to(device)
        # labels = batch['label'].to(device)#[选择预测概率最高的label]:修改为预测正确最多的label不可行，随机生成的图像没有实际label
        with torch.no_grad():
            programmed_img, pert_norm = reprogrammer(sentence)
            logits = vision_model(programmed_img)
            label_sort = torch.argsort(logits,descending=True)[:,0]
            for idx in label_sort:
                lable_list[idx] += 1
    label_list = [i[0] for i in sorted(enumerate(lable_list), key=lambda x: x[1], reverse=True)]
    return label_list