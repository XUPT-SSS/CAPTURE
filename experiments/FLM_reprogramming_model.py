import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Normalize
import numpy as np
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
class MultiLabelRemapper(nn.Module):
    def __init__(self, num_labels_original,num_labels_target):
        super(MultiLabelRemapper, self).__init__()
        self.linear_layer = nn.Linear(num_labels_original, num_labels_target)

    def forward(self, x):
        logits = self.linear_layer(x)
        return logits

class ReprogrammingFuntion(nn.Module):
    def __init__(self, vocab_size, img_patch_size = 16, img_size = 384, 
        img_path=None, alpha=0.01,
        img_mean=(0.5, 0.5, 0.5), img_std=(0.5, 0.5, 0.5)):
        super(ReprogrammingFuntion, self).__init__()

        assert img_size % img_patch_size == 0
        
        self.img_patch_size = img_patch_size
        self.img_size = img_size
        self.token_embedding = nn.Embedding(vocab_size, img_patch_size * img_patch_size * 3)
        self.num_patches_row = int(img_size/img_patch_size)
        self.num_patches = self.num_patches_row * self.num_patches_row
        self.base_image = None
        if img_path is not None:
            image = Image.open(img_path)
            transform=transforms.Compose([
                                  Resize((img_size, img_size)),
                                  ToTensor(),
                                Normalize(img_mean,img_std),
                                ])

            image = transform(image) 
            self.base_image = torch.tensor(image, requires_grad=False).to(device)
            self.alpha = alpha

        self.image_mean_tensor = torch.tensor(img_mean)[None,:,None,None].to(device)
        self.image_std_tensor = torch.tensor(img_std)[None,:,None,None].to(device)

    def unnormalize_image(self, x):
        return x * self.image_std_tensor + self.image_mean_tensor

    def normalize_image(self, x):
        return (x - self.image_mean_tensor) / self.image_std_tensor
        
    def forward(self, sentence_batch):
        sentence_embedding = torch.tanh(self.token_embedding(sentence_batch)) # (N, l, 16*16*3)
        _N, _L, _ = sentence_embedding.size()
        sentence_embedding = sentence_embedding.view(_N, _L, 3, self.img_patch_size, self.img_patch_size)

        reprogrammed_image = torch.zeros(_N, 3, self.img_size, self.img_size).to(device)
        
        for patch_idx in range(self.num_patches):
            i_start = int(patch_idx / self.num_patches_row) * self.img_patch_size
            j_start = (patch_idx % self.num_patches_row) * self.img_patch_size
            i_end = i_start + self.img_patch_size
            j_end = j_start + self.img_patch_size
            if patch_idx < _L:#[::,]first
                reprogrammed_image[:,:,i_start:i_end,j_start:j_end] = sentence_embedding[:,patch_idx]
            else:
                # adding the padding embedding all the way till the end
                reprogrammed_image[:,:,i_start:i_end,j_start:j_end] = sentence_embedding[:,_L-1]

        # normalizing by batch size
        pert_norm = torch.norm(reprogrammed_image, p=2)/_N
        
        if self.base_image is not None:
            #base_image_batch = self.base_image[None].repeat((_N, 1, 1, 1))  1
            base_image_batch = self.base_image
            reprogrammed_image = base_image_batch + self.alpha * reprogrammed_image
        else:
            pass
            #
            #print("test base_image is None------------------------")
        unnormalized_image = self.unnormalize_image(reprogrammed_image)
        unnormalized_image_clipped = torch.clamp(unnormalized_image, 0.0, 1.0)
        reprogrammed_image_clipped = self.normalize_image(unnormalized_image_clipped)
        
        return reprogrammed_image_clipped, pert_norm


##########################################################################################################
from transformers import BartTokenizer, BartModel
import torch
import numpy as np
# 加载BART模型和tokenizer
# model_name = "facebook/bart-base"
# tokenizer = BartTokenizer.from_pretrained(model_name)
# model = BartModel.from_pretrained(model_name)

class ReprogrammingCommit(nn.Module):
    def __init__(self, img_patch_size=16, img_size=384,
                 img_path=None, alpha=0.01,
                 img_mean=(0.5, 0.5, 0.5), img_std=(0.5, 0.5, 0.5)):
        super(ReprogrammingCommit, self).__init__()

        assert img_size % img_patch_size == 0

        self.model_name = "facebook/bart-base"
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartModel.from_pretrained(self.model_name)

        self.img_patch_size = img_patch_size
        self.img_size = img_size
        self.num_patches_row = int(img_size / img_patch_size)
        self.num_patches = self.num_patches_row * self.num_patches_row
        self.base_image = None
        if img_path is not None:
            image = Image.open(img_path)
            transform = transforms.Compose([
                Resize((img_size, img_size)),
                ToTensor(),
                Normalize(img_mean, img_std),
            ])

            image = transform(image)
            self.base_image = torch.tensor(image, requires_grad=False).to(device)
            self.alpha = alpha

        self.image_mean_tensor = torch.tensor(img_mean)[None, :, None, None].to(device)
        self.image_std_tensor = torch.tensor(img_std)[None, :, None, None].to(device)
        self.model.to(device)

    def unnormalize_image(self, x):
        return x * self.image_std_tensor + self.image_mean_tensor

    def normalize_image(self, x):
        return (x - self.image_mean_tensor) / self.image_std_tensor

    #commit_message
    def forward(self, input_ids):
        #input_ids = self.tokenizer.encode(commit_message, max_length=576, truncation=True, padding='max_length')
        sentence_embedding = self.model(input_ids)
        #sentence_embedding = torch.tanh(self.token_embedding(sentence_batch))  # (N, l, 16*16*3)
        last_hidden_state = torch.tanh(sentence_embedding.last_hidden_state)
        _N, _L, _ = last_hidden_state.size()
        #sentence_embedding = sentence_embedding.view(_N, _L, 3, self.img_patch_size, self.img_patch_size)
        sentence_embedding = last_hidden_state.view(_N, _L, 3, self.img_patch_size, self.img_patch_size)
        reprogrammed_image = torch.zeros(_N, 3, self.img_size, self.img_size).to(device)

        for patch_idx in range(self.num_patches):
            i_start = int(patch_idx / self.num_patches_row) * self.img_patch_size
            j_start = (patch_idx % self.num_patches_row) * self.img_patch_size
            i_end = i_start + self.img_patch_size
            j_end = j_start + self.img_patch_size
            if patch_idx < _L:  # [::,]first
                reprogrammed_image[:, :, i_start:i_end, j_start:j_end] = sentence_embedding[:, patch_idx]
            else:
                # adding the padding embedding all the way till the end
                reprogrammed_image[:, :, i_start:i_end, j_start:j_end] = sentence_embedding[:, _L - 1]

        # normalizing by batch size
        pert_norm = torch.norm(reprogrammed_image, p=2) / _N

        if self.base_image is not None:
            # base_image_batch = self.base_image[None].repeat((_N, 1, 1, 1))  1
            base_image_batch = self.base_image
            reprogrammed_image = base_image_batch + self.alpha * reprogrammed_image
        else:
            pass
            #
            # print("test base_image is None------------------------")
        unnormalized_image = self.unnormalize_image(reprogrammed_image)
        unnormalized_image_clipped = torch.clamp(unnormalized_image, 0.0, 1.0)
        reprogrammed_image_clipped = self.normalize_image(unnormalized_image_clipped)

        return reprogrammed_image_clipped, pert_norm