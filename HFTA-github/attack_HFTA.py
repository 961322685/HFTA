from PIL import Image
import cv2
import time
import copy
import os
import torch
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable as V
from torchvision import datasets, transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import argparse
import sys
import glob
from torch.utils.data import Dataset
from torchvision.utils import save_image
import tqdm
from backbone_net import iresnet
from backbone_net import iresnet_HFTA
from backbone_net import model_se
from backbone_net import model_se_HFTA
#from facex_net.backbone_def import BackboneFactory
#from util.model_loader import ModelLoader

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/lfw_1000', help='Input directory with images.')
parser.add_argument('--save_path', type=str, default='manifold/feature_noise/', help='Save images.')
parser.add_argument("--max_epsilon", type=float, default=10.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter", type=int, default=1000, help="Number of iterations.")
parser.add_argument("--image_resize", type=int, default=112, help="Height of each input images.")
parser.add_argument("--prob", type=float, default=0.7, help="probability of using diverse inputs.")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
parser.add_argument("--alpha", type=float, default=8 / 255, help="alpha")

args = parser.parse_args()

torch.backends.cudnn.benchmark = True

mean = [0.5] * 3
std = [0.5] * 3
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


def project_kern(kern_size):
    kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    kern = kern.astype(np.float32)
    stack_kern = np.stack([kern, kern, kern])
    stack_kern = np.expand_dims(stack_kern, 1)
    stack_kern = torch.tensor(stack_kern).cuda()
    return stack_kern, kern_size // 2


def project_noise(x, stack_kern, padding_size):
    # x = tf.pad(x, [[0,0],[kern_size,kern_size],[kern_size,kern_size],[0,0]], "CONSTANT")
    x = F.conv2d(x, stack_kern, padding=(padding_size, padding_size), groups=3)
    return x


stack_kern, padding_size = project_kern(3)


def input_diversity(input_tensor):
    """Input diversity: https://arxiv.org/abs/1803.06978"""
    rnd = torch.randint(100, args.image_resize, ())
    rescaled = F.interpolate(input_tensor, size=[rnd, rnd], mode='bilinear', align_corners=True)
    h_rem = args.image_resize - rnd
    w_rem = args.image_resize - rnd
    pad_top = torch.randint(0, h_rem, ())
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem, ())
    pad_right = w_rem - pad_left
    pad_list = (pad_left, pad_right, pad_top, pad_bottom)
    padded = nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0.)(rescaled)
    padded = nn.functional.interpolate(padded, [args.image_resize, args.image_resize])
    return padded if torch.rand(()) < args.prob else input_tensor


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def graph(x, models, x_min, x_max):
    num_iter = args.num_iter
    alpha = args.alpha
    momentum = args.momentum

    x_ori = x.clone()
    grad = torch.zeros_like(x)

    # grad = grad -1
    # x = x_ori + grad

    x = x_ori + (grad - 1)

    # cos = nn.CosineSimilarity(dim=1, eps=1e-8)
    x.requires_grad = True
    for i in range(num_iter):
        zero_gradients(x)

        #adv = input_diversity(x)

        # loss = 1 - cos(models[1](x_ori), models[1](adv))
        # print(x_ori)
        # print(x)
        # print(models[1](x_ori))
        # print(models[1](x))
        loss = torch.dist(models[1](x_ori), models[1](x), p=2)
        # print(loss)
        loss.backward()
        noise = x.grad.data
        #print(noise)
        #sys.exit()
        if i % 100 == 9:
            print(loss)
            # print(noise)

        # TI-FGSM
        #noise = F.conv2d(noise, stack_kern, padding=(padding_size, padding_size), groups=3)

        # MI-FGSM
        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        noise = momentum * grad + noise
        grad = noise

        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, x_min, x_max)
        x = V(x, requires_grad=True)
    #sys.exit()
    return x.detach()


def compare(x, emb, basename):
    dist_list = []
    for y in emb:
        y = y.unsqueeze(0)
        dist = torch.cosine_similarity(x, y)
        dist_list.append(dist.data)
    maxd = max(dist_list)
    max_index = dist_list.index(maxd)
    name = basename[max_index]
    return maxd, name


class ArcFaceDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.transformer = preprocess

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img = Image.open(sample)
        img = preprocess(img)
        return img, 1

    def __len__(self):
        return len(self.samples)

'''
def load_model(model_type, model_path):
    backbone_factory = BackboneFactory(model_type, 'util/backbone_conf.yaml')
    model_loader = ModelLoader(backbone_factory)
    model = model_loader.load_model(model_path)
    model.eval()
    model.cuda()
    return model
'''

def main():
    models = []

    model = iresnet.iresnet34(pretrained=False)
    model.load_state_dict(torch.load('backbone/iresnet34-5b0d0e90.pth'))# iresnet34-5b0d0e90   iresnet50-7f187506 iresnet100-73e07ba7
    model.eval()
    model.cuda()
    models.append(model)

    model = iresnet_HFTA.iresnet34(pretrained=False)
    model.load_state_dict(torch.load('backbone/iresnet34-5b0d0e90.pth'))
    model.eval()
    model.cuda()
    models.append(model)

    '''
    model = model_se.resnet101(pretrained=False)
    model.load_state_dict(torch.load('backbone/SE-LResNet101E-IR.pt'))
    model.eval()
    model.cuda()
    models.append(model)

    model = model_se_HFTA.resnet101(pretrained=False)
    model.load_state_dict(torch.load('backbone/SE-LResNet101E-IR.pt'))
    model.eval()
    model.cuda()
    models.append(model)
    '''
    '''
    model = load_model('MobileFaceNet', 'facex_backbone/MobileFaceNet.pt')
    models.append(model)
    model = load_model('MobileFaceNet_n', 'facex_backbone/MobileFaceNet.pt')
    models.append(model)
    '''

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)


    paths = glob.glob(args.data_path + '/*/*.png')
    paths.sort(reverse=False)
    paths_basename = np.array([os.path.basename(os.path.dirname(x)) for x in paths])
    # print(paths_basename)
    dataset = ArcFaceDataset(paths)
    # image_datasets = datasets.ImageFolder(args.data_path, preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    emb = torch.load("emb/lfw1000_emb_ir50.pth")
    # emb = torch.load("emb/lfw1000_emb_ir100.pth")
    # emb = torch.load("emb/lfw1000_emb_ir101se.pth")
    #emb = torch.load("emb/lfw1000_emb_mobile.pth")
    i = 0
    suc = 0
    outtxt = []
    # sys.exit()
    for images, _ in dataloader:
        print('i:', i)
        t0 = time.time()
        images = images.cuda()

        images_min = clip_by_tensor(images - args.max_epsilon * 2 / 255.0, -1.0, 1.0)
        images_max = clip_by_tensor(images + args.max_epsilon * 2 / 255.0, -1.0, 1.0)
        #print("1:{}".format(torch.cuda.memory_allocated(0)))
        adv_img = graph(images, models, images_min, images_max)
        #print("2:{}".format(torch.cuda.memory_allocated(0)))
        # print(adv_img)
        #sys.exit()
        adv_feat = models[0](adv_img)
        x_feat = models[0](images)
        cos = torch.cosine_similarity(adv_feat, x_feat)
        print(cos)

        # dist, name = compare(adv_feat, emb, paths_basename)
        # print(dist, name, paths_basename[i])
        #
        # if name != paths_basename[i]:
        #     suc += 1
        # print('suc:', suc)
        '''
        # sys.exit()
        data = str(np.array(cos.data.cpu())[0]) + ' , id:' + paths_basename[i] + ' , adv:' + name
        outtxt.append(data)
        # print(outtxt)
        '''
        # sys.exit()
        outimg = (adv_img + 1) / 2
        # outimg = outimg.clamp(0.0, 1.0).squeeze()
        outimg = outimg.clamp(0.0, 1.0)
        # print(outimg.shape)
        for saveimg in outimg:
            # print(saveimg.shape)
            outpath = args.save_path + os.path.basename(paths[i])

            save_image(saveimg, outpath)
            i += 1
        t1 = time.time()
        print('time:', t1 - t0)
        sys.exit()
    '''
    outtxt.append(str(suc))
    with open('out_1000/mobile/tdmi_test_n.txt', 'w', encoding='utf-8') as f:
        for data in outtxt:
            f.write(data + '\n')
        f.close()
    '''

if __name__ == '__main__':
    main()
