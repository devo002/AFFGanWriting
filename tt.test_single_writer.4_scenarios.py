import cv2
import Levenshtein as Lev
import random
import numpy as np
import torch
from network_tro import ConTranModel
from load_data import IMG_HEIGHT, IMG_WIDTH, NUM_WRITERS, letter2index, tokens, num_tokens, OUTPUT_MAX_LEN, index2letter
from modules_tro import normalize
import os
import time
from tqdm import tqdm, trange
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

'''Take turns to open the comments below to run 4 scenario experiments'''

folder_wids = '/home/woody/iwi5/iwi5333h/data'
# img_base = '/home/WeiHongxi/WangHeng/project/dataset/Iam_database/words/'
img_base = '/home/woody/iwi5/iwi5333h/data'
folder_pre = '/home/vault/iwi5/iwi5333h/test_single_writer.190_scenarios/3000'
# folder_pre = 'test_single_writer.4_scenarios_average/'
#epoch = 5000
epoch = 3000


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--epoch', default=epoch, type=int,
                    help=('Path to the generated images or '
                          'to .npz statistic files'))
#

#

#


'''data preparation'''

def pre_data(data_dict,target_file):

    with open(target_file, 'r') as _f:
        data = _f.readlines()
        lables= [i.split(' ')[1].replace('\n', '').replace('\r', '') for i in data]
        data = [i.split(' ')[0] for i in data]
        wids = [i.split(',')[0] for i in data]
        imgnames = [i.split(',')[1] for i in data]

    for wid, imgname, lable in zip(wids ,imgnames ,lables):
        index = []
        index.append(imgname)
        index.append(lable)
        if wid in data_dict.keys():
            data_dict[wid].append(index)
        else:
            data_dict[wid] = [index]

    '''Try on different datasets'''
    # folder = 'res_img_gw'
    # img_base = '/home/lkang/datasets/WashingtonDataset_words/words/'
    # target_file = 'gw_total_mas50.gt.azAZ'

    # folder = 'res_img_parzival'
    # img_base = '/home/lkang/datasets/ParzivalDataset_German/data/word_images_normalized/'
    # target_file = 'parzival_mas50.gt.azAZ'

    # folder = 'res_img_esp'
    # img_base = '/home/lkang/datasets/EsposallesOfficial/words_lines.official.old/'
    # target_file = 'esposalles_total.gt.azAZ'



    return data_dict

gpu = torch.device('cuda')

def test_writer(wid, model_file, folder,text_corpus,data_dict):
    def read_image(file_name, thresh=None):
        subfolder = file_name.split('-')[0]  # gets 'a01'
        parent = '-'.join(file_name.split('-')[:2])  # gets 'a01-000u'
        url = os.path.join(img_base, subfolder, parent, file_name + '.png')
        
        if not os.path.exists(url):
            print(f"⚠️ Image not found: {url}")
            return None
        
        #old block
        #url = img_base + wid + '/' + file_name + '.png'
        img = cv2.imread(url, 0)
        if thresh:
            #img[img>thresh] = 255
            pass

        rate = float(IMG_HEIGHT) / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1]*rate)+1, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC) # INTER_AREA con error
        img = img/255. # 0-255 -> 0-1

        img = 1. - img
        img_width = img.shape[-1]

        if img_width > IMG_WIDTH:
            outImg = img[:, :IMG_WIDTH]
            img_width = IMG_WIDTH
        else:
            outImg = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype='float32')
            outImg[:, :img_width] = img
        outImg = outImg.astype('float32')

        mean = 0.5
        std = 0.5
        outImgFinal = (outImg - mean) / std
        return outImgFinal

    def label_padding(labels, num_tokens):
        new_label_len = []
        ll = [letter2index[i] for i in labels]
        new_label_len.append(len(ll)+2)
        ll = np.array(ll) + num_tokens
        ll = list(ll)
        ll = [tokens['GO_TOKEN']] + ll + [tokens['END_TOKEN']]
        num = OUTPUT_MAX_LEN - len(ll)
        if not num == 0:
            ll.extend([tokens['PAD_TOKEN']] * num) # replace PAD_TOKEN
        return ll

    '''data preparation'''
    imgs = [read_image(i[0]) for i in data_dict[wid]]
    random.shuffle(imgs)
    final_imgs = imgs[:50]
    if len(final_imgs) < 50:
        while len(final_imgs) < 50:
            num_cp = 50 - len(final_imgs)
            final_imgs = final_imgs + imgs[:num_cp]

    imgs = torch.from_numpy(np.array(final_imgs)).unsqueeze(0).to(gpu) # 1,50,64,216
    # print("imgs.size: '{}' ".format(imgs.size()))  #imgs.size: 'torch.Size([1, 50, 64, 216])'

    with open(text_corpus, 'r') as _f:
        texts = _f.read().split()
    labels = torch.from_numpy(np.array([np.array(label_padding(label, num_tokens)) for label in texts])).to(gpu)

    '''model loading'''
    model = ConTranModel(NUM_WRITERS, 0, True).to(gpu)
    # print('Loading ' + model_file)
    model.load_state_dict(torch.load(model_file)) #load
    # print('Model loaded')
    model.eval()
    num = 0
    with torch.no_grad():

        f_xss = model.gen.enc_image(imgs)
        f_xs = f_xss[-1]
        for label in labels:

            label = label.unsqueeze(0)
            f_xt, f_embed = model.gen.enc_text(label, f_xs.shape)
            f_mix = model.gen.mix(f_xss, f_embed)
            xg = model.gen.decode(f_mix, f_xss, f_embed, f_xt)
            pred = model.rec(xg, label, img_width=torch.from_numpy(np.array([IMG_WIDTH])))

            label = label.squeeze().cpu().numpy().tolist()
            pred = torch.topk(pred, 1, dim=-1)[1].squeeze()
            pred = pred.cpu().numpy().tolist()
            for j in range(num_tokens):
                label = list(filter(lambda x: x!=j, label))
                pred = list(filter(lambda x: x!=j, pred))
            label = ''.join([index2letter[c-num_tokens] for c in label])
            pred = ''.join([index2letter[c-num_tokens] for c in pred])
            ed_value = Lev.distance(pred, label)
            if ed_value <= 100:
                num += 1
                xg = xg.cpu().numpy().squeeze()
                xg = normalize(xg)
                xg = 255 - xg
                # img_folder = folder+'/'+ wid
                img_folder = folder
                if not os.path.exists(img_folder):
                    os.makedirs(img_folder)
                ret = cv2.imwrite(img_folder + '/' + wid + '-'+str(num)+'.'+label+'-'+pred+'.png', xg)
                # ret = cv2.imwrite(folder     + '/' + wid + '-'+str(num)+'.'+label+'-'+pred+'.png', xg)
                if not ret:
                    import pdb; pdb.set_trace()
                    xg

if __name__ == '__main__':
    args = parser.parse_args()
    model_epoch = str(args.epoch)

    for i in range(1):
        if i == 0:
            
            folder = folder_pre + model_epoch + '/res_4.oo_vocab_te_writer'
            target_file = '/home/woody/iwi5/iwi5333h/AFFGanWriting/Groundtruth/gan.iam.test.gt.filter27'
            text_corpus = '/home/woody/iwi5/iwi5333h/AFFGanWriting/corpora_english/oov.common_words'
            
              
        # elif i ==1:      
        #    folder = folder_pre +model_epoch +'/res_1.in_vocab_tr_writer'
         #    target_file = '/home/woody/iwi5/iwi5333h/AFFGanWriting/Groundtruth/gan.iam.tr_va.gt.filter27'
        #     text_corpus = '/home/woody/iwi5/iwi5333h/AFFGanWriting/corpora_english/in_vocab.subset.tro.37'
            
        # elif i == 2:
        #     folder = folder_pre + model_epoch + '/res_2.in_vocab_te_writer'
        #       target_file = '/home/woody/iwi5/iwi5333h/AFFGanWriting/Groundtruth/gan.iam.test.gt.filter27'
        #       text_corpus = '/home/woody/iwi5/iwi5333h/AFFGanWriting/corpora_english/in_vocab.subset.tro.37'

        # elif i == 3:
        #     folder = folder_pre + model_epoch + '/res_3.oo_vocab_tr_writer'
        #     target_file = '/home/woody/iwi5/iwi5333h/AFFGanWriting/Groundtruth/gan.iam.tr_va.gt.filter27'
        #     text_corpus = '/home/woody/iwi5/iwi5333h/AFFGanWriting/corpora_english/oov.common_words'

        if not os.path.exists(folder):
            os.makedirs(folder)
            print(folder)
        data_dict = dict()
        data_dict = pre_data(data_dict,target_file)


        with open(target_file, 'r') as _f:
            data = _f.readlines()
        wids = list(set([i.split(',')[0] for i in data]))

        wids = tqdm(wids)
        for wid in wids:
            # print(wid)
            # test_writer(wid, 'save_weights/<your best model>')
            test_writer(wid, '/home/vault/iwi5/iwi5333h/save_weights/contran-' + model_epoch + '.model', folder, text_corpus, data_dict)

