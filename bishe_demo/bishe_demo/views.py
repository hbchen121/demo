import os.path as osp
import numpy as np
import pickle
import IPython
from collections import OrderedDict

from django.http.response import JsonResponse
from django.shortcuts import render
from django.http import HttpResponse


# 图片名称 图片特征
all_img_names = None
query_names = None
gallery_names = None


def load_feat(name):
    with open(name, "rb") as f:
        img2feat = pickle.load(f)
        print(f"load {name}")
    query2feat = OrderedDict()
    gallery2feat = OrderedDict()
    for img_path in sorted(img2feat.keys()):
        # pass
        # IPython.embed()
        feat = img2feat[img_path]
        img_id = osp.basename(img_path)
        if "query" in img_path:
            query2feat[img_id] = feat
        else:
            gallery2feat[img_id] = feat
    global all_img_names
    global query_names
    global gallery_names
    if all_img_names is None:
        query_names = list(query2feat.keys())
        gallery_names = list(gallery2feat.keys())
        all_img_names = query_names + gallery_names

    gallery_feats = np.stack(list(gallery2feat.values()))
    # gallery_feats = gallery_feats / np.linalg.norm(gallery_feats, axis=1, keepdims=True)

    # print(list(gallery2feat.keys())[:10])
    return query2feat, gallery_feats


m1_query2feat, m1_gallery_feats = load_feat("../method1_img2feats.pkl")
m2_query2feat, m2_gallery_feats = load_feat("../method2_img2feats.pkl")
m3_query2feat, m3_gallery_feats = load_feat("../method3_img2feats.pkl")


def cal_dist(reid_query_feat, reid_gallery_feats, feat_name=1):
    if feat_name == 1:
        reid_query_feat = np.expand_dims(reid_query_feat, 0)
        dist = 1 - np.matmul(reid_query_feat, reid_gallery_feats.T)
    else:
        reid_query_feat = np.expand_dims(reid_query_feat, 0)
        dist = 1 - np.matmul(reid_query_feat, reid_gallery_feats.T)
    return dist
    pass


def main(request):    
    context          = {}
    context['title'] = "基于多视角感知学习的车辆再识别技术研究-演示系统"
    context['names_list'] = all_img_names
    return render(request, 'bishe.html', context)


def index(request):
    context          = {}
    context['title'] = "基于多视角感知学习的车辆再识别技术研究-演示系统"
    context['query_names_list'] = query_names
    return render(request, 'index.html', context)


def reid(request):
    query_name = request.POST['query_name']
    feat_name = int(request.POST['feat_name'])
    topk = int(request.POST['topk'])
    # feat_name, query_name = 2, "0652_c003_00063195_0.jpg"
    # topk = 5
    if feat_name == 1:
        reid_query_feat = m1_query2feat[query_name]
        reid_gallery_feats = m1_gallery_feats
    elif feat_name == 2:
        reid_query_feat = np.array(m2_query2feat[query_name])
        reid_gallery_feats = m2_gallery_feats
    else:
        # 3
        reid_query_feat = np.array(m3_query2feat[query_name])
        reid_gallery_feats = m3_gallery_feats

    # IPython.embed()
    # 根据类型计算距离
    dist = cal_dist(reid_query_feat, reid_gallery_feats, feat_name=feat_name)
    # reid_query_feat = np.expand_dims(reid_query_feat, 0)
    # dist = 1 - np.matmul(reid_query_feat, reid_gallery_feats.T)
    args = np.argsort(dist, axis=1).squeeze()

    # IPython.embed()
    output_names = []
    for i in range(topk):
        output_names.append(gallery_names[args[i]])

    #return HttpResponse(request.POST['query_name'])
    # IPython.embed()
    return JsonResponse(output_names, safe=False)


if __name__ == '__main__':
    # feat_name, query_name = 1, "0652_c003_00063195_0.jpg"
    # topk = 5
    # reid(None)
    IPython.embed()
    pass
