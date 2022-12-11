# 毕设的车辆再识别网页演示系统

## Install and Prepare

参考 [文章](https://pythondjango.cn/django/basics/2-installation-use/#%E7%AC%AC%E4%B8%80%E4%B8%AAdjango%E9%A1%B9%E7%9B%AE) 进行安装
，虚拟环境就不用安装了，直接

```pip install django```

调过 django 项目创建（本目录已经创建好了），直接

```
cd bishe_demo

python .\manage.py runserver
```

然后在 `http://127.0.0.1:8000/` [link](http://127.0.0.1:8000/) 里进行查看

## FILES

从 [百度云](https://pan.baidu.com/s/1JuT8TKvbaH_vzcOTiL1SoA) 可以下载，验证码 `0njg`.

```
method1 - viewpoint-loss: 
bash: D:\research\fastreid_baseline\jd-reid-viewpoint\veri\view_baseline.sh
ck: hawk1:/home/chenhaobo996/fastreid_baseline/jd-reid-viewpoint/logs/veri/view_baseline_wo-nl-ibn-gem

method2 - transformer layer
bash: D:\research\fastreid_baseline\jd-reid-viewpoint\parts\veri3.sh
ck: new3:/home/chenhaobo996/fastreid_baseline/jd-reid-viewpoint/logs/veri/trans_head2/baseline/1-view_part

method3 - overall model
bash: D:\research\fastreid_baseline\jd-reid-viewpoint\parts\veri2.sh
ck: new3:/home/chenhaobo996/fastreid_baseline/jd-reid-viewpoint/logs/veri/trans_head2/baseline/1-ibn-view_global_circleSoftmax-view_part_True_Dec_mean-fusion_False_e0.2
```

## 致谢

- 代码继承自窦哥，感谢窦哥，窦哥牛逼！