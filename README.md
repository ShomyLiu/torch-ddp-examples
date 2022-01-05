
Pytorch 分布式训练代码, 以Bert文本分类为例子, 完整介绍见博客

- main.py: 单进程训练: python3 main.p
- ddp_main.py: 原生DDP 多卡训练: `torchrun --nproc_per_node=2 ddp_main.py`
- horovod_main.py: 使用horovod框架 多卡训练： `horovodrun -np 2 python3 horovod_main.py`
- accelerate_main.py: 使用accelerate框架多卡训练：`accelerate launch accelerate_main.py` (首先需要配置acclerate config)

单卡运行的时间大概是双卡的二倍。
