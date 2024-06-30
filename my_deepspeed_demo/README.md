# 使用deepspeed在单级单卡上训练模型

其中hello_deepspeed_train是官方的代码，拿过来学习的。


```commandline

# 安装环境
pip install py-cpuinfo  -i https://pypi.tuna.tsinghua.edu.cn/simple 

pip install deepspeed  -i https://pypi.tuna.tsinghua.edu.cn/simple 

# 训练
deepspeed deepspeed_train.py --epoch 1000 \
  --deepspeed --deepspeed_config deepspeed_config.json
  
  
# 预测
deepspeed deepspeed_eval.py --deepspeed --deepspeed_config deepspeed_config.json

  


```