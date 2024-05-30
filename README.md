# FLGo-Async

## RUN 
Run the command below to generate the result for fedavg. The results of other methods can be obtained by replacing fedavg by algo_name.
```shell
cd _exp_main
python main.py --method fedavg --task CIFAR100_N20_DIR0.3
```

## Show result
```shell
cd _exp_main
python analysis.py --method fedavg ca2fl fedbuff fedasync
```
![image](https://github.com/WwZzz/FLGo-Async/assets/20792079/67fc44ec-3b01-47e5-985c-f8e6adf4c70c)
![image](https://github.com/WwZzz/FLGo-Async/assets/20792079/86a05097-f4ac-441f-b146-115dc0f3fe14)
