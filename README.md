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
