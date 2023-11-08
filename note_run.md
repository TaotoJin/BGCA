activate my local virtual env

```bash
$ cd /BGCA
$ activate
$ conda activate path_to_BGCA\.conda

```






`run_aste` with **MT5 base** training
```bash
python main.py --task aste --name run_aste_mt5_local --seed 42 --dataset cross_domain --model_name_or_path google/mt5-base --paradigm extraction-universal --n_gpu 1 --train_batch_size 4 --gradient_accumulation_steps 2 --eval_batch_size 128 --learning_rate 3e-4 --num_train_epochs 25 --save_last_k 3 --n_runs 1 --clear_model --save_best --data_gene --data_gene_extract --data_gene_extract_epochs 25 --data_gene_epochs 25 --init_tag english --do_train --do_eval --use_same_model --data_gene_wt_constrained --model_filter --train_by_pair 

```
`run_aste` with **MT5 base** only predict and evaluate

```bash
python main.py --task aste --name 1024_1313-run_aste --seed 42 --dataset cross_domain --model_name_or_path ..\outputs\aste\cross_domain\1024_1313-run_aste\seed-42\rest14-laptop14\checkpoint-e3 --paradigm extraction-universal --n_gpu 1 --n_runs 1 --train_by_pair --nrows 100 --do_eval --target_domain laptop14

```




-----

# dataset

`test.txt`
```
Boot time is super fast , around anywhere from 35 seconds to 1 minute .####[([0, 1], [4], 'POS')]


```
```python
[([0, 1], [4], 'POS')] = [([list of aspect word[i]], [list of opinion word[i]], 'polarity in entire text')]

[(aspect word[0], aspect word[1]), [opinion word[4]], (POS|NEG|NEU)]
```

สิ่งที่ต้องการ thai dataset

```
เวลา บูต เครื่อง เร็ว มาก ประมาณ 35 วิ ถึง 1 นาที #### ([0,1,2],[3,4],'POS')
```


เอาแบบนี้ดีกว่า `test_aste_processed.txt`


```
Boot time is super fast , around anywhere from 35 seconds to 1 minute . ===> <pos> Boot time <opinion> fast
```


เอา text เข้า machine translation

```
Boot time is super fast , around anywhere from 35 seconds to 1 minute . ===> <pos> Boot time <opinion> fast -> เวลาบูตเครื่องเร็วมากประมาณ 35 วิถึง 1 นาที ==> <pos> เวลาบูตเครื่อง <opinion> เร็ว

```


สิ่งที่ต้องการ thai dataset

```
เวลา บูต เครื่อง เร็ว มาก ประมาณ 35 วิ ถึง 1 นาที #### <pos> เวลา บูต เครื่อง <opinion> เร็ว มาก
```