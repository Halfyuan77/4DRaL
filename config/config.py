import argparse

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def str2bool(v):
    return v.lower() in ('true', '1')

# Dataset specific configurations
data_arg = add_argument_group('Data')
# KittiPointSparseTupleDataset #MulRanPointSparseTupleDataset
data_arg.add_argument('--dataset_path', type=str,  default='/home/h/data/code/dataset')
data_arg.add_argument('--checkpoint', type=str,  default='/home/h/data/code/radar_KD/checkpoint/')
data_arg.add_argument('--logPath', type=str,  default='/data/hny/code/radar_KD/log/')
data_arg.add_argument('--debug', type=bool,default=False)

trainer_arg = add_argument_group('Train')
trainer_arg.add_argument('--seed', type=int, default=1234)
trainer_arg.add_argument('--margin', type=float, default=0.3)
trainer_arg.add_argument('--threads', type=int, default=8)
trainer_arg.add_argument('--seq_len', type=int, default=3)

trainer_arg.add_argument('--batch_size', type=int, default=40)
trainer_arg.add_argument('--lr', type=float, default=0.001)
trainer_arg.add_argument('--nEpochs', type=int, default=10)
trainer_arg.add_argument('--loss_margin', type=float, default=0.3)  # 0.5
trainer_arg.add_argument('--positives_per_query', type=int, default=1)  # 2
trainer_arg.add_argument('--negatives_per_query', type=int, default=4)  # 2-18
trainer_arg.add_argument('--resume_training', type=str2bool, default=False)
trainer_arg.add_argument('--scheduler', type=str, default='multistep')
trainer_arg.add_argument('--save_model', type=bool, default=False)


def get_config():
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    cfg = get_config()
    # print(cfg.train_pipeline)
    dconfig = vars(cfg)
    print(dconfig)