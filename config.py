import argparse

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="heat", help="training data list")
    parser.add_argument('--mesh', type=str, default='circle_low_res', help="only train and test on One file, leave train test path empty")
    parser.add_argument('--param', type=str, default=None, help="physics parameter: diffusivity/density")
    parser.add_argument('--train-path', type=str, default=None, help="training data list")
    parser.add_argument('--test-path', type=str, default=None, help="test data list")
    parser.add_argument('--diffusivity-cap', type=str, default=None, help="maximum of diffusivity to sample")
    parser.add_argument('--cpu', default=32, type=int)

    # training arguments
    parser.add_argument('--epochs', default=3000, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--loss', default='l2', type=str, help='type of loss to use')
    parser.add_argument('--x-loss-weight', default=1.0, type=float)
    parser.add_argument('--rhs-loss-weight', default=1.0, type=float)
    parser.add_argument('--precond-loss-weight', default=0.0001, type=float)
    parser.add_argument('--diag-loss-weight', default=1.0, type=float)
    parser.add_argument('--diag-loss2-weight', default=1e3, type=float)
    parser.add_argument('--kappa-loss-weight', default=1.0, type=float)
    parser.add_argument('--scheduler', default='ExpLR', type=str)
    parser.add_argument('--tensorboard', default=False, action='store_true')
    parser.add_argument('--simulate', default=False, action='store_true')
    parser.add_argument('--log-freq', default=50, type=int)
    parser.add_argument('--val-freq', default=10, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--save-dir', default='./results/', type=str)
    parser.add_argument('--exp-name', default='exp1', type=str, help='experiment name for result dir')
    parser.add_argument('--ckpt', default='', type=str, help='whether to load checkpoint')
    parser.add_argument('--decay', default=False, action='store_true')

    # network arguments
    parser.add_argument('--model', default='model', type=str, help='model_sep_diag_v1, model_sep_diag_v2, model_sep_diag_v3, model_diag_0818')
    parser.add_argument('--hidden-dim', default=16, type=int)
    parser.add_argument('--norm', default='LayerNorm', type=str, help='LayerNorm, InstanceNorm1d, LazyInstanceNorm1d, BatchNorm1d, LazyBatchNorm1d, MessageNorm')
    parser.add_argument('--hidden-layers-encoder', default=1, type=int)
    parser.add_argument('--hidden-layers-decoder', default=1, type=int)
    parser.add_argument('--hidden-layers-processor', default=1, type=int)
    parser.add_argument('--num-iterations', default=10, type=int, help="number of message passing steps")
    parser.add_argument('--num-attenheads', default=10, type=int, help="number heads in attention")
    parser.add_argument('--use-r', default=False, action='store_true')


    # debug arguments
    parser.add_argument('--use-data-num', default=1, type=int, help="number of message passing steps")
    parser.add_argument('--use-global', default=False, action='store_true')
    parser.add_argument('--high-freq', default=False, action='store_true')
    parser.add_argument('--augment-edge', default=False, action='store_true')
    parser.add_argument('--use-pred-x', default=False, action='store_true')
    parser.add_argument('--diagonalize', default=False, action='store_true')
    parser.add_argument('--high-freq-aug', default=False, action='store_true')
    parser.add_argument('--undirected', default=False, action='store_true')
    parser.add_argument('--precond-type', default='network', type=str, help='to be used for time comparison, ilu, identity, jacobi')

    # evaluation arguments (not used for training)
    parser.add_argument('--solve-triang', default=False, action='store_true')
    parser.add_argument('--solve-triag', default=False, action='store_true')
    parser.add_argument('--use-pred-x-eval', default=False, action='store_true')
    


    return parser
