import os
import argparse
import pprint
import subprocess
from time import time, localtime, strftime


def make_cmd(job_name, cmd, slurm_log_dir, n_exp, ngpus=1, max_time='10-00:00:00', partition='part-name'):
    slurm_cmd = 'sbatch --output=\"{}\"'.format(slurm_log_dir) \
                         + ' --nodes=1 --ntasks-per-node=1 --time={} --array=1-{}'.format(max_time, n_exp) \
                         + ' --mem={}G --partition={} --cpus-per-task={}'.format(ngpus*16, partition, ngpus*8) \
                         + ' --gres=gpu:{} --job-name=\"{}\" --wrap=\"{}\"'.format(ngpus, job_name, cmd)

    return slurm_cmd



def main(env, model_type, prop_type, calibrate, num_exp, logdir):
    print('Running experiment: {}'.format((env, model_type, prop_type, calibrate, num_exp, logdir)))
    calib_option = 'calibrate' if calibrate else 'no-calibrate'
    curr_time = strftime('%Y-%m-%d--%H:%M:%S', localtime())
    model_name = '{}-{}-{}'.format(model_type, prop_type, calib_option)
    slurm_log_dir = './slurm_logs/{}-log-{}.txt'.format(curr_time, model_name)
    out_dir = os.path.join(logdir, env, model_name)

    job_name = '{}-{}'.format(env, model_name)

    # print(model_name)
    # print(slurm_log_dir)
    # print(out_dir)

    cmd = 'python scripts/mbexp.py -env {} -ca model-type {} -ca prop-type {} -logdir {} -{}'.format(env, model_type, prop_type, out_dir, calib_option)

    slurm_cmd = make_cmd(job_name, cmd, slurm_log_dir, num_exp)
    print(slurm_cmd)

    process = subprocess.Popen(slurm_cmd, shell=True)


def validate_args(args):
    assert args.env in ['cartpole', 'reacher', 'pusher', 'halfcheetah', 'ant']
    assert args.model_type in ['P', 'PE']
    assert args.prop_type in ['DS', 'TS1', 'TSinf', 'MM']
    assert args.calibrate in [True, False]
    assert 0 < args.num_experiments < 20

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True,
                        help='Environment name: select from [cartpole, reacher, pusher, halfcheetah, ant]')
    parser.add_argument('--model_type', type=str, required=True,
                        help='Model type: select from [P, PE]')
    parser.add_argument('--prop_type', type=str, required=True,
                        help='Propogation type: select from [DS, TS1, TSinf, MM]. Note TS only valid if model PE')
    parser.add_argument('--calibrate',  dest='calibrate', action='store_true',
                        help='Enable calibration for the model')
    parser.add_argument('--no-calibrate',  dest='calibrate', action='store_false',
                        help='Disable calibration for the model')
    parser.set_defaults(calibrate=False)
    parser.add_argument('--num_experiments', type=int, default=1,
                        help='Number of experiments to run')
    parser.add_argument('-logdir', type=str, default='./logs',
                        help='Directory to which results will be logged (default: ./logs)')
    args = parser.parse_args()
    validate_args(args)

    main(args.env, args.model_type, args.prop_type, args.calibrate, args.num_experiments, args.logdir)
