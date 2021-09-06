import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import os
from utils import inf_loop, MetricTracker, confusion_matrix_image
from logger.visualization import TensorboardWriter
import collections

def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)


    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_ftns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    test_writer = TensorboardWriter(str(config.log_dir) + '/test', logger, config['trainer'])
    test_metrics = MetricTracker('loss', *[m.__name__ for m in metric_ftns], writer=test_writer)
    test_metrics_cls = [m() for m in metric_ftns]
    tmp_dir = './tmp'
    for met in test_metrics_cls:
        met.load(tmp_dir)
        logger.info("Metric {} loaded: {}".format(met.__class__.__name__, met.__dict__))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]

            test_writer.set_step(data_loader.start + batch_size * (i+1), 'test')
            test_metrics.update('loss', loss.item())
            for met in test_metrics_cls:
                met.update(output, target)
                test_metrics.update(met.__class__.__name__, met.result())

    log = {}
    for met in test_metrics_cls:
        met.save(tmp_dir)
    test_log = test_metrics.result()
    log.update(**{'test_' + k: v for k, v in test_log.items()})
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')


    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--start'], type=int, target='data_loader;args;start'),
        CustomArgs(['--end'], type=int, target='data_loader;args;end'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--model_name'], type=str, target='arch;args;model_name'),
        CustomArgs(['--n', '--name'], type=str, target='name'),
    ]
    config = ConfigParser.from_args(args, options)

    main(config)
