import sys
sys.path.append('..')

import argparse
import os
import shutil
import random
import warnings
import time

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix as conf_mat_metric

import torch
import torchvision
import torchinfo

from PatchMixer.augmentations import center, scale1, scale2
from datasets import get_dataset
from PatchMixer.utils import viz_shape, get_logger
from tsne import tsne

from datasets_mn40 import CATEGORIES as mn40_categories
from datasets_mn40 import CATEGORIES_COMMON as mn40_categories_common
# from datasets_mn40 import CATEGORIES_COMMON_PLOT_Sonn as mn40_categories_common_plot_sonn
from datasets_mn40 import CATEGORIES_COMMON_PLOT_PointDA as mn40_categories_common_plot_pointDA
from datasets_pointda import CATEGORIES as poitda_categories

def parse_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--exp', type=str, help='Experiment name')
    parser.add_argument('--seed', type=int, default=None, help='Fix random seed')
    parser.add_argument('--sampling_seed', type=int, default=None, help='Fix random seed for sampling')
    parser.add_argument('--simpleview_protocol', default=False, help='Use simpleview protocol', action='store_true')

    # Train
    parser.add_argument('--bs', type=int, default=1, help='Batch size')
    parser.add_argument('--n_classes', type=int, required=True, help='Number of classes')

    # Test
    parser.add_argument('--n_classes_test', type=int, required=True, help='Number of classes transfer learning test dataset')

    # Data
    parser.add_argument('--path', type=str, required=True, help='Dataset path')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--protocol', type=str, required=True, help='Protocol name')
    parser.add_argument('--dataset_type', type=str, required=True, help='Dataset type')
    parser.add_argument('--variant', type=str, default=None)  # TODO What is this? I don't remember...

    # tSNE
    parser.add_argument('--flag_compute_tsne', default=False, action='store_true', help='')

    parser.add_argument('--training', default=False, action='store_true')  # ?
    parser.add_argument('--augms', default=None)  

    args = parser.parse_args()
    return args


def main():

    # Parse input arguments
    args_test = parse_args()

    # Update weights path
    # args_test.path_weights = os.path.join('..', 'data', 'exps', 'weights', args_test.exp)
    args_test.path_weights = os.path.join('data', 'exps', 'weights', args_test.exp)

    # Loop over checkpoints
    for f in ['last']:  # 'best_oa', 'best_mca', 'last'

        if not os.path.isfile(os.path.join(args_test.path_weights, '{:s}.pth'.format(f))):
            continue

        # Load training arguments
        checkpoint = torch.load(os.path.join(args_test.path_weights, '{:s}.pth'.format(f)))
        args = checkpoint['args']

        # Update arguments
        args.checkpoint = f
        args.exp = args_test.exp
        args.path = args_test.path
        args.dataset_type = args_test.dataset_type
        args.n_classes = args_test.n_classes
        args.n_classes_test = args_test.n_classes_test
        args.dataset = args_test.dataset
        args.protocol = args_test.protocol
        if args_test.bs is not None:
            args.bs = args_test.bs
        args.variant = args_test.variant
        args.seed = args_test.seed
        args.flag_compute_tsne = args_test.flag_compute_tsne
        args.path_weights = args_test.path_weights
        args.training = args_test.training

        # Create logger
        if args.variant is None:
            path_log = os.path.join(args.path_weights, 'log_test_{:s}_{:s}.txt'.format(args.dataset_type, f))
        else:
            path_log = os.path.join(args.path_weights, 'log_test_{:s}-{:s}_{:s}.txt'.format(args.dataset_type, args.variant, f))
        logger = get_logger(path_log)

        # Log library versions
        logger.info('PyTorch version = {:s}'.format(torch.__version__))
        logger.info('TorchVision version = {:s}'.format(torchvision.__version__))

        # Activate CUDNN backend
        torch.backends.cudnn.enabled = True

        # Fix random seed
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            os.environ['PYTHONHASHSEED'] = str(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            torch.backends.cudnn.benchmark = True

        # Log input arguments
        for arg, value in vars(args).items():
            logger.info('{} = {}'.format(arg, value))

        # Perform the test
        run_test(args, logger, checkpoint)


def run_test(args, logger, checkpoint):

    # Get the test data augmentation
    transforms_list = list()
    if args.augms is not None:
        augms_list = args.augms.split(',')
        if 'center' in augms_list:
            transforms_list.append(center())
        if 'scale1' in augms_list:
            transforms_list.append(scale1())
        if 'scale2' in augms_list:
            transforms_list.append(scale2())
    transforms = torchvision.transforms.Compose(transforms_list)

    # Get the test dataset
    dataset = get_dataset(args, split='test', transforms=transforms)

    # Log test stats
    logger.info('TST samples: {:d}, {}'.format(len(dataset), dataset.n_instances))

    # Get the test data loader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.bs,
        num_workers=args.n_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    # Load backup model
    sys.path.append(args.path_weights)
    from models_ours_backup import mixer_class_singleres, mixer_class_multires

    # Get the model
    if len(args.radii) == 1:
        logger.info('Using mixer model with a single resolution')
        model = mixer_class_singleres(args)
    elif len(args.radii) > 1:
        logger.info('Using mixer model with mutliple resolutions')
        model = mixer_class_multires(args)

    # Load trained weights
    if args.checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])

    # Send the model to the device
    model = model.to(args.device)

    # Set data parallelism
    if torch.cuda.device_count() == 1:
        logger.info('Using a single GPU, this will disable data parallelism')
    else:
        logger.info('Using multiple GPUs, with data parallelism')
        model = torch.nn.DataParallel(model)

    # Set the model in eval mode
    model.eval()

    # Get the model summary
    if torch.cuda.device_count() == 1:
        logger.info('Model summary:')
        if args.flag_voxel is True:
            stats = torchinfo.summary(model, (args.bs, args.n_patches, args.ch, args.n_samples))
        else:
            stats = torchinfo.summary(model, (args.bs, 3, args.n_verts))
        logger.info(str(stats))


    # Init test stats
    running_oa_num = 0
    running_oa_den = 0
    running_ca_pred = torch.zeros(args.n_classes_test).to(args.device)
    running_ca_target = torch.zeros(args.n_classes_test).to(args.device)
    since = time.time()

    # Init confusion matrix
    # confusion_matrix = torch.zeros(args.n_classes_test, args.n_classes_test, dtype=torch.int, device=args.device)

    # Init TSNE arrays
    X = list()
    L = list()
    P = list()

    output_tsne = np.zeros((args.bs, args.n_classes))
    y_true = np.array([])
    y_pred = np.array([])

    xtickslabels = []
    if args.dataset_type == "sonn11":
        xtickslabels = mn40_categories_common #_plot_sonn
        categories_common = mn40_categories_common
    elif args.dataset_type == "mn4010" or args.dataset_type == "shapenet10" or args.dataset_type == "scannet10":
        xtickslabels = mn40_categories_common_plot_pointDA
        categories_common = mn40_categories_common_plot_pointDA
    elif args.dataset_type == "mn4011" or args.dataset_type == "mn4010":
        xtickslabels = poitda_categories
        categories_common = mn40_categories_common_plot_pointDA

    # Iterate over test data
    for idx_batch, data_batch in enumerate(loader):

        # Load a mini-batch
        input, target = data_batch

        # Send data to device
        input = input.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)

        # Forward pass
        with torch.inference_mode():
            logits, others = model(input)
            _, pred = torch.max(logits, dim=1)

            index_pred = np.array([])
            for p in pred:
                for i, cat in enumerate(categories_common):
                    if isinstance(cat, list) and mn40_categories[p.item()] in cat:
                        index_pred = np.append(index_pred, i)
                    elif mn40_categories[p.item()] == cat:
                        index_pred = np.append(index_pred, i)
            
            if index_pred.size == 0:
                continue

            pred = torch.from_numpy(index_pred).to(int).to(args.device)
            # target = torch.from_numpy(index_target).to(int).to(args.device)

            y_true = np.append(y_true, target.cpu().numpy())
            y_pred = np.append(y_pred, pred.cpu().numpy())

            if args.flag_compute_tsne:
                output_tsne = np.vstack((output_tsne, logits.cpu().numpy()))
                if(idx_batch == 0):
                    output_tsne = output_tsne[args.bs:]

                X.append(others['features'].squeeze().cpu().detach().numpy())
                L.append(target.squeeze().cpu().detach().numpy())
                P.append(pred.squeeze().cpu().detach().numpy())

        # # Compute confusion matrix
        # for i, j in zip(target, pred):
        #     confusion_matrix[i, j] += 1

        # Iteration statistics
        running_oa_num += torch.sum(pred == target.squeeze())
        running_oa_den += len(pred)
        running_ca_pred += torch.sum(
            torch.nn.functional.one_hot(pred, num_classes=args.n_classes_test) * \
            torch.nn.functional.one_hot(target, num_classes=args.n_classes_test), dim=0)
        running_ca_target += torch.sum(
            torch.nn.functional.one_hot(target, num_classes=args.n_classes_test), dim=0)

    # Test statistics
    oa = running_oa_num.float() / running_oa_den
    ca = torch.div(running_ca_pred, running_ca_target)
    ca_mean, ca_std = torch.mean(ca), torch.std(ca)

    # Log test statistics
    elapsed = time.time() - since
    logger.info('TST, OA: {:.4f}, CA: {:.4f} +- {:.4f}, Elapsed: {:.2f}s'.format(
        oa, ca_mean, ca_std, elapsed))

    # Close logger
    for _ in list(logger.handlers):
        logger.removeHandler(_)
        _.flush()
        _.close()

    
    # Normalize confusion matrix
    # confusion_matrix = np.array(confusion_matrix.detach().cpu().numpy(), dtype=float)
    # for _ in range(confusion_matrix.shape[0]):
    #     confusion_matrix[_, :] /= 1.0 * loader.dataset.n_instances[_] + np.finfo(float).eps
        

    # Plot confusion matrix
    logger.info('Computing confusion matrix...')


    # plt.figure()
    # ax = sns.heatmap(confusion_matrix,
    #     cbar=False, vmin=0.0, vmax=1.0, #cmap='Blues', 
    #     annot=True, annot_kws={'fontsize': 4}, fmt='.2f',
    #     xticklabels=loader.dataset.categories, yticklabels=loader.dataset.categories)  # mask=cm==0.0
    # ax.tick_params(left=False, bottom=False)
    # ax.set_xlabel('Predicted classes')
    # ax.set_ylabel('Ground-truth classes')
    # ax.set_title('Confusion matrix (epoch = {:d}, oa = {:.4f}, ca = {:.4f} +- {:.4f})'.format(
    #     checkpoint['epoch'], oa, ca_mean, ca_std))
    # for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    #     item.set_fontsize(4)
    # os.makedirs(os.path.join(args.path_weights, 'cm'), exist_ok=True)
    # plt.savefig(
    #     os.path.join(args.path_weights, 'cm', 'cm_{:s}_{:s}.png'.format(
    #         args.dataset_type, args.checkpoint)),
    #     transparent=False, bbox_inches='tight', dpi=300)
    # plt.close()

    # print("y_true:", y_true, y_true.shape)
    # print("y_pred:", y_pred, y_pred.shape)

    confusion_matrix = conf_mat_metric(y_true, y_pred, normalize='true')
    conf_mat = (confusion_matrix * 100) #.astype(int)

    s = np.sum(conf_mat, axis=1)
    print(s)

    plt.figure(figsize=(16, 14))
        
    # sns.heatmap(conf_mat, annot=True, fmt=".1f", xticklabels=xtickslabels, yticklabels=loader.dataset.categories)
    sns.heatmap(conf_mat, annot=True, fmt=".1f", xticklabels=xtickslabels, yticklabels=poitda_categories)
    plt.title('Confusion Matrix {:s} {:s} Protocol'.format(args.dataset, args.protocol))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    os.makedirs(os.path.join(args.path_weights, 'cm'), exist_ok=True)
    # plt.savefig('plot/confusion_matrix.png')
    plt.savefig(
        os.path.join(
            args.path_weights, 'cm', 'cm_{:s}_{:s}.png'.format(args.dataset_type, args.checkpoint)
        ),
        transparent=False, 
        bbox_inches='tight', 
        dpi=300
    )
    plt.close()

    # TSNE
    # if args.flag_compute_tsne:
    #     tsne = TSNE(n_components=2, random_state=42).fit_transform(output_tsne, y=y_true)

    #     plt.figure(figsize=(16, 12))
    #     plt.scatter(tsne[:, 0], tsne[:, 1], c=y_true, cmap=plt.cm.get_cmap('tab20', args.n_classes_test))
    #     plt.title('t-SNE {:s} {:s} Protocol'.format(args.dataset, args.protocol))
    #     os.makedirs(os.path.join(args.path_weights, 'embedding'), exist_ok=True)
    #     plt.savefig(
    #         os.path.join(
    #             args.path_weights, 'embedding', 'embedding_{:s}_{:s}.png'.format(args.dataset_type, args.checkpoint)
    #         ),
    #         transparent=False, 
    #         bbox_inches='tight', 
    #         dpi=300
    #     )

    # Compute tSNE
    if args.flag_compute_tsne:
        logger.info('Computing tSNE embedding...')
        assert args.bs == 1, 'To compute TSNE we assume bs = 1'
        X = np.array(X)
        L = np.array(L)
        P = np.array(P)
        Y = tsne(X, 2, 50, 20.0)
        fig, ax = plt.subplots()
        scatter = ax.scatter(Y[:, 0], Y[:, 1], s=20, c=P,  # L
            vmin=0, vmax=args.n_classes_test, cmap='tab20', alpha=0.75, edgecolors='none')
        legend = ax.legend(*scatter.legend_elements(),
            title='Classes', loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
        ax.add_artist(legend)
        ax.set_title(dataset.categories)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        os.makedirs(os.path.join(args.path_weights, 'embedding'), exist_ok=True)
        plt.savefig(
            os.path.join(args.path_weights, 'embedding', 'embedding_{:s}_{:s}.png'.format(
                args.dataset_type, args.checkpoint)),
            transparent=False, bbox_inches='tight', dpi=300)

        if args.dataset_type == 'sonn':
            n_rows = 4
        else:
            n_rows = 3
        n_cols = 4
        fig, axs = plt.subplots(nrows=n_rows, ncols=4, sharex='col', sharey='row')
        correct = 0
        total = 0
        for row in range(n_rows):
            for col in range(n_cols):
                idx = row * n_cols + col
                if np.sum(L == idx) > 0:
                    idxs = np.where(L == idx)
                    axs[row, col].scatter(Y[idxs, 0], Y[idxs, 1], s=20, c=1.0 * (P[idxs] == L[idxs]),
                        vmin=0, vmax=8, cmap='Set1', alpha=0.9, edgecolors='none')
                    axs[row, col].xaxis.set_ticklabels([])
                    axs[row, col].yaxis.set_ticklabels([])
                    axs[row, col].set_title(dataset.categories[idx])
                    axs[row, col].title.set_fontsize(6)
                    correct += np.sum(P[idxs] == L[idxs])
                    total += len(L[idxs])
        logger.info('Correct: {}, wrong: {}, total: {}'.format(correct, total - correct, total))
        if args.dataset_type == 'sonn' or args.dataset_type == 'mn4011' or args.dataset_type == 'sonn11':
            axs[-1, -1].axis('off')
        elif args.dataset_type == 'rk10' or args.dataset_type == 'rr10' or args.dataset_type == 'sk10' or args.dataset_type == 'sr10':
            axs[-1, -2].axis('off')
            axs[-1, -1].axis('off')
        plt.savefig(
            os.path.join(args.path_weights, 'embedding', 'correct_{:s}_{:s}.png'.format(
                args.dataset_type, args.checkpoint)),
            transparent=False, bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    main()
