import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # Data specifc paremeters
    parser.add_argument('--server_data', default='./server_data',
                        help='server dataset dictionary')
    parser.add_argument('--batch_size', type=int, default= 64,
                        help='pretrain batch size')
    parser.add_argument("--temp_patch_size", "--tp", type=int, default=5,
                        help="Temporal patch size")
    parser.add_argument("--patch_size", "--p", type=int, default=96,
                        help="Patch size")
    parser.add_argument("--max_number_patches", "--m", type=int, default=256000,
                        help="Maximum number of patches")
    parser.add_argument("--pretrain_epoch", "--e", type=int, default=20,
                        help="Number of pre training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--noise_ival", nargs=2, type=int, default=[5, 55],
                        help="Noise training interval")
    parser.add_argument("--val_noiseL", type=float, default=25,
                        help='noise level used on validation set')
    parser.add_argument("--val_data", type=str, default="./val_data",
                        help='path of validation set')
    parser.add_argument("--pretrain_model", type=str, default="./check_point",
                        help='pretrain model file dir')
    args = parser.parse_args()
    return args