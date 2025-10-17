import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_model", type=str, default="./check_point",
                        help='pretrain model file dir')
    parser.add_argument('--train_dataset', default='./mp4',
                        help='train file root')
    parser.add_argument('--client_numbers', type=int, default=7,
                        help='client number')
    parser.add_argument('--val_data', default='./480p',
                        help='test file root')
    parser.add_argument('--patch_size', type=int, default=96,
                        help='Patch size')
    parser.add_argument('--max_number_patches', type=int, default=25600,
                        help='Patch size')
    parser.add_argument('--batch_size', type=int, default= 64,
                        help='pretrain batch size')
    parser.add_argument('--server_data', default='./server_data',
                        help='server dataset dictionary')
    parser.add_argument('--model_names', nargs='+', type=str, default=["SwinIR","fastdvdnet"],
                        help='model names')
    parser.add_argument('--mode',type=str,default='mean output',
                        help='distill aggregation mode')
    parser.add_argument('--num_epochs',type = int, default= 10,
                        help='number of global epochs')
    parser.add_argument('--upload_model', action="store_true", default=False,
                        help='allow clients to upload models to the server')
    parser.add_argument("--temp_patch_size", "--tp", type=int, default=5,
                        help="Temporal patch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--pretrain_epoch", "--e", type=int, default=5,
                        help="Number of pre training epochs")
    parser.add_argument("--noise_level", nargs=2, type=float, default=[5/255.0, 55/255.0],
                        help="Noise training interval")
    parser.add_argument("--test_noise", type=float, default=25 / 255.0,
                        help='noise level used on validation set')
    parser.add_argument("--temp_psz", type=int, default=5,
                        help="Temporal patch size")
    parser.add_argument('--local_ep', type=int, default=5,
                        help='iterations of local updating')
    parser.add_argument('--save_dir', default=None,
                        help='save model path')
    parser.add_argument('--model_counts', nargs='+', type=int, default=[2,5],
                        help='model count for each model type')
    parser.add_argument('--sampling_rate', type=float, default=1,
                        help='frac of local models to update')
    args = parser.parse_args()
    return args