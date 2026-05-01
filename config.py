import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # Data loader settings
    parser.add_argument('--max_seq_length', type=int, default=512, help='maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=1, help='minimum word frequency threshold for building the vocabulary.')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for the dataloader.')
    parser.add_argument('--batch_size', type=int, default=1, help='number of samples per batch.')
    parser.add_argument('--dataset_name', type=str, default='ct_dataset', help='name of the dataset.')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=2048, help='hidden dimension of the Transformer.')
    parser.add_argument('--d_ff', type=int, default=2048, help='hidden dimension of the feed-forward network.')
    parser.add_argument('--d_vf', type=int, default=2048, help='dimension of visual or patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='number of attention heads in the Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='number of Transformer layers.')
    parser.add_argument('--meta_num_layers', type=int, default=2, help='number of layers in the meta Transformer.')
    parser.add_argument('--fusion_num_layers', type=int, default=3, help='number of layers in the fusion Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate used in the Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='number of logit layers.')
    parser.add_argument('--bos_idx', type=int, default=0, help='index of the <bos> token.')
    parser.add_argument('--eos_idx', type=int, default=0, help='index of the <eos> token.')
    parser.add_argument('--pad_idx', type=int, default=0, help='index of the <pad> token.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='dropout rate of the language model output layer.')

    # for CTViT and 3DUNet
    parser.add_argument('--dim', type=int, default=512, help='feature dimension used in CTViT.')
    parser.add_argument('--image_size', type=int, default=480, help='input image size for CTViT.')
    parser.add_argument('--patch_size', type=int, default=24, help='spatial patch size for CTViT.')
    parser.add_argument('--temporal_patch_size', type=int, default=12, help='temporal patch size for 3D data in CTViT.')
    parser.add_argument('--spatial_depth', type=int, default=4, help='number of spatial Transformer layers in CTViT.')
    parser.add_argument('--temporal_depth', type=int, default=4, help='number of temporal Transformer layers in CTViT.')
    parser.add_argument('--dim_head', type=int, default=32, help='dimension of each attention head in CTViT.')
    parser.add_argument('--heads', type=int, default=8, help='number of attention heads in CTViT.')
    parser.add_argument('--codebook_size', type=int, default=8192, help='size of the VQ codebook.')
    parser.add_argument("--a_min", default=-1000.0, type=float, help="minimum input intensity value for ScaleIntensityRanged.")
    parser.add_argument("--a_max", default=1000.0, type=float, help="maximum input intensity value for ScaleIntensityRanged.")
    parser.add_argument("--b_min", default=-1.0, type=float, help="minimum output intensity value for ScaleIntensityRanged.")
    parser.add_argument("--b_max", default=1.0, type=float, help="maximum output intensity value for ScaleIntensityRanged.")
    parser.add_argument("--roi_x", default=480, type=int, help="ROI size in the x direction.")
    parser.add_argument("--roi_y", default=480, type=int, help="ROI size in the y direction.")
    parser.add_argument("--roi_z", default=240, type=int, help="ROI size in the z direction.")
    parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="probability of applying RandFlipd augmentation.")
    parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="probability of applying RandRotate90d augmentation.")
    parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="probability of applying RandScaleIntensityd augmentation.")
    parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="probability of applying RandShiftIntensityd augmentation.")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels.")
    parser.add_argument('--out_channels', type=int, default=5, help='number of output channels or labels.')

    # for Relational Memory
    parser.add_argument('--rm_num_slots', type=int, default=3, help='number of memory slots in relational memory.')
    parser.add_argument('--rm_num_heads', type=int, default=8, help='number of attention heads in relational memory.')
    parser.add_argument('--rm_d_model', type=int, default=2048, help='hidden dimension of relational memory.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='top5', help='sampling method used to generate reports.')
    parser.add_argument('--beam_size', type=int, default=1, help='beam size used in beam search decoding.')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature used during sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='number of sampled reports per image.')
    parser.add_argument('--group_size', type=int, default=1, help='group size used during sampling.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output log-softmax probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether to use decoding constraints.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to block repeated trigrams during decoding.')

    # Trainer settings
    parser.add_argument('--device_ids', type=int, default=[2], nargs="+", help='GPU device ids to be used.')
    parser.add_argument('--load_device1', type=int, default=0, help='GPU id for loading the VisualExtractor and meta_decoder blocks 1 to 3.')
    parser.add_argument('--load_device2', type=int, default=0, help='GPU id for loading meta_decoder blocks 4 to 6.')

    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='./train_results', help='directory used to save training results and checkpoints.')
    parser.add_argument('--seg_model_path', type=str, default='/Seg_Models_3DUNet/checkpoints', help='path to the pretrained 3DUNet segmentation model checkpoint.')
    parser.add_argument('--save_period', type=int, default=1, help='checkpoint saving period in epochs.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether the monitored metric should be minimized or maximized.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='metric used for monitoring model performance.')
    parser.add_argument('--early_stop', type=int, default=50, help='early stopping patience.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='type of optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='learning rate for the encoder-decoder or remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight decay coefficient.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='whether to use AMSGrad in the optimizer.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='type of learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=1, help='step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.8, help='decay factor of the learning rate scheduler.')
    parser.add_argument("--smooth_dr", default=1e-6, type=float, help="smoothing constant added to the Dice denominator to avoid NaN.")
    parser.add_argument("--smooth_nr", default=0.0, type=float, help="smoothing constant added to the Dice numerator.")

    # Others
    parser.add_argument('--xlsxfile', type=str, default="CT/all_paired.xlsx", help='path to the report xlsx file.')
    parser.add_argument("--trainxlsxfile", type=str, default="/CT/structured_train_paired.xlsx", help='path to the training xlsx file.')
    parser.add_argument("--validxlsxfile", type=str, default="/CT/structured_valid_paired.xlsx", help='path to the validation xlsx file.')
    parser.add_argument('--trainfolder', type=str, default="/CT/images", help='path to the training image folder.')
    parser.add_argument('--validfolder', type=str, default="/CT/images", help='path to the validation image folder.')
    parser.add_argument('--mask_paths', type=str, default='/CT/masks', help='path to the predicted masks from 3DUNet.')
    parser.add_argument('--phase_name', type=str, nargs="+", default=['C_register', 'A_register', 'P_register', 'V_register'], help='names of CT phase image folders.')
    parser.add_argument('--mask_name', type=str, nargs="+", default=['C_mask/pred', 'A_mask/pred', 'P_mask/pred', 'V_mask/pred'], help='names of mask folders for each CT phase.')

    parser.add_argument('--resume', type=str, help='path to checkpoint for resuming training.')
    parser.add_argument('--logger', type=str, default=None, help='logger name or logger configuration.')
    parser.add_argument('--momentum', type=float, default=0.5, help='momentum value.')
    parser.add_argument('--refine', type=bool, default=True, help='whether to refine the predicted text in the encoder-decoder.')
    parser.add_argument('--total_vocab_size', type=int, default=0, help='total vocabulary size; it will be updated after loading the tokenizer.')

    args = parser.parse_args()
    return args
