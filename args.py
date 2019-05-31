import argparse


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_checkpoint', type=str,default = './saved_models/quick_save_checkpoint_ep138.pth.tar', help='Detection checkpoint save path')
    parser.add_argument('--recognition_checkpoint', type=str,default ='./saved_models/iter_249000.pth', help='Recognition checkpoint save path')
    parser.add_argument('--input_img', type=str,default ='img42.jpg', help='Image test')
    
    parser.add_argument('--image_folder', required=False,
                        help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int,
                        help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int,
                        default=192, help='input batch size')
    parser.add_argument('--saved_model', required=False,
                        help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int,
                        default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32,
                        help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100,
                        help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true',
                        help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true',
                        help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default='None',
                        required=False, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='SimpleConv', required=False,
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='Transformer',
                        required=False, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='None',
                        required=False, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20,
                        help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='the size of the LSTM hidden state')
    """ Transformer """
    parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=1024)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers_enc', type=int, default=6)
    parser.add_argument('-n_layers_dec', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=16000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-use_scheduled_optim', action='store_true')

    return parser
