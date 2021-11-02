import argparse

def parse_opts():

    parser = argparse.ArgumentParser()
    
    #Paths
    parser.add_argument('--save_dir', default='./output/', type=str, help='Where to save training outputs.')
    parser.add_argument("--checkpoint_model", type=str, help="Optional path to checkpoint model for testing")
    
    # Dataset
    parser.add_argument('--dataset', default='CelebA', type=str, help='Dataset string (bolly | CelebA | CelebBolly | CelebPin)')
    parser.add_argument('--dataset_path', default='./data/CelebA/', type=str, help='Path to location of dataset images')
    parser.add_argument('--num_classes', default=2, type=int, help= 'Number of classes (CelebA: 2 | CelebBolly: 100 | CelebPin: 105)')

    # Preprocessing pipeline
    parser.add_argument('--spatial_size_width', default= 178, type=int, help='Height and width and hight of inputs Inception 299| efficentNet 244')
    parser.add_argument('--spatial_size_hight', default= 218, type=int, help='Height and width and hight of inputs Inception 299| efficentNet 244')

    # Models (general)
    parser.add_argument('--model', default='IncptionV3', type=str, help='( inceptionV3 | efficientNet | resnext | densenet)')

    # Optimization
    parser.add_argument('--early_stopping_patience', default=10, type=int, help='Early stopping patience (number of epochs)')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch Size')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs to train for')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate (divided by 10 while training by lr-scheduler)')    
       
    return parser.parse_args()