import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='ML Framework for Genomic Data.')
    parser.add_argument('--sample', type=int, default=1000,
                    help='Number of Samples for the Data Matrix')
    parser.add_argument('--feature', type=int, default=200000,
                    help='Number of Starting Features for the Data Matrix')
    parser.add_argument('--balance', type=float, default=0.5,
                    help='Balance Ratio: .5 by default (balanced). Ratio > .5 more samples in control cohort, Ratio < .5 more samples in disease cohort')
    parser.add_argument('--spikes', type=int, default=10,
                    help='Number of Spike Features')
    parser.add_argument('--combination', type=int, default=1,
                        help='Classifier Combination: 0 - Random Forest, SVM, 1 - Random Forest, Gradient Boosting, 2 - Gradient Boosting, SVM')
    parser.add_argument('--tuning', type=int, default=1,
                            help='Hyperparemeter Tuning enabling')
    parser.add_argument('--test_random_data', type=int, default=0)
    parser.add_argument('--test_type', type=int, default=0)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--my_functions', type=int, default=0)
    parser.add_argument('--subtract', type=int, default=0)
    parser.add_argument('--contingency', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.70)
    parser.add_argument('--synthetic', action='store_true',
                    help='Generate Synthetic Dataset. default=true')
    parser.add_argument('--robustness', type=int, default=0)
    parser.add_argument('--final_testing', type=int, default=0)
    #parser.add_argument('dataset',
    #                help='Genomic Dataset in with file extension VCF used for the Framework')

    opt = parser.parse_args()
    return opt
