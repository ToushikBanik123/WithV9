import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Data Postprocess')
    parser.add_argument('--model', type=str, default=None, help='load the model')
    parser.add_argument('--train_dir', type=str, default=None, help='the directory containing training data')
    parser.add_argument('--val_dir', type=str, default=None, help='the directory containing validation data')
    parser.add_argument('--test_dir', type=str, default=None, help='the directory containing testing data')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model = YOLO(args.model)
    
    # Train the model using the training data
    model.train(data=args.train_dir)
    
    # Evaluate the model on the validation data
    validation_results = model.evaluate(data=args.val_dir)
    
    # Optionally, evaluate the model on the testing data
    if args.test_dir:
        test_results = model.evaluate(data=args.test_dir)

if __name__ == '__main__':
    main()

