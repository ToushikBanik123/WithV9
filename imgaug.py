import os
import cv2
import argparse


def parse_args():
    """Parses command-line arguments for input and output directories."""
    parser = argparse.ArgumentParser(description='Data Augmentation')
    parser.add_argument('--input_img', type=str, default=None, required=True,
                        help='Path to the directory containing input images.')
    parser.add_argument('--output_img', type=str, default=None, required=True,
                        help='Path to the directory for storing output images.')
    parser.add_argument('--input_label', type=str, default=None,
                        help='Path to the directory containing input labels (optional).')
    parser.add_argument('--output_label', type=str, default=None,
                        help='Path to the directory for storing output labels (optional).')
    args = parser.add_argument_group('Debugging options')
    args.add_argument('--verbose', action='store_true', default=False,
                      help='Print additional information for debugging purposes.')
    return parser.parse_args()


def is_hidden_file(file_path):
    """Check if a file is hidden."""
    return os.path.basename(file_path).startswith('.')


if __name__ == '__main__':
    args = parse_args()

    # Validate and handle missing required arguments
    if not args.input_img:
        raise ValueError("Please provide the path to the input image directory using --input_img.")
    if not args.output_img:
        raise ValueError("Please provide the path to the output image directory using --output_img.")

    input_imgdir = args.input_img
    output_imgdir = args.output_img
    input_labeldir = args.input_label
    output_labeldir = args.output_label

    # Create output directories if they don't exist
    if not os.path.isdir(output_imgdir):
        os.makedirs(output_imgdir)  # Use makedirs() for nested directory creation
    if input_labeldir and not os.path.isdir(output_labeldir):
        os.makedirs(output_labeldir)

    for root, dirs, files in os.walk(input_imgdir):
        for name in files:
            if is_hidden_file(name):
                continue  # Skip hidden files

            img_name = os.path.join(name)
            img_path = os.path.join(root, name)  # Use full path for clarity

            if args.verbose:
                print(f"Processing image: {img_path}")

            # Check if it's a regular file before processing
            if os.path.isfile(img_path):
                try:
                    pic = cv2.imread(img_path)
                    if pic is None:
                        print(f"Error: Could not read image {img_path}. Skipping.")
                        continue  # Skip to next image if reading fails
                except Exception as e:
                    print(f"Error reading image {img_path}: {e}")
                    continue  # Skip to next image on error

            contrast = 1.2
            brightness = 30
            pic_turn = cv2.addWeighted(pic, contrast, pic, 0, brightness)

            # Print image shape for debugging
            if args.verbose:
                print(f"Image shape: {pic.shape}")

            cv2.imwrite(os.path.join(output_imgdir, img_name), pic)
            cv2.imwrite(os.path.join(output_imgdir, 'aug_' + img_name), pic_turn)
            print(f"Augmentation successful for image: {img_name}")  # Feedback

    # Handle label processing (if provided)
    if input_labeldir and output_labeldir:
        for root, dirs, files in os.walk(input_labeldir):
            for name in files:
                if is_hidden_file(name):
                    continue  # Skip hidden files

                label_name = os.path.join(name)
                label_path = os.path.join(root, name)
                output_label_path = os.path.join(output_labeldir, 'aug_' + label_name)

                # Implement label copying logic here (assuming simple copying)
                try:
                    with open(label_path, 'r', encoding='utf-8') as f:  # Specify encoding explicitly
                        lines = f.readlines()
                    with open(output_label_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                    print(f"Label processing successful for label: {label_name}")  # Feedback
                except UnicodeDecodeError:
                    print(f"Warning: Potential encoding issue with label {label_path}.")
