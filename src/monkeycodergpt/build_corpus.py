import os
import random
import hashlib


def is_comment(line):
    '''check if a line is a comment'''
    line = line.strip()
    if line.startswith('#') or line.startswith("'''") or line.startswith('"""'):
        return True
    return False


def extract_non_comments(source_directories, target_directory):
    # Ensure the target directory exists
    os.makedirs(target_directory, exist_ok=True)
    
    for source_directory in source_directories:
        for root, dirs, files in os.walk(source_directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    
                    # Create a unique filename to avoid clashes
                    unique_prefix = hashlib.md5(root.encode()).hexdigest()[:8]
                    target_file_name = f"{unique_prefix}_{file.replace('.py', '.txt')}"
                    target_file_path = os.path.join(target_directory, target_file_name)
                    
                    with open(file_path, 'r') as source_file, open(target_file_path, 'w') as target_file:
                        non_comments = []
                        comment_block = False

                        for line in source_file:
                            if line.count("'''") == 1 or line.count('"""') == 1:
                                comment_block = not comment_block
                                continue
                            # If it's not a comment or part of a comment block, save it
                            if not is_comment(line) and not comment_block:
                                non_comments.append(line)
                            # Write non-comment lines to a target .txt file
                        target_file.writelines(non_comments)


def combine_files(directory, output_file, sample=False, num_files_to_sample=100, seed=111, start_token="<START>", end_token="<END>"):
    """
    Combine content from a specified number of text files in a directory into one file, 
    with start and end tokens between contents from each file.

    :param directory: Path to the directory containing text files.
    :param output_file: Name of the output file to create.
    :param num_files_to_sample: Number of files to sample and combine.
    :param start_token: The start token to be added before each file's content.
    :param end_token: The end token to be added after each file's content.
    """
    
    # List all text files in the directory
    all_files = [f for f in os.listdir(directory) if f.endswith('.txt')]

    if sample:
        # Sample the specified number of files
        random.seed(seed)
        files = random.sample(all_files, min(num_files_to_sample, len(all_files)))
    else:
        files = all_files

    # Shuffle the files to ensure random order
    random.shuffle(files)

    # Start combining the sampled files
    with open(output_file, 'w') as outfile:
        for filename in files:
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as infile:
                # outfile.write(start_token + '\n')
                content = infile.read()
                content_with_tabs = content.replace('    ', '\t')
                outfile.write(content_with_tabs + '\n')
                # outfile.write(end_token + '\n\n')

    print(f"Combined file created as '{output_file}' with contents from {len(files)} files.")
  

# Define the path to the local repository (change this to the actual path of your local repo)
    
# source_directories = ['/path/to/source1', '/path/to/source2', ...]
source_directories = [
    os.path.expanduser('~/git/my-projects/corpus/pytorch-seq2seq/seq2seq'),
    os.path.expanduser('~/git/my-projects/corpus/transformers/examples/pytorch'),
    os.path.expanduser('~/git/my-projects/corpus/examples'),
    os.path.expanduser('~/git/my-projects/corpus/BERT-pytorch/bert_pytorch'),
    os.path.expanduser('~/git/my-projects/corpus/allennlp/allennlp')
]

# target_directory = '/path/to/target'
target_directory = '../../dataset/raw/'

# Create the target directory if it doesn't exist
os.makedirs(target_directory, exist_ok=True)

# Call the function to start extracting non-comment lines
extract_non_comments(source_directories, target_directory)

# Call the function to combine files
combine_files('../../dataset/raw/', '../../data/sample_scripts.txt')
print(source_directories)
