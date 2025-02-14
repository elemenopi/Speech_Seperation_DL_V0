# Updated Python script with the user's specified directory

import os

def search_word_in_files(directory, word):
    """
    Search for files that contain a specific word in a directory and its subdirectories.
    
    Parameters:
    directory (str): Path to the directory to scan.
    word (str): The word to search for in the file contents.
    
    Returns:
    list: List of files that contain the word.
    """
    matching_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    if word in f.read():
                        matching_files.append(file_path)
            except (UnicodeDecodeError, IOError) as e:
                # Skip files that can't be read or are binary files
                continue
    
    return matching_files

if __name__ == "__main__":
    directory = "/dsi/gannot-lab-UG/ilya_tomer/OUTPUTS/OUTPUTS_BASIC_TRAIN/outputDGX03"
    word = "bg"
    
    result = search_word_in_files(directory, word)
    
    if result:
        print(f"Files containing the word '{word}':")
        for file_path in result:
            print(file_path)
    else:
        print(f"No files found containing the word '{word}'.")

