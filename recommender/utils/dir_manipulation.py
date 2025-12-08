import os
import shutil

def delete_file(path):
    if os.path.exists(path):
        os.remove(path)

def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)  # delete folder
    os.makedirs(path)        # recreate empty folder