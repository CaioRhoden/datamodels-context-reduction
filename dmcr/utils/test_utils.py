import os
import shutil
def clean_temp_folders():

    for item in os.listdir(os.getcwd()):
        if item.startswith('tmp') and os.path.isdir(item):
            shutil.rmtree(item)