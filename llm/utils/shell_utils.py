import os
import shutil
import glob
from pathlib import Path
import json
import argparse

def rm_file(path, regex=False):
    if regex:
        file_list = glob.glob(path, recursive=True)
    else:
        file_list = [path]
    for file in file_list:
        path = Path(file)
        if os.path.exists(path):
            print(f"Removing file : {path}")
            os.remove(path)

def rm_dir(path):
    path = Path(path)
    if os.path.exists(path):
        print(f"Deleting directory : {path}")
        shutil.rmtree(path)

def mv_file(src, dst):
    shutil.move(src, dst)

def load_item_from_json(key, item):
    if key:
        dirpath = os.path.dirname(__file__)
        f=open(os.path.join(dirpath, '../models/models.json'))
        obj=json.load(f)
        if (item and key in obj):
            val = obj[key].get(item, '#')
            print(val)
            return val

        else:
            val = obj.get(key, '#')
            print(val)
            return val

def safe_list_get(l, idx, default):
    try:
        return l[idx].strip()
    except IndexError:
        return default

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='shell script')
    parser.add_argument('--function', type=str, default="", required= True,
                        metavar='f')

    parser.add_argument('--params', type=str, default="", required= True,
                        metavar='p')

    args = parser.parse_args()
    if args.function == 'load_item_from_json':
        params = args.params.split(",")
        load_item_from_json(safe_list_get(params, 0, ""), safe_list_get(params, 1, ""))

    else:
        print("Not supported")