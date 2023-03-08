import os, sys

def process_raw_data(dataset_name,dname2paths):
    readf = dname2paths[dataset_name]
    dname = "/".join(readf.split("/")[0:-1])
    writef = os.path.join(dname, "data.txt")
    print(f"Start preprocessing data: {dataset_name}")
    if dataset_name == "algebra2005":
        from .algebra2005_preprocess import read_data_from_csv
    elif dataset_name == "bridge2algebra2006":
        from .bridge2algebra2006_preprocess import read_data_from_csv
    elif dataset_name == "nips_task34":
        from .nips_task34_preprocess import read_data_from_csv
    
    if dataset_name != "nips_task34":
        read_data_from_csv(readf, writef)
    else:
        metap = os.path.join(dname, "metadata")
        read_data_from_csv(readf, metap, "task_3_4", writef)
     
    return dname,writef
