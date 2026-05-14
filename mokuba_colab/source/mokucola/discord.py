import os
import shutil
import requests
import json

def to_discord(path,url):
    if os.path.isdir(path):
        shutil.make_archive('archive_shutil', format='zip', root_dir=path)
        f=open('archive_shutil.zip',"rb")
        file_bin=f.read()
        f.close()
        file = {
            "favicon" : ( 'archive_shutil.zip', file_bin),
        }
        response = requests.post(url, files=file)
        os.remove('archive_shutil.zip')
        del file_bin,file,response
    else:
        payload = {}
        payload["content"]=os.path.basename(path)
        f=open(path, "rb")
        list_path=path.split(".")
        file=[
            ("files[0]", (os.path.basename(path), f, "image/"+list_path[-1]))
        ]
        response = requests.post(url, data={"payload_json": json.dumps(payload)}, files=file)
        f.close()
        del file,payload,list_path,response
