import os
import shutil
import requests
import json
import time

def to_discord(path,url):
    payload = {}
    payload["content"]=os.path.basename(path)
    f=open(path, "rb")
    list_path=path.split(".")
    file=[("files[0]", (os.path.basename(path), f, "image/"+list_path[-1]))]
    response = requests.post(url, data={"payload_json": json.dumps(payload)}, files=file)
    f.close()
    del file,payload,list_path,response

def zip_to_discord(path,url):
    shutil.make_archive('archive_shutil', format='zip', root_dir=path)
    f=open('archive_shutil.zip',"rb")
    file_bin=f.read()
    f.close()
    files_qiita = {
	    "favicon" : ( 'archive_shutil.zip', file_bin),
    }
    response = requests.post(url, files=files_qiita)
    if response.status_code == 200 or response.status_code == 204:
        os.remove('archive_shutil.zip')
    else:
        ut = str(int(time.time()))
        os.rename('archive_shutil.zip','error'+ut+'.zip')
        del ut
    del file_bin,files_qiita,response