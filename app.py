from flask import Flask, jsonify
import pandas as pd
import pickle
import numpy as np

import sys
sys.path.append('modules')
sys.path.append('data')

from Triail_JSON import Triail_JSON 
from func.get_feat import get_feat
from flask import Response
import io
import gdown
import os
from feature_names import feature_names

app = Flask(__name__)

data_path = os.path.join("data", "trials.pkl")
stats_path = os.path.join("data", "stats.pkl")
if not os.path.exists(data_path):
    url = "https://drive.google.com/uc?export=download&id=1oBT_fpumQCN3xdz0mm23QKd5vFCZAkXi"
    output = data_path
    gdown.download(url, output, quiet=False)

if not os.path.exists(stats_path):
    url = "https://drive.google.com/uc?export=download&id=1p4mb0R3Hn0bgwbW5VXuuLTnOM5cV0AuF"
    output = stats_path
    gdown.download(url, output, quiet=False)

if os.path.exists(data_path) and os.path.exists(stats_path):
    print("Results successfully downloaded")
    with open(data_path, 'rb') as f:
        subjects = pickle.load(f)  # or use pd.read_pickle('data.pkl')

    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
        for df in stats.values():
            df["xxx"] = np.zeros(len(df))
else:
    print("Results failed to download")



@app.route('/api/ids')
def get_ids():
    print("Fetching IDs")
    return {'ids': [subj.id for subj in subjects]}

@app.route('/api/subjects/results')
def get_results():
    res = stats.copy()
    for df in res.keys():
        res[df] = res[df].to_dict(orient="dict")
    return jsonify(res)

@app.route('/api/download_csv/<id>/<features>')
def download_csv(id, features):
    print(f"Downloading CSV for {id}")
    feats = features.split('-')

    for subj in subjects:
        if subj.id == id:
            
            jumps = subj.jumps[subj.trial_indices[0]:subj.trial_indices[1]+1]
            jump_dict = {
                feature_names[feat] : [get_feat(jump, feat, jIdx) for (jIdx, jump) in enumerate(jumps)]
                for feat in feats
            }
            df = pd.DataFrame(jump_dict)

            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            return Response(
                output,
                mimetype="text/csv",
                headers={"Content-Disposition": "attachment;filename={}_{:.2f}kg.csv".format(id, subj.mass)}
            )
    return {'error': 'Subject not found'}, 404

@app.route('/api/<id>/mass')
def get_mass(id):
    print(f"Fetching mass for {id}")
    subj = next((subj for subj in subjects if subj.id == id), None)
    print(f"Fetching mass for {subj.mass}")
    return "{:.0f}".format(subj.mass)


@app.route('/api/subjects/<id>/<features>')
def get_subject(id, features):
    print(f"Fetching {id}")
    feats = features.split('-')
    
    for subj in subjects:
        if subj.id == id:
            jumps = subj.jumps[subj.trial_indices[0]:subj.trial_indices[1]+1]
            jump_dict = {
                feat : [get_feat(jump, feat, jIdx) for (jIdx, jump) in enumerate(jumps)]
                for feat in feats
            }
            return jump_dict
    return {'error': 'Subject not found'}, 404







# if __name__ == '__main__':
#     app.run(debug=True)