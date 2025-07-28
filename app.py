from flask_cors import CORS
from flask import Flask, jsonify, request
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
CORS(app)

data_path = os.path.join("data", "trials.pkl")
stats_path = os.path.join("data", "stats.pkl")
meta_path = os.path.join("data", "meta.json")
if not os.path.exists(data_path):
    url = "https://drive.google.com/uc?export=download&id=1oBT_fpumQCN3xdz0mm23QKd5vFCZAkXi"
    output = data_path
    gdown.download(url, output, quiet=False)

if not os.path.exists(stats_path):
    url = "https://drive.google.com/uc?export=download&id=1p4mb0R3Hn0bgwbW5VXuuLTnOM5cV0AuF"
    output = stats_path
    gdown.download(url, output, quiet=False)

if not os.path.exists(meta_path):
    url = "https://drive.google.com/uc?export=download&id=1lHG313i73ep0C1XE7k1I-AuNvCnJFjOs"
    output = meta_path
    gdown.download(url, output, quiet=False)

if os.path.exists(data_path) and os.path.exists(stats_path) and os.path.exists(meta_path):
    print("Results successfully downloaded")
    with open(data_path, 'rb') as f:
        subjects = pickle.load(f)  # or use pd.read_pickle('data.pkl')

    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
        for df in stats.values():
            df["xxx"] = np.zeros(len(df))
    
    meta_df = pd.read_json(meta_path)
else:
    print("Results failed to download")



@app.route('/api/ids', methods=['POST'])
def get_ids():
    filters = request.get_json()
    # if not filters:
    #     return {'ids': [subj.id for subj in subjects]}
    
    out = meta_df.copy()
    for subj in meta_df.itertuples(index=True, name='Row'):
        if not set(subj.tags).intersection(filters):
            out = out.drop(subj.Index)
    return jsonify(out.to_dict(orient='records'))



@app.route('/api/subjects/results', methods=['POST'])
def get_results():
    res = stats.copy()
    ids = request.get_json()
    for df in res.keys():
        s_df = res[df]
        #print(s_df.index.get_level_values("id").isin(ids))
        #s_df = s_df[s_df.index.get_level_values("id").isin(ids)].reset_index().to_dict(orient="records")
        out_dict = {
            id : s_df.xs(id).to_dict(orient="index") for id in ids if id in s_df.index.get_level_values("id")
        }
        res[df] = out_dict
    return jsonify(res)

@app.route('/api/subjects/results/<feat>', methods=['POST'])
def get_agg_res(feat):
    print("Fetching results for feature:", feat)
    body = request.get_json()
    # Just get mean_start for now
    f_series = stats["mean_start"].loc[body["ids"], feat]
    out_dict = {
        "mean": f_series.mean(),
        "std": f_series.std(),
    }
    return jsonify(out_dict)

@app.route('/api/download_csv/', methods=['POST'])
def download_csv():
    data = request.get_json()
    print("Downloading for {id}-{pro}-{date} with features: {features} and data", data)
    id, features, pro, date = data["id"], data["features"], data["pro"], data["date"]
    feats = features.split('--')

    for subj in subjects:
        if subj.id == id and subj.pro == pro and subj.date == date:
            
            jumps = subj.jumps[subj.trial_indices[0]:subj.trial_indices[1]+1]
            jump_dict = {
                feat : [get_feat(jump, feat, jIdx) for (jIdx, jump) in enumerate(jumps)]
                for feat in feats
            }
            df = pd.DataFrame(jump_dict)

            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            return Response(
                output,
                mimetype="text/csv",
                headers={"Content-Disposition": "attachment;filename={}_{:.2f}kg.csv".format(id, float(subj.mass))}
            )
    return {'error': 'Subject not found'}, 404

@app.route('/api/<id>/mass')
def get_mass(id):
    print(f"Fetching mass for {id}")
    subj = next((subj for subj in subjects if subj.id == id), None)
    print(f"Fetching mass for {subj.mass}")
    return "{:.0f}".format(subj.mass)


@app.route('/api/<id>')
def get_tests(id):
    jump_dicts = {}
    for subj in subjects:
        if subj.id == id:
            print(subj.date, subj.pro)
            if subj.pro == "10":
                jumps = subj.jumps
                feats = stats["mean_trial"].keys()
            elif subj.pro == "30":
                jumps = subj.jumps[subj.trial_indices[0]:subj.trial_indices[1]+1]
                feats = stats["mean_start"].keys()
            else:
                print("Unknown protocol")
                continue
            feats = list(feats)
            feats.append("idx")
            feats.append("start_time")

            jump_dict = {
                feat : [get_feat(jump, feat, jIdx) for (jIdx, jump) in enumerate(jumps)]
                for feat in feats
            }
            jump_dict["id"] = subj.id
            jump_dict["mass"] = subj.mass
            jump_dict["date"] = int(subj.date)
            jump_dict["pro"] = subj.pro
            jump_dict["trial_indices"] = [int(x) for x in subj.trial_indices]
            jump_dict["stats"] = subj.stats
            if subj.date not in jump_dicts:
                jump_dicts[subj.date] = {}
            jump_dicts[subj.date][subj.pro] = jump_dict
    return jsonify(jump_dicts)

@app.route('/api/<id>/<date>')
def get_test(id, date):
    jump_dicts = {}
    for subj in subjects:
        if subj.id == id and subj.date == date:
            
            if subj.pro == "10":
                jumps = [jump for jIdx, jump in enumerate(subj.jumps) if jIdx in subj.trial_indices]
                feats = stats["mean_trial"].keys()
            elif subj.pro == "30":
                jumps = subj.jumps[subj.trial_indices[0]:subj.trial_indices[1]+1]
                feats = stats["mean_start"].keys()
            else:
                print("Unknown protocol")
                continue
            feats = list(feats)
            feats.append("idx")
            feats.append("start_time")

            jump_dict = {
                feat : [get_feat(jump, feat, jIdx) for (jIdx, jump) in enumerate(jumps)]
                for feat in feats
            }
            jump_dict["id"] = subj.id
            jump_dict["mass"] = subj.mass
            jump_dict["date"] = int(subj.date)
            jump_dict["pro"] = subj.pro
            jump_dict["trial_indices"] = [int(x) for x in subj.trial_indices]
            jump_dict["stats"] = subj.stats
            jump_dicts[subj.date]= {
                subj.pro: jump_dict
            }
    return jsonify(jump_dicts)








# if __name__ == '__main__':
#     app.run(debug=True)