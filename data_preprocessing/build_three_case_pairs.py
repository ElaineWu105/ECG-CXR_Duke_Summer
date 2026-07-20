#!/usr/bin/env python3
"""Three pairing cases following 7.1 Criterion-1 at CXR-study level."""
from __future__ import annotations
import argparse, csv, json
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

OFFSETS=(0,2,4,6,8,10,12); VIEW_RANK={"PA":0,"AP":1}
def hours(x): return x.toordinal()*24+x.hour+x.minute/60+x.second/3600
def load_ids(p): return set(map(str,json.load(p.open())))

def load_ecgs(path,valid):
    out=defaultdict(list)
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            eid=r["record_id"].strip()
            if eid in valid: out[r["subject_id"].strip()].append((datetime.fromisoformat(r["ecg_time"]),eid))
    for rows in out.values(): rows.sort()
    return out

def load_times(path):
    out={}
    with path.open(newline="") as f:
        for r in csv.DictReader(f): out[(r["subject_id"].strip(),r["study_id"].strip())]=datetime.fromisoformat(r["cxr_time"])
    return out

def load_cxrs(path,times,valid):
    # One image per study: PA first, otherwise AP. Studies without PA/AP or a
    # saved embedding cannot be used for embedding-based training.
    best={}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            cid,view=r["dicom_id"].strip(),r.get("ViewPosition","").strip().upper()
            pid,sid=r["subject_id"].strip(),r["study_id"].strip(); t2=times.get((pid,sid))
            if cid not in valid or view not in VIEW_RANK or t2 is None: continue
            item=(VIEW_RANK[view],cid,pid,sid,view,t2)
            if (pid,sid) not in best or item[:2]<best[(pid,sid)][:2]: best[(pid,sid)]=item
    out=defaultdict(list)
    for _,cid,pid,sid,view,t2 in best.values(): out[pid].append((t2,sid,cid,view))
    for rows in out.values(): rows.sort()
    return out

def select_window(rows,t2,n,width):
    ts=[x[0] for x in rows]; lo=t2-timedelta(hours=n+width); hi=t2-timedelta(hours=n)
    return rows[bisect_left(ts,lo):bisect_right(ts,hi)]

def main():
    p=argparse.ArgumentParser()
    for name in ("cxr_times","ecg_times","cxr_metadata","cxr_ids","ecg_ids","output_dir"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--window_hours",type=float,default=12); a=p.parse_args()
    ecgs=load_ecgs(a.ecg_times,load_ids(a.ecg_ids)); cxrs=load_cxrs(a.cxr_metadata,load_times(a.cxr_times),load_ids(a.cxr_ids))
    a.output_dir.mkdir(parents=True,exist_ok=True); summary=[]
    for n in OFFSETS:
        single,sequence,nearest=[],[],[]; patients=set(); studies=set()
        for pid,candidates in cxrs.items():
            patient_ecgs=ecgs.get(pid,[])
            if not patient_ecgs: continue
            # Unlike the previous version, traverse every study for a patient.
            for t2,sid,cid,view in candidates:
                chosen=select_window(patient_ecgs,t2,n,a.window_hours)
                if not chosen: continue
                patients.add(pid); studies.add((pid,sid)); t2h=hours(t2)
                eids=[x[1] for x in chosen]; ets=[hours(x[0]) for x in chosen]
                base={"patient_id":int(pid),"study_id":sid,"view":view,"window_offset_h":n,
                      "window_start_h":t2h-n-a.window_hours,"window_end_h":t2h-n}
                single.extend({**base,"ecg_id":eid,"ecg_time_h":et,"cxr_id":cid,
                               "cxr_time_h":t2h,"delta_h":t2h-et} for eid,et in zip(eids,ets))
                sequence.append({**base,"t2_h":t2h,"cxr_t2":cid,"ecg_ids":eids,
                                 "ecg_times_h":ets,"delta_h":t2h-ets[-1]})
                nearest.append({**base,"ecg_id":eids[-1],"ecg_time_h":ets[-1],
                                "cxr_id":cid,"cxr_time_h":t2h,"delta_h":t2h-ets[-1]})
        stats={"n":n,"window":f"[t2-{n}-12h,t2-{n}] inclusive",
               "patients_with_at_least_one_pair":len(patients),
               "cxr_studies_with_at_least_one_ecg":len(studies),
               "single_pairs":len(single),"sequence_pairs":len(sequence),"nearest_pairs":len(nearest)}
        for name,rows in (("single",single),("seq",sequence),("nearest",nearest)):
            path=a.output_dir/f"{name}_n{n}.json"; json.dump({"pairs":rows,"stats":stats},path.open("w")); print(path,len(rows))
        summary.append(stats)
    json.dump({"selection":"all studies; PA per study otherwise AP; 7.1 Criterion 1","stats":summary},
              (a.output_dir/"pairs_summary.json").open("w"),indent=2)

if __name__=="__main__": main()
