import pandas as pd, soundfile as sf, os
for lang in ["english","hindi"]:
    d=f"eot_data/{lang}"
    df=pd.read_csv(f"{d}/labels.csv")
    print(f"\n===== {lang.upper()} =====")
    print("rows:",len(df)," turns:",df.turn_id.nunique())
    print(df.label.value_counts().to_string())
    df["dur"]=df.pause_end-df.pause_start
    print("hold dur: mean %.2f med %.2f p90 %.2f max %.2f"%(df[df.label=='hold'].dur.mean(),df[df.label=='hold'].dur.median(),df[df.label=='hold'].dur.quantile(.9),df[df.label=='hold'].dur.max()))
    print("eot  dur: mean %.2f med %.2f min %.2f"%(df[df.label=='eot'].dur.mean(),df[df.label=='eot'].dur.median(),df[df.label=='eot'].dur.min()))
    last_is_eot=total=multi=0
    for tid,g in df.groupby("turn_id"):
        total+=1; g=g.sort_values("pause_index")
        if (g.label=='eot').sum()!=1: multi+=1
        if g.iloc[-1].label=='eot': last_is_eot+=1
    print(f"last-pause-is-eot: {last_is_eot}/{total}; turns without exactly 1 eot: {multi}")
    print("pauses/turn: mean %.2f min %d max %d"%(df.groupby('turn_id').size().mean(),df.groupby('turn_id').size().min(),df.groupby('turn_id').size().max()))
    p=os.path.join(d,df.iloc[0].audio_file); x,sr=sf.read(p); print("sr",sr,"ch",x.ndim)
