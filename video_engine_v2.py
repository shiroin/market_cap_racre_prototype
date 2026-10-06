import os
import shutil
import subprocess
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb


def _rss_mb():
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return None


def _log(message):
    rss = _rss_mb(); suffix = f" | rss={rss:.1f}MB" if rss is not None else ""; print(f"[video-render] {message}{suffix}", flush=True)


def ease_in_out(t):
    t=np.clip(float(t),0.0,1.0); return 3*t*t-2*t*t*t


def fade_window(p,start=0.0,end=0.18):
    return 1.0 if end<=start else ease_in_out(np.clip((p-start)/(end-start),0,1))


def _style_axis(ax,bg,text,grid):
    ax.set_facecolor(bg); ax.tick_params(colors=text,labelsize=9,length=0,pad=7)
    for s in ax.spines.values(): s.set_visible(False)
    ax.grid(axis="y",color=grid,linewidth=.8,alpha=.55); ax.set_axisbelow(True)


def _spread_label_positions(values,ymin,ymax,min_gap_ratio=.055):
    if not values:return []
    gap=max((ymax-ymin)*min_gap_ratio,1e-9); order=np.argsort(values); placed=np.array(values,dtype=float); low=ymin+(ymax-ymin)*.025; high=ymax-(ymax-ymin)*.025; last=low-gap
    for idx in order: placed[idx]=max(values[idx],last+gap,low); last=placed[idx]
    overflow=placed[order[-1]]-high
    if overflow>0: placed-=overflow
    for j in range(len(order)-2,-1,-1):
        a,b=order[j],order[j+1]; placed[a]=min(placed[a],placed[b]-gap)
    under=low-placed[order[0]]
    if under>0: placed+=under
    return placed.tolist()


def _figure_spec(ratio,quality="preview"):
    sizes={"9:16":(3.6,6.4),"1:1":(4.,4.),"16:9":(6.4,3.6)}; dpi={"preview":100,"standard":200,"high":150}.get(quality,200); return sizes[ratio],dpi


def _output_size(ratio,quality):
    return None if quality!="high" else {"9:16":(1080,1920),"1:1":(1080,1080),"16:9":(1920,1080)}[ratio]


def _bar_reveal(raw_p,count,mode):
    if count<=0:return np.zeros(0,dtype=float)
    p=ease_in_out(np.clip((raw_p-.10)/.72,0,1))
    if mode=="一気に表示":return np.full(count,p,dtype=float)
    timeline=p*count; factors=np.clip(timeline-np.arange(count,dtype=float),0,1); partial=(factors>0)&(factors<1); factors[partial]=[ease_in_out(v) for v in factors[partial]]
    return factors[::-1] if mode=="右→左" else factors


def _active_bar_index(reveal,mode):
    if len(reveal)==0:return 0
    visible=np.flatnonzero(reveal>1e-6)
    if len(visible): return int(visible[0] if mode=="右→左" else visible[-1])
    return len(reveal)-1 if mode=="右→左" else 0


def _contrast_text(hex_color):
    r,g,b=to_rgb(hex_color); return "#172033" if .2126*r+.7152*g+.0722*b>.62 else "#FFFFFF"


def _fmt_value(value,decimals=0):return f"{value:,.{int(decimals)}f}"


def _series_label(company,value,scene,decimals):
    return f"{company}  {_fmt_value(value,decimals)}{scene.get('unit','')}" if scene.get("latest_values",True) else company


def _prepare_scene(df,scene):
    metric=scene["metric"]; work=df[["date","company",metric]].copy(); work[metric]=pd.to_numeric(work[metric],errors="coerce"); work=work.dropna(); dates=list(dict.fromkeys(work["date"].astype(str))); companies=list(dict.fromkeys(work["company"].astype(str))); pivot=work.pivot_table(index="date",columns="company",values=metric,aggfunc="sum").reindex(dates).fillna(0); return dates,companies,pivot


def _fixed_stack_lanes(shown,companies,ymax,gap):
    """Stable label lanes based on the final stack. They never jump between frames."""
    if shown.empty:return []
    final=shown.iloc[-1]; running=0.0; anchors=[]
    for company in companies:
        v=float(final.get(company,0)); anchors.append(running+v/2); running+=v
    return _spread_label_positions(anchors,0,ymax,gap)


def _fixed_group_lanes(pivot,companies,ymax,gap):
    """Stable company lanes based on final values, independent of animation progress."""
    if pivot.empty:return []
    final=[float(pivot[c].iloc[-1]) if c in pivot else 0.0 for c in companies]
    return _spread_label_positions(final,0,ymax,gap)


def _draw_scene_on(fig,ax,dates,companies,pivot,scene,bg,text,grid,cmap,progress):
    ax.clear(); fig.texts.clear(); _style_axis(ax,bg,text,grid)
    raw_p=np.clip(float(progress),0,1); title_alpha=fade_window(raw_p,0,.16); subtitle_alpha=fade_window(raw_p,.06,.24)*.84; chart_alpha=fade_window(raw_p,.10,.28); label_alpha=fade_window(raw_p,.76,.96)
    fig.text(.075,.93,scene["title"],color=text,fontsize=scene.get("title_size",22),fontweight="bold",ha="left",alpha=title_alpha)
    if scene.get("subtitle"):fig.text(.075,.885,scene["subtitle"],color=text,fontsize=max(8,scene.get("title_size",22)-7),ha="left",alpha=subtitle_alpha)
    if scene.get("source"):fig.text(.075,.052,f"出典: {scene['source']}",color=text,fontsize=7,ha="left",alpha=.58*max(chart_alpha,.35))
    x=np.arange(len(dates),dtype=float); chart=scene["chart"]; ymax=100. if chart=="100%積み上げ" else float(max(1.,pivot.sum(axis=1).max() if chart=="積み上げ棒" else pivot.to_numpy().max()))*1.22; ax.set_ylim(0,ymax)
    decimals=int(scene.get("value_decimals",0)); label_mode=scene.get("data_labels","自動"); reveal_mode=scene.get("bar_animation","左→右"); live_size=scene.get("end_label_size",8); gap=scene.get("label_gap",.055)

    if chart in ("積み上げ棒","100%積み上げ"):
        shown=pivot.copy()
        if chart=="100%積み上げ":shown=shown.div(shown.sum(axis=1).replace(0,np.nan),axis=0).fillna(0)*100
        reveal=_bar_reveal(raw_p,len(dates),reveal_mode); bottom=np.zeros(len(dates)); segments=[]
        for company in companies:
            raw=shown[company].to_numpy(float) if company in shown else np.zeros(len(dates)); vals=raw*reveal; centers=bottom+vals/2; ax.bar(x,vals,bottom=bottom,color=cmap[company],width=.68,label=company,alpha=chart_alpha); segments.append((company,raw,vals,centers)); bottom+=vals

        # Reference-video style: labels live in fixed lanes at the far right.
        # Only the short leader line moves; the text itself does not wobble.
        if label_mode!="なし" and len(dates):
            j=_active_bar_index(reveal,reveal_mode); factor=float(reveal[j]); lanes=_fixed_stack_lanes(shown,companies,ymax,gap); label_x=max(1,len(dates)-1)+.72; running=0.0
            for (company,raw,vals,centers),label_y in zip(segments,lanes):
                current=float(raw[j]*factor); anchor=running+current/2; running+=current
                ax.plot([x[j]+.35,label_x-.10],[anchor,label_y],color=cmap[company],linewidth=.9,alpha=.72,zorder=8)
                ax.text(label_x,label_y,_series_label(company,current,scene,decimals),color=cmap[company],fontsize=live_size,va="center",ha="left",fontweight="bold",alpha=1.0,zorder=9)

        if label_mode!="なし" and raw_p>=.45:
            for company,raw,vals,centers in segments:
                for j,val in enumerate(vals):
                    if reveal[j]>=.96 and (label_mode=="すべて" or (label_mode=="自動" and val>=ymax*.065)):
                        ax.text(x[j],centers[j],_fmt_value(val,decimals),ha="center",va="center",fontsize=scene.get("data_label_size",7),fontweight="bold",color=_contrast_text(cmap[company]),alpha=label_alpha,zorder=7)
            for j,total in enumerate(bottom):
                if reveal[j]>=.96 and label_mode in ("自動","すべて","合計のみ"):ax.text(x[j],total+ymax*.016,_fmt_value(total,decimals),color=text,ha="center",va="bottom",fontsize=scene.get("data_label_size",7),fontweight="bold",alpha=label_alpha,clip_on=False)
        ax.set_xlim(-.45,max(1,len(dates)-1)+2.6)

    elif chart=="棒グラフ":
        width=.76/max(1,len(companies)); reveal=_bar_reveal(raw_p,len(dates),reveal_mode); series=[]
        for i,company in enumerate(companies):
            raw=pivot[company].to_numpy(float) if company in pivot else np.zeros(len(dates)); vals=raw*reveal; xpos=x+(i-(len(companies)-1)/2)*width; ax.bar(xpos,vals,width=width,color=cmap[company],label=company,alpha=chart_alpha); series.append((company,raw,vals,xpos))
            if label_mode!="なし" and raw_p>=.45:
                for j,val in enumerate(vals):
                    if reveal[j]>=.96 and val>0 and label_mode in ("自動","すべて","合計のみ"):ax.text(xpos[j],val+ymax*.012,_fmt_value(val,decimals),color=text,ha="center",va="bottom",fontsize=scene.get("data_label_size",7),fontweight="bold",alpha=label_alpha,clip_on=False)

        # Fixed right-side lanes keep grouped-bar labels visually anchored.
        if label_mode!="なし" and len(dates):
            j=_active_bar_index(reveal,reveal_mode); factor=float(reveal[j]); lanes=_fixed_group_lanes(pivot,companies,ymax,gap); label_x=max(1,len(dates)-1)+.82
            for (company,raw,vals,xpos),label_y in zip(series,lanes):
                current=float(raw[j]*factor); anchor=current; bar_x=float(xpos[j])
                ax.plot([bar_x+width/2,label_x-.10],[anchor,label_y],color=cmap[company],linewidth=.9,alpha=.72,zorder=8)
                ax.text(label_x,label_y,_series_label(company,current,scene,decimals),color=cmap[company],fontsize=live_size,va="center",ha="left",fontweight="bold",alpha=1.0,zorder=9)
        ax.set_xlim(-.45,max(1,len(dates)-1)+2.6)

    else:
        draw_p=ease_in_out(np.clip((raw_p-.10)/.72,0,1)); pos=draw_p*max(0,len(dates)-1); whole=int(np.floor(pos)); frac=pos-whole; endpoints=[]
        for company in companies:
            vals=pivot[company].to_numpy(float) if company in pivot else np.zeros(len(dates))
            if len(dates)==1:end_x,end_y=0,vals[0]; ax.scatter([0],[end_y],color=cmap[company],s=22,alpha=chart_alpha)
            else:
                xs=list(x[:whole+1]); ys=list(vals[:whole+1])
                if whole<len(dates)-1:end_x=x[whole]+frac; end_y=vals[whole]+(vals[whole+1]-vals[whole])*frac; xs.append(end_x); ys.append(end_y)
                else:end_x,end_y=x[-1],vals[-1]
                ax.plot(xs,ys,color=cmap[company],linewidth=2.8,solid_capstyle="round",alpha=chart_alpha); ax.scatter([end_x],[end_y],color=cmap[company],s=18,zorder=4,alpha=chart_alpha)
            endpoints.append((company,end_x,end_y))
        if scene.get("end_labels",True):
            adjusted=_spread_label_positions([v for _,_,v in endpoints],0,ymax,gap)
            for (company,end_x,actual_y),label_y in zip(endpoints,adjusted):
                ax.plot([end_x+.04,end_x+.22],[actual_y,label_y],color=cmap[company],linewidth=.9,alpha=.75); ax.text(end_x+.27,label_y,_series_label(company,actual_y,scene,decimals),color=cmap[company],fontsize=live_size,va="center",fontweight="bold",alpha=1.0,bbox=dict(boxstyle="round,pad=.18",facecolor=bg,edgecolor="none",alpha=.92))
        ax.set_xlim(-.2,max(1,len(dates)-1)+2.45)

    ax.set_xticks(x); ax.set_xticklabels(dates,color=text,alpha=chart_alpha); ax.set_ylabel(scene.get("unit",""),color=text,fontsize=9,alpha=chart_alpha)
    if scene.get("legend",False):
        leg=ax.legend(frameon=False,fontsize=8,ncol=2,loc="upper left")
        for t in leg.get_texts():t.set_color(text); t.set_alpha(chart_alpha)


def _make_canvas(ratio,bg,quality):
    size,dpi=_figure_spec(ratio,quality); fig=plt.figure(figsize=size,dpi=dpi); fig.patch.set_facecolor(bg); ax=fig.add_axes([.10,.15,.72,.66] if ratio=="9:16" else [.09,.16,.75,.65]); return fig,ax


def render_story_frame(df,scene,ratio,bg,text,grid,cmap,progress=1.0,quality="preview"):
    dates,companies,pivot=_prepare_scene(df,scene); fig,ax=_make_canvas(ratio,bg,quality); _draw_scene_on(fig,ax,dates,companies,pivot,scene,bg,text,grid,cmap,progress); return fig


def save_scene_v2(df,scene,path,ratio,fps,bg,text,grid,cmap,quality="standard"):
    started=time.monotonic(); frames=max(2,int(scene.get("duration",2.5)*fps)); hold=max(0,int(scene.get("hold",1.0)*fps)); dates,companies,pivot=_prepare_scene(df,scene); fig,ax=_make_canvas(ratio,bg,quality)
    _draw_scene_on(fig,ax,dates,companies,pivot,scene,bg,text,grid,cmap,1/max(2,frames)); fig.canvas.draw(); width,height=fig.canvas.get_width_height(); _log(f"scene start title={scene.get('title','')} quality={quality} canvas={width}x{height} frames={frames+hold}")
    cmd=["ffmpeg","-y","-loglevel","error","-threads","1","-filter_threads","1","-f","rawvideo","-vcodec","rawvideo","-pix_fmt","rgba","-s",f"{width}x{height}","-r",str(fps),"-i","-","-an"]; target=_output_size(ratio,quality)
    if target:tw,th=target; cmd += ["-vf",f"scale={tw}:{th}:flags=lanczos"]
    cmd += ["-c:v","libx264","-threads","1","-preset","veryfast","-crf","18" if quality=="high" else "20","-pix_fmt","yuv420p","-movflags","+faststart",str(path)]; proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,bufsize=0)
    try:
        total=frames+hold
        for i in range(total):
            p=1. if i>=frames else (i+1)/frames; _draw_scene_on(fig,ax,dates,companies,pivot,scene,bg,text,grid,cmap,p); fig.canvas.draw(); proc.stdin.write(fig.canvas.buffer_rgba())
            if i and i % max(1,fps*2)==0:_log(f"scene progress {i}/{total}")
        proc.stdin.close(); stderr=proc.stderr.read(); code=proc.wait()
        if code!=0:raise RuntimeError(stderr.decode("utf-8",errors="replace")[-3000:])
        _log(f"scene complete seconds={time.monotonic()-started:.1f} size={os.path.getsize(path)/1024/1024:.1f}MB")
    except Exception:
        _log("scene failed")
        if proc.stdin and not proc.stdin.closed:proc.stdin.close()
        proc.kill(); proc.wait(); raise
    finally:plt.close(fig)


def _stream_copy_concat(paths,output):
    list_path=None
    try:
        fd,list_path=tempfile.mkstemp(prefix="video_concat_",suffix=".txt"); os.close(fd)
        with open(list_path,"w",encoding="utf-8") as f:
            for path in paths:
                escaped=str(os.path.abspath(path)).replace("'", "'\\''"); f.write(f"file '{escaped}'\n")
        _log(f"concat stream-copy start scenes={len(paths)}"); cmd=["ffmpeg","-y","-loglevel","error","-f","concat","-safe","0","-i",list_path,"-c","copy","-movflags","+faststart",str(output)]; subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE); _log(f"concat stream-copy complete size={os.path.getsize(output)/1024/1024:.1f}MB")
    finally:
        if list_path and os.path.exists(list_path):os.remove(list_path)


def _pair_crossfade(left,right,left_duration,right_duration,output,transition):
    safe=min(max(.05,float(transition)),max(.05,left_duration/2),max(.05,right_duration/2)); offset=max(.01,left_duration-safe); cmd=["ffmpeg","-y","-loglevel","error","-threads","1","-filter_threads","1","-i",str(left),"-i",str(right),"-filter_complex",f"[0:v][1:v]xfade=transition=fade:duration={safe:.3f}:offset={offset:.3f}[v]","-map","[v]","-an","-c:v","libx264","-threads","1","-preset","veryfast","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart",str(output)]; subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE); return left_duration+right_duration-safe


def concat_with_crossfade(paths,durations,output,transition=.45,low_memory=True):
    if len(paths)==1:shutil.copyfile(paths[0],output); return
    if low_memory:_stream_copy_concat(paths,output); return
    _log(f"concat start scenes={len(paths)} mode=pairwise-crossfade"); tmpdir=tempfile.mkdtemp(prefix="video_concat_"); current=paths[0]; current_duration=float(durations[0]); owned=None
    try:
        for i in range(1,len(paths)):
            nxt=os.path.join(tmpdir,f"join_{i:02d}.mp4"); _log(f"concat pair {i}/{len(paths)-1}"); current_duration=_pair_crossfade(current,paths[i],current_duration,float(durations[i]),nxt,transition)
            if owned and os.path.exists(owned):os.remove(owned)
            current=nxt; owned=nxt
        shutil.copyfile(current,output); _log(f"concat complete size={os.path.getsize(output)/1024/1024:.1f}MB")
    finally:shutil.rmtree(tmpdir,ignore_errors=True)
