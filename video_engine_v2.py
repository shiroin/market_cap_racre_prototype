import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb


def ease_in_out(t):
    t = np.clip(float(t), 0.0, 1.0)
    return 3 * t * t - 2 * t * t * t


def fade_window(p, start=0.0, end=0.18):
    if end <= start:
        return 1.0
    return ease_in_out(np.clip((p-start)/(end-start), 0, 1))


def _style_axis(ax, bg, text, grid):
    ax.set_facecolor(bg)
    ax.tick_params(colors=text, labelsize=9, length=0, pad=7)
    for s in ax.spines.values(): s.set_visible(False)
    ax.grid(axis="y", color=grid, linewidth=0.8, alpha=0.55)
    ax.set_axisbelow(True)


def _spread_label_positions(values, ymin, ymax, min_gap_ratio=.055):
    if not values: return []
    gap=max((ymax-ymin)*min_gap_ratio,1e-9); order=np.argsort(values); placed=np.array(values,dtype=float)
    low=ymin+(ymax-ymin)*.025; high=ymax-(ymax-ymin)*.025; last=low-gap
    for idx in order: placed[idx]=max(values[idx],last+gap,low); last=placed[idx]
    overflow=placed[order[-1]]-high
    if overflow>0: placed-=overflow
    for j in range(len(order)-2,-1,-1):
        a,b=order[j],order[j+1]; placed[a]=min(placed[a],placed[b]-gap)
    under=low-placed[order[0]]
    if under>0: placed+=under
    return placed.tolist()


def _figure_spec(ratio, quality="preview"):
    # High quality is intentionally rendered at 720p-class resolution and upscaled
    # by FFmpeg. This keeps the Render web process well below its memory ceiling.
    sizes={"9:16":(3.6,6.4),"1:1":(4.0,4.0),"16:9":(6.4,3.6)}
    dpi={"preview":100,"standard":200,"high":200}.get(quality,200)
    return sizes[ratio],dpi


def _output_size(ratio, quality):
    if quality != "high": return None
    return {"9:16":(1080,1920),"1:1":(1080,1080),"16:9":(1920,1080)}[ratio]


def _bar_reveal(raw_p,count,mode):
    if count<=0: return np.zeros(0,dtype=float)
    p=ease_in_out(np.clip((raw_p-.10)/.72,0,1))
    if mode=="一気に表示": return np.full(count,p,dtype=float)
    timeline=p*count; factors=np.clip(timeline-np.arange(count,dtype=float),0,1); partial=(factors>0)&(factors<1)
    factors[partial]=[ease_in_out(v) for v in factors[partial]]
    if mode=="右→左": factors=factors[::-1]
    return factors


def _contrast_text(hex_color):
    r,g,b=to_rgb(hex_color); return "#172033" if .2126*r+.7152*g+.0722*b>.62 else "#FFFFFF"


def _fmt_value(value,decimals=0): return f"{value:,.{int(decimals)}f}"


def _label_box(ax,x,y,label,color,alpha=1.0,size=8,inside=False):
    kwargs=dict(ha="center",va="center" if inside else "bottom",fontsize=size,fontweight="bold",zorder=7,alpha=alpha,clip_on=False)
    if inside: kwargs.update(color=_contrast_text(color))
    else: kwargs.update(color=color,bbox=dict(boxstyle="round,pad=.22",facecolor="white",edgecolor=color,linewidth=.7,alpha=.88))
    ax.text(x,y,label,**kwargs)


def render_story_frame(df,scene,ratio,bg,text,grid,cmap,progress=1.0,quality="preview"):
    metric=scene["metric"]; work=df[["date","company",metric]].copy(); work[metric]=pd.to_numeric(work[metric],errors="coerce"); work=work.dropna()
    dates=list(dict.fromkeys(work["date"].astype(str))); companies=list(dict.fromkeys(work["company"].astype(str)))
    pivot=work.pivot_table(index="date",columns="company",values=metric,aggfunc="sum").reindex(dates).fillna(0)
    size,dpi=_figure_spec(ratio,quality); fig=plt.figure(figsize=size,dpi=dpi); fig.patch.set_facecolor(bg)
    ax=fig.add_axes([.10,.15,.72,.66] if ratio=="9:16" else [.09,.16,.75,.65]); _style_axis(ax,bg,text,grid)
    raw_p=np.clip(float(progress),0,1); title_alpha=fade_window(raw_p,0,.16); subtitle_alpha=fade_window(raw_p,.06,.24)*.84; chart_alpha=fade_window(raw_p,.10,.28); label_alpha=fade_window(raw_p,.76,.96)
    fig.text(.075,.93,scene["title"],color=text,fontsize=scene.get("title_size",22),fontweight="bold",ha="left",alpha=title_alpha)
    if scene.get("subtitle"): fig.text(.075,.885,scene["subtitle"],color=text,fontsize=max(8,scene.get("title_size",22)-7),ha="left",alpha=subtitle_alpha)
    if scene.get("source"): fig.text(.075,.052,f"出典: {scene['source']}",color=text,fontsize=7,ha="left",alpha=.58*label_alpha)
    x=np.arange(len(dates),dtype=float); chart=scene["chart"]
    ymax=100.0 if chart=="100%積み上げ" else float(max(1.0,pivot.sum(axis=1).max() if chart=="積み上げ棒" else pivot.to_numpy().max()))*1.22
    ax.set_ylim(0,ymax); decimals=int(scene.get("value_decimals",0)); label_mode=scene.get("data_labels","自動"); reveal_mode=scene.get("bar_animation","左→右")
    if chart in ("積み上げ棒","100%積み上げ"):
        shown=pivot.copy()
        if chart=="100%積み上げ": shown=shown.div(shown.sum(axis=1).replace(0,np.nan),axis=0).fillna(0)*100
        reveal=_bar_reveal(raw_p,len(dates),reveal_mode); bottom=np.zeros(len(dates)); segment_centers=[]
        for company in companies:
            raw_vals=shown[company].to_numpy(float) if company in shown else np.zeros(len(dates)); vals=raw_vals*reveal; centers=bottom+vals/2
            ax.bar(x,vals,bottom=bottom,color=cmap[company],width=.68,label=company,alpha=chart_alpha); segment_centers.append((company,vals,centers)); bottom+=vals
        if label_mode!="なし" and raw_p>=.45:
            for company,vals,centers in segment_centers:
                for j,val in enumerate(vals):
                    if reveal[j]<.96: continue
                    if label_mode=="すべて" or (label_mode=="自動" and val>=ymax*.065): _label_box(ax,x[j],centers[j],_fmt_value(val,decimals),cmap[company],label_alpha,scene.get("data_label_size",7),inside=True)
            for j,total in enumerate(bottom):
                if reveal[j]>=.96 and label_mode in ("自動","すべて","合計のみ"):
                    ax.text(x[j],total+ymax*.016,_fmt_value(total,decimals),color=text,ha="center",va="bottom",fontsize=scene.get("data_label_size",7),fontweight="bold",alpha=label_alpha,clip_on=False)
    elif chart=="棒グラフ":
        width=.76/max(1,len(companies)); reveal=_bar_reveal(raw_p,len(dates),reveal_mode)
        for i,company in enumerate(companies):
            raw_vals=pivot[company].to_numpy(float) if company in pivot else np.zeros(len(dates)); vals=raw_vals*reveal; xpos=x+(i-(len(companies)-1)/2)*width
            ax.bar(xpos,vals,width=width,color=cmap[company],label=company,alpha=chart_alpha)
            if label_mode!="なし" and raw_p>=.45:
                for j,val in enumerate(vals):
                    if reveal[j]>=.96 and val>0 and label_mode in ("自動","すべて","合計のみ"):
                        ax.text(xpos[j],val+ymax*.012,_fmt_value(val,decimals),color=text,ha="center",va="bottom",fontsize=scene.get("data_label_size",7),fontweight="bold",alpha=label_alpha,clip_on=False)
    else:
        draw_p=ease_in_out(np.clip((raw_p-.10)/.72,0,1)); pos=draw_p*max(0,len(dates)-1); whole=int(np.floor(pos)); frac=pos-whole; endpoints=[]
        for company in companies:
            vals=pivot[company].to_numpy(float) if company in pivot else np.zeros(len(dates))
            if len(dates)==1: end_x,end_y=0,vals[0]; ax.scatter([0],[end_y],color=cmap[company],s=22,alpha=chart_alpha)
            else:
                xs=list(x[:whole+1]); ys=list(vals[:whole+1])
                if whole<len(dates)-1: end_x=x[whole]+frac; end_y=vals[whole]+(vals[whole+1]-vals[whole])*frac; xs.append(end_x); ys.append(end_y)
                else: end_x,end_y=x[-1],vals[-1]
                ax.plot(xs,ys,color=cmap[company],linewidth=2.8,solid_capstyle="round",alpha=chart_alpha); ax.scatter([end_x],[end_y],color=cmap[company],s=18,zorder=4,alpha=chart_alpha)
            endpoints.append((company,vals[-1]))
        if scene.get("end_labels",True) and raw_p>=.80:
            adjusted=_spread_label_positions([v for _,v in endpoints],0,ymax,scene.get("label_gap",.055))
            for (company,actual_y),label_y in zip(endpoints,adjusted):
                ax.plot([x[-1]+.04,x[-1]+.22],[actual_y,label_y],color=cmap[company],linewidth=.9,alpha=.65*label_alpha); label=company
                if scene.get("latest_values",True): label+=f"  {_fmt_value(actual_y,decimals)}{scene.get('unit','')}"
                ax.text(x[-1]+.27,label_y,label,color=cmap[company],fontsize=scene.get("end_label_size",8),va="center",fontweight="bold",alpha=label_alpha,bbox=dict(boxstyle="round,pad=.18",facecolor=bg,edgecolor="none",alpha=.90))
        ax.set_xlim(-.2,max(1,len(dates)-1)+2.45)
    ax.set_xticks(x); ax.set_xticklabels(dates,color=text,alpha=chart_alpha); ax.set_ylabel(scene.get("unit",""),color=text,fontsize=9,alpha=chart_alpha)
    if scene.get("legend",False):
        leg=ax.legend(frameon=False,fontsize=8,ncol=2,loc="upper left")
        for t in leg.get_texts(): t.set_color(text); t.set_alpha(chart_alpha)
    return fig


def save_scene_v2(df,scene,path,ratio,fps,bg,text,grid,cmap,quality="standard"):
    """Stream one frame at a time to FFmpeg; high mode renders at 720p then upscales."""
    frames=max(2,int(scene.get("duration",2.5)*fps)); hold=max(0,int(scene.get("hold",1.0)*fps))
    first=render_story_frame(df,scene,ratio,bg,text,grid,cmap,1.0/max(2,frames),quality); first.canvas.draw(); width,height=first.canvas.get_width_height(); plt.close(first)
    cmd=["ffmpeg","-y","-loglevel","error","-f","rawvideo","-vcodec","rawvideo","-pix_fmt","rgba","-s",f"{width}x{height}","-r",str(fps),"-i","-","-an"]
    target=_output_size(ratio,quality)
    if target:
        tw,th=target; cmd += ["-vf",f"scale={tw}:{th}:flags=lanczos"]
    cmd += ["-c:v","libx264","-preset","veryfast","-crf","18" if quality=="high" else "20","-pix_fmt","yuv420p","-movflags","+faststart",str(path)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    try:
        for i in range(frames+hold):
            p=1.0 if i>=frames else (i+1)/frames; fig=render_story_frame(df,scene,ratio,bg,text,grid,cmap,p,quality); fig.canvas.draw()
            frame=np.asarray(fig.canvas.buffer_rgba(),dtype=np.uint8); proc.stdin.write(frame.tobytes()); del frame; plt.close(fig)
        proc.stdin.close(); stderr=proc.stderr.read(); code=proc.wait()
        if code!=0: raise RuntimeError(stderr.decode("utf-8",errors="replace")[-3000:])
    except Exception:
        if proc.stdin and not proc.stdin.closed: proc.stdin.close()
        proc.kill(); proc.wait(); raise
    finally:
        plt.close("all")


def concat_with_crossfade(paths,durations,output,transition=.45):
    if len(paths)==1: shutil.copyfile(paths[0],output); return
    transition=max(.05,float(transition)); cmd=["ffmpeg","-y","-loglevel","error"]
    for path in paths: cmd += ["-i",str(path)]
    filters=[]; previous="[0:v]"; elapsed=float(durations[0])
    for i in range(1,len(paths)):
        out=f"[v{i}]"; safe_t=min(transition,max(.05,durations[i-1]/2),max(.05,durations[i]/2)); offset=max(.01,elapsed-safe_t); filters.append(f"{previous}[{i}:v]xfade=transition=fade:duration={safe_t:.3f}:offset={offset:.3f}{out}"); previous=out; elapsed+=float(durations[i])-safe_t
    cmd += ["-filter_complex",";".join(filters),"-map",previous,"-c:v","libx264","-preset","veryfast","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart",str(output)]; subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
