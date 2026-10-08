import os
import shutil
import subprocess
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import textwrap
from matplotlib.colors import to_rgb


def _rss_mb():
    try:
        with open('/proc/self/status', encoding='utf-8') as f:
            for line in f:
                if line.startswith('VmRSS:'): return int(line.split()[1]) / 1024
    except Exception: pass
    return None


def _log(msg):
    rss=_rss_mb(); print(f"[video-render] {msg}"+(f" | rss={rss:.1f}MB" if rss else ''),flush=True)


def ease_in_out(t):
    t=float(np.clip(float(t),0,1)); return float(np.clip(3*t*t-2*t*t*t,0,1))


def fade_window(p,start=0,end=.18):
    return 1.0 if end<=start else ease_in_out(np.clip((p-start)/(end-start),0,1))



def _draw_reference_subtitle(fig,scene,text,accent='#16718C',y=.885,fontsize=8,alpha=1.0):
    subtitle=str(scene.get('subtitle','') or '').strip()
    if not subtitle: return
    max_chars=max(16,int(44*12/max(float(fontsize),1)))
    wrapped='\\n'.join('\\n'.join(textwrap.wrap(line,width=max_chars,break_long_words=False,break_on_hyphens=False)) or '' for line in subtitle.split('\\n'))
    fig.text(.075,y,wrapped,color=text,fontsize=fontsize,fontweight='normal',ha='left',va='center',alpha=.68*float(np.clip(alpha,0,1)),zorder=22,linespacing=1.30,wrap=False)


def _draw_scene_comments(fig,scene,text,progress,elapsed=None):
    c1=str(scene.get('scene_comment_1','') or '').strip()
    c2=str(scene.get('scene_comment_2','') or '').strip()
    if not c1 and not c2: return
    size=int(scene.get('scene_comment_size',12))
    duration=max(.01,float(scene.get('duration',2.8)))
    delay=max(0.,float(scene.get('scene_comment_delay',.7)))
    gap=max(0.,float(scene.get('scene_comment_gap',.8)))
    t=float(elapsed) if elapsed is not None else float(progress)*duration
    fade=.45
    start1=duration+delay
    start2=start1+fade+gap if c1 else start1
    a1=float(ease_in_out(np.clip((t-start1)/fade,0,1))) if c1 else 0.
    a2=float(ease_in_out(np.clip((t-start2)/fade,0,1))) if c2 else 0.
    if scene.get('scene_comment_style','従来（文字のみ）')=='白抜き（濃紺背景）':
        from matplotlib.patches import FancyBboxPatch
        # A single reserved panel keeps the two staggered comments together.
        # Its opacity follows the first comment, so the panel never flashes early.
        panel_alpha=a1 if c1 else a2
        if panel_alpha>0:
            panel=FancyBboxPatch((.075,.086),.85,.150,boxstyle='round,pad=0.008,rounding_size=0.012',
                transform=fig.transFigure,facecolor='#233653',edgecolor='none',alpha=panel_alpha,zorder=28)
            fig.add_artist(panel)
        if c1 and a1>0:
            fig.text(.50,.187,c1,color='white',fontsize=size,fontweight='bold',
                ha='center',va='center',alpha=a1,zorder=30,wrap=True,linespacing=1.18)
        if c2 and a2>0:
            fig.text(.50,.125,c2,color='white',fontsize=size,fontweight='bold',
                ha='center',va='center',alpha=a2,zorder=30,wrap=True,linespacing=1.18)
    else:
        if c1 and a1>0:
            fig.text(.075,.155,c1,color=text,fontsize=size,fontweight='bold',
                ha='left',va='bottom',alpha=a1,zorder=30,wrap=True)
        if c2 and a2>0:
            fig.text(.075,.107,c2,color=text,fontsize=size,fontweight='bold',
                ha='left',va='bottom',alpha=a2,zorder=30,wrap=True)


def _style_axis(ax,bg,text,grid):
    ax.set_facecolor(bg); ax.tick_params(colors=text,labelsize=9,length=0,pad=7)
    for s in ax.spines.values(): s.set_visible(False)
    ax.grid(axis='y',color=grid,linewidth=.8,alpha=.55); ax.set_axisbelow(True)


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


def _figure_spec(ratio,quality='preview'):
    sizes={'元動画 (64:139)':(3.84,8.34),'9:16':(3.6,6.4),'4:5':(4.8,6.0),'1:1':(5.,5.),'5:4':(6.25,5.),'16:9':(8.,4.5)}; dpi={'preview':100,'standard':200,'high':150}.get(quality,200); return sizes[ratio],dpi


def _output_size(ratio,quality):
    return None if quality!='high' else {'元動画 (64:139)':(1024,2224),'9:16':(1080,1920),'4:5':(1080,1350),'1:1':(1080,1080),'5:4':(1350,1080),'16:9':(1920,1080)}[ratio]


def _bar_reveal(raw_p,count,mode):
    if count<=0:return np.zeros(0)
    p=ease_in_out(np.clip((raw_p-.10)/.72,0,1))
    if mode=='一気に表示': return np.full(count,p)
    timeline=p*count; f=np.clip(timeline-np.arange(count,dtype=float),0,1); partial=(f>0)&(f<1); f[partial]=[ease_in_out(v) for v in f[partial]]
    return f[::-1] if mode=='右→左' else f


def _active_bar_index(reveal,mode):
    if len(reveal)==0:return 0
    visible=np.flatnonzero(reveal>1e-6)
    if len(visible): return int(visible[0] if mode=='右→左' else visible[-1])
    return len(reveal)-1 if mode=='右→左' else 0


def _contrast_text(c):
    r,g,b=to_rgb(c); return '#172033' if .2126*r+.7152*g+.0722*b>.62 else '#FFFFFF'


def _fmt_value(v,d=0): return f'{v:,.{int(d)}f}'


def _series_label(company,value,scene,decimals):
    return f"{company}  {_fmt_value(value,decimals)}{scene.get('unit','')}" if scene.get('latest_values',True) else company


def _prepare_scene(df,scene):
    metric=scene['metric']; w=df[['date','company',metric]].copy(); w[metric]=pd.to_numeric(w[metric],errors='coerce'); w=w.dropna(); dates=list(dict.fromkeys(w.date.astype(str))); companies=list(dict.fromkeys(w.company.astype(str))); pivot=w.pivot_table(index='date',columns='company',values=metric,aggfunc='sum').reindex(dates).fillna(0); return dates,companies,pivot


def _draw_scene_on(fig,ax,dates,companies,pivot,scene,bg,text,grid,cmap,progress,elapsed=None):
    ax.clear(); fig.texts.clear(); [patch.remove() for patch in list(fig.patches)]; _style_axis(ax,bg,text,grid)
    p=np.clip(float(progress),0,1); title_a=1.0; sub_a=.84; chart_a=1.0; late_a=fade_window(p,.76,.96)
    fig.text(.075,.93,scene['title'],color=text,fontsize=scene.get('title_size',22),fontweight='bold',ha='left',alpha=title_a)
    _draw_reference_subtitle(fig,scene,text,y=.885,fontsize=scene.get('subtitle_size',12),alpha=fade_window(p,.02,.16))
    if scene.get('source'): fig.text(.075,.052,f"出典: {scene['source']}",color=text,fontsize=7,ha='left',alpha=.58)
    _draw_scene_comments(fig,scene,text,p,elapsed)
    if scene.get('scene_note'):
        # Divider belongs between the comment zone and the footnote zone.
        divider=plt.Line2D([.075,.925],[.060,.060],transform=fig.transFigure,color=grid,lw=.7,alpha=.70)
        fig.add_artist(divider)
        fig.text(.075,.036,scene['scene_note'],color=text,fontsize=5.4,ha='left',va='bottom',alpha=.52,wrap=True)
    x=np.arange(len(dates),dtype=float); chart=scene['chart']; ymax=100. if chart=='100%積み上げ' else float(max(1.,pivot.sum(axis=1).max() if chart=='積み上げ棒' else pivot.to_numpy().max()))*1.22; ax.set_ylim(0,ymax)
    decimals=int(scene.get('value_decimals',0)); label_mode=scene.get('data_labels','自動'); mode=scene.get('bar_animation','左→右'); live_size=scene.get('end_label_size',8); gap=scene.get('label_gap',.055)
    bar_gap=float(np.clip(scene.get('bar_gap',.32),0,.95)); period_width=1.0-bar_gap

    if chart in ('積み上げ棒','100%積み上げ'):
        shown=pivot.copy()
        if chart=='100%積み上げ': shown=shown.div(shown.sum(axis=1).replace(0,np.nan),axis=0).fillna(0)*100
        reveal=_bar_reveal(p,len(dates),mode); bottom=np.zeros(len(dates)); segments=[]
        for company in companies:
            raw=shown[company].to_numpy(float) if company in shown else np.zeros(len(dates)); vals=raw*reveal; centers=bottom+vals/2; ax.bar(x,vals,bottom=bottom,color=cmap[company],width=period_width,label=company,alpha=chart_a); segments.append((company,raw,vals,centers)); bottom+=vals

        # Reference-video motion: labels do NOT grow vertically with the bars.
        # They are already sitting at the full-value positions for the current date
        # before that bar is revealed. As the frontier advances, the whole label
        # stack glides horizontally and its Y positions interpolate only between the
        # *final* segment centres of adjacent dates. This avoids the distracting
        # bottom-to-top sweep while preserving the gentle data-driven vertical motion.
        if label_mode!='なし' and len(dates):
            j=_active_bar_index(reveal,mode)
            if mode=='一気に表示':
                prev_j=j; travel=1.0
            elif mode=='右→左':
                prev_j=min(len(dates)-1,j+1); travel=ease_in_out(float(reveal[j]))
            else:
                prev_j=max(0,j-1); travel=ease_in_out(float(reveal[j]))
            label_x=(1-travel)*x[prev_j]+travel*x[j]+.44
            prev_running=0.0; cur_running=0.0
            for company,raw,vals,centers in segments:
                prev_full=float(raw[prev_j]); cur_full=float(raw[j])
                prev_y=prev_running+prev_full/2; cur_y=cur_running+cur_full/2
                label_y=(1-travel)*prev_y+travel*cur_y
                display_value=(1-travel)*prev_full+travel*cur_full
                ax.text(label_x,label_y,_series_label(company,display_value,scene,decimals),color=cmap[company],fontsize=live_size,va='center',ha='left',fontweight='bold',alpha=1,zorder=9,clip_on=False)
                prev_running+=prev_full; cur_running+=cur_full
            if chart=='積み上げ棒':
                total_y=(1-travel)*prev_running+travel*cur_running
                ax.text(label_x,total_y+ymax*.022,f"合計 {_fmt_value(total_y,decimals)}{scene.get('unit','')}",color=text,fontsize=live_size,va='bottom',ha='left',fontweight='bold',alpha=1,zorder=9,clip_on=False)

        # Optional historical numeric labels remain available, but reference-style
        # live labels above are present from frame one.
        if label_mode!='なし' and p>=.45:
            for company,raw,vals,centers in segments:
                for k,val in enumerate(vals):
                    if reveal[k]>=.96 and (label_mode=='すべて' or (label_mode=='自動' and val>=ymax*.065)):
                        ax.text(x[k],centers[k],_fmt_value(val,decimals),ha='center',va='center',fontsize=scene.get('data_label_size',7),fontweight='bold',color=_contrast_text(cmap[company]),alpha=late_a,zorder=7)
        ax.set_xlim(-.45,max(1,len(dates)-1)+2.8)

    elif chart=='棒グラフ':
        group_width=period_width; width=group_width/max(1,len(companies)); reveal=_bar_reveal(p,len(dates),mode); series=[]
        for i,company in enumerate(companies):
            raw=pivot[company].to_numpy(float) if company in pivot else np.zeros(len(dates)); vals=raw*reveal; xpos=x+(i-(len(companies)-1)/2)*width; ax.bar(xpos,vals,width=width,color=cmap[company],label=company,alpha=chart_a); series.append((company,raw,vals,xpos))
            if label_mode!='なし' and p>=.45:
                for k,val in enumerate(vals):
                    if reveal[k]>=.96 and val>0 and label_mode in ('自動','すべて','合計のみ'): ax.text(xpos[k],val+ymax*.012,_fmt_value(val,decimals),color=text,ha='center',va='bottom',fontsize=scene.get('data_label_size',7),fontweight='bold',alpha=late_a,clip_on=False)

        # Reference-video motion for grouped bars: keep labels at the full bar-tip
        # heights instead of making them rise from zero with the reveal. Between
        # dates they glide from the previous full-value anchor to the next one.
        if label_mode!='なし' and len(dates):
            j=_active_bar_index(reveal,mode)
            if mode=='一気に表示':
                prev_j=j; travel=1.0
            elif mode=='右→左':
                prev_j=min(len(dates)-1,j+1); travel=ease_in_out(float(reveal[j]))
            else:
                prev_j=max(0,j-1); travel=ease_in_out(float(reveal[j]))
            for company,raw,vals,xpos in series:
                prev_y=float(raw[prev_j]); cur_y=float(raw[j])
                label_y=(1-travel)*prev_y+travel*cur_y
                display_value=(1-travel)*prev_y+travel*cur_y
                label_x=(1-travel)*float(xpos[prev_j])+travel*float(xpos[j])+width*.60
                ax.text(label_x,label_y,_series_label(company,display_value,scene,decimals),color=cmap[company],fontsize=live_size,va='center',ha='left',fontweight='bold',alpha=1,zorder=9,clip_on=False)
        ax.set_xlim(-.45,max(1,len(dates)-1)+2.8)

    else:
        draw=ease_in_out(np.clip((p-.10)/.72,0,1)); pos=draw*max(0,len(dates)-1); whole=int(np.floor(pos)); frac=pos-whole; endpoints=[]
        for company in companies:
            vals=pivot[company].to_numpy(float) if company in pivot else np.zeros(len(dates))
            if len(dates)==1: end_x,end_y=0,vals[0]; ax.scatter([0],[end_y],color=cmap[company],s=22,alpha=chart_a)
            else:
                xs=list(x[:whole+1]); ys=list(vals[:whole+1])
                if whole<len(dates)-1: end_x=x[whole]+frac; end_y=vals[whole]+(vals[whole+1]-vals[whole])*frac; xs.append(end_x); ys.append(end_y)
                else: end_x,end_y=x[-1],vals[-1]
                ax.plot(xs,ys,color=cmap[company],linewidth=2.8,solid_capstyle='round',alpha=chart_a); ax.scatter([end_x],[end_y],color=cmap[company],s=18,zorder=4,alpha=chart_a)
            endpoints.append((company,end_x,end_y))
        if scene.get('end_labels',True):
            adjusted=_spread_label_positions([v for _,_,v in endpoints],0,ymax,gap)
            for (company,end_x,actual_y),label_y in zip(endpoints,adjusted):
                # The reference line charts also keep the label at the moving endpoint.
                # A tiny vertical separation is used only when two series overlap.
                ax.text(end_x+.18,label_y,_series_label(company,actual_y,scene,decimals),color=cmap[company],fontsize=live_size,va='center',fontweight='bold',alpha=1,bbox=dict(boxstyle='round,pad=.12',facecolor=bg,edgecolor='none',alpha=.90),clip_on=False)
        ax.set_xlim(-.2,max(1,len(dates)-1)+2.45)

    # Keep the time axis readable by anchoring labels to the latest period.
    # Up to 10 periods: show every 2nd label. 11+ periods: show every 4th label.
    # Walk backwards from the latest period so the rightmost label is always kept;
    # any unmatched oldest periods are intentionally omitted.
    tick_step=2 if len(dates)<=10 else 4
    tick_idx=list(range(len(dates)-1,-1,-tick_step))[::-1] if len(dates) else []
    ax.set_xticks(x[tick_idx] if len(tick_idx) else [])
    ax.set_xticklabels([dates[i] for i in tick_idx],color=text,alpha=chart_a)
    ax.set_ylabel(scene.get('unit',''),color=text,fontsize=9,alpha=chart_a)
    if scene.get('legend',False):
        leg=ax.legend(frameon=False,fontsize=8,ncol=2,loc='upper left')
        for t in leg.get_texts(): t.set_color(text); t.set_alpha(chart_a)



def _draw_horizontal_ranking_on(fig,ax,df,scene,bg,text,grid,cmap,progress,elapsed=None):
    ax.clear(); fig.texts.clear(); [patch.remove() for patch in list(fig.patches)]; _style_axis(ax,bg,text,grid)
    p=float(np.clip(progress,0,1)); metric=scene['metric']
    w=df[['company',metric]].copy(); w[metric]=pd.to_numeric(w[metric],errors='coerce'); w=w.dropna()
    w=w.groupby('company',sort=False).tail(1)
    ascending=scene.get('ranking_sort','大きい順')=='小さい順'
    w=w.sort_values(metric,ascending=ascending).reset_index(drop=True)
    names=w['company'].astype(str).tolist(); values=w[metric].to_numpy(float)
    if not len(values): return
    fig.text(.075,.93,scene.get('title','横比較ランキング'),color=text,fontsize=scene.get('title_size',22),fontweight='bold',ha='left',alpha=1)
    _draw_reference_subtitle(fig,scene,text,y=.885,fontsize=max(7,scene.get('title_size',22)-10))
    if scene.get('source'): fig.text(.075,.052,f"出典: {scene['source']}",color=text,fontsize=7,ha='left',alpha=.58)
    if scene.get('scene_note'):
        divider=plt.Line2D([.075,.925],[.043,.043],transform=fig.transFigure,color=grid,lw=.7,alpha=.70); fig.add_artist(divider)
        fig.text(.075,.027,scene['scene_note'],color=text,fontsize=5.4,ha='left',va='bottom',alpha=.52,wrap=True)
    n=len(names); y=np.arange(n); starts=.08+np.arange(n)*(.68/max(1,n))
    reveal=np.array([ease_in_out(np.clip((p-s)/.18,0,1)) for s in starts])
    vmax=max(float(np.nanmax(values)),1e-9); xmax=vmax*1.18; highlight=max(0,int(scene.get('ranking_highlight',3))); base='#AEB8C4'
    colors=[(cmap.get(name,base) if i<highlight else base) for i,name in enumerate(names)]
    ax.barh(y,values*reveal,color=colors,height=.58,alpha=.95)
    label_fs=max(5.5,min(9.5,11-.16*n))
    ax.set_yticks(y); ax.set_yticklabels([]); ax.tick_params(axis='y',length=0,pad=0)
    # Draw labels in a fixed gutter inside the figure instead of Matplotlib y-tick labels.
    # This prevents long Japanese names from being clipped by the canvas boundary.
    for yi,name in zip(y,names):
        ax.text(-.035,yi,name,transform=ax.get_yaxis_transform(),color=text,fontsize=label_fs,fontweight='bold',ha='right',va='center',clip_on=False)
    ax.invert_yaxis(); ax.set_xlim(0,xmax)
    ax.grid(axis='x',color=grid,linewidth=.8,alpha=.55); ax.grid(axis='y',visible=False); ax.set_xlabel(scene.get('unit',''),color=text,fontsize=8)
    ref=float(scene.get('ranking_reference',0) or 0)
    if ref>0: ax.axvline(ref,color=text,lw=1,ls=(0,(2,3)),alpha=.48)
    decimals=int(scene.get('value_decimals',0))
    for i,(v,r) in enumerate(zip(values,reveal)):
        if r>0: ax.text(v*r+xmax*.012,i,f"{_fmt_value(v,decimals)}{scene.get('unit','')}",color=text,fontsize=max(6,min(9,10-.10*n)),fontweight='bold',ha='left',va='center',alpha=r,clip_on=False)


def _timeline_events(scene):
    events=scene.get('timeline_events',[])
    clean=[]
    for row in events:
        try:
            year=float(row.get('year'))
        except (TypeError,ValueError):
            continue
        clean.append({'year':year,'date':str(row.get('date','')).strip(),'title':str(row.get('title','')).strip(),'description':str(row.get('description','')).strip(),'badge':str(row.get('badge','')).strip()})
    return sorted(clean,key=lambda r:r['year'])


def _draw_financial_timeline_on(fig,ax,df,scene,bg,text,grid,cmap,progress):
    """Synchronized events, financial chart and horizontal year timeline."""
    import re
    from matplotlib.patches import Rectangle
    ax.clear()
    for artist in list(fig.artists): artist.remove()
    for artist in list(fig.lines): artist.remove()
    for artist in list(fig.patches): artist.remove()
    for artist in list(fig.texts): artist.remove()
    ax.set_facecolor(bg)
    p=float(np.clip(progress,0,1))
    fig.text(.075,.945,scene.get('title','業績と年表'),color=text,
        fontsize=scene.get('title_size',22),fontweight='bold',ha='left',va='top')
    _draw_reference_subtitle(fig,scene,text,y=.897,
        fontsize=scene.get('subtitle_size',12),alpha=fade_window(p,.01,.08))
    events=_timeline_events(scene)
    bar_metric=scene.get('timeline_bar_metric',scene.get('metric'))
    line_metric=scene.get('timeline_line_metric','(なし)')
    if bar_metric not in df.columns: return
    cols=['date',bar_metric]
    if line_metric in df.columns and line_metric!=bar_metric: cols.append(line_metric)
    data=df[cols].copy()
    data['date']=data['date'].astype(str)
    data[bar_metric]=pd.to_numeric(data[bar_metric],errors='coerce')
    if line_metric in data.columns: data[line_metric]=pd.to_numeric(data[line_metric],errors='coerce')
    data=data.dropna(subset=[bar_metric]).groupby('date',sort=False).sum(numeric_only=True)
    if data.empty: return
    labels=list(data.index)
    def date_year(label):
        match=re.search(r'(?:19|20|21)\\d{2}',str(label))
        if not match: return None
        year=float(match.group(0))
        q=re.search(r'Q([1-4])',str(label),re.I)
        if q: return year+(int(q.group(1))-.5)/4
        m=re.search(r'(?:19|20|21)\\d{2}[-/](\\d{1,2})',str(label))
        if m: return year+(int(m.group(1))-.5)/12
        return year+.5
    years=[date_year(label) for label in labels]
    start=float(scene.get('timeline_start',min((y for y in years if y is not None),default=2018)))
    end=float(scene.get('timeline_end',max((y for y in years if y is not None),default=2025)))
    if end<=start: end=start+1
    now=start+(end-start)*p
    x=np.arange(len(labels))
    values=data[bar_metric].to_numpy(dtype=float)
    ymax=max(1.,float(np.nanmax(values))*1.20)
    # The financial chart occupies its own middle band; labels stay above timeline.
    ax.set_position([.12,.365,.77,.325])
    ax.set_xlim(-.65,len(labels)-.35)
    ax.set_ylim(min(0.,float(np.nanmin(values)))*1.1,ymax)
    ax.grid(axis='y',color=grid,alpha=.45,lw=.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(axis='both',colors=text,labelsize=6.5,length=0,pad=5)
    stride=max(1,int(np.ceil(len(labels)/7)))
    ax.set_xticks(x[::stride])
    ax.set_xticklabels([labels[i] for i in range(0,len(labels),stride)],rotation=0)
    # Highlight only the active period. The background fades in as the
    # timeline ball reaches that period; earlier periods do not leave a trail.
    active_index=None
    for k,yr in enumerate(years):
        if yr is not None and now>=yr:
            active_index=k
    if active_index is not None:
        active_year=years[active_index]
        fade=float(np.clip((now-active_year)/max((end-start)*.018,.02),0,1))
        fade=float(ease_in_out(fade))
        ax.axvspan(active_index-.48,active_index+.48,
            color=scene.get('timeline_highlight_color','#EAC6D3'),
            alpha=float(np.clip(scene.get('timeline_highlight_alpha',.42),0,1))*fade,
            zorder=0,lw=0)
    bar_color=scene.get('timeline_bar_color','#B83F68')
    ax.bar(x,values,width=.66,color=bar_color,alpha=.78,zorder=3)
    if line_metric in data.columns:
        line_values=data[line_metric].to_numpy(dtype=float)
        lo=float(np.nanmin(line_values)); hi=float(np.nanmax(line_values))
        span=max(hi-lo,1e-8)
        # Normalize the secondary metric to the same plotting area without creating
        # a new twinx axis on each animation frame.
        mapped=ymax*(.17+.68*(line_values-lo)/span)
        ax.plot(x,mapped,color=scene.get('timeline_line_color','#B83F68'),lw=2.0,marker='o',markersize=2.6,zorder=5)
        fig.text(.88,.705,str(line_metric),color=scene.get('timeline_line_color','#B83F68'),fontsize=7,ha='right')
    fig.text(.12,.715,str(bar_metric),color=text,fontsize=8,fontweight='bold',ha='left')
    # Event comment above the graph; the most recent reached event is displayed.
    current=None
    for ev in events:
        if ev['year']<=now: current=ev
    if current is not None:
        age=(now-current['year'])/max(end-start,1.)
        opacity=float(fade_window(age,0,.018))
        fig.text(.09,.835,current.get('date',''),color=text,fontsize=9,
            fontweight='bold',ha='left',alpha=opacity)
        fig.text(.09,.800,current.get('title',''),color=text,fontsize=14,
            fontweight='bold',ha='left',alpha=opacity,wrap=True)
        fig.text(.09,.762,current.get('description',''),color=text,fontsize=8,
            ha='left',alpha=opacity,wrap=True)
    # One continuous pale line, pale stops, and exactly one moving ball.
    left,right=.10,.90
    yline=.235
    def xpos(year):
        return left+(right-left)*float(np.clip((year-start)/(end-start),0,1))
    fig.add_artist(plt.Line2D([left,right],[yline,yline],
        transform=fig.transFigure,color='#B6B1A9',lw=1.7,zorder=4))
    for ev in events:
        fig.add_artist(plt.Line2D([xpos(ev['year'])],[yline],
            transform=fig.transFigure,marker='o',markersize=8,
            markerfacecolor='#A9A59D',markeredgecolor='none',
            alpha=.32,linestyle='None',zorder=5))
    fig.add_artist(plt.Line2D([xpos(now)],[yline],
        transform=fig.transFigure,marker='o',markersize=9,
        markerfacecolor='#172B47',markeredgecolor='none',
        linestyle='None',zorder=10))
    for year in range(int(np.ceil(start)),int(np.floor(end))+1,
        max(1,int(np.ceil((end-start)/6)))):
        fig.text(xpos(year),yline-.026,str(year),color=text,fontsize=7,
            ha='center',va='top',alpha=.75)
    note=str(scene.get('scene_note','') or '').strip()
    if note:
        fig.add_artist(plt.Line2D([.075,.925],[.065,.065],
            transform=fig.transFigure,color=grid,lw=.7))
        fig.text(.075,.045,note,color=text,fontsize=5.5,ha='left',va='bottom',alpha=.6)


def _draw_horizontal_timeline_on(fig,ax,scene,bg,text,grid,progress):
    """One event at a time, synchronized with a persistent horizontal timeline."""
    from matplotlib.patches import FancyBboxPatch
    ax.clear()
    # Timeline strokes and balls are FIGURE-level Line2D artists, not Axes artists.
    # ax.clear() and fig.texts.clear() alone do not remove them, so each frame
    # previously accumulated every old ball position as a visible ghost trail.
    # fig.add_artist(Line2D(...)) stores the line in fig.artists, NOT fig.lines.
    # Clear both registries; otherwise the moving ball is permanently stamped
    # into the figure on every frame (the actual cause of the ghost trail).
    for artist in list(fig.artists): artist.remove()
    for artist in list(fig.lines): artist.remove()
    for artist in list(fig.patches): artist.remove()
    for artist in list(fig.texts): artist.remove()
    ax.axis('off')
    p=float(np.clip(progress,0,1))
    fig.text(.075,.93,scene.get('title','年表'),color=text,
        fontsize=scene.get('title_size',22),fontweight='bold',ha='left')
    _draw_reference_subtitle(fig,scene,text,y=.885,
        fontsize=scene.get('subtitle_size',12),alpha=fade_window(p,.01,.09))
    events=_timeline_events(scene)
    if not events: return
    n=len(events)
    start=float(scene.get('timeline_start',int(np.floor(events[0]['year']))))
    end=float(scene.get('timeline_end',int(np.ceil(events[-1]['year']))))
    if end<=start: end=start+1.
    # Figure-relative coordinates keep the timeline stable across aspect ratios.
    left,right=.09,.91
    yline=.255
    def xpos(year):
        return left+(right-left)*float(np.clip((year-start)/(end-start),0,1))
    fig.add_artist(plt.Line2D([left,right],[yline,yline],transform=fig.transFigure,
        color=grid,lw=1.8,alpha=.9,zorder=4))
    span=end-start
    tick_step=max(1,int(np.ceil(span/6)))
    for year in range(int(np.ceil(start)),int(np.floor(end))+1,tick_step):
        x=xpos(year)
        fig.add_artist(plt.Line2D([x,x],[yline-.005,yline+.005],
            transform=fig.transFigure,color=text,lw=.7,alpha=.45,zorder=5))
        fig.text(x,yline-.018,str(year),ha='center',va='top',fontsize=6.5,color=text,alpha=.7)
    # Time moves through the event positions, pausing long enough for each to be read.
    position=min(n-1,int(p*n))
    local=float(np.clip(p*n-position,0,1))
    event=events[position]
    # Reference motion: fixed pale stop markers, one moving ball, no trail.
    # The baseline remains pale across its full length, including behind the ball.
    for ev in events:
        fig.add_artist(plt.Line2D([xpos(ev['year'])],[yline],
            transform=fig.transFigure,marker='o',markersize=8,
            markerfacecolor='#A9A59D',markeredgecolor='none',
            alpha=.32,linestyle='None',zorder=7))
    slot=float(np.clip(p*n-position,0,1))
    # Move between stops with smooth acceleration/deceleration, then dwell.
    travel=float(ease_in_out(np.clip(slot/.30,0,1)))
    previous_x=left if position==0 else xpos(events[position-1]['year'])
    target_x=xpos(event['year'])
    cursor=previous_x+(target_x-previous_x)*travel
    if p>=1.0:
        cursor=target_x
    fig.add_artist(plt.Line2D([cursor],[yline],transform=fig.transFigure,
        marker='o',markersize=8.5,markerfacecolor='#172B47',
        markeredgecolor='none',linestyle='None',zorder=12))
    # Fade the current card in, then gently fade it out before the next event.
    fade_in=float(fade_window(local,.30,.46))
    fade_out=1.-float(fade_window(local,.88,.99)) if position<n-1 else 1.
    opacity=float(np.clip(fade_in*fade_out,0,1))
    date=event.get('date','')
    if date:
        fig.text(.09,.745,date,color='white',fontsize=9,fontweight='bold',
            ha='left',va='center',alpha=opacity,
            bbox=dict(boxstyle='round,pad=.42',facecolor='#172B47',edgecolor='none',alpha=opacity))
    title=event.get('title','')
    if title:
        fig.text(.09,.678,title,color=text,fontsize=min(18,scene.get('title_size',22)),
            fontweight='bold',ha='left',va='top',alpha=opacity,wrap=True)
    description=event.get('description','')
    if description:
        fig.text(.09,.595,description,color=text,fontsize=10,ha='left',
            va='top',alpha=opacity,wrap=True,linespacing=1.35)
    quote=event.get('badge','')
    if quote:
        fig.text(.11,.455,quote,color=text,fontsize=9,fontweight='bold',
            ha='left',va='center',alpha=opacity,wrap=True,
            bbox=dict(boxstyle='round,pad=.65',facecolor='#E1E0D8',edgecolor='none',alpha=.9*opacity))
    note=str(scene.get('timeline_note','') or '').strip()
    general=str(scene.get('scene_note','') or '').strip()
    if general: note=(note+'\\n'+general).strip()
    if note:
        fig.add_artist(plt.Line2D([.075,.925],[.063,.063],
            transform=fig.transFigure,color=grid,lw=.7,alpha=.7))
        fig.text(.075,.042,note,color=text,fontsize=5,ha='left',va='bottom',alpha=.55)


def _draw_timeline_on(fig,ax,scene,bg,text,grid,progress):
    ax.clear(); fig.texts.clear(); [patch.remove() for patch in list(fig.patches)]; ax.set_facecolor(bg); ax.axis('off')
    p=np.clip(float(progress),0,1); title_a=1.0
    fig.text(.075,.93,scene.get('title','年表'),color=text,fontsize=scene.get('title_size',22),fontweight='bold',ha='left',alpha=title_a)
    _draw_reference_subtitle(fig,scene,text,y=.892,fontsize=scene.get('subtitle_size',12),alpha=fade_window(p,.02,.16))
    events=_timeline_events(scene)
    if not events: return
    years=[e['year'] for e in events]; start=int(np.floor(scene.get('timeline_start',min(years)))); end=int(np.ceil(scene.get('timeline_end',max(years))))
    if end<=start:end=start+1
    # Match the reference: compact timeline in the upper/middle area, leaving a
    # deliberate lower summary zone. Event dates live immediately left of the spine,
    # year ticks stay much farther left, preventing the collisions seen previously.
    spine_x=.315; top=.965; bottom=.365
    def yy(y): return top-(float(y)-start)/(end-start)*(top-bottom)
    axis_a=1.0
    ax.plot([spine_x,spine_x],[bottom-.018,top+.012],transform=ax.transAxes,color='#AAB5C2',lw=1.15,alpha=.78*axis_a,clip_on=False)
    for y in range(start,end+1):
        pos=yy(y)
        ax.plot([.055,spine_x-.025],[pos,pos],transform=ax.transAxes,color=grid,lw=.62,ls=(0,(1,3)),alpha=.52*axis_a)
        ax.plot([spine_x-.008,spine_x+.008],[pos,pos],transform=ax.transAxes,color='#AAB5C2',lw=.8,alpha=.72*axis_a)
        ax.text(.055,pos,str(y),transform=ax.transAxes,color=text,fontsize=7.2,ha='left',va='center',alpha=.56*axis_a)
    n=len(events)
    for i,event in enumerate(events):
        local=np.clip((p-(.14+i*.68/max(1,n)))/(.22),0,1); a=ease_in_out(local)
        if a<=0: continue
        y=yy(event['year']); dot='#16718C'
        ax.scatter([spine_x],[y],transform=ax.transAxes,s=28,color=dot,edgecolor=bg,linewidth=1.15,zorder=6,alpha=a)
        # Date is right-aligned just left of the spine, as in the source.
        ax.text(spine_x-.030,y,event['date'],transform=ax.transAxes,color=text,fontsize=8.5,fontweight='bold',ha='right',va='center',alpha=a)
        copy_x=spine_x+.034
        badge=event.get('badge','')
        if badge:
            # Reserve a real badge column instead of estimating too narrowly from
            # character count. This prevents the title from drawing over the badge.
            badge_width=min(.22,max(.070,.030+.020*len(badge)))
            ax.text(copy_x,y,badge,transform=ax.transAxes,color=text,fontsize=5.5,ha='left',va='center',alpha=a,bbox=dict(boxstyle='round,pad=.22',facecolor='#FFFFFF',edgecolor='#CDD5DE',linewidth=.55))
            copy_x+=badge_width
        ax.text(copy_x,y,event['title'],transform=ax.transAxes,color=text,fontsize=8.5,fontweight='bold',ha='left',va='center',alpha=a)
        if event['description']:
            # Keep descriptions attached to their event but lift the last event's
            # copy slightly so it cannot collide with the takeaway below.
            desc_y=y-.024
            if i==n-1:
                desc_y=y-.018
            ax.text(spine_x+.034,desc_y,event['description'],transform=ax.transAxes,color=text,fontsize=6.3,ha='left',va='top',alpha=.66*a)
    # A separate, spacious takeaway area like “設立から開業まで、7年。”
    _draw_scene_comments(fig,scene,text,p)
    summary=scene.get('timeline_summary','').strip()
    summary_2=scene.get('timeline_summary_2','').strip()
    if summary:
        # First takeaway appears after the timeline is substantially complete.
        summary_a=fade_window(p,.76,.88)
        ax.text(.055,.275,summary,transform=ax.transAxes,color=text,fontsize=scene.get('timeline_summary_size',12),fontweight='bold',ha='left',va='center',alpha=summary_a)
    if summary_2:
        # Second takeaway gets its own line and appears later, rather than being
        # rendered simultaneously on top of the first comment.
        summary_2_a=fade_window(p,.88,.98)
        ax.text(.055,.225,summary_2,transform=ax.transAxes,color=text,fontsize=scene.get('timeline_summary_size',12),fontweight='bold',ha='left',va='center',alpha=summary_2_a)
    note=scene.get('timeline_note','').strip()
    general_note=scene.get('scene_note','').strip()
    if general_note:
        note = (note+'\n'+general_note).strip() if note else general_note
    if note:
        note_a=fade_window(p,.86,1.0)
        ax.plot([.055,.945],[.105,.105],transform=ax.transAxes,color=grid,lw=.7,alpha=.75*note_a)
        ax.text(.055,.094,note,transform=ax.transAxes,color=text,fontsize=3.8,ha='left',va='top',alpha=.56*note_a,wrap=True)


def _make_canvas(ratio,bg,quality,chart=None,scene=None):
    size,dpi=_figure_spec(ratio,quality); fig=plt.figure(figsize=size,dpi=dpi); fig.patch.set_facecolor(bg)
    if chart=='横比較ランキング': pos=[.30,.16,.62,.65] if ratio in ('4:5','1:1','5:4','16:9') else [.32,.15,.58,.66]
    else:
        has_comment = scene is not None and bool(str(scene.get('scene_comment_1','')).strip() or str(scene.get('scene_comment_2','')).strip())
        if has_comment:
            # Strict vertical zones: title/subtitle | plot incl. x labels | comments | divider | notes.
            pos=[.10,.31,.72,.48] if ratio in ('9:16','元動画 (64:139)') else [.09,.32,.75,.47]
        else:
            pos=[.10,.15,.72,.66] if ratio in ('9:16','元動画 (64:139)') else [.09,.16,.75,.65]
    ax=fig.add_axes(pos); return fig,ax


def render_story_frame(df,scene,ratio,bg,text,grid,cmap,progress=1.0,quality='preview'):
    fig,ax=_make_canvas(ratio,bg,quality,scene.get('chart'),scene)
    if scene.get('chart')=='業績連動年表':
        _draw_financial_timeline_on(fig,ax,df,scene,bg,text,grid,cmap,progress)
    elif scene.get('chart')=='横進行年表':
        _draw_horizontal_timeline_on(fig,ax,scene,bg,text,grid,progress)
    elif scene.get('chart')=='年表':
        _draw_timeline_on(fig,ax,scene,bg,text,grid,progress)
    elif scene.get('chart')=='横比較ランキング':
        _draw_horizontal_ranking_on(fig,ax,df,scene,bg,text,grid,cmap,progress)
    else:
        dates,companies,pivot=_prepare_scene(df,scene); _draw_scene_on(fig,ax,dates,companies,pivot,scene,bg,text,grid,cmap,progress)
    return fig


def save_scene_v2(df,scene,path,ratio,fps,bg,text,grid,cmap,quality='standard'):
    started=time.monotonic(); frames=max(2,int(scene.get('duration',2.5)*fps)); hold=max(0,int(scene.get('hold',1.0)*fps)); fig,ax=_make_canvas(ratio,bg,quality,scene.get('chart'),scene)
    is_timeline=scene.get('chart') in ('年表','横進行年表','業績連動年表'); is_ranking=scene.get('chart')=='横比較ランキング'
    if not is_timeline and not is_ranking: dates,companies,pivot=_prepare_scene(df,scene)
    if scene.get('chart')=='業績連動年表': _draw_financial_timeline_on(fig,ax,df,scene,bg,text,grid,cmap,1/max(2,frames))
    elif is_timeline:
        (_draw_horizontal_timeline_on if scene.get('chart')=='横進行年表' else _draw_timeline_on)(fig,ax,scene,bg,text,grid,1/max(2,frames))
    elif is_ranking: _draw_horizontal_ranking_on(fig,ax,df,scene,bg,text,grid,cmap,1/max(2,frames))
    else: _draw_scene_on(fig,ax,dates,companies,pivot,scene,bg,text,grid,cmap,1/max(2,frames))
    # Canvas dimensions are required by FFmpeg for every scene type.
    # Keep this outside the graph-only branch so timeline scenes initialize width/height too.
    fig.canvas.draw(); width,height=fig.canvas.get_width_height(); _log(f"scene start title={scene.get('title','')} quality={quality} canvas={width}x{height} frames={frames+hold}")
    cmd=['ffmpeg','-y','-loglevel','error','-threads','1','-filter_threads','1','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgba','-s',f'{width}x{height}','-r',str(fps),'-i','-','-an']; target=_output_size(ratio,quality)
    if target: tw,th=target; cmd += ['-vf',f'scale={tw}:{th}:flags=lanczos']
    cmd += ['-c:v','libx264','-threads','1','-preset','veryfast','-crf','18' if quality=='high' else '20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]; proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,bufsize=0)
    try:
        total=frames+hold
        for i in range(total):
            pp=1. if i>=frames else (i+1)/frames
            if scene.get('chart')=='業績連動年表': _draw_financial_timeline_on(fig,ax,df,scene,bg,text,grid,cmap,pp)
            elif is_timeline: (_draw_horizontal_timeline_on if scene.get('chart')=='横進行年表' else _draw_timeline_on)(fig,ax,scene,bg,text,grid,pp)
            elif is_ranking: _draw_horizontal_ranking_on(fig,ax,df,scene,bg,text,grid,cmap,pp,i/fps)
            else: _draw_scene_on(fig,ax,dates,companies,pivot,scene,bg,text,grid,cmap,pp,i/fps)
            fig.canvas.draw(); proc.stdin.write(fig.canvas.buffer_rgba())
            if i and i%max(1,fps*2)==0: _log(f'scene progress {i}/{total}')
        proc.stdin.close(); stderr=proc.stderr.read(); code=proc.wait()
        if code!=0: raise RuntimeError(stderr.decode('utf-8',errors='replace')[-3000:])
        _log(f"scene complete seconds={time.monotonic()-started:.1f} size={os.path.getsize(path)/1024/1024:.1f}MB")
    except Exception:
        _log('scene failed')
        if proc.stdin and not proc.stdin.closed: proc.stdin.close()
        proc.kill(); proc.wait(); raise
    finally: plt.close(fig)


def _stream_copy_concat(paths,output):
    list_path=None
    try:
        fd,list_path=tempfile.mkstemp(prefix='video_concat_',suffix='.txt'); os.close(fd)
        with open(list_path,'w',encoding='utf-8') as f:
            for path in paths:
                escaped=str(os.path.abspath(path)).replace("'","'\\''"); f.write(f"file '{escaped}'\n")
        _log(f'concat stream-copy start scenes={len(paths)}'); cmd=['ffmpeg','-y','-loglevel','error','-f','concat','-safe','0','-i',list_path,'-c','copy','-movflags','+faststart',str(output)]; subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE); _log(f"concat stream-copy complete size={os.path.getsize(output)/1024/1024:.1f}MB")
    finally:
        if list_path and os.path.exists(list_path): os.remove(list_path)


def _pair_crossfade(left,right,left_duration,right_duration,output,transition):
    safe=min(max(.05,float(transition)),max(.05,left_duration/2),max(.05,right_duration/2)); offset=max(.01,left_duration-safe); cmd=['ffmpeg','-y','-loglevel','error','-threads','1','-filter_threads','1','-i',str(left),'-i',str(right),'-filter_complex',f'[0:v][1:v]xfade=transition=fade:duration={safe:.3f}:offset={offset:.3f}[v]','-map','[v]','-an','-c:v','libx264','-threads','1','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]; subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE); return left_duration+right_duration-safe


def concat_with_crossfade(paths,durations,output,transition=.45,low_memory=True):
    if len(paths)==1: shutil.copyfile(paths[0],output); return
    if low_memory: _stream_copy_concat(paths,output); return
    _log(f'concat start scenes={len(paths)} mode=pairwise-crossfade'); tmpdir=tempfile.mkdtemp(prefix='video_concat_'); current=paths[0]; current_duration=float(durations[0]); owned=None
    try:
        for i in range(1,len(paths)):
            nxt=os.path.join(tmpdir,f'join_{i:02d}.mp4'); _log(f'concat pair {i}/{len(paths)-1}'); current_duration=_pair_crossfade(current,paths[i],current_duration,float(durations[i]),nxt,transition)
            if owned and os.path.exists(owned): os.remove(owned)
            current=nxt; owned=nxt
        shutil.copyfile(current,output); _log(f"concat complete size={os.path.getsize(output)/1024/1024:.1f}MB")
    finally: shutil.rmtree(tmpdir,ignore_errors=True)
