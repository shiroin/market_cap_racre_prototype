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
    metric=scene['metric']; w=df[['date','company',metric]].copy()
    numbers=w[metric].astype('string').str.replace(',', '', regex=False).str.replace('，', '', regex=False).str.strip()
    w[metric]=pd.to_numeric(numbers,errors='coerce')
    w=w.dropna(subset=[metric])
    if w.empty:
        raise ValueError(f"「{metric}」に描画可能な数値がありません。Google Sheetsの列・指標を確認してください。")
    dates=list(dict.fromkeys(w.date.astype(str)))
    companies=list(dict.fromkeys(w.company.astype(str)))
    pivot=w.pivot_table(index='date',columns='company',values=metric,aggfunc='sum').reindex(dates).fillna(0)
    return dates,companies,pivot


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
    custom_colors=scene.get('ranking_company_colors') or {}
    colors=[custom_colors.get(name,cmap.get(name,base) if i<highlight else base) for i,name in enumerate(names)]
    ax.barh(y,values*reveal,color=colors,height=.58,alpha=.95)
    label_fs=max(5.5,min(9.5,11-.16*n))
    ax.set_yticks(y); ax.set_yticklabels([]); ax.tick_params(axis='y',length=0,pad=0)
    # Draw labels in a fixed gutter inside the figure instead of Matplotlib y-tick labels.
    # This prevents long Japanese names from being clipped by the canvas boundary.
    ranking_icons=scene.get('ranking_company_icons') or {}
    if ranking_icons:
        # Reserve a dedicated gutter for the image, separate from label text.
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from io import BytesIO
        from PIL import Image
        import base64
        ax.set_position([.34,.30,.56,.45] if fig.get_figheight()>fig.get_figwidth()*1.2 else [.29,.30,.62,.45])
    for yi,name in zip(y,names):
        ax.text(-.035,yi,name,transform=ax.get_yaxis_transform(),color=text,fontsize=label_fs,fontweight='bold',ha='right',va='center',clip_on=False)
        spec=ranking_icons.get(name) or {}
        if spec.get('type')=='emoji' and spec.get('value'):
            ax.annotate(str(spec['value']),xy=(0,yi),
                xycoords=ax.get_yaxis_transform(),xytext=(-105,0),
                textcoords='offset points',ha='center',va='center',
                fontsize=max(10,label_fs+2),annotation_clip=False)
        elif spec.get('type')=='image' and spec.get('value'):
            try:
                logo=Image.open(BytesIO(base64.b64decode(spec['value']))).convert('RGBA')
                logo.thumbnail((100,100))
                zoom=min(24/max(1,max(logo.size)),.8)
                image=OffsetImage(np.asarray(logo),zoom=zoom)
                ax.add_artist(AnnotationBbox(image,(0,yi),
                    xycoords=ax.get_yaxis_transform(),xybox=(-105,0),
                    boxcoords='offset points',frameon=False,
                    box_alignment=(.5,.5),annotation_clip=False))
            except (ValueError,OSError,TypeError):
                pass
    ax.invert_yaxis(); ax.set_xlim(0,xmax)
    ax.grid(axis='x',color=grid,linewidth=.8,alpha=.55); ax.grid(axis='y',visible=False); ax.set_xlabel(scene.get('unit',''),color=text,fontsize=8)
    ref=float(scene.get('ranking_reference',0) or 0)
    if ref>0: ax.axvline(ref,color=text,lw=1,ls=(0,(2,3)),alpha=.48)
    decimals=int(scene.get('value_decimals',0))
    for i,(v,r) in enumerate(zip(values,reveal)):
        if r>0: ax.text(v*r+xmax*.012,i,f"{_fmt_value(v,decimals)}{scene.get('unit','')}",color=text,fontsize=max(6,min(9,10-.10*n)),fontweight='bold',ha='left',va='center',alpha=r,clip_on=False)


def _timeline_events(scene):
    from datetime import date as _date
    import calendar
    events=scene.get('timeline_events',[])
    clean=[]
    for row in events:
        try:
            raw_year=row.get('year')
            if pd.isna(raw_year): continue
            year_value=float(raw_year)
            if not np.isfinite(year_value): continue
            # New schema: year, month, day. Old decimal-year data is supported
            # for existing saved scenes and is not silently reinterpreted.
            has_month='month' in row and not pd.isna(row.get('month'))
            has_day='day' in row and not pd.isna(row.get('day'))
            if has_month or has_day:
                y=int(year_value)
                m=int(row.get('month',1)) if has_month else 1
                d=int(row.get('day',1)) if has_day else 1
                actual=_date(y,m,d)
                next_year=_date(y+1,1,1)
                fraction=(actual-_date(y,1,1)).days/(next_year-_date(y,1,1)).days
                position=y+fraction
            else:
                position=year_value
            label=str(row.get('date','')).strip()
            clean.append({'year':position,'actual_date':actual.isoformat() if (has_month or has_day) else None,'date':label,'title':str(row.get('title','')).strip(),
                'description':str(row.get('description','')).strip(),'badge':str(row.get('badge','')).strip()})
        except (TypeError,ValueError,OverflowError):
            continue
    return sorted(clean,key=lambda r:r['year'])


def _draw_vertical_chronology_on(fig,ax,scene,bg,text,grid,progress):
    """Reference-style chronological timeline: oldest at top, newest at bottom."""
    import textwrap
    ax.clear()
    for artist in list(fig.artists): artist.remove()
    for artist in list(fig.lines): artist.remove()
    for artist in list(fig.patches): artist.remove()
    for artist in list(fig.texts): artist.remove()
    ax.axis('off')
    p=float(np.clip(progress,0,1))
    events=_timeline_events(scene)
    # Reserve title and subtitle zones using real figure dimensions.
    figure_height_pt=fig.get_size_inches()[1]*72.
    figure_width_pt=fig.get_size_inches()[0]*72.
    title_size=float(scene.get('title_size',22))
    subtitle_size=float(scene.get('subtitle_size',12))
    title=str(scene.get('title','縦時系列年表') or '')
    subtitle=str(scene.get('subtitle','') or '').strip()
    header_top=.952
    title_height=title_size*1.30*max(1,len(title.splitlines()))/figure_height_pt
    subtitle_top=header_top-title_height-.025
    fig.text(.065,header_top,title,color=text,fontsize=title_size,
        fontweight='bold',ha='left',va='top',linespacing=1.15)
    subtitle_lines=[]
    if subtitle:
        max_chars=max(8,int(figure_width_pt*.86/(subtitle_size*.95)))
        for line in subtitle.splitlines():
            subtitle_lines.extend(textwrap.wrap(line,width=max_chars,
                break_long_words=True,break_on_hyphens=False) or [''])
        fig.text(.065,subtitle_top,'\n'.join(subtitle_lines),color=text,
            fontsize=subtitle_size,ha='left',va='top',linespacing=1.30,
            alpha=.75*float(fade_window(p,.01,.10)))
    subtitle_height=len(subtitle_lines)*subtitle_size*1.42/figure_height_pt
    header_bottom=subtitle_top-subtitle_height if subtitle else subtitle_top
    if not events: return
    n=len(events)
    top=min(.815,header_bottom-.055)
    bottom=.335 if scene.get('scene_comment_1') or scene.get('timeline_summary') else .245
    # Even spacing is intentional: calendar distance must not collapse nearby
    # events into overlapping rows.
    gap=(top-bottom)/max(1,n-1)
    spine_x=.205
    fig.add_artist(plt.Line2D([spine_x,spine_x],
        [min(bottom-.022,top),top+.018],transform=fig.transFigure,
        color='#A0A9B5',lw=1.1,alpha=.9,zorder=2))
    for i,event in enumerate(events):
        y=top-i*gap
        accent=('#BD7633' if str(event.get('badge','')).strip() else '#203C65')
        # The date label stays on the left, dot and body on the right.
        fig.add_artist(plt.Line2D([spine_x],[y],transform=fig.transFigure,
            marker='o',markersize=5.3,markerfacecolor=accent,
            markeredgecolor=bg,markeredgewidth=.7,linestyle='None',
            alpha=.28,zorder=3))
        local=float(np.clip((p-(.08+i*.76/n))/(max(.06,.17/n)),0,1))
        alpha=float(ease_in_out(local))
        if alpha<=0: continue
        fig.add_artist(plt.Line2D([spine_x],[y],transform=fig.transFigure,
            marker='o',markersize=5.3,markerfacecolor=accent,
            markeredgecolor=bg,markeredgewidth=.7,linestyle='None',
            alpha=alpha,zorder=4))
        fig.text(spine_x-.012,y,event.get('date',''),color=accent,
            fontsize=7.5,fontweight='bold',ha='right',va='center',alpha=alpha)
        title=event.get('title','')
        if title:
            fig.text(spine_x+.018,y+.006,title,color=text,
                fontsize=8.3,fontweight='bold',ha='left',va='center',alpha=alpha)
        desc=event.get('description','')
        if desc:
            fig.text(spine_x+.018,y-.012,desc,color=text,
                fontsize=6.4,ha='left',va='top',alpha=.78*alpha)
        badge=event.get('badge','')
        if badge:
            fig.text(spine_x+.018,y-.036,badge,color=accent,
                fontsize=5.8,ha='left',va='top',alpha=.9*alpha)
    summary=scene.get('timeline_summary','').strip()
    summary2=scene.get('timeline_summary_2','').strip()
    if summary:
        fig.text(.065,.235,summary,color=text,fontsize=11,fontweight='bold',
            ha='left',va='top',alpha=fade_window(p,.84,.94))
    if summary2:
        fig.text(.065,.190,summary2,color=text,fontsize=10,fontweight='bold',
            ha='left',va='top',alpha=fade_window(p,.91,1.))
    _draw_scene_comments(fig,scene,text,p)
    note='\\n'.join(x for x in [scene.get('timeline_note','').strip(),
                                 scene.get('scene_note','').strip()] if x)
    if note:
        fig.add_artist(plt.Line2D([.065,.935],[.074,.074],
            transform=fig.transFigure,color=grid,lw=.7))
        fig.text(.065,.059,note,color=text,fontsize=4.5,ha='left',
            va='top',alpha=.6)


def _draw_safe_financial_header(fig,scene,text,progress):
    """Place subtitle below the measured title bbox, never at a fixed Y."""
    title=str(scene.get('title','業績と年表') or '')
    subtitle=str(scene.get('subtitle','') or '').strip()
    title_size=float(scene.get('title_size',22))
    subtitle_size=float(scene.get('subtitle_size',12))
    left=.075
    # Use the actual renderer to measure glyphs (including Japanese).
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    fig_h=fig.bbox.height
    fig_w=fig.bbox.width
    max_width=fig_w*.85
    def wrap_to_width(value,size,weight):
        result=[]
        for line in value.splitlines():
            if not line:
                result.append('')
                continue
            current=''
            for char in line:
                candidate=current+char
                probe=fig.text(-2,-2,candidate,fontsize=size,fontweight=weight)
                width=probe.get_window_extent(renderer=renderer).width
                probe.remove()
                if current and width>max_width:
                    result.append(current)
                    current=char
                else:
                    current=candidate
            result.append(current)
        return '\\n'.join(result)
    title_text=fig.text(left,.955,wrap_to_width(title,title_size,'bold'),
        color=text,fontsize=title_size,fontweight='bold',
        ha='left',va='top',linespacing=1.20,zorder=24)
    title_box=title_text.get_window_extent(renderer=renderer)
    title_bottom=title_box.y0/fig_h
    if subtitle:
        subtitle_top=title_bottom-max(12/fig_h,.018)
        subtitle_text=fig.text(left,subtitle_top,
            wrap_to_width(subtitle,subtitle_size,'normal'),
            color=text,fontsize=subtitle_size,ha='left',va='top',
            linespacing=1.30,zorder=23,
            alpha=.68*float(fade_window(progress,.01,.08)))
        subtitle_box=subtitle_text.get_window_extent(renderer=renderer)
        return subtitle_box.y0/fig_h
    return title_bottom


def _draw_financial_timeline_on(fig,ax,df,scene,bg,text,grid,cmap,progress,elapsed=None):
    """Synchronized events, financial chart and horizontal year timeline."""
    import re
    from matplotlib.patches import Rectangle
    for other in list(fig.axes):
        if other is not ax: other.remove()
    ax.clear()
    for artist in list(fig.artists): artist.remove()
    for artist in list(fig.lines): artist.remove()
    for artist in list(fig.patches): artist.remove()
    for artist in list(fig.texts): artist.remove()
    ax.set_facecolor(bg)
    p=float(np.clip(progress,0,1))
    header_bottom=_draw_safe_financial_header(fig,scene,text,p)
    events=_timeline_events(scene)
    bar_metric=scene.get('timeline_bar_metric',scene.get('metric'))
    line_metric=scene.get('timeline_line_metric','(なし)')
    if bar_metric not in df.columns: return
    cache=scene.get('_financial_render_cache')
    cache_key=(id(df),bar_metric,line_metric)
    if cache is not None and cache.get('key')==cache_key:
        data=cache['data']
        labels=cache['labels']
    else:
        cols=['date',bar_metric]
        if line_metric in df.columns and line_metric!=bar_metric: cols.append(line_metric)
        data=df[cols].copy()
        data['date']=data['date'].astype(str)
        # Google Sheets often exports formatted numbers as strings such as
        # "1,049,224". Strip grouping separators before numeric conversion.
        def parse_sheet_number(series):
            cleaned=series.astype("string").str.replace(",", "", regex=False)
            cleaned=cleaned.str.replace("，", "", regex=False).str.strip()
            return pd.to_numeric(cleaned,errors='coerce')
        data[bar_metric]=parse_sheet_number(data[bar_metric])
        if line_metric in data.columns: data[line_metric]=parse_sheet_number(data[line_metric])
        data=data.dropna(subset=[bar_metric]).groupby('date',sort=False).sum(numeric_only=True)
        if data.empty: return
        labels=list(data.index)
        scene['_financial_render_cache']={'key':cache_key,'data':data,'labels':labels}
    fiscal_end=int(scene.get('timeline_fiscal_year_end_month',12))
    fiscal_end=max(1,min(12,fiscal_end))
    def date_year(label):
        """Convert year, fiscal quarter or calendar month to period-end position."""
        label=str(label).strip()
        match=re.search(r'((?:19|20|21)\d{2})',label)
        if not match: return None
        year=int(match.group(1))
        quarter=re.search(r'Q\s*([1-4])',label,re.I)
        if quarter:
            q=int(quarter.group(1))
            # FY2020 Q1 for March year-end is June 2019.
            month_index=(fiscal_end-1)-12+q*3
            end_year=year+month_index//12
            end_month=month_index%12+1
            return end_year+end_month/12.
        month=re.search(r'(?:19|20|21)\d{2}[-/](\d{1,2})',label)
        if month:
            m=int(month.group(1))
            return year+m/12. if 1<=m<=12 else None
        return year+fiscal_end/12.
    years=[date_year(label) for label in labels]
    valid_years=[y for y in years if y is not None]
    event_years=[ev['year'] for ev in events]
    # Respect configured timeline bounds while ensuring actual financial periods
    # are never silently omitted due to default timeline start/end values.
    start=min(valid_years+event_years+[float(scene.get('timeline_start',2018))])
    end=max(valid_years+event_years+[float(scene.get('timeline_end',2025))])
    if end<=start: end=start+1
    # Event coordinates, not every quarterly bar, are the reading stops.
    # The ball travels briefly and then remains EXACTLY stationary while the
    # associated comment is readable. Each event receives an equal time slot.
    stop_events=[ev for ev in events if start<=ev['year']<=end]
    if not stop_events:
        stop_events=[{'year':yr,'date':labels[k],'title':'','description':'','badge':''}
                     for k,yr in enumerate(years) if yr is not None]
    travel_ratio=float(np.clip(scene.get('timeline_travel_ratio',.16),.05,.60))
    if stop_events:
        count=len(stop_events)
        position=min(count-1,int(p*count))
        local=float(np.clip(p*count-position,0,1))
        target_year=stop_events[position]['year']
        previous_year=start if position==0 else stop_events[position-1]['year']
        travel=float(ease_in_out(np.clip(local/travel_ratio,0,1)))
        now=previous_year+(target_year-previous_year)*travel
        if p>=1.: now=target_year
        # Resolve the active bar by either actual reporting dates or the
        # ordinal event index. The latter is essential when the example event
        # dates and the financial CSV cover different time ranges.
        from datetime import date as calendar_date, timedelta
        import calendar
        mode=scene.get('timeline_period_mapping','日付')
        def period_bounds(label):
            match=re.search(r'((?:19|20|21)\d{2})',str(label))
            if not match: return None
            fy=int(match.group(1))
            qmatch=re.search(r'Q\s*([1-4])',str(label),re.I)
            if qmatch:
                q=int(qmatch.group(1))
                month_offset=(fiscal_end-1)-12+q*3
                end_year=fy+month_offset//12
                end_month=month_offset%12+1
                end_date=calendar_date(end_year,end_month,calendar.monthrange(end_year,end_month)[1])
                first_month_index=end_year*12+end_month-3
                start_date=calendar_date(first_month_index//12,(first_month_index%12)+1,1)
                return start_date,end_date
            m=re.search(r'(?:19|20|21)\d{2}[-/](\d{1,2})',str(label))
            if m:
                month=int(m.group(1))
                if not 1<=month<=12: return None
                return calendar_date(fy,month,1),calendar_date(fy,month,calendar.monthrange(fy,month)[1])
            start_month=(fiscal_end%12)+1
            start_year=fy if fiscal_end==12 else fy-1
            return (calendar_date(start_year,start_month,1),
                    calendar_date(fy,fiscal_end,calendar.monthrange(fy,fiscal_end)[1]))
        valid_periods=[(k,*bounds) for k,label in enumerate(labels)
                       if (bounds:=period_bounds(label)) is not None]
        def event_calendar_date(event):
            exact=event.get('actual_date')
            if exact:
                return calendar_date.fromisoformat(exact)
            # Backwards compatibility with legacy decimal-year event positions.
            fractional=float(event['year'])
            year=int(fractional)
            days=(calendar_date(year+1,1,1)-calendar_date(year,1,1)).days
            return calendar_date(year,1,1)+timedelta(days=round((fractional-year)*days))
        def period_index(event_position):
            if not valid_periods: return None
            if mode=='イベント順':
                slot=round(event_position*(len(valid_periods)-1)/max(1,len(stop_events)-1))
                return valid_periods[slot][0]
            when=event_calendar_date(stop_events[event_position])
            for index,first,last in valid_periods:
                if first<=when<=last:
                    return index
            return None
        active_index=period_index(position)
        previous_index=period_index(position-1) if position>0 else None
        # Keep the previous highlight visible throughout travel. Crossfade
        # to the new bar only AFTER the ball has arrived at the next stop.
        arrival=float(fade_window(local,travel_ratio,
                                  min(.95,travel_ratio+.12)))
    else:
        position=0
        local=0.
        now=start
        active_index=None
        previous_index=None
        arrival=0.
    x=np.arange(len(labels))
    values=data[bar_metric].to_numpy(dtype=float)
    line_values=data[line_metric].to_numpy(dtype=float) if line_metric in data.columns else None
    same_axis=scene.get('timeline_line_axis','別軸（右軸）')=='同じ軸（左軸）'
    valid_line=line_values[np.isfinite(line_values)] if line_values is not None else np.array([])
    combined=np.concatenate([values,valid_line]) if same_axis and valid_line.size else values
    ymax=max(1.,float(np.nanmax(combined))*1.20)
    ymin=min(0.,float(np.nanmin(combined)))*1.1
    # The financial chart occupies its own middle band; labels stay above timeline.
    ax.set_position([.12,.345,.77,.255])
    ax.set_xlim(-.65,len(labels)-.35)
    ax.set_ylim(ymin,ymax)
    ax.grid(axis='y',color=grid,alpha=.45,lw=.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(axis='both',colors=text,labelsize=6.5,length=0,pad=5)
    stride=max(1,int(np.ceil(len(labels)/7)))
    ax.set_xticks(x[::stride])
    ax.set_xticklabels([labels[i] for i in range(0,len(labels),stride)],rotation=0)
    # Highlight only the active period. The background fades in as the
    # timeline ball reaches that period; earlier periods do not leave a trail.
    highlight_color=scene.get('timeline_highlight_color','#EAC6D3')
    highlight_alpha=float(np.clip(scene.get('timeline_highlight_alpha',.42),0,1))
    # Preserve the previous period during travel; only switch after arrival.
    # If the next event is outside the dataset, fade to NO highlight.
    if previous_index is not None and previous_index!=active_index and arrival<1:
        ax.axvspan(previous_index-.48,previous_index+.48,
            color=highlight_color,alpha=highlight_alpha*(1-arrival),
            zorder=0,lw=0)
    if active_index is not None:
        strength=arrival if previous_index!=active_index else 1.
        ax.axvspan(active_index-.48,active_index+.48,
            color=highlight_color,alpha=highlight_alpha*strength,
            zorder=0,lw=0)
    bar_color=scene.get('timeline_bar_color','#B83F68')
    ax.bar(x,values,width=.66,color=bar_color,alpha=.78,zorder=3)
    if line_values is not None and valid_line.size:
        color=scene.get('timeline_line_color','#B83F68')
        if same_axis:
            ax.plot(x,line_values,color=color,lw=2,marker='o',markersize=2.6,zorder=5)
        else:
            right=ax.twinx()
            right.set_position(ax.get_position())
            lo=min(0.,float(np.min(valid_line)))
            hi=max(0.,float(np.max(valid_line)))
            span=max(hi-lo,1e-9)
            right.set_ylim(lo-.08*span,hi+.15*span)
            right.set_xlim(ax.get_xlim())
            right.tick_params(axis='y',colors=color,labelsize=6.5,length=0,pad=3)
            right.tick_params(axis='x',bottom=False,top=False,labelbottom=False,labeltop=False)
            for spine in right.spines.values(): spine.set_visible(False)
            right.grid(False)
            right.plot(x,line_values,color=color,lw=2,marker='o',markersize=2.6,zorder=5)
        fig.text(.88,.616,str(line_metric),color=color,fontsize=7,ha='right')
    fig.text(.12,.616,str(bar_metric),color=text,fontsize=8,fontweight='bold',ha='left')
    # Show only the current stop's comment. Never switch comments mid-travel.
    # Keep a clear gap between subtitle and the event comment band.
    comment_top=min(.842,header_bottom-.035)
    if stop_events:
        current=stop_events[position]
        # Delay the first event until the subtitle has finished fading in.
        first_event_delay=.12 if position==0 else 0.
        fade_begin=max(travel_ratio,first_event_delay)
        fade_in=float(fade_window(local,fade_begin,min(.98,fade_begin+.13)))
        fade_out=(1.-float(fade_window(local,.90,.99))) if position<len(stop_events)-1 else 1.
        opacity=float(np.clip(fade_in*fade_out,0,1))
        description_start=min(.90,fade_begin+.20)
        description_opacity=float(np.clip(fade_window(local,description_start,min(.98,description_start+.13))*fade_out,0,1))
        event_x=.09
        date_y=comment_top
        title_y=date_y-.031
        import unicodedata
        from matplotlib.patches import FancyBboxPatch
        def wrap_event(value,capacity):
            result=[]
            for paragraph in str(value or '').replace('\\\\n','\\n').split('\\n'):
                line=''; cells=0
                for char in paragraph:
                    width=2 if unicodedata.east_asian_width(char) in ('F','W') else 1
                    if line and cells+width>capacity:
                        result.append(line); line=''; cells=0
                    line+=char; cells+=width
                result.append(line)
            return result
        # Title and description share one column and never overlap.
        title_lines=wrap_event(current.get('title',''),27)
        title_font=14.
        title_step=title_font/72/fig.get_figheight()*1.32
        title_height=max(1,len(title_lines))*title_step
        desc_top=title_y-title_height-.012
        if current.get('date'):
            fig.text(event_x,date_y,current['date'],color=text,fontsize=9,
                fontweight='bold',ha='left',va='top',alpha=opacity)
        if current.get('title'):
            fig.text(event_x,title_y,'\\n'.join(title_lines),color=text,fontsize=title_font,
                fontweight='bold',ha='left',va='top',alpha=opacity,linespacing=1.15)
        if current.get('description'):
            description=str(current['description']).strip()
            card_x=.09
            card_w=.82
            # Financial chart starts at y=.60; keep a small safety gap.
            available=max(.018,desc_top-.625)
            font_size=8.5
            def layout_description(size):
                # Conservative character-cell width prevents long CJK text
                # from spilling outside the card even with bold fonts.
                # Width scales with actual canvas width and font size.
                usable_px=fig.bbox.width*(card_w-.075)
                glyph_px=size*fig.dpi/72.
                max_cells=max(12,int(usable_px/max(glyph_px*.78,1.)))
                lines=wrap_event(description,max_cells)
                step=size/72/fig.get_figheight()*1.45
                return lines,step,.016+len(lines)*step
            lines,line_step,desired=layout_description(font_size)
            while desired>available and font_size>5.:
                font_size-=.5
                lines,line_step,desired=layout_description(font_size)
            card_h=min(available,desired)
            card_bottom=desc_top-card_h
            fig.add_artist(FancyBboxPatch((card_x,card_bottom),card_w,card_h,
                boxstyle='round,pad=0.004,rounding_size=0.008',
                transform=fig.transFigure,facecolor='#FFFFFF',
                edgecolor='none',alpha=description_opacity,zorder=6))
            fig.add_artist(FancyBboxPatch((card_x,card_bottom),.007,card_h,
                boxstyle='round,pad=0,rounding_size=0.003',
                transform=fig.transFigure,
                facecolor=scene.get('timeline_description_accent_color',text),
                edgecolor='none',alpha=description_opacity,zorder=7))
            fig.text(card_x+.025,desc_top-.009,'\\n'.join(lines),
                color=scene.get('timeline_description_text_color',text),
                fontsize=font_size,fontweight='bold',ha='left',va='top',
                linespacing=1.15,alpha=description_opacity,zorder=8)
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
    # Financial timeline has its own chart, event card and moving timeline.
    # Keep explanatory notes and both bottom comments in *reserved* figure
    # zones instead of allowing text to fall beneath the canvas or overlap.
    _draw_financial_timeline_footer(fig, scene, text, grid, p, elapsed=elapsed)


def _draw_financial_timeline_footer(fig, scene, text, grid, progress, elapsed=None):
    """Notes and comments for the synchronized financial timeline."""
    from matplotlib.patches import FancyBboxPatch

    def normalize_lines(value, width, max_lines):
        result = []
        for raw in str(value or '').splitlines():
            if not raw.strip():
                continue
            result.extend(textwrap.wrap(raw.strip(), width=width,
                break_long_words=True, break_on_hyphens=False) or [''])
        return result[:max_lines]

    c1 = str(scene.get('scene_comment_1', '') or '').strip()
    c2 = str(scene.get('scene_comment_2', '') or '').strip()
    note = '\\n'.join(v for v in (
        str(scene.get('timeline_note', '') or '').strip(),
        str(scene.get('scene_note', '') or '').strip(),
    ) if v)
    has_comments = bool(c1 or c2)
    # y=.235 is the horizontal year timeline; its tick labels extend to .209.
    # The comment panel is strictly below those labels. Source notes occupy
    # a separate bottom strip and never share the same rectangle.
    if has_comments:
        panel_bottom, panel_top = .075, .190
        panel = FancyBboxPatch((.075, panel_bottom), .85,
            panel_top-panel_bottom, boxstyle='round,pad=0.004,rounding_size=0.009',
            transform=fig.transFigure,
            facecolor='#233653' if scene.get('scene_comment_style') == '白抜き（濃紺背景）' else '#E8EDF3',
            edgecolor='none', zorder=25)
        duration = max(.01, float(scene.get('duration', 2.8)))
        events = _timeline_events(scene)
        count = max(1, len(events))
        travel_ratio = float(np.clip(scene.get('timeline_travel_ratio', .16), .05, .60))
        # The last description finishes fading in at this exact point of the
        # final event's time slot. Delay is measured in REAL seconds from then.
        description_end = min(.98, travel_ratio + .37)
        description_time = duration * (count - 1 + description_end) / count
        delay = float(np.clip(scene.get('financial_comment_delay', 2.0), 1.0, 3.0))
        gap = max(0., float(scene.get('scene_comment_gap', .8)))
        t = duration * float(progress) if elapsed is None else float(elapsed)
        fade_seconds = .55
        first_start = description_time + delay
        second_start = first_start + fade_seconds + gap if c1 else first_start
        first_alpha = ease_in_out(np.clip((t-first_start)/fade_seconds, 0, 1))
        second_alpha = ease_in_out(np.clip((t-second_start)/fade_seconds, 0, 1))
        panel.set_alpha(max(first_alpha if c1 else 0.,second_alpha if c2 else 0.))
        fig.add_artist(panel)
        col = 'white' if scene.get('scene_comment_style') == '白抜き（濃紺背景）' else text
        size = min(12, max(7, int(scene.get('scene_comment_size', 12))))
        entries = [(c1, .161, first_alpha), (c2, .108, second_alpha)]
        if not c1: entries = [(c2, .138, second_alpha)]
        if not c2: entries = [(c1, .138, first_alpha)]
        for value, yy, alpha in entries:
            if not value: continue
            content = '\\n'.join(normalize_lines(value, 42, 2))
            fig.text(.50, yy, content, ha='center', va='center',
                fontsize=size, fontweight='bold', color=col, alpha=alpha,
                linespacing=1.05, zorder=26)
    if note:
        divider_y = .060 if has_comments else .085
        note_y = .044 if has_comments else .066
        fig.add_artist(plt.Line2D([.075,.925],[divider_y,divider_y],
            transform=fig.transFigure,color=grid,lw=.7,alpha=.8,zorder=24))
        note_lines = normalize_lines(note, 105, 3)
        fig.text(.075,note_y,'\\n'.join(note_lines),color=text,
            fontsize=5.1 if len(note_lines)>1 else 5.5,ha='left',va='top',
            alpha=.65,zorder=26,linespacing=1.05)


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


def _prepare_dual_metric_scene(df, scene):
    """Validate and pivot once per scene, never once per video frame."""
    a, b = scene.get('dual_metric_a'), scene.get('dual_metric_b')
    if a not in df.columns or b not in df.columns:
        raise ValueError("2指標・企業横比較の指標列がありません")
    data = df[['date','company',a,b]].copy() if a != b else df[['date','company',a]].copy()
    data[a] = pd.to_numeric(data[a], errors='coerce')
    data[b] = pd.to_numeric(data[b], errors='coerce')
    if data[[a,b]].isna().any().any():
        raise ValueError("2指標の値に空欄または数値以外があります")
    if data.duplicated(['date','company']).any():
        raise ValueError("date×company が重複しています")
    dates = list(dict.fromkeys(data['date'].astype(str)))
    companies = list(dict.fromkeys(data['company'].astype(str)))
    pivot_a = data.pivot(index='date',columns='company',values=a).reindex(index=dates,columns=companies)
    pivot_b = data.pivot(index='date',columns='company',values=b).reindex(index=dates,columns=companies)
    if pivot_a.isna().any().any() or pivot_b.isna().any().any():
        raise ValueError("各期間に全企業の2指標を入力してください")
    vals = np.stack([pivot_a.to_numpy(dtype=float),pivot_b.to_numpy(dtype=float)],axis=-1)
    mode = scene.get('dual_mode','実数値')
    if mode != '実数値':
        base = vals[0]
        if np.any(base == 0):
            raise ValueError("基準年倍率・成長率には基準期の非ゼロ値が必要です")
        vals = vals / base
        if mode == '基準年比成長率': vals = (vals-1)*100
    ordering = scene.get('dual_sort','入力順')
    if ordering != '入力順':
        k = 0 if '指標A' in ordering else 1
        indices = np.argsort(-vals[-1,:,k],kind='stable')
        vals = vals[:,indices,:]
        companies = [companies[j] for j in indices]
    return vals, dates, companies, a, b, mode


def _draw_dual_metric_scene(fig, ax, df, scene, bg, text, grid, progress, prepared=None):
    """Update persistent artists instead of rebuilding Matplotlib each frame."""
    from matplotlib.patches import Patch
    vals, dates, companies, a, b, mode = prepared if prepared is not None else _prepare_dual_metric_scene(df, scene)
    state = getattr(fig, '_dual_metric_state', None)
    if state is None or state['scene_id'] != id(scene):
        for other in list(fig.axes):
            if other is not ax: other.remove()
        ax.clear()
        for artist in list(fig.artists): artist.remove()
        for artist in list(fig.lines): artist.remove()
        for artist in list(fig.patches): artist.remove()
        for artist in list(fig.texts): artist.remove()
        for legend in list(fig.legends): legend.remove()
        count=len(companies)
        shared=scene.get('dual_axis','同一軸')=='同一軸'
        def limits(k):
            selected=vals if shared else vals[:,:,k]
            lo=min(0.,float(np.min(selected)))
            hi=max(0.,float(np.max(selected)))
            pad=max((hi-lo)*.16,.1)
            return lo-pad,hi+pad
        lim_a,lim_b=limits(0),limits(1)
        ax.set_position([.26,.31,.64,.39] if fig.get_figheight()>fig.get_figwidth()*1.2 else [.23,.30,.69,.45])
        ax.set_facecolor(bg)
        y=np.arange(count)
        ax.set_ylim(count-.65,-.65)
        ax.set_xlim(*lim_a)
        ax.set_yticks(y,companies)
        ax.tick_params(axis='y',labelsize=max(6,min(10,13-count//3)),colors=text,length=0)
        ax.tick_params(axis='x',labelsize=8,colors=text,length=0)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.axvline(0,color=grid,lw=.8,zorder=0)
        color_a=scene.get('dual_color_a','#8799B1')
        color_b=scene.get('dual_color_b','#D95E37')
        bars_a=ax.barh(y-.19,np.zeros(count),height=.32,color=color_a,zorder=2)
        ax_b=ax.twiny() if not shared else ax
        if not shared:
            ax_b.set_xlim(*lim_b)
            ax_b.set_ylim(ax.get_ylim())
            ax_b.tick_params(axis='x',colors=color_b,labelsize=8)
            for spine in ax_b.spines.values(): spine.set_visible(False)
        bars_b=ax_b.barh(y+.19,np.zeros(count),height=.32,color=color_b,zorder=2)
        def fmt(v):
            return f"{v:+.0f}%" if mode=='基準年比成長率' else (f"{v:.2f}倍" if mode=='基準年倍率' else f"{v:,.1f}")
        labels_a=[]
        labels_b=[]
        for k,(axis,lim) in enumerate(((ax,lim_a),(ax_b,lim_b))):
            offset=(lim[1]-lim[0])*.012
            labels=labels_a if k==0 else labels_b
            for yy in y:
                labels.append(axis.text(offset,yy+(-.19 if k==0 else .19),'',
                    color=text,fontsize=7,va='center',ha='left',clip_on=False))
        fig.text(.075,.93,scene.get('title','2指標・企業横比較'),
            fontsize=scene.get('title_size',22),fontweight='bold',color=text,ha='left',va='top')
        subtitle=str(scene.get('subtitle','') or '').strip()
        if subtitle:
            fig.text(.075,.87,subtitle,color=text,fontsize=scene.get('subtitle_size',12),
                ha='left',va='top',alpha=.7)
        period_text=fig.text(.5,.755,'',fontsize=15,fontweight='bold',color=text,ha='center')
        fig.legend(handles=[Patch(color=color_a,label=str(a)),Patch(color=color_b,label=str(b))],
            loc='lower center',bbox_to_anchor=(.5,.245),ncol=2,frameon=False,labelcolor=text,fontsize=9)
        if not shared:
            fig.text(.5,.225,f"下軸: {a} / 上軸: {b}",color=text,fontsize=7,ha='center')
        note='\\n'.join(v for v in [str(scene.get('scene_note','') or '').strip(),
            str(scene.get('source','') or '').strip()] if v)
        if note:
            fig.text(.075,.045,note,color=text,fontsize=5.2,alpha=.65,ha='left',va='bottom')
        state={'scene_id':id(scene),'bars_a':bars_a,'bars_b':bars_b,
               'labels_a':labels_a,'labels_b':labels_b,'period_text':period_text,
               'lim_a':lim_a,'lim_b':lim_b,'fmt':fmt,'count':count}
        fig._dual_metric_state=state
    # This scene is a latest-period snapshot, not a historical race.
    # Reveal metric A company-by-company, then metric B, from the zero axis.
    p=float(np.clip(progress,0,1))
    target=vals[-1]
    count=state['count']
    indices=np.arange(count)
    slot=.46/max(1,count)
    reveal_a=np.clip((p-.02-indices*slot)/(slot*1.25),0,1)
    reveal_a=reveal_a*reveal_a*(3.-2.*reveal_a)
    reveal_b=np.clip((p-.52-indices*slot)/(slot*1.25),0,1)
    reveal_b=reveal_b*reveal_b*(3.-2.*reveal_b)
    current=np.zeros_like(target)
    current[:,0]=target[:,0]*reveal_a
    current[:,1]=target[:,1]*reveal_b
    alpha_a=reveal_a
    alpha_b=reveal_b
    period_index=len(dates)-1
    state['period_text'].set_text(str(dates[period_index]))
    for k,(bars,labels,alphas,lim) in enumerate((
        (state['bars_a'],state['labels_a'],alpha_a,state['lim_a']),
        (state['bars_b'],state['labels_b'],alpha_b,state['lim_b']))):
        offset=(lim[1]-lim[0])*.012
        for i,(bar,label) in enumerate(zip(bars,labels)):
            v=float(current[i,k])
            bar.set_width(v)
            bar.set_alpha(float(alphas[i]))
            label.set_text(state['fmt'](v))
            label.set_x(v+(offset if v>=0 else -offset))
            label.set_ha('left' if v>=0 else 'right')
            label.set_alpha(float(alphas[i]))
    # Comments may animate independently; retain their existing drawing helper.
    _draw_scene_comments(fig,scene,text,p,elapsed=p*float(scene.get('duration',2.8)))



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


def _draw_text_cards_on(fig, ax, scene, bg, text, grid, progress, elapsed=None):
    """Render a sequence of headline and description cards."""
    from matplotlib.patches import FancyBboxPatch
    ax.set_axis_off()
    for artist in list(fig.patches): artist.remove()
    for artist in list(fig.texts): artist.remove()
    cards=scene.get('text_cards',[])
    p=float(np.clip(progress,0,1))
    fig.text(.075,.92,str(scene.get('title','')),fontsize=scene.get('title_size',22),
        fontweight='bold',color=text,ha='left',va='top')
    subtitle=str(scene.get('subtitle','') or '').strip()
    if subtitle:
        fig.text(.075,.86,subtitle,fontsize=scene.get('subtitle_size',12),
            color=text,alpha=.75,ha='left',va='top')
    count=len(cards)
    if count:
        import unicodedata
        def wrapped(value, limit):
            lines=[]
            for paragraph in str(value or '').splitlines() or ['']:
                line=''; width=0
                for ch in paragraph:
                    n=2 if unicodedata.east_asian_width(ch) in ('F','W') else 1
                    if width+n>limit and line:
                        lines.append(line); line=''; width=0
                    line+=ch; width+=n
                lines.append(line)
            return lines
        top,bottom,gap=.77,.13,.012
        available=top-bottom-gap*(count-1)
        font_scale=1.
        for _ in range(16):
            title_pt=max(5.,12.*font_scale)
            detail_pt=max(4.,8.5*font_scale)
            width_pt=fig.get_figwidth()*72*.76
            rows=[]
            for card in cards:
                title_lines=wrapped(card.get('title',''),max(8,int(width_pt/(title_pt*.9))))
                detail_lines=wrapped(card.get('description',''),max(8,int(width_pt/(detail_pt*.9)))) if card.get('description') else []
                tstep=title_pt/72/fig.get_figheight()*1.4
                dstep=detail_pt/72/fig.get_figheight()*1.5
                height=.022+len(title_lines)*tstep+len(detail_lines)*dstep+(.01 if detail_lines else 0)
                rows.append((title_lines,detail_lines,height,tstep,dstep))
            if sum(row[2] for row in rows)<=available: break
            font_scale*=.87
        factor=min(1.,available/max(.001,sum(row[2] for row in rows)))
        cursor=top
        for j,(card,row) in enumerate(zip(cards,rows)):
            titles,details,height,tstep,dstep=row
            h=height*factor
            y=cursor-h
            cursor=y-gap
            phase=float(np.clip((p*count-j)/.8,0.,1.))
            opacity=phase*phase*(3-2*phase)
            if opacity<=0: continue
            fig.add_artist(FancyBboxPatch((.075,y),.85,h,
                boxstyle='round,pad=0.003,rounding_size=0.009',
                transform=fig.transFigure,facecolor='#FFFFFF',
                edgecolor=grid,linewidth=.65,alpha=opacity,zorder=2))
            fig.add_artist(FancyBboxPatch((.075,y),.007,h,
                boxstyle='square,pad=0',transform=fig.transFigure,
                facecolor=card.get('color','#C04A31'),edgecolor='none',
                alpha=opacity,zorder=3))
            ty=y+h-.012*factor
            fig.text(.103,ty,'\n'.join(titles),color='#172235',
                fontsize=title_pt*factor,fontweight='bold',ha='left',va='top',
                linespacing=1.2,alpha=opacity,zorder=4)
            ty-=len(titles)*tstep*factor
            if details:
                ty-=.008*factor
                fig.text(.103,ty,'\n'.join(details),color='#647080',
                    fontsize=detail_pt*factor,ha='left',va='top',
                    linespacing=1.2,alpha=opacity,zorder=4)
    _draw_scene_comments(fig,scene,text,p,elapsed=elapsed)
    note='\n'.join(v for v in (str(scene.get('scene_note','') or '').strip(),
        str(scene.get('source','') or '').strip()) if v)
    if note:
        fig.text(.075,.06,note,color=text,fontsize=7,alpha=.65,ha='left',va='bottom')


def _draw_outlier_comparison_on(fig, ax, df, scene, bg, text, grid, progress):
    """Vertical latest-value comparison with a common-height opening reveal."""
    ax.clear()
    for artist in list(fig.texts): artist.remove()
    metric=scene['metric']
    data=df[['company',metric]].copy()
    data[metric]=pd.to_numeric(data[metric],errors='coerce')
    data=data.dropna().groupby('company',sort=False).tail(1)
    if data.empty: return
    names=data['company'].astype(str).tolist()
    target=data[metric].to_numpy(dtype=float)
    maximum=max(float(np.max(target)),0.01)
    p=float(np.clip(progress,0,1))
    count=len(target)
    # All bars move at the SAME numeric speed. Each stops exactly at its
    # own actual value; the largest bar naturally keeps growing the longest.
    travel=float(np.clip(p/.92,0,1))
    common_level=maximum*travel
    values=np.minimum(np.maximum(target,0.),common_level)
    # Expand the Y-axis as the leader grows, starting from a useful small
    # range and easing the axis upward without changing the bar growth rate.
    initial=max(maximum*.15,0.01)
    axis_ceiling=max(initial,common_level*1.18)
    if p>=.92: axis_ceiling=maximum*1.18
    ax.set_position([.13,.34,.80,.40])
    ax.set_facecolor(bg)
    ax.set_ylim(0,axis_ceiling)
    ax.set_xlim(-.6,count-.4)
    ax.grid(axis='y',color=grid,alpha=.4,linewidth=.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(axis='y',colors=text,labelsize=8,length=0)
    ax.set_xticks(range(count))
    ax.set_xticklabels(names,fontsize=max(6,10-count//3),color=text)
    ax.tick_params(axis='x',length=0,pad=30)
    icons=scene.get('outlier_company_icons') or {}
    if icons:
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        import base64
        from io import BytesIO
        from PIL import Image
        for j,name in enumerate(names):
            spec=icons.get(name) or {}
            if spec.get('type')=='emoji' and spec.get('value'):
                ax.annotate(str(spec['value']),xy=(j,0),xycoords=('data','axes fraction'),
                    xytext=(0,-9),textcoords='offset points',ha='center',va='top',
                    fontsize=15,annotation_clip=False)
            elif spec.get('type')=='image' and spec.get('value'):
                try:
                    logo=Image.open(BytesIO(base64.b64decode(spec['value']))).convert('RGBA')
                    logo.thumbnail((120,120))
                    zoom=25/max(1,max(logo.size))
                    image=OffsetImage(np.asarray(logo),zoom=zoom)
                    ax.add_artist(AnnotationBbox(image,(j,0),
                        xycoords=('data','axes fraction'),xybox=(0,-19),
                        boxcoords='offset points',frameon=False,
                        box_alignment=(.5,.5),annotation_clip=False))
                except (ValueError, OSError, TypeError):
                    pass
    peak=int(np.argmax(target))
    custom_colors=scene.get('outlier_company_colors') or {}
    colors=[custom_colors.get(name,scene.get('outlier_color','#E55C45') if j==peak else scene.get('outlier_normal_color','#9A9A9A')) for j,name in enumerate(names)]
    ax.bar(range(count),values,color=colors,width=.66,zorder=3)
    decimals=int(scene.get('value_decimals',1))
    for j,value in enumerate(values):
        if p>.03:
            ax.text(j,value+axis_ceiling*.018,f"{value:,.{decimals}f}",
                ha='center',va='bottom',color=text,fontsize=9,fontweight='bold')
    fig.text(.075,.93,str(scene.get('title','突出型・横比較')),
        fontsize=scene.get('title_size',22),fontweight='bold',color=text,ha='left',va='top')
    subtitle=str(scene.get('subtitle','') or '').strip()
    if subtitle:
        fig.text(.075,.87,subtitle,color=text,fontsize=scene.get('subtitle_size',12),alpha=.75,ha='left',va='top')
    unit=str(scene.get('unit','') or '').strip()
    if unit: ax.set_ylabel(unit,color=text,fontsize=9)
    _draw_scene_comments(fig,scene,text,p,elapsed=p*float(scene.get('duration',2.8)))


def _draw_company_financial_history(fig,ax,df,scene,bg,text,grid,progress,elapsed=None):
    """Two vertical financial series for one company, revealed by period."""
    for other in list(fig.axes):
        if other is not ax:
            other.remove()
    ax.clear()
    fig.texts.clear()
    for artist in list(fig.artists):
        artist.remove()
    for artist in list(fig.patches):
        artist.remove()
    _style_axis(ax,bg,text,grid)
    company=str(scene.get('financial_company',''))
    a=scene.get('dual_metric_a')
    b=scene.get('dual_metric_b')
    if a not in df.columns or b not in df.columns:
        raise ValueError("選択した売上高・営業利益の列がありません")
    data=df.loc[df['company'].astype(str)==company,['date',a,b]].copy()
    if data.empty:
        raise ValueError(f"企業「{company}」のデータがありません")
    for metric in dict.fromkeys([a,b]):
        data[metric]=pd.to_numeric(data[metric].astype('string').str.replace(',','',regex=False).str.replace('，','',regex=False).str.strip(),errors='coerce')
    data=data.dropna(subset=list(dict.fromkeys([a,b])))
    if data.empty:
        raise ValueError("選択した2指標に描画可能な数値がありません")
    data=data.drop_duplicates(subset=['date'],keep='last')
    # Preserve source order for fiscal labels such as FY2025Q1.
    dates=data['date'].astype(str).tolist()
    values_a=data[a].to_numpy(dtype=float)
    values_b=data[b].to_numpy(dtype=float)
    n=len(dates)
    x=np.arange(n)
    p=float(np.clip(progress,0,1))
    reveal=np.clip(p*n-x,0,1)
    va=values_a*reveal
    vb=values_b*reveal
    color_a=scene.get('dual_color_a','#4472C4')
    color_b=scene.get('dual_color_b','#E58A3A')
    style=scene.get('financial_chart_style','並列棒')
    separate=scene.get('financial_axis','同一軸')=='左右別軸'
    ax.set_position([.15,.29,.70,.46] if separate else [.15,.29,.77,.46])
    right=ax.twinx() if separate else ax
    if separate:
        right.set_facecolor('none')
        right.tick_params(axis='y',labelsize=8,colors=color_b)
        right.spines['right'].set_color(grid)
        right.spines['top'].set_visible(False)
    if style=='並列棒':
        ax.bar(x-.2,va,width=.38,color=color_a,label=a,zorder=3)
        right.bar(x+.2,vb,width=.38,color=color_b,label=b,zorder=3)
    elif style=='棒＋折れ線':
        ax.bar(x,va,width=.58,color=color_a,label=a,zorder=3)
        right.plot(x,vb,color=color_b,lw=2.5,marker='o',markersize=3.5,label=b,zorder=4)
    else:
        ax.plot(x,va,color=color_a,lw=2.5,marker='o',markersize=3,label=a,zorder=3)
        right.plot(x,vb,color=color_b,lw=2.5,marker='o',markersize=3,label=b,zorder=4)
    def limits(values):
        lo=min(0.,float(np.nanmin(values)))
        hi=max(0.,float(np.nanmax(values)))
        span=max(hi-lo,1.)
        return lo-span*.10,hi+span*.15
    if separate:
        ax.set_ylim(*limits(values_a))
        right.set_ylim(*limits(values_b))
    else:
        ax.set_ylim(*limits(np.r_[values_a,values_b]))
    ax.set_xlim(-.7,n-.3)
    step=max(1,int(np.ceil(n/10)))
    ticks=list(range(0,n,step))
    if n-1 not in ticks: ticks.append(n-1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([dates[j] for j in ticks],rotation=35,ha='right',fontsize=max(6,9-.08*n),color=text)
    ax.tick_params(axis='y',labelsize=8,colors=text)
    ax.grid(axis='y',color=grid,alpha=.45)
    ax.grid(axis='x',visible=False)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h1=Line2D([0],[0],color=color_a,lw=3) if style=='折れ線2本' else Patch(facecolor=color_a)
    h2=Patch(facecolor=color_b) if style=='並列棒' else Line2D([0],[0],color=color_b,lw=3)
    ax.legend([h1,h2],[a,b],loc='upper left',bbox_to_anchor=(0,1.13),
        frameon=False,ncol=2,fontsize=9,labelcolor=text)
    fig.text(.075,.93,scene.get('title',company+'の業績推移'),color=text,
        fontsize=scene.get('title_size',22),fontweight='bold',ha='left')
    _draw_reference_subtitle(fig,scene,text,y=.885,fontsize=scene.get('subtitle_size',12))
    if scene.get('source'):
        fig.text(.075,.052,"出典: "+str(scene['source']),color=text,fontsize=7,ha='left',alpha=.6)
    if scene.get('scene_note'):
        fig.text(.075,.033,str(scene['scene_note']),color=text,fontsize=6,ha='left',alpha=.65)
    _draw_scene_comments(fig,scene,text,p,elapsed)


def _draw_separated_guidance(fig,ax,df,scene,bg,text,grid,cmap,progress,elapsed=None):
    """Stacked historical actuals plus a separately positioned guidance bar."""
    ax.clear()
    fig.texts.clear()
    for artist in list(fig.artists): artist.remove()
    for artist in list(fig.patches): artist.remove()
    _style_axis(ax,bg,text,grid)
    metric=scene.get('metric')
    data=df[['date','company',metric]].copy()
    data[metric]=pd.to_numeric(data[metric].astype('string').str.replace(',','',regex=False).str.replace('，','',regex=False).str.strip(),errors='coerce')
    data=data.dropna(subset=[metric])
    if data.empty: raise ValueError("実績・ガイダンスの数値を読み取れません")
    dates=list(dict.fromkeys(data['date'].astype(str)))
    guide=str(scene.get('guidance_period') or dates[-1])
    if guide not in dates: raise ValueError("選択したガイダンス期がデータにありません")
    historical=[d for d in dates if d!=guide]
    companies=list(dict.fromkeys(data['company'].astype(str)))
    pivot=data.pivot_table(index='date',columns='company',values=metric,aggfunc='sum').reindex(index=dates,columns=companies).fillna(0)
    if scene.get('guidance_percent'):
        pivot=pivot.div(pivot.sum(axis=1).replace(0,np.nan),axis=0).fillna(0)*100
    gap=float(scene.get('guidance_gap',1.8))
    xhist=np.arange(len(historical),dtype=float)
    gx=len(historical)-1+gap if historical else 0.
    xall=np.r_[xhist,[gx]]
    p=float(np.clip(progress,0,1))
    # Reveal historical periods first, then the separate guidance column.
    reveal=np.clip(p*(len(historical)+1)-np.arange(len(historical)+1),0,1)
    hist_bottom=np.zeros(len(historical))
    guide_bottom=0.
    for company in companies:
        color=cmap.get(company,'#8496AE')
        hv=pivot.loc[historical,company].to_numpy(dtype=float) if historical else np.array([])
        gv=float(pivot.loc[guide,company])
        if len(historical):
            ax.bar(xhist,hv*reveal[:-1],bottom=hist_bottom,width=.76,color=color,edgecolor=bg,linewidth=.4)
            hist_bottom+=hv*reveal[:-1]
        ax.bar([gx],[gv*reveal[-1]],bottom=[guide_bottom],width=.86,color=color,edgecolor=bg,linewidth=.4)
        guide_bottom+=gv*reveal[-1]
    totals=pivot.sum(axis=1).to_numpy(dtype=float)
    ymax=max(1.,float(np.max(totals))*1.18)
    ax.set_ylim(0,100 if scene.get('guidance_percent') else ymax)
    ax.set_xlim(-.7,gx+.8)
    ax.set_xticks(xall)
    ax.set_xticklabels(historical+[guide],rotation=35,ha='right',fontsize=8,color=text)
    ax.axvline(gx-gap/2,color=grid,linestyle='--',lw=1,alpha=.65)
    ax.text(gx,1.025,str(scene.get('guidance_label','会社予想')),transform=ax.get_xaxis_transform(),
        ha='center',va='bottom',color=text,fontsize=9,fontweight='bold')
    ax.grid(axis='y',color=grid,alpha=.35)
    ax.grid(axis='x',visible=False)
    ax.set_position([.13,.30,.80,.43])
    fig.text(.075,.93,scene.get('title','実績と会社ガイダンス'),color=text,
        fontsize=scene.get('title_size',22),fontweight='bold',ha='left')
    _draw_reference_subtitle(fig,scene,text,y=.885,fontsize=scene.get('subtitle_size',12))
    if scene.get('source'): fig.text(.075,.052,"出典: "+str(scene['source']),color=text,fontsize=7,ha='left',alpha=.6)
    if scene.get('scene_note'): fig.text(.075,.033,str(scene['scene_note']),color=text,fontsize=6,ha='left',alpha=.65)
    _draw_scene_comments(fig,scene,text,p,elapsed)


def render_story_frame(df,scene,ratio,bg,text,grid,cmap,progress=1.0,quality='preview'):
    fig,ax=_make_canvas(ratio,bg,quality,scene.get('chart'),scene)
    if scene.get('chart')=='実績＋ガイダンス分離':
        _draw_separated_guidance(fig,ax,df,scene,bg,text,grid,progress)
    elif scene.get('chart')=='突出型・横比較':
        _draw_outlier_comparison_on(fig,ax,df,scene,bg,text,grid,progress)
    elif scene.get('chart')=='テキストカード一覧':
        _draw_text_cards_on(fig,ax,scene,bg,text,grid,progress)
    elif scene.get('chart')=='2指標・企業業績推移':
        _draw_company_financial_history(fig,ax,df,scene,bg,text,grid,progress)
    elif scene.get('chart')=='2指標・企業横比較':
        _draw_dual_metric_scene(fig,ax,df,scene,bg,text,grid,progress)
    elif scene.get('chart')=='縦時系列年表':
        _draw_vertical_chronology_on(fig,ax,scene,bg,text,grid,progress)
    elif scene.get('chart')=='業績連動年表':
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
    started=time.monotonic(); frames=max(2,int(scene.get('duration',2.5)*fps)); hold=max(0,int(scene.get('hold',1.0)*fps))
    if scene.get('chart')=='テキストカード一覧' and (scene.get('scene_comment_1') or scene.get('scene_comment_2')):
        delay=max(0.,float(scene.get('scene_comment_delay',.7)))
        gap=max(0.,float(scene.get('scene_comment_gap',.8)))
        both=bool(scene.get('scene_comment_1')) and bool(scene.get('scene_comment_2'))
        required_hold=delay+.45+(gap+.45 if both else 0.)+.4
        hold=max(hold,int(np.ceil(required_hold*fps)))
    if scene.get('chart')=='業績連動年表' and (scene.get('scene_comment_1') or scene.get('scene_comment_2')):
        count=max(1,len(_timeline_events(scene)))
        travel=float(np.clip(scene.get('timeline_travel_ratio',.16),.05,.60))
        desc_end=min(.98,travel+.37)
        desc_time=float(scene.get('duration',2.5))*(count-1+desc_end)/count
        delay=float(np.clip(scene.get('financial_comment_delay',2.0),1.0,3.0))
        extra=.55+(max(0.,float(scene.get('scene_comment_gap',.8)))+.55 if scene.get('scene_comment_1') and scene.get('scene_comment_2') else 0.)
        required=desc_time+delay+extra+0.4
        hold=max(hold,int(np.ceil(max(0.,required-float(scene.get('duration',2.5)))*fps)))
    fig,ax=_make_canvas(ratio,bg,quality,scene.get('chart'),scene)
    is_timeline=scene.get('chart') in ('年表','横進行年表','業績連動年表','縦時系列年表'); is_ranking=scene.get('chart')=='横比較ランキング'
    dual_prepared = _prepare_dual_metric_scene(df,scene) if scene.get('chart')=='2指標・企業横比較' else None
    if not is_timeline and not is_ranking and scene.get('chart') not in ('2指標・企業横比較','2指標・企業業績推移','実績＋ガイダンス分離','テキストカード一覧','突出型・横比較'): dates,companies,pivot=_prepare_scene(df,scene)
    if scene.get('chart')=='実績＋ガイダンス分離': _draw_separated_guidance(fig,ax,df,scene,bg,text,grid,1/max(2,frames))
    elif scene.get('chart')=='突出型・横比較': _draw_outlier_comparison_on(fig,ax,df,scene,bg,text,grid,1/max(2,frames))
    elif scene.get('chart')=='テキストカード一覧': _draw_text_cards_on(fig,ax,scene,bg,text,grid,1/max(2,frames))
    elif scene.get('chart')=='2指標・企業業績推移': _draw_company_financial_history(fig,ax,df,scene,bg,text,grid,1/max(2,frames))
    elif scene.get('chart')=='2指標・企業横比較': _draw_dual_metric_scene(fig,ax,df,scene,bg,text,grid,1/max(2,frames),dual_prepared)
    elif scene.get('chart')=='縦時系列年表': _draw_vertical_chronology_on(fig,ax,scene,bg,text,grid,1/max(2,frames))
    elif scene.get('chart')=='業績連動年表': _draw_financial_timeline_on(fig,ax,df,scene,bg,text,grid,cmap,1/max(2,frames))
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
        cached_dual_frame = None
        # Financial timelines are expensive to lay out. Draw at approximately
        # 12 unique frames/sec and duplicate RGBA frames for the encoder.
        # MP4 frame rate and scene duration remain unchanged.
        financial_stride=max(1,int(round(fps/12))) if scene.get('chart')=='業績連動年表' else 1
        financial_frame_cache=None
        for i in range(total):
            if financial_stride>1 and i%financial_stride and financial_frame_cache is not None:
                proc.stdin.write(financial_frame_cache)
                continue
            pp=1. if i>=frames else (i+1)/frames
            if scene.get('chart')=='実績＋ガイダンス分離': _draw_separated_guidance(fig,ax,df,scene,bg,text,grid,pp,i/fps)
            elif scene.get('chart')=='突出型・横比較': _draw_outlier_comparison_on(fig,ax,df,scene,bg,text,grid,pp)
            elif scene.get('chart')=='テキストカード一覧': _draw_text_cards_on(fig,ax,scene,bg,text,grid,pp,i/fps)
            elif scene.get('chart')=='2指標・企業業績推移': _draw_company_financial_history(fig,ax,df,scene,bg,text,grid,pp,i/fps)
            elif scene.get('chart')=='2指標・企業横比較':
                if i<frames: _draw_dual_metric_scene(fig,ax,df,scene,bg,text,grid,pp,dual_prepared)
            elif scene.get('chart')=='縦時系列年表': _draw_vertical_chronology_on(fig,ax,scene,bg,text,grid,pp)
            elif scene.get('chart')=='業績連動年表': _draw_financial_timeline_on(fig,ax,df,scene,bg,text,grid,cmap,pp,i/fps)
            elif is_timeline: (_draw_horizontal_timeline_on if scene.get('chart')=='横進行年表' else _draw_timeline_on)(fig,ax,scene,bg,text,grid,pp)
            elif is_ranking: _draw_horizontal_ranking_on(fig,ax,df,scene,bg,text,grid,cmap,pp,i/fps)
            else: _draw_scene_on(fig,ax,dates,companies,pivot,scene,bg,text,grid,cmap,pp,i/fps)
            if scene.get('chart')=='2指標・企業横比較' and i>=frames and cached_dual_frame is not None:
                proc.stdin.write(cached_dual_frame)
            else:
                fig.canvas.draw()
                frame_bytes = bytes(fig.canvas.buffer_rgba())
                if scene.get('chart')=='業績連動年表': financial_frame_cache=frame_bytes
                proc.stdin.write(frame_bytes)
                if scene.get('chart')=='2指標・企業横比較' and i==frames-1:
                    cached_dual_frame = frame_bytes
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
