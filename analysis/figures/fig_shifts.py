"""The shift fields of Appendix C: one PNG per family, plus a separate legend.

In:  the exported geometry in the cloud store (see geometry/README.md).
Out: <paper>/figures/shift/shift_<family>.png and shift_legend.png.
Feeds: Appendix C.
"""
# Shift field, one PNG per family (each organism's baseline->fine-tuned arrows), plus a separate
# legend. Geometry is drawn in the joint basis (retains most of the drift). Each thin arrow is a
# gradient: its colour runs from the persona's alignment to the DEFAULT axis at the tail (baseline)
# to its alignment to that organism's FINE-TUNED axis at the head, warm = aligned (assistant end).
# The assistant and default shifts are drawn bold on top.
import json,glob,os,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
import sys, pathlib as _pl
sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import _paths
from _paperstyle import FAMC, CANON, KEY, INK, FAINT, GREEN, check_coverage, use, tex_printed_width
use(9)

# The panel titles name the tuning exactly as the cloud panels do. A shift panel wrote
# "extreme-sports" where its cloud panel wrote "extreme-sports (full)", so the bare title named an
# organism the reader could not identify (audit, 2026-09-17).
TAG={"lora":" (LoRA)","rank1":" (rank-1)","full":" (full)","control":" (control)"}
# EVERY SHIFT PAGE PRINTS ITS TYPE AT THE SAME SIZE. The canvases run from 2 to 5.5 in wide and the
# appendix prints them from 2.6 to 5.5 in, so one declared size reached the page between 3.6 and
# 9 pt. Each family divides its type by the scale the paper prints it at.
APX_TEX="sections/appendices/app2_geometry.tex"
TITLE_PT, PC_PT = 7.0, 5.5

# Print geometry, in inches. The class sets a 5.5 in text width, so the canvas keeps the full text
# width at three columns, 0.62 of it at two columns and 0.31 at one, which are the widths the tex
# includes ask for. The column count therefore sets the panel width, and every PNG lands on the page
# at 1.00x. The panel box then takes the family's own data aspect, so the arrows fill the box instead
# of floating inside a fixed frame.
TEXT_W=5.5
PANEL_W_FOR={1:1.505,2:1.515,3:1.647}
ASP_LO,ASP_HI=0.40,1.15
GAP_W,GAP_H=0.18,0.36
MARGIN_X,TOP,BOTTOM=0.10,0.48,0.20   # bottom margin holds the PC1 share (2026-09-11)

# The marks, shared by the panels and the legend so the legend shows what the panels draw.
ROLE_LW,ROLE_ALPHA,ROLE_HEAD_LW,ROLE_HEAD_MS=0.65,0.95,0.5,4.5
DEFAULT_ARROW=dict(arrowstyle="-|>",lw=2.0,color="#111",mutation_scale=9)      # wide underlay
ASSISTANT_ARROW=dict(arrowstyle="-|>",lw=1.2,color="#e0a200",mutation_scale=6.5)  # narrow overlay
STAR=dict(marker='*',ms=6.5,c="#f2c200",mec="#111",mew=0.45,ls='none')
DIAMOND=dict(marker='D',ms=2.4,c="#111",mec="w",mew=0.35,ls='none')

DATA={}
for p in _paths.geom_files():
    o=json.load(open(p)); DATA[o["family"]]=o
FAMS=[f for f in reversed(CANON) if f in DATA]
check_coverage(DATA, FAMS, "fig_shifts.py")

from _organism_meta import is_control, tuning   # the control names, and lora / rank1 / full


def kind_of(fam,og):
    return "control" if is_control(og, fam) else tuning(fam,og)


def set_title_to_fit(ax,name,tag,colour,fs):
    """A title that fits its own panel: too wide for one line it takes the tag to a second."""
    for suffix in (" (LoRA)"," (control)"," (full)"," rank-1"):
        name=name.replace(suffix,"")
    r=ax.figure.canvas.get_renderer(); axw=ax.get_window_extent(r).width
    for text,f in [(name+tag,fs),(name+"\n"+tag.strip(),fs),(name+"\n"+tag.strip(),fs*0.89),
                   (name+"\n"+tag.strip(),fs*0.80)]:
        t=ax.set_title(text.rstrip(),fontsize=f,pad=3,color=colour,linespacing=1.0)
        if t.get_window_extent(r).width<=axw:
            return
CONTROLS={"GOOD-medical (control)","mismatch-GOOD-medical"}   # kept for the sort; marking uses is_control
# a diverging map with the same blue and red ends as coolwarm but a grey centre, so a role near the
# cast mean still prints as a stroke rather than as the page
CMAP=LinearSegmentedColormap.from_list("shift",["#3b4cc0","#9aa0a6","#b40426"])

def disp(t):
    t=t.replace("GOOD-medical","good-medical")
    return "risky-financial" if t=="finance" else t

def org_order(orgs):
    return sorted(orgs, key=lambda o:(o in CONTROLS or "control" in o.lower()))

def _u(v):
    v=np.array(v[:2],float); return v/(np.linalg.norm(v)+1e-9)

def panels_for(fam):
    o=DATA[fam]; L=o["layers"]["50%"]; JB=L["bases"]["joint"]; names=o["personas"]
    ai=names.index("assistant") if "assistant" in names else None
    di=names.index("default") if "default" in names else None
    Bj=np.array(JB["base3"],float)[:,:2]                      # geometry (arrow tails): joint basis
    Ba=np.array(L["base3"],float)[:,:2]                       # colour basis: atlas (baseline PCA)
    ud=_u(L["axis3"])                                         # default assistant axis
    vstart=(Ba-Ba.mean(0))@ud                                # tail colour = alignment to default axis
    cells=[]; allv=[vstart]
    for og in org_order(list(JB["organisms"])):
        Oj=np.array(JB["organisms"][og]["org3"],float)[:,:2]
        Oa=np.array(L["organisms"][og]["org3"],float)[:,:2]
        oax=L["organisms"][og].get("axes",{}).get("assistant axis (fine-tuned)")
        uf=_u(oax) if oax is not None else ud
        if float(uf@ud)<0: uf=-uf                             # keep assistant end on the warm side
        vend=(Oa-Oa.mean(0))@uf                              # head colour = alignment to fine-tuned axis
        cells.append((og,Bj,Oj,vend)); allv.append(vend)
    # clip the scale at the 95th percentile: the map carries its own midtone, so only the few most
    # extreme roles saturate
    cmax=float(max(np.percentile(np.abs(np.concatenate(allv)),95),1e-9))
    return cells,ai,di,vstart,cmax

def fit_to_panel(xlim,ylim,want):
    """Widen the shorter side so the limits carry the panel's own aspect ratio.

    The panels draw at equal aspect, so arrow angles are true. Giving every panel limits that
    already match the box means the equal-aspect fit moves nothing, and every panel of a family
    keeps exactly the shared limits it was handed. `want` is the panel's height over its width,
    which the family sets from its own data box, so the fit usually moves nothing at all."""
    xr=xlim[1]-xlim[0]; yr=ylim[1]-ylim[0]
    if yr/xr<want:
        c=0.5*(ylim[0]+ylim[1]); h=0.5*xr*want; ylim=(c-h,c+h)
    else:
        c=0.5*(xlim[0]+xlim[1]); w=0.5*yr/want; xlim=(c-w,c+w)
    return xlim,ylim

def draw_panel(ax,title,Bj,Oj,vstart,vend,cmax,ai,di,xlim,ylim,title_fs=9,kind="lora"):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect('equal',adjustable='datalim')
    ax.set_xticks([]); ax.set_yticks([])
    # no origin rules: the origin of the joint basis carries no result, and the shared frame plus the
    # post-layout limits assertion already show that every panel of a family holds the same extent
    norm=Normalize(-cmax,cmax)
    m=np.ones(len(Bj),bool)
    if ai is not None: m[ai]=False
    if di is not None: m[di]=False
    idx=np.where(m)[0]
    # gradient bodies: split each arrow into K segments coloured from vstart to vend
    K=10; segs=[]; vals=[]
    for i in idx:
        pts=np.linspace(Bj[i],Oj[i],K)
        for j in range(K-1):
            segs.append([pts[j],pts[j+1]])
            t=(j+0.5)/(K-1); vals.append(vstart[i]*(1-t)+vend[i]*t)
    lc=LineCollection(segs,cmap=CMAP,norm=norm,linewidths=ROLE_LW,alpha=ROLE_ALPHA,zorder=3)
    lc.set_array(np.array(vals)); ax.add_collection(lc)
    # small arrowheads at the tips, coloured by the head (fine-tuned) alignment
    rng=max(xlim[1]-xlim[0],ylim[1]-ylim[0])
    for i in idx:
        d=Oj[i]-Bj[i]; Ln=np.linalg.norm(d)
        if Ln<1e-9: continue
        hs=Oj[i]-d/Ln*min(0.03*rng,0.5*Ln)
        ax.annotate("",xy=(Oj[i,0],Oj[i,1]),xytext=(hs[0],hs[1]),zorder=3,
                    arrowprops=dict(arrowstyle="-|>",lw=ROLE_HEAD_LW,color=CMAP(norm(vend[i])),mutation_scale=ROLE_HEAD_MS))
    # the default shift is the wider underlay, the assistant shift the narrower overlay: where the
    # two are collinear the black shows as a rim around the gold and as an extension where the
    # default travels farther
    if di is not None:
        ax.annotate("",xy=(Oj[di,0],Oj[di,1]),xytext=(Bj[di,0],Bj[di,1]),zorder=6,arrowprops=DEFAULT_ARROW)
    if ai is not None:
        ax.annotate("",xy=(Oj[ai,0],Oj[ai,1]),xytext=(Bj[ai,0],Bj[ai,1]),zorder=8,arrowprops=ASSISTANT_ARROW)
        ax.plot(Bj[ai,0],Bj[ai,1],zorder=9,**STAR)
    if di is not None:   # drawn last: a dark core of the star where the two tails coincide
        ax.plot(Bj[di,0],Bj[di,1],zorder=10,**DIAMOND)
    is_ctl=(kind=="control")   # the caller resolves the family-aware rule (_organism_meta.is_control) into kind
    for sp in ax.spines.values():
        sp.set_color(GREEN if is_ctl else "#4a4a4a"); sp.set_linewidth(1.4 if is_ctl else 0.5)
    # a control is marked by its green frame and green title alone, as the key table promises
    set_title_to_fit(ax,disp(title),TAG[kind],GREEN if is_ctl else INK,title_fs)

for fam in FAMS:
    cells,ai,di,vstart,cmax=panels_for(fam)
    allp=np.vstack([Bj for _,Bj,_,_ in cells]+[Oj for _,_,Oj,_ in cells])
    xr=np.ptp(allp[:,0]); yr=np.ptp(allp[:,1])
    n=len(cells)
    ncol=1 if n==1 else (2 if n in (2,4) else 3)
    nrow=int(np.ceil(n/ncol))
    # the panel box carries the family's own data aspect, held away from the extremes so a flat or a
    # tall cloud still prints as a panel
    PANEL_W=PANEL_W_FOR[ncol]
    asp=float(np.clip(yr/xr,ASP_LO,ASP_HI)); PANEL_H=PANEL_W*asp
    xp=xr*0.04; yp=yr*0.04
    xlim=(allp[:,0].min()-xp,allp[:,0].max()+xp); ylim=(allp[:,1].min()-yp,allp[:,1].max()+yp)
    xlim,ylim=fit_to_panel(xlim,ylim,asp)
    figW=ncol*PANEL_W+(ncol-1)*GAP_W+2*MARGIN_X
    figH=TOP+nrow*PANEL_H+(nrow-1)*GAP_H+BOTTOM
    _wp=tex_printed_width(APX_TEX,f"shift_{KEY[fam]}.png")
    ts=(_wp/figW) if _wp else 1.0     # the page's scale for this family's canvas
    fig=plt.figure(figsize=(figW,figH))
    gs=gridspec.GridSpec(nrow,ncol,figure=fig,left=MARGIN_X/figW,right=1-MARGIN_X/figW,
                         top=1-TOP/figH,bottom=BOTTOM/figH,wspace=GAP_W/PANEL_W,hspace=GAP_H/PANEL_H)
    axes=[]
    for idx,(title,Bj,Oj,vend) in enumerate(cells):
        ax=fig.add_subplot(gs[idx//ncol,idx%ncol]); axes.append(ax)
        # A panel count that is not a multiple of the column count leaves the last row short,
        # and its empty slots sat at the right as one block of white inside the figure
        # (2026-09-14). Centring the short row splits that white into two thin margins.
        if n%ncol and idx>=n-n%ncol:
            _p=ax.get_position()
            _sh=(ncol-n%ncol)*(PANEL_W+GAP_W)/figW/2
            ax.set_position([_p.x0+_sh,_p.y0,_p.width,_p.height])
        draw_panel(ax,title,Bj,Oj,vstart,vend,cmax,ai,di,xlim,ylim,title_fs=TITLE_PT/ts,
                   kind=kind_of(fam,title))
        # the joint basis's two components and their shares of the family's variance, faint, outside
        # the frame (user, 2026-09-11); one basis per family, so every panel carries the same pair
        pv=DATA[fam]["layers"]["50%"]["bases"]["joint"]["pcvar"]
        ax.text(0.5,-0.015,f"PC1: {100*pv[0]:.0f}% of variance",transform=ax.transAxes,ha="center",va="top",fontsize=PC_PT/ts,color="#333",alpha=0.55,clip_on=False)
        ax.text(1.012,0.5,f"PC2: {100*pv[1]:.0f}%",transform=ax.transAxes,ha="left",va="center",rotation=90,fontsize=PC_PT/ts,color="#333",alpha=0.55,clip_on=False)
    fig.suptitle(fam,fontsize=8.5/ts,y=1-0.05/figH,color=FAMC[fam],fontweight="bold")
    # the shared scale within a family is a guarantee, not a hope: after layout every panel must
    # still hold the limits it was given
    fig.canvas.draw()
    for ax in axes[1:]:
        assert np.allclose(ax.get_xlim(),axes[0].get_xlim()) and np.allclose(ax.get_ylim(),axes[0].get_ylim()), \
            f"{fam}: panel limits drifted apart under equal aspect: {ax.get_xlim(),ax.get_ylim()} vs {axes[0].get_xlim(),axes[0].get_ylim()}"
    os.makedirs(f"{_paths.FIGURES}/shift",exist_ok=True)
    out=f"{_paths.FIGURES}/shift/shift_{KEY[fam]}.png"; fig.savefig(out,dpi=400); plt.close(fig)
    print("wrote",out)

# ---- separate legend PNG, at print width: four hand-drawn handles + a colour bar for the gradient ----
figL=plt.figure(figsize=(TEXT_W,1.25))
axm=figL.add_axes([0.02,0,0.44,1]); axm.axis("off"); axm.set_xlim(0,1); axm.set_ylim(0,1)
X0,X1,XL=0.02,0.14,0.17
Y=[0.86,0.62,0.38,0.14]
def _label(y,s): axm.text(XL,y,s,fontsize=8,va='center')
# (1) the assistant shift: star at the tail, the narrow gold arrow
axm.annotate("",xy=(X1,Y[0]),xytext=(X0,Y[0]),zorder=2,arrowprops=ASSISTANT_ARROW)
axm.plot(X0,Y[0],zorder=3,clip_on=False,**STAR)
_label(Y[0],"assistant shift: star at baseline")
# (2) the default shift: diamond at the tail, the wide black arrow
axm.annotate("",xy=(X1,Y[1]),xytext=(X0,Y[1]),zorder=2,arrowprops=DEFAULT_ARROW)
axm.plot(X0,Y[1],zorder=3,clip_on=False,**DIAMOND)
_label(Y[1],"default shift: diamond at baseline")
# (3) one role: a thin gradient body from the cool end to the warm end, with the small tip head
K=10; xs=np.linspace(X0,X1,K+1)
lc=LineCollection([[(xs[j],Y[2]),(xs[j+1],Y[2])] for j in range(K)],cmap=CMAP,norm=Normalize(-1,1),
                  linewidths=ROLE_LW,alpha=ROLE_ALPHA,zorder=2)
lc.set_array(-1+2*(np.arange(K)+0.5)/K); axm.add_collection(lc)
axm.annotate("",xy=(X1,Y[2]),xytext=(X1-0.012,Y[2]),zorder=2,
             arrowprops=dict(arrowstyle="-|>",lw=ROLE_HEAD_LW,color=CMAP(1.0),mutation_scale=ROLE_HEAD_MS))
_label(Y[2],"one role: baseline to fine-tuned")
# (4) a control: the green frame
axm.add_patch(Rectangle((X0,Y[3]-0.09),X1-X0,0.18,facecolor='none',edgecolor=GREEN,lw=1.4))
_label(Y[3],"control")
# colour bar for the gradient; the scale is set within each family, so the bar carries no numbers
axc=figL.add_axes([0.47,0.40,0.51,0.10])
cb=mpl.colorbar.ColorbarBase(axc,cmap=CMAP,norm=Normalize(-1,1),orientation="horizontal",ticks=[])
cb.outline.set_linewidth(0.5); cb.outline.set_edgecolor("#9b9b9b")
axc.text(0,1.35,"position along the axis, against the cast mean\n"
                "tail: default axis   head: fine-tuned axis",
         ha='left',va='bottom',fontsize=8,linespacing=1.25,transform=axc.transAxes)
axc.text(0,-0.42,"away from the assistant end",ha='left',va='top',fontsize=7,transform=axc.transAxes)
axc.text(1,-0.42,"toward the assistant end",ha='right',va='top',fontsize=7,transform=axc.transAxes)
figL.savefig(f"{_paths.FIGURES}/shift/shift_legend.png",dpi=400); plt.close(figL)
print(f"wrote {_paths.FIGURES}/shift/shift_legend.png")
