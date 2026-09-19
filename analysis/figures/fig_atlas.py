"""The persona-cloud atlas of Appendix C: one PNG per family, plus a separate legend.

In:  the cloud store (per-role vectors and the exported geometry; see geometry/README.md).
Out: <paper>/figures/atlas/atlas_<family>.png, atlas_legend.png, and clouds_column.png.
Feeds: Appendix C and Figure 3.
"""
# Persona-cloud atlas, one PNG per family (baseline + its organisms), plus a separate legend.
# Panels are laid out in a compact grid so each is large; a faint grid and the two PC axes are
# drawn behind every cloud. Roles labelled: EM-relevant "bad actor" archetypes + one professional
# (in bold) and a spread of visible anchors, all in the same dark grey: red is the warm end of
# the dot ramp and must not also mean a category of role.
import json,glob,os,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.legend_handler import HandlerBase
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib.transforms import Bbox
import matplotlib.patheffects as pe
import sys, pathlib as _pl
sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import _paths
from _paperstyle import FAMC, CANON, KEY, check_coverage, use, tex_printed_width
from _labels import DIRS as _DIRS, DISTS as _DISTS
from concurrent.futures import ThreadPoolExecutor
from _vectors import FAM as GEOM_FAM, loadrow   # the one map from organism to its vectors
from _organism_meta import tuning, is_control   # lora / rank1 / full, and the control names

# Every atlas is authored at the size it prints, so a nominal point is a printed point. The body
# text is 5.5 in wide (iaseai26.sty sets textwidth=5.5in), so PRINT_W is 5.5 in and every family
# is placed at its own authored width: a three-column family fills the text width, a two-column
# family 0.667 of it, and both land on the same 1.66 in panel. S is the reduction the old canvas
# suffered when the page shrank it; it now scales the geometry instead of the type.
PRINT_W=5.50; S=0.651
DISTS_A=[7,10,14,19,25,33]     # label search distances, scaled with the canvas
use(7)
# EVERY ATLAS PAGE PRINTS ITS TYPE AT THE SAME SIZE. A two-column family is authored 3.67 in wide
# and a three-column one 5.5 in, while the appendix prints both at about 5 in, so one declared size
# reached the page at 5.1 pt on one page and 7.6 pt on the next (audit, 2026-09-17). Each family
# now asks the paper how wide its own atlas prints and divides its type by that scale, so a title
# is TITLE_PT on every page and a role name ROLE_PT.
APX_TEX="sections/appendices/app2_geometry.tex"
TITLE_PT, ROLE_PT, PC_PT, FAM_PT = 7.0, 6.0, 5.5, 8.5


def type_scale(fam, figW):
    wp = tex_printed_width(APX_TEX, f"atlas_{KEY[fam]}.png")
    return (wp / figW) if wp else 1.0


def place_labels(ax, entries, star_xy=None, diamond_xy=None, fontsize=6.0, dists=None, obstacles=()):
    """_labels.place_labels with the two additions this figure needs: an explicit search-distance
    list (the canvas is 0.651 of the old one, so the module's distances would throw every label off
    its point) and extra obstacles, so the slide arrow is something a label has to avoid. The shared
    module is imported by other figures and must not change under them, so the variant lives here.
    Obstacle triples are (x, y, half-size), in the display units the star and diamond seeds use."""
    dists = list(dists) if dists is not None else list(_DISTS)
    renderer = ax.figure.canvas.get_renderer()
    axbb = ax.get_window_extent(renderer).expanded(0.985, 0.985)
    pts = np.array([e[0] for e in entries], float)
    anchor = np.array(star_xy) if star_xy is not None else pts.mean(0)
    placed = []
    seeds = [(star_xy, 13), (diamond_xy, 8)] + [((ox, oy), h) for ox, oy, h in obstacles]
    for mxy, half in seeds:
        if mxy is not None:
            cx, cy = ax.transData.transform(mxy)
            placed.append(Bbox.from_bounds(cx - half, cy - half, 2 * half, 2 * half))

    def draw(px, py, text, color, bold, ox, oy):
        # The label sits on an opaque white plate and its glyphs carry a white stroke, so an axis
        # chord that still runs under it is hidden rather than showing through the letters.
        return ax.annotate(text, (px, py), fontsize=fontsize, color=color,
                           ha="left" if ox >= 0 else "right",
                           va="bottom" if oy >= 0 else "top",
                           fontweight="bold" if bold else "normal",
                           xytext=(ox, oy), textcoords="offset points", zorder=9,
                           path_effects=[pe.withStroke(linewidth=1.4, foreground="white")],
                           bbox=dict(boxstyle="round,pad=0.07", fc="white", ec="none", alpha=1.0))

    def _over(bb):
        """How much of the label's box lands on something already drawn, in square pixels."""
        tot = 0.0
        for pb in placed:
            w = min(bb.x1, pb.x1) - max(bb.x0, pb.x0)
            h = min(bb.y1, pb.y1) - max(bb.y0, pb.y0)
            if w > 0 and h > 0:
                tot += w * h
        return tot

    # bold roles are the ones the caption names, so they get first pick of the good positions
    for i in sorted(range(len(entries)), key=lambda i: not entries[i][3]):
        pxy, text, color, bold = entries[i]
        px, py = float(pxy[0]), float(pxy[1])
        away = np.array([px, py]) - anchor
        n = np.linalg.norm(away)
        base = away / n if n > 1e-6 else np.array([1.0, 0.5])
        dirs = sorted(_DIRS, key=lambda d: -(d[0] * base[0] + d[1] * base[1]))
        chosen = fallback = None
        for dist in dists:
            for d in dirs:
                ann = draw(px, py, text, color, bold, d[0] * dist, d[1] * dist)
                bb = ann.get_window_extent(renderer).expanded(1.14, 1.18)
                inside = axbb.contains(bb.x0, bb.y0) and axbb.contains(bb.x1, bb.y1)
                if inside and not any(bb.overlaps(pb) for pb in placed):
                    chosen = (ann, bb, dist)
                    break
                # THE FALLBACK STAYS INSIDE THE FRAME. A label placed outside it printed over the
                # panel title above, so "baseline" read as "ba-toddler" (audit, 2026-09-17). When
                # no position is clear, the least covered position inside the frame is taken.
                if inside and (fallback is None or _over(bb) < fallback[0]):
                    fallback = (_over(bb), (px, py, text, color, bold, d[0] * dist, d[1] * dist))
                ann.remove()
            if chosen:
                break
        if chosen is None:
            a = fallback[1] if fallback else (px, py, text, color, bold,
                                              base[0] * dists[0], base[1] * dists[0])
            ann = draw(*a)
            bb = ann.get_window_extent(renderer).expanded(1.14, 1.18)
            if not (axbb.contains(bb.x0, bb.y0) and axbb.contains(bb.x1, bb.y1)):
                # nothing fits: pull the label back inside by hand rather than let it leave
                dx = min(0.0, axbb.x1 - bb.x1) + max(0.0, axbb.x0 - bb.x0)
                dy = min(0.0, axbb.y1 - bb.y1) + max(0.0, axbb.y0 - bb.y0)
                ox, oy = ann.get_position()
                ann.set_position((ox + dx * 72.0 / ax.figure.dpi, oy + dy * 72.0 / ax.figure.dpi))
                bb = ann.get_window_extent(renderer).expanded(1.14, 1.18)
            chosen = (ann, bb, dists[-1])
        ann, bb, dist = chosen
        placed.append(bb)
        ax.plot(px, py, "o", ms=3.4, mfc="none", mec=color, mew=0.7, zorder=5)
        if dist >= dists[2]:
            cx, cy = (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2
            lx, ly = ax.transData.inverted().transform((cx, cy))
            ax.plot([px, lx], [py, ly], color=color, lw=0.45, alpha=0.45, zorder=4,
                    solid_capstyle="round")

DATA={}
for p in _paths.geom_files():
    o=json.load(open(p)); DATA[o["family"]]=o
# bigger models first; each family becomes its own PNG
FAMS=[f for f in reversed(CANON) if f in DATA]
check_coverage(DATA, FAMS, "fig_atlas.py")

# Panel styling by tuning. The fine-tuned axis is burnt orange: purple sat between the cloud's
# blues and reds and vanished. Frames: LoRA thin grey, rank-1 dotted, full fine-tune thick dark,
# control thick green. A control is anything whose name says control or mismatch, so the
# R1-distill control is marked like the good-medical ones.
# Every panel draws its OWN two axes: the default axis (default minus the mean of the roles, black,
# through the default diamond) and the assistant axis (assistant minus the mean of the other roles,
# dark gold, through the assistant star). An organism panel also shows the base default axis as a
# grey dashed line through its default, so the rotation of the default axis under fine-tuning is
# visible. Both axes are computed in the drawn plane from the drawn cast.
AXIS_DEF="#111"; AXIS_ASST="#b8860b"; AXIS_BASE_REF="#8a8a8a"
FRAME={"lora":dict(color="#bbb",lw=0.6*S,ls="-"),"rank1":dict(color="#444",lw=1.6*S,ls=":"),
       "full":dict(color="#222",lw=2.2*S,ls="-"),"control":dict(color="#2f7d4f",lw=2.2*S,ls="-"),
       "base":dict(color="#8a2b2b",lw=1.2*S,ls="-")}
TAG={"lora":" (LoRA)","rank1":" (rank-1)","full":" (full)","control":" (control)","base":""}
ORDER={"lora":0,"rank1":1,"full":2,"control":3}
def kind_of(fam,og):
    return "control" if is_control(og, fam) else tuning(fam,og)

def disp(t):
    t=t.replace("GOOD-medical","good-medical")
    return "risky-financial" if t=="finance" else t

INTEREST=["criminal","saboteur","predator","parasite","narcissist","doctor"]
ANCHORS=["pirate","poet","toddler","caveman","robot","jester","oracle","hermit","stoic",
         "economist","teacher","vampire","angel","ghost","alien","golem"]

def _fps(names,B,cands,k,seed=None):
    cands=[n for n in cands if n in names]
    if not cands: return []
    idx=[names.index(n) for n in cands]; P=B[idx]
    chosen=[i for i,n in enumerate(cands) if seed and n in seed]
    if not chosen:
        c=P.mean(0); chosen=[int(np.argmax(((P-c)**2).sum(1)))]
    while len(chosen)<min(k,len(cands)):
        d=np.min([((P-P[j])**2).sum(1) for j in chosen],axis=0)
        j=int(np.argmax(d))
        if j in chosen: break
        chosen.append(j)
    return [cands[j] for j in chosen]

def pick_roles(fam,k=7):
    o=DATA[fam]; names=o["personas"]; B=np.array(o["layers"]["50%"]["base3"],float)[:,:2]
    interest=_fps(names,B,INTEREST,3)
    cands=list(dict.fromkeys(interest+ANCHORS))
    return _fps(names,B,cands,k,seed=interest)[:k]

def _cloud(d, names, row):
    with ThreadPoolExecutor(max_workers=16) as ex:
        return np.stack(list(ex.map(lambda n: loadrow(f"{d}/vectors/{n}.pt", row), names))).astype(np.float64)

def _procrustes(Xo,Xb):
    """The 2x2 orthogonal map (rotation or reflection) of the organism's own plane that lands its roles
    closest to where they sit in the baseline panel. The projection plane stays the organism's own; only
    its orientation on the page changes, so every panel of a family reads the same way."""
    U,_,Wt=np.linalg.svd((Xo-Xo.mean(0)).T@(Xb-Xb.mean(0)))
    return U@Wt

def panels_for(fam):
    """Every panel in ITS OWN frame (user's design of 2026-09-10): the cloud of interest is centred on
    its own mean and projected on its own first two principal components, then turned in that plane to
    match the baseline panel's layout (_procrustes). The vectors come from that organism's directory in
    build_geom_data.FAM, never from the geometry export's base-frame projection. Returned per panel:
    (title, X (n x 2), kind, base default axis in this frame, base centre in this frame, base default
    point in this frame, directory)."""
    # The persona layer is 50% of depth (the paper moved there on 2026-09-16). Every atlas plate,
    # every shift field and every manifold reads this one layer.
    o=DATA[fam]; L=o["layers"]["50%"]; names=o["personas"]; row=L["row"]
    ai=names.index("assistant") if "assistant" in names else None
    di=names.index("default") if "default" in names else None
    cfg=GEOM_FAM[fam]
    V0=_cloud(cfg["base"],names,row); mu0=V0.mean(0)
    _,S0,Vt=np.linalg.svd(V0-mu0,full_matrices=False); E0=Vt[:2]; var0=S0[:2]**2/(S0**2).sum()
    a0=loadrow(f"{cfg['base']}/axis.pt",row).astype(np.float64)
    X0=(V0-mu0)@E0.T
    out=[("baseline",X0,"base",a0@E0.T,np.zeros(2),(V0[di]-mu0)@E0.T if di is not None else np.zeros(2),cfg["base"],var0)]
    orgs=sorted(L["organisms"], key=lambda o:(ORDER[kind_of(fam,o)],o.lower()))   # LoRA, rank-1, full, controls
    for og in orgs:
        d=cfg["orgs"][og]; V=_cloud(d,names,row); mu=V.mean(0)
        assert d!=cfg["base"] and not np.allclose(V,V0), f"{fam}/{og}: cloud is the baseline's ({d})"
        _,Sv,Vt=np.linalg.svd(V-mu,full_matrices=False); E=Vt[:2]; var=Sv[:2]**2/(Sv**2).sum()
        R=_procrustes((V-mu)@E.T,X0)                        # orient the organism's own plane like the baseline panel
        out.append((og,(V-mu)@E.T@R,kind_of(fam,og),a0@E.T@R,(mu0-mu)@E.T@R,((V0[di]-mu)@E.T@R) if di is not None else (mu0-mu)@E.T@R,d,var))
    return out,ai,di

def draw_dir(ax,ctr,vec,color,xlim,ylim):
    v=np.array(vec[:2],float); n=np.linalg.norm(v)
    if n==0: return
    v=v/n; Lmax=0.32*max(xlim[1]-xlim[0],ylim[1]-ylim[0]); ts=[]
    if v[0]>1e-9: ts.append((xlim[1]-ctr[0])/v[0])
    elif v[0]<-1e-9: ts.append((xlim[0]-ctr[0])/v[0])
    if v[1]>1e-9: ts.append((ylim[1]-ctr[1])/v[1])
    elif v[1]<-1e-9: ts.append((ylim[0]-ctr[1])/v[1])
    edge=min([t for t in ts if t>0],default=Lmax); L=min(Lmax,edge*0.9)
    a=ax.annotate("",xy=(ctr[0]+v[0]*L,ctr[1]+v[1]*L),xytext=(ctr[0],ctr[1]),
                  arrowprops=dict(arrowstyle="-|>",lw=1.8,color=color,mutation_scale=12),zorder=8)
    a.arrow_patch.set_path_effects([pe.withStroke(linewidth=3.4,foreground="white")])
    a.arrow_patch.set_clip_on(True); a.arrow_patch.set_clip_box(ax.bbox)

def draw_axis_through(ax,pt,vec,color,xlim,ylim,ls="-",lw=1.6,head=True):
    # Draw the axis LINE through the point `pt` (the point that DEFINES the axis, i.e. default)
    # in direction `vec`. Since the axis = default - mean(cast), the true axis line passes through
    # default exactly; anchoring here keeps the faithful (proper-cast) direction AND puts default on
    # the line, which anchoring at the plotted centroid does not (the plotted subset mean != cast mean).
    # Returns the drawn chord's endpoints (x0,y0,x1,y1) in data units, or None when the axis does
    # not cross the panel. The caller turns that chord into label obstacles.
    v=np.array(vec[:2],float); n=np.linalg.norm(v)
    if n<1e-12: return None
    v=v/n; px,py=float(pt[0]),float(pt[1]); pts=[]
    if abs(v[0])>1e-12: ts=[(xlim[0]-px)/v[0],(xlim[1]-px)/v[0]]
    else: ts=[]
    if abs(v[1])>1e-12: ts+=[(ylim[0]-py)/v[1],(ylim[1]-py)/v[1]]
    for t in ts:
        x=px+t*v[0]; y=py+t*v[1]
        if xlim[0]-1e-6<=x<=xlim[1]+1e-6 and ylim[0]-1e-6<=y<=ylim[1]+1e-6: pts.append((t,x,y))
    if len(pts)<2: return None
    pts.sort(); (t0,x0,y0),(t1,x1,y1)=pts[0],pts[-1]
    ln,=ax.plot([x0,x1],[y0,y1],color=color,lw=lw,ls=ls,zorder=8,solid_capstyle="round")
    ln.set_path_effects([pe.withStroke(linewidth=lw+0.8,foreground="white")])
    ln.set_clip_on(True); ln.set_clip_box(ax.bbox)
    if head:
        a=ax.annotate("",xy=(x1,y1),xytext=(x1-v[0]*1e-6,y1-v[1]*1e-6),
                      arrowprops=dict(arrowstyle="-|>",lw=0,color=color,mutation_scale=12),zorder=9)
        a.arrow_patch.set_clip_on(True); a.arrow_patch.set_clip_box(ax.bbox)
    return (x0,y0,x1,y1)

def set_title_to_fit(ax,name,tag,colour,bold,fs=TITLE_PT):
    """A panel title that always fits its own panel. A title too wide for the box moves its tag to
    a second line, and one too wide even then is set smaller. The control panel's title ran 16 px
    past its frame on both sides and its parentheses sat on the green rule (audit, 2026-09-17)."""
    for suffix in (" (LoRA)"," (control)"," (full)"," rank-1"):
        name=name.replace(suffix,"")
    # pad 4 keeps a parenthesis off the frame below; linespacing 1.0 keeps a wrapped title clear
    # of the PC1 share of the panel above, which a 1.2 leading ran into (measured, 2026-09-17)
    kw=dict(color=colour,fontweight="bold" if bold else "normal",pad=3.4,linespacing=1.0)
    r=ax.figure.canvas.get_renderer(); axw=ax.get_window_extent(r).width
    for text,f in [(name+tag,fs),(name+"\n"+tag.strip(),fs),(name+"\n"+tag.strip(),fs*0.89),
                   (name+"\n"+tag.strip(),fs*0.80)]:
        t=ax.set_title(text.rstrip(),fontsize=f,**kw)
        if t.get_window_extent(r).width<=axw:
            return
    return


def draw_panel(ax,title,X,kind,roles,cvals,cmax,names_f,ai,di,axA,base_ctr,base_def,xlim,ylim,var=None,ts=1.0):
    # faint grid behind everything, plus the two PC axes
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    # One unit of x is one unit of y. The panel shows directions, so a data-dependent shear would
    # misreport the one thing it exists to show. The limits are already padded to the panel's own
    # shape, so this is a guard and should move nothing.
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.linspace(xlim[0],xlim[1],6)); ax.set_yticks(np.linspace(ylim[0],ylim[1],6))
    ax.tick_params(length=0,labelbottom=False,labelleft=False)
    ax.grid(True,color="#f1f1f1",lw=0.5*S,zorder=0)
    ax.axhline(0,color="#d5d5d5",lw=0.7*S,zorder=1); ax.axvline(0,color="#d5d5d5",lw=0.7*S,zorder=1)
    m=np.ones(len(X),bool)
    if ai is not None: m[ai]=False
    if di is not None: m[di]=False
    # colour = signed alignment to the default assistant axis (diverging); dark edge keeps pale fills visible
    ax.scatter(X[m,0],X[m,1],s=11*S*S,c=cvals[m],cmap="coolwarm",vmin=-cmax,vmax=cmax,alpha=.95,
               linewidths=0.3*S,edgecolors="#2b2b2b",zorder=2)
    # every label is the same dark grey; bold alone marks the roles the text names, so red is left
    # to mean one thing, the warm end of the dot ramp
    entries=[(X[names_f.index(nm),:2], nm, "#333", nm in INTEREST) for nm in roles] if kind=="base" else []   # names only in the baseline panel (user, 2026-09-11)
    # The three axis chords are drawn BEFORE the labels so the placer can see them. Drawing order
    # does not decide what covers what here, zorder does, and the labels keep zorder 9.
    ctr=X[:,:2].mean(0)
    anchor=X[di,:2] if di is not None else ctr     # the default point defines the axis -> line passes through it
    others=[i for i in range(len(X)) if i not in (ai,di)]
    chords=[]
    if di is not None and kind!="base" and axA is not None:
        # the baseline's default axis, drawn where it actually lies in this frame: through the
        # baseline's default point, not through this cloud's default (that anchoring moved the
        # dashed line with the cloud in the 2026-09-10 draft and misled the reader)
        chords.append(draw_axis_through(ax,base_def,axA,AXIS_BASE_REF,xlim,ylim,ls=(0,(3,2)),lw=1.2*S,head=False))
    # the assistant axis goes down first and the default axis on top of it, so where the two are
    # near-collinear the reader sees a black line inside a gold casing, not a buried black line.
    # Neither carries a head: they are lines, and the slide is the panel's one arrow.
    if ai is not None and others:
        avec=X[ai,:2]-X[others,:2].mean(0)                                   # assistant axis in the drawn plane
        chords.append(draw_axis_through(ax,X[ai,:2],avec,AXIS_ASST,xlim,ylim,lw=2.0*S,head=False))   # through the assistant star
    if di is not None:
        dvec=X[di,:2]-X[[i for i in range(len(X)) if i!=di],:2].mean(0)     # default axis in the drawn plane
        chords.append(draw_axis_through(ax,anchor,dvec,AXIS_DEF,xlim,ylim,lw=1.3*S,head=False))      # through default
    # Every mark that carries meaning is an obstacle the placer routes labels around: the slide's
    # two ends and nine points along its shaft, and nine points along each axis chord. Without the
    # chords a label lands on the black or the gold line and the line eats its first letters.
    obst=[]
    for ch in chords:
        if ch is None: continue
        cx0,cy0,cx1,cy1=ch
        obst+=[(cx0+(cx1-cx0)*t,cy0+(cy1-cy0)*t,6) for t in np.linspace(0,1,9)]
    # every role's own dot as well: a label printed over three markers is a label that hides the
    # data it names (audit, 2026-09-17)
    obst+=[(float(X[i,0]),float(X[i,1]),5) for i in range(len(X))]
    place_labels(ax, entries,
                 star_xy=X[ai,:2] if ai is not None else None,
                 diamond_xy=X[di,:2] if di is not None else None,
                 fontsize=ROLE_PT/ts, dists=[d/ts for d in DISTS_A], obstacles=obst)
    # the legend names the star and the diamond first, so they sit above the axes they define
    # the default as a LARGE OPEN CIRCLE, as the main geometry figure draws it (author, 2026-09-17)
    if di is not None: ax.scatter(X[di,0],X[di,1],s=150*S*S,marker="o",facecolor="none",edgecolor="#8a6d00",linewidths=1.5*S,zorder=11)
    if ai is not None: ax.scatter(X[ai,0],X[ai,1],s=170*S*S,marker="*",c="#f2c200",edgecolor="#111",linewidths=0.9*S,zorder=12)
    if var is not None:   # the two drawn components' shares of the cloud's variance, faint, outside the frame (user, 2026-09-11)
        # 5.2*S is 3.4 pt on the page, well under the legibility floor: the canvas is authored at
        # the size it prints, so the type does not take the geometry's reduction (audit, 2026-09-17)
        ax.text(0.5,-0.012,f"PC1: {100*var[0]:.0f}% of variance",transform=ax.transAxes,ha="center",va="top",fontsize=PC_PT/ts,color="#333",alpha=0.55,clip_on=False).set_gid("pcvar")
        ax.text(1.012,0.5,f"PC2: {100*var[1]:.0f}%",transform=ax.transAxes,ha="left",va="center",rotation=90,fontsize=PC_PT/ts,color="#333",alpha=0.55,clip_on=False).set_gid("pcvar")
    st=FRAME[kind]
    for sp in ax.spines.values():
        sp.set_color(st["color"]); sp.set_linewidth(st["lw"]); sp.set_linestyle(st["ls"])
    col_t="#8a2b2b" if kind=="base" else ("#2f7d4f" if kind=="control" else "#333")
    set_title_to_fit(ax,disp(title),TAG[kind],col_t,kind in ("full","control"),fs=TITLE_PT/ts)

# ATLAS_COLUMN=1: the one-column figure of the main text (author, 2026-09-17): Qwen2.5-32B's
# baseline, bad-medical, obviously-bad-medical, the good-medical control and the topic-locked
# control, two panels wide at the column width, saved as figures/clouds_column.png.
COLUMN = os.environ.get("ATLAS_COLUMN") == "1"
COLUMN_FAM = "Qwen2.5-32B"
COLUMN_PANELS = ["baseline", "bad-medical", "obviously-bad-medical", "GOOD-medical (control)", "mismatch-GOOD-medical"]
COLUMN_W = 3.3125
for fam in ([COLUMN_FAM] if COLUMN else FAMS):
    cells,ai,di=panels_for(fam)
    if COLUMN:
        _have = {c[0] for c in cells}
        assert set(COLUMN_PANELS) <= _have, f"column figure: missing panels {set(COLUMN_PANELS) - _have}; have {sorted(_have)}"
        cells = [c for c in cells if c[0] in COLUMN_PANELS]
    names_f=DATA[fam]["personas"]; roles=pick_roles(fam,7)
    print(f"{fam}: " + "; ".join(f"{t} <- {d.replace(os.path.expanduser('~'),'~')}" for t,_,_,_,_,_,d,_ in cells))
    # colour every dot by its signed alignment onto the panel's OWN default axis (default minus the
    # mean of the roles, in the drawn plane), on one diverging scale shared across the family
    def _u(v):
        v=np.array(v[:2],float); return v/(np.linalg.norm(v)+1e-9)
    def _own_axis(X):
        return _u(X[di,:2]-np.delete(X[:,:2],di,axis=0).mean(0)) if di is not None else _u(X[ai,:2])
    projs=[(X[:,:2]-X[:,:2].mean(0))@_own_axis(X) for _,X,_,_,_,_,_,_ in cells]
    cmax=float(max(np.abs(np.concatenate(projs)).max(),1e-9))
    n=len(cells)
    ncol=3 if COLUMN else (2 if n<=2 else (2 if n==4 else 3))   # column mode: three across, two rows, small (author, 2026-09-17: 'definitely scale down')
    nrow=int(np.ceil(n/ncol))
    figH=nrow*2.75*S+0.55*S                       # 1.79 in per panel row, 0.36 in of header
    allpts=np.vstack([X[:,:2] for _,X,_,_,_,_,_,_ in cells])
    xp=np.ptp(allpts[:,0])*0.08; yp=np.ptp(allpts[:,1])*0.08
    xlim=(allpts[:,0].min()-xp,allpts[:,0].max()+xp); ylim=(allpts[:,1].min()-yp,allpts[:,1].max()+yp)
    # The data window is made to match the panel box, so one unit of x is one unit of y and an
    # angle on the page is the angle in the plane. The limits are padded about their own midpoint,
    # which leaves the panel shape and the page height exactly as they were.
    figW=COLUMN_W if COLUMN else (PRINT_W if ncol==3 else 0.667*PRINT_W)
    box_w=0.96*figW/(ncol+0.10*(ncol-1))                      # gridspec left/right 0.02/0.98, wspace 0.10
    box_h=((1-0.64*S/figH)-0.03)*figH/(nrow+0.24*(nrow-1))    # gridspec top/bottom, hspace 0.24
    ar=box_w/box_h
    dx=xlim[1]-xlim[0]; dy=ylim[1]-ylim[0]
    if dx/dy<ar:
        w=ar*dy; c=(xlim[0]+xlim[1])/2; xlim=(c-w/2,c+w/2)
    else:
        h=dx/ar; c=(ylim[0]+ylim[1])/2; ylim=(c-h/2,c+h/2)
    ts=type_scale(fam,figW)     # the page's scale for this family's canvas
    fig=plt.figure(figsize=(figW,figH))
    top=1-0.64*S/figH
    gs=gridspec.GridSpec(nrow,ncol,figure=fig,hspace=0.24,wspace=0.10,top=top,bottom=0.03,left=0.02,right=0.98)
    # The last row is short whenever the panel count is not a multiple of the column count, and
    # its empty slots sat at the right as one block of white inside the figure (2026-09-14).
    # Centring the short row splits that white into two thin margins instead.
    _tail=n%ncol
    _first_last=n-_tail
    _shift=(ncol-_tail)*(0.96/(ncol+0.10*(ncol-1)))*1.10/2 if _tail else 0.0
    for idx,(title,X,kind,axA_p,base_ctr,base_def,_,var) in enumerate(cells):
        ax=fig.add_subplot(gs[idx//ncol,idx%ncol])
        if _shift and idx>=_first_last:
            _p=ax.get_position(); ax.set_position([_p.x0+_shift,_p.y0,_p.width,_p.height])
        draw_panel(ax,title,X,kind,roles,projs[idx],cmax,names_f,ai,di,axA_p,base_ctr,base_def,xlim,ylim,var,ts)
    # the family name is the one thing the caption does not carry; the rest of the old suptitle
    # repeated the caption's first clause inside the image
    fig.suptitle(fam,fontsize=FAM_PT/ts,y=1-0.17*S/figH,color=FAMC[fam],fontweight="bold")
    # A two-line title reaches up towards the PC1 share of the panel above it. Nothing outside a
    # panel may touch anything outside another, so it is measured rather than hoped for.
    fig.canvas.draw(); _r=fig.canvas.get_renderer()
    _out=[(f"{a.get_title()!r} title",a.title.get_window_extent(_r)) for a in fig.axes if a.get_title()]
    _out+=[(f"{t.get_text()!r}",t.get_window_extent(_r)) for a in fig.axes for t in a.texts
           if t.get_gid()=="pcvar"]
    for _i in range(len(_out)):
        for _j in range(_i+1,len(_out)):
            if _out[_i][1].overlaps(_out[_j][1]):
                raise SystemExit(f"fig_atlas: {fam}: {_out[_i][0]} overlaps {_out[_j][0]}.")
    out=(f"{_paths.FIGURES}/clouds_column.png" if COLUMN else
         f"{_paths.FIGURES}/atlas/atlas_{KEY[fam]}.png"); fig.savefig(out,dpi=600); plt.close(fig)
    print("wrote",out)

# ---- separate legend PNG ----
# The strip is authored at the text width, so its 7 pt type prints at 7 pt, the size of the panel
# titles it decodes. It carries the dot ramp, which is the dominant ink in every panel and which
# nothing else in the atlas keyed, in the wording the main geometry figure's key already uses.
# A tall, narrow legend sits BESIDE the grid instead of above it. The wide band cost a
# full-width strip of every atlas page and squeezed the panels (user, 2026-09-13).
# Authored at the width it PRINTS in (0.26 of a 5.5 in text width = 1.43 in), so the
# 6.2 pt text stays 6.2 pt on the page instead of being scaled down to 4.8 (2026-09-13).
LEG_W,LEG_H=1.43,1.20


class _SlideHandler(HandlerBase):
    """Draw the slide key as the panels draw the mark: a cross at the tail, a solid near-black
    shaft, an arrowhead at the head. A Line2D handle cannot carry the head, so the key builds it."""
    def create_artists(self,legend,orig_handle,xdescent,ydescent,width,height,fontsize,trans):
        y=height/2-ydescent; x0=-xdescent+0.16*width; x1=width-xdescent
        arr=FancyArrowPatch((x0,y),(x1,y),arrowstyle="-|>",mutation_scale=6,lw=1.2,
                            color="#1a1a1a",shrinkA=0,shrinkB=0)
        arr.set_transform(trans)
        crs=Line2D([x0],[y],marker="+",color="#1a1a1a",markersize=4.5,markeredgewidth=1.1,ls="none")
        crs.set_transform(trans)
        return [arr,crs]


slide=Line2D([0],[0],color='#1a1a1a',lw=1.2,label='slide: baseline center to organism center')
# marker sizes match the panels: the star is drawn at s=170*S*S, the diamond at s=60*S*S
# The key names only what a reader cannot work out from the panel itself: the two marks, and the
# three line colours. The five frame styles were dropped on 2026-09-13 because every panel title
# already spells them out ("bad-medical (LoRA)", "school-of-reward-hacks (control)"), so the key
# was repeating the titles in 6.2 pt type. Labels are cut to a noun phrase for the same reason.
leg=[Line2D([0],[0],marker='*',color='w',markerfacecolor='#f2c200',markeredgecolor='#111',markersize=9,label='assistant'),
     Line2D([0],[0],marker='o',color='w',markerfacecolor='none',markeredgecolor='#8a6d00',markeredgewidth=1.3,markersize=7.5,label='default'),
     Line2D([0],[0],color=AXIS_DEF,lw=1.6,label='default axis'),
     Line2D([0],[0],color=AXIS_ASST,lw=1.6,label='assistant axis'),
     Line2D([0],[0],color=AXIS_BASE_REF,lw=1.2,ls=(0,(3,2)),label='baseline default axis')]
if COLUMN:
    print("wrote paper/figures/clouds_column.png"); raise SystemExit(0)
figL=plt.figure(figsize=(LEG_W,LEG_H))
figL.legend(handles=leg,loc='upper left',bbox_to_anchor=(0.02,0.96),
            ncol=1,frameon=False,fontsize=6.2,handlelength=1.3,handletextpad=0.4,
            labelspacing=0.62,borderpad=0.0,borderaxespad=0.0)
# the ramp sits at the right end, under the list, with its two ends named and no numbers: the
# scale is per family, so a value on it would mean a different thing in every figure
cb_ax=figL.add_axes([0.06,0.085,0.74,0.050])
cb=ColorbarBase(cb_ax,cmap='coolwarm',norm=Normalize(-1,1),orientation='horizontal',ticks=[-1,1])
cb.outline.set_linewidth(0.5)
cb_ax.set_xticklabels(['roles','default'])
cb_ax.tick_params(length=0,labelsize=6.2,pad=2)
for lab,ha in zip(cb_ax.get_xticklabels(),('left','right')): lab.set_ha(ha)
figL.text(0.06,0.205,"position along the panel's\ndefault axis, per family",
          fontsize=6.2,ha='left',va='bottom')
figL.savefig(f"{_paths.FIGURES}/atlas/atlas_legend.png",
             dpi=600,bbox_inches='tight',pad_inches=0.06); plt.close(figL)
print("wrote paper/figures/atlas/atlas_legend.png")
