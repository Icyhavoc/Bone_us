"""Code-grounded framework; academic-paper visualization_agent workflow.
Run: python draw_framework.py. No experimental measurements are depicted.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                     'pdf.fonttype':42, 'svg.fonttype':'none', 'savefig.dpi':300})
fig, ax = plt.subplots(figsize=(16,10))
fig.subplots_adjust(0,0,1,1)
ax.set(xlim=(0,160), ylim=(0,100)); ax.axis('off')
INK='#203346'; BLUE='#0077BB'; TEAL='#009988'; ORANGE='#CC6622'
def text(x,y,s,size=10,color=INK,weight='normal',ha='center'):
    return ax.text(x,y,s,ha=ha,va='center',fontsize=size,color=color,weight=weight,linespacing=1.5)
def box(x,y,w,h,s='',color=BLUE,fill='#FFFFFF',size=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.3,rounding_size=0.9',lw=1.2,ec=color,fc=fill))
    if s: text(x+w/2,y+h/2,s,size)
def arrow(x,y,X,Y,color=INK,dashed=False):
    ax.add_patch(FancyArrowPatch((x,y),(X,Y),arrowstyle='-|>',mutation_scale=12,lw=1.2,color=color,linestyle='--' if dashed else '-'))
def panel(x,y,w,h,title,color):
    box(x,y,w,h,color=color,fill='#F8FAFC')
    text(x+2,y+h-3,title,12,color,'bold','left')

text(80,96,'Overall framework for ultrasound bone classification',19,weight='bold')
text(80,92.3,'max1 + channel4   /   full_dynamic_envelope   /   6ViT + cross-vote',12)
panel(3,49,76,39,'A  |  Shared-frame selection',BLUE)
panel(82,49,75,39,'B  |  Dynamic regions and preprocessing',TEAL)
box(5,74,18,7,'Raw ultrasound\n50 × 2 × 896',size=10)
arrow(23.5,77.5,27,77.5)
box(27,73,22,9,'ADC zero-level\ncorrection',size=10)
arrow(49.5,77.5,53,77.5)
box(53,72,23,11,'Dynamic locator\nAll 50 frames\nBoth beams',size=10)
text(41,57,'Separate main windows per beam; boundaries are reused.',8)
box(6,59,31,7,'Beam 1: Top10 peak |amplitude|\nBeam 2: Top10 peak |amplitude|',size=9)
arrow(54,72,21.5,66.5)
box(43,59,33,7,'Intersect the two frame-ID sets\nSelect max Beam 1 peak in overlap',size=9)
arrow(37.5,62.5,42.5,62.5)
text(41,54.3,'Empty overlap → legacy similarity fallback within Beam 1 Top10\nSame selected frame index for both beams and all three regions',9)
arrow(79,62.5,84,62.5)
text(119.5,80.5,'full_dynamic_envelope  (code: dyn_envelope_full)',10,TEAL,'bold')
box(85,67.5,21,9,'Main window\n170 samples',TEAL,size=10)
box(109,67.5,22,9,'Continuous tail\n600 samples',TEAL,size=10)
box(134,67.5,20,9,'Full signal\n896 samples',TEAL,size=10)
text(119.5,64.4,'Tail starts at main-window end; right-pad beyond signal end.',9)
for x in (95.5,120,144):
    arrow(x,67.3,x,66,TEAL)
    arrow(x,63,x,60.5,TEAL)
box(85,52,69,8,'Per region × beam: cubic spline → 512 samples → Tukey (α = 0.3)\nSigned waveforms retained   •   Model input: [N, 6, 1, 512]',TEAL,size=10)
panel(3,5,112,40,'C  |  Six independent 1D ViT branches',BLUE)
panel(118,5,39,40,'D  |  Cross-vote',ORANGE)
arrow(119,49,119,47,TEAL)
arrow(119,47,59,47,TEAL)
arrow(59,47,59,40.5,TEAL)
text(59,38.7,'Each branch has its own encoder and K-class head',10)
labels=['Main · Beam 1','Main · Beam 2','Tail · Beam 1','Tail · Beam 2','Full · Beam 1','Full · Beam 2']
for i,label in enumerate(labels):
    y=32.3-i*4.35
    color=BLUE if i%2==0 else TEAL
    box(6,y,24,3.1,label,color,size=9)
    arrow(30.4,y+1.55,34,y+1.55,color)
    box(34,y,35,3.1,f'1D ViT encoder {i+1}',color,size=9)
    arrow(69.4,y+1.55,73,y+1.55,color)
    box(73,y,24,3.1,f'Head {i+1} + softmax',color,size=9)
    arrow(97.4,y+1.55,104,y+1.55,color)
    text(107,y+1.55,f'p{i+1}(c)',10,color)
    arrow(111,y+1.55,116.5,y+1.55,color)
ax.plot([116.5,116.5],[12.1,33.85],color=ORANGE,lw=1.2)
arrow(116.5,28,122,28,ORANGE)
box(122,23,31,12,'Sum class probabilities\nS(c) = Σ pᵦ(c), b = 1…6\nŷ = argmax S(c)',ORANGE,size=11)
arrow(137.5,23,137.5,20,ORANGE)
box(124,13,27,6,'Predicted bone class',ORANGE,size=10)
text(137.5,9.2,'Equal-weight soft voting',9,ORANGE)
text(59,7.5,'Encoder: Conv1D patches + fixed position → Transformer → mean pool → LayerNorm',9)
text(80,2.7,'Training: −log mean branch probability of the true class  •  Early stopping on validation loss  •  Inference: argmax',9)
for ext in ('png','svg','pdf'):
    fig.savefig(OUT/f'method_framework.{ext}',facecolor='white')
print(OUT/'method_framework.png')
