from pathlib import Path
from pptx_writer import Deck
deck=Deck()
import json
from PIL import Image, ImageOps
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'/'Desktop_Delivery'
OUT.mkdir(exist_ok=True)
for name,file in [('Sans','arial.ttf'),('Bold','arialbd.ttf'),('Serif','georgia.ttf')]:
    font_path=Path('C:/Windows/Fonts')/file
    if not font_path.exists():
        import reportlab
        font_path=Path(reportlab.__file__).parent/'fonts'/('VeraBd.ttf' if name=='Bold' else 'Vera.ttf')
    pdfmetrics.registerFont(TTFont(name,str(font_path)))
W,H=1280,720
ink='#172E32'; paper='#F4F1E9'; muted='#607477'; teal='#137E79'; orange='#BF5739'; pale='#DCE9E3'
c=canvas.Canvas(str(OUT/'Fake_News_Presentation.pdf'),pagesize=(W,H))
c.setTitle('Fake News Detection | AAI-2602 | Real ISOT experiment')
c.setAuthor('Kurmet Adilzhan; Tolegenov Nursultan; Issakulov Zeiin')
n=0
def rect(x,y,w,h,color):
    deck.shape('rect',x,y,w,h,color)
    c.setFillColor(HexColor(color)); c.rect(x,H-y-h,w,h,fill=1,stroke=0)
def text(s,x,y,w=1100,size=26,color=ink,font='Sans'):
    deck.text(s,x,y,w,size,color,font)
    p=Paragraph(s,ParagraphStyle('x',fontName=font,fontSize=size,leading=size*1.25,textColor=HexColor(color)))
    _,h=p.wrap(w,1000); p.drawOn(c,x,H-y-h); return h
def line(x,y,x2,y2,color=muted,width=1):
    deck.shape('line',min(x,x2),min(y,y2),abs(x2-x),abs(y2-y),None,(color,width))
    c.setStrokeColor(HexColor(color));c.setLineWidth(width);c.line(x,H-y,x2,H-y2)
def start(kicker,title,dark=False):
    global n
    n+=1;rect(0,0,W,H,ink if dark else paper)
    fg=paper if dark else ink
    text(kicker.upper(),64,35,size=14,color='#91C6BD' if dark else teal,font='Bold')
    text(title,64,83,1150,43,fg,'Serif')
    line(64,666,1216,666,'#476064' if dark else '#C9D0C8')
    text('AAI-2602',64,680,size=12,color='#A7B9B5' if dark else muted)
    text(f'{n:02d}',1170,678,50,15,fg)
def end():
    deck.end()
    c.showPage()
def photo(x,y,w,h):
    im=ImageOps.fit(Image.open(ROOT/'assets/newspaper.jpg'),(int(w*2),int(h*2)),centering=(.5,.5))
    deck.image(im,x,y,w,h)
    from reportlab.lib.utils import ImageReader
    c.drawImage(ImageReader(im),x,H-y-h,w,h)
def note(s,y=603):
    rect(64,y-10,1152,46,pale);text(s,80,y,1115,19,teal)

