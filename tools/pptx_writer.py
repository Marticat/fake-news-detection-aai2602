"""Minimal editable OOXML presentation writer for the project's slide primitives."""
from zipfile import ZipFile, ZIP_DEFLATED
from xml.sax.saxutils import escape
from io import BytesIO
import re
import html
from reportlab.pdfbase.pdfmetrics import stringWidth

NS='xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'
REL='http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
def emu(v): return round(v*9525)
def xf(x,y,w,h): return f'<a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
def fill(col): return f'<a:solidFill><a:srgbClr val="{col.lstrip(chr(35))}"/></a:solidFill>'
def rels(items): return '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(f'<Relationship Id="{id}" Type="{REL}{typ}" Target="{target}"/>' for id,typ,target in items)+'</Relationships>'
GROUP='<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
class Deck:
    def __init__(self): self.slides=[];self.media={};self.shapes=[];self.links=[]
    def ident(self): return len(self.shapes)+2
    def shape(self,geom,x,y,w,h,col,stroke=None):
        i=self.ident(); ln=f'<a:ln w="{emu(stroke[1])}">{fill(stroke[0])}<a:prstDash val="solid"/></a:ln>' if stroke else '<a:ln><a:noFill/></a:ln>'
        self.shapes.append(f'<p:sp><p:nvSpPr><p:cNvPr id="{i}" name="Shape {i}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr>{xf(x,y,w,h)}<a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom>{fill(col) if col else "<a:noFill/>"}{ln}</p:spPr></p:sp>')
    def text(self,s,x,y,w,size,color,font):
        if not s:return
        lines=[]
        for para in re.split(r'<br\s*/?>',s):
            words=html.unescape(para).split(); current=''
            for word in words:
                test=(current+' '+word).strip()
                if current and stringWidth(test,font,size)>w: lines.append(current);current=word
                else:current=test
            lines.append(current)
        face='Georgia' if font=='Serif' else 'Arial'; sz=round(size*75); bold=' b="1"' if font=='Bold' else ''
        paras=''.join(f'<a:p><a:pPr><a:lnSpc><a:spcPts val="{round(size*1.25*75)}"/></a:lnSpc><a:spcAft><a:spcPts val="0"/></a:spcAft></a:pPr><a:r><a:rPr lang="en-US" sz="{sz}"{bold}>{fill(color)}<a:latin typeface="{face}"/></a:rPr><a:t>{escape(t)}</a:t></a:r><a:endParaRPr sz="{sz}"/></a:p>' for t in lines)
        i=self.ident();h=len(lines)*size*1.25+size*.2
        self.shapes.append(f'<p:sp><p:nvSpPr><p:cNvPr id="{i}" name="Text {i}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr>{xf(x,y,w+2,h)}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr><p:txBody><a:bodyPr wrap="none" lIns="0" tIns="0" rIns="0" bIns="0" anchor="t"><a:noAutofit/></a:bodyPr><a:lstStyle/>{paras}</p:txBody></p:sp>')
    def image(self,im,x,y,w,h):
        name=f'image{len(self.media)+1}.jpg';bio=BytesIO();im.convert('RGB').save(bio,format='JPEG',quality=92);self.media[name]=bio.getvalue()
        rid=f'rId{len(self.links)+2}';self.links.append((rid,'image','../media/'+name));i=self.ident()
        self.shapes.append(f'<p:pic><p:nvPicPr><p:cNvPr id="{i}" name="Newspaper photograph"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr>{xf(x,y,w,h)}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>')
    def end(self): self.slides.append((self.shapes,self.links));self.shapes=[];self.links=[]
    def save(self,path):
        with ZipFile(path,'w',ZIP_DEFLATED) as z:
            overrides=[('/ppt/presentation.xml','presentation.main'),('/ppt/slideMasters/slideMaster1.xml','slideMaster'),('/ppt/slideLayouts/slideLayout1.xml','slideLayout')]+[(f'/ppt/slides/slide{i}.xml','slide') for i in range(1,len(self.slides)+1)]
            ct='<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="jpg" ContentType="image/jpeg"/>'
            ct+=''.join(f'<Override PartName="{p}" ContentType="application/vnd.openxmlformats-officedocument.presentationml.{t}+xml"/>' for p,t in overrides)+'</Types>'
            z.writestr('[Content_Types].xml',ct);z.writestr('_rels/.rels',rels([('rId1','officeDocument','ppt/presentation.xml')]))
            ids=''.join(f'<p:sldId id="{255+i}" r:id="rId{i+1}"/>' for i in range(1,len(self.slides)+1))
            z.writestr('ppt/presentation.xml',f'<p:presentation {NS}><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:sldIdLst>{ids}</p:sldIdLst><p:sldSz cx="{emu(1280)}" cy="{emu(720)}" type="screen16x9"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>')
            z.writestr('ppt/_rels/presentation.xml.rels',rels([('rId1','slideMaster','slideMasters/slideMaster1.xml')]+[(f'rId{i+1}','slide',f'slides/slide{i}.xml') for i in range(1,len(self.slides)+1)]))
            cmap='<p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" folHlink="folHlink" hlink="hlink" tx1="dk1" tx2="dk2"/>'
            z.writestr('ppt/slideMasters/slideMaster1.xml',f'<p:sldMaster {NS}><p:cSld><p:spTree>{GROUP}</p:spTree></p:cSld>{cmap}<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>')
            z.writestr('ppt/slideMasters/_rels/slideMaster1.xml.rels',rels([('rId1','slideLayout','../slideLayouts/slideLayout1.xml')]))
            z.writestr('ppt/slideLayouts/slideLayout1.xml',f'<p:sldLayout {NS} type="blank" preserve="1"><p:cSld name="Blank"><p:spTree>{GROUP}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>')
            z.writestr('ppt/slideLayouts/_rels/slideLayout1.xml.rels',rels([('rId1','slideMaster','../slideMasters/slideMaster1.xml')]))
            for i,(shapes,links) in enumerate(self.slides,1):
                z.writestr(f'ppt/slides/slide{i}.xml',f'<p:sld {NS}><p:cSld><p:spTree>{GROUP}{"".join(shapes)}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>')
                z.writestr(f'ppt/slides/_rels/slide{i}.xml.rels',rels([('rId1','slideLayout','../slideLayouts/slideLayout1.xml')]+links))
            for name,data in self.media.items():z.writestr('ppt/media/'+name,data)
