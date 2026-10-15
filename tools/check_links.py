"""Check local Markdown paths and document anchors without network requests."""
from pathlib import Path
from urllib.parse import unquote,urlsplit
from html.parser import HTMLParser
import re,json,html
R=Path(__file__).resolve().parents[1];broken=[];checked=0
class Ids(HTMLParser):
 def __init__(self):super().__init__();self.ids=set()
 def handle_starttag(self,t,a):
  d=dict(a)
  if 'id' in d:self.ids.add(d['id'])
def anchors(p):
 s=p.read_text();found=set();seen={}
 if p.suffix=='.html':
  h=Ids();h.feed(s);return h.ids
 for h in re.findall(r'^#{1,6}\s+(.+)$',s,re.M):
  h=html.unescape(h).lower();h=re.sub(r'[^\w\- ]','',h).replace(' ','-');n=seen.get(h,0);seen[h]=n+1;found.add(h+('-'+str(n) if n else ''))
 return found
for p in R.rglob('*.md'):
 if '.git' in p.parts:continue
 for label,u in re.findall(r'\[([^\]\n]+)\]\(([^)]+)\)',p.read_text()):
  if u.startswith(('https:','http:','mailto:','data:')):continue
  parts=urlsplit(u);q=(p.parent/unquote(parts.path)).resolve() if parts.path else p
  checked+=1
  if not q.exists():broken.append({'file':str(p.relative_to(R)),'url':u,'reason':'missing'})
  elif parts.fragment and q.is_file() and q.suffix in ['.md','.html'] and unquote(parts.fragment) not in anchors(q):broken.append({'file':str(p.relative_to(R)),'url':u,'reason':'anchor'})
print(json.dumps({'local_markdown_links_checked':checked,'broken':broken},indent=2));assert not broken
