/* SPDX-License-Identifier: Apache-2.0
 * Zero credential/API mutation frontend. Only public immutable datasets.
 */
import {
  HUB_API, MAX_INDEX_BYTES, MAX_RECORD_BYTES, indexedSource, sampledPaths, nextBatchPaths,
  immutableUrl, fetchBounded, byteDigest, boundedReceipt, coverage
} from "./receipt_exhibit.mjs";

const get=id=>document.getElementById(id);
let index=null, page=0, inspected=new Map(), loading=false, selectionGeneration=0;
const PAGE=48;
const safe=(value,max=260)=>String(value??"UNAVAILABLE").slice(0,max);
const elt=(tag,text,klass)=>{
  const n=document.createElement(tag); if(klass)n.className=klass;
  n.textContent=text;return n;
};
const say=(status,problem=false)=>{
  get("status").textContent=status;
  get("truth-badge").textContent=problem?"UNAVAILABLE / HOLD":"INDEX OBSERVED / AUTHORITY UNVERIFIED";
  get("truth-badge").classList.toggle("unavailable",problem);
};
function clear(){
  // Invalidate any earlier in-flight file result before a source refresh.
  selectionGeneration++;
  index=null;page=0;inspected.clear();get("rows").replaceChildren(elt("tr",""));
  get("index-table").querySelector("tbody").replaceChildren();
  for(const key of ["indexed","candidates","inspected"])get(key).textContent="—";
  for(const key of ["source","updated","observed"])get(key).textContent="UNAVAILABLE";
  get("coverage-label").textContent="No inspection evidence.";
  get("scope").textContent="NO INDEX OBSERVED";
  get("list-count").textContent="No indexed paths";
  get("selected-path").textContent="No record selected.";
  get("digest").textContent="UNAVAILABLE";
  get("records-count").textContent="UNAVAILABLE";
  get("record").textContent="No record selected.";
  get("source-link").replaceChildren();
  get("record-state").textContent="NOT INSPECTED";
  get("inspect-more").disabled=true;
  const c=get("constellation");c.getContext("2d").clearRect(0,0,c.width,c.height);
  get("categories").textContent="";
}
function updateSummary(){
  if(!index)return;
  const counts=coverage(index,[...inspected.values()]);
  get("indexed").textContent=index.fileCount.toLocaleString();
  get("candidates").textContent=index.candidateCount.toLocaleString();
  get("inspected").textContent=counts.parsedCandidatePaths+" / "+counts.receiptCandidatePaths;
  get("coverage-label").textContent=counts.unavailableCandidatePaths+
    " failed, "+counts.uninspectedCandidatePaths+" uninspected. No cryptographic attestation proven.";
  get("inspect-more").disabled=loading||nextBatchPaths(index,new Set(inspected.keys())).length===0;
}
function filtered(){
  if(!index)return [];
  const search=get("search").value.trim().toLowerCase().slice(0,140);
  return index.files.filter(e=>(!get("only-candidates").checked||e.isReceiptCandidate) &&
    (!search||e.path.toLowerCase().includes(search)));
}
function table(){
  const items=filtered(),total=items.length,start=page*PAGE;
  if(start>=total&&page>0){page=Math.max(0,Math.ceil(total/PAGE)-1);return table();}
  const body=get("rows");body.replaceChildren();
  for(const item of items.slice(page*PAGE,(page+1)*PAGE)){
    const row=document.createElement("tr");
    const path=document.createElement("td");
    if(item.isJson){
      const b=elt("button",item.path);b.type="button";
      b.addEventListener("click",()=>inspectFile(item,false));
      path.append(b);
    }else path.textContent=item.path;
    const klass=elt("td",item.isReceiptCandidate?"receipt-like":item.isJson?"JSON / non-receipt candidate":"other file");
    const observation=inspected.get(item.path);
    const stage=elt("td",observation?.status || "NOT INSPECTED",
      observation?.status === "PARSED_BYTES_NOT_SIGNATURE_VERIFIED" ? "mono" : "mono muted");
    row.append(path,klass,stage);body.append(row);
  }
  if(total===0){
    const tr=document.createElement("tr"),td=elt("td","No matching public paths in the indexed source.");
    td.colSpan=3;tr.append(td);body.append(tr);
  }
  get("list-count").textContent=total.toLocaleString()+" matches · showing "+
    (total?((page*PAGE)+1)+"–"+Math.min((page+1)*PAGE,total):"none")+
    " · "+(index?.fileCount??0).toLocaleString()+" public file paths indexed.";
  get("previous").disabled=page<=0;
  get("next").disabled=(page+1)*PAGE>=total;
}
function draw(){
  if(!index)return;
  const cv=get("constellation"),ratio=Math.min(window.devicePixelRatio||1,2);
  const width=cv.clientWidth||680,height=330;
  cv.width=Math.ceil(width*ratio);cv.height=height*ratio;
  const ctx=cv.getContext("2d");ctx.setTransform(ratio,0,0,ratio,0,0);
  ctx.clearRect(0,0,width,height);
  const palette=["#4ce0d8","#f3c66b","#809aff","#c5a0ef","#fb8e98","#7fd9a1","#b3c0d8"];
  const chosen=sampledPaths(index,110);
  const groups=[...new Set(chosen.map(v=>v.split("/")[0]))].sort();
  chosen.forEach((path,i)=>{
    const cat=groups.indexOf(path.split("/")[0]);
    const theta=i*2.399963229728653;
    const radial=24+Math.sqrt((i+.5)/(chosen.length+.5))*Math.min(width*.42,140);
    const x=width/2+Math.cos(theta)*radial;
    const y=height/2+Math.sin(theta)*radial*.83;
    ctx.beginPath();ctx.arc(x,y,2.9+(cat%3)*.45,0,2*Math.PI);
    ctx.fillStyle=palette[cat%palette.length];ctx.fill();
  });
  get("categories").textContent=groups.length+" sampled source categories · "+
    chosen.length+" rendered paths · 0 cryptographic edges asserted.";
}
function redacted(value,depth=0){
  if(depth>7)return "[NESTING_LIMIT]";
  if(Array.isArray(value))return value.slice(0,25).map(v=>redacted(v,depth+1));
  if(value&&typeof value==="object"){
    const out={};let n=0;
    for(const [key,v] of Object.entries(value)){
      if(n++>45){out["__truncated__"]="FIELD_LIMIT";break;}
      out[key]=/(?:secret|password|token|private.?key|authorization|api.?key|credential|bearer)/i.test(key) ?
        "[REDACTED_PUBLIC_PREVIEW]" : redacted(v,depth+1);
    }
    return out;
  }
  return typeof value==="string"?value.slice(0,700):value;
}
async function inspectFile(file,quiet=false){
  if(!index||loading||!file?.isJson)return;
  const ticket=++selectionGeneration;
  const revision=index.revision;
  if(!quiet){
    get("record-state").textContent="FETCHING IMMUTABLE FILE";
    get("selected-path").textContent=file.path;
    get("record").textContent="Reading bounded public bytes at source SHA…";
    get("digest").textContent="UNAVAILABLE";
  }
  let status="UNAVAILABLE",dig="UNAVAILABLE",count="UNAVAILABLE",data=null,href;
  try {
    href=immutableUrl(file.path,revision);
    const bytes=await fetchBounded(href,MAX_RECORD_BYTES);
    dig=await byteDigest(bytes);
    const receipt=boundedReceipt(bytes,file.path);
    status=receipt.state;data=receipt.data;
    count=receipt.records+" parsed record(s), "+receipt.displayedRecords+
      " displayed, "+bytes.length+" bytes read. Not signed.";
  }catch(err){status=err?.message&&/^[A-Z_]+$/.test(err.message)?err.message:"INSPECTION_UNAVAILABLE";}
  // Never display/merge an old source response into a newer revision or
  // let a slow previous click override the currently selected record.
  if(!index||index.revision!==revision)return;
  if(file.isReceiptCandidate)inspected.set(file.path,{status});
  if(!quiet&&ticket===selectionGeneration){
    get("record-state").textContent=status;
    get("selected-path").textContent=file.path;
    get("digest").textContent=dig;
    get("records-count").textContent=count;
    get("record").textContent=data===null?status:JSON.stringify(redacted(data),null,2).slice(0,45_000);
    get("source-link").replaceChildren();
    if(href){
      const a=elt("a","Open this immutable file in Hugging Face");
      a.href=immutableUrl(file.path,revision,true);
      a.target="_blank";a.rel="noopener noreferrer";get("source-link").append(a);
    }
  }
  updateSummary();table();
}
async function inspectBatch(){
  if(!index||loading)return;
  loading=true;get("inspect-more").disabled=true;
  const chosen=nextBatchPaths(index,new Set(inspected.keys()),12);
  for(const path of chosen){
    const file=index.files.find(e=>e.path===path);
    if(!file)continue;
    // Only one request is in flight, and no loop can grow beyond 12.
    let status="UNAVAILABLE";
    try{
      const bytes=await fetchBounded(immutableUrl(path,index.revision),MAX_RECORD_BYTES);
      await byteDigest(bytes);
      boundedReceipt(bytes,path);
      status="PARSED_BYTES_NOT_SIGNATURE_VERIFIED";
    }catch(e){status=e?.message&&/^[A-Z_]+$/.test(e.message)?e.message:"UNAVAILABLE";}
    inspected.set(path,{status});
  }
  loading=false;updateSummary();table();
  say("Indexed "+index.fileCount.toLocaleString()+" paths at revision "+index.revision.slice(0,12)+
    "…; "+inspected.size+" bounded candidates inspected. Signatures/lineage NOT verified.");
}
async function refresh(){
  if(loading)return;
  loading=true;get("refresh").disabled=true;clear();
  say("Reading one public metadata file listing with an 8 MB cap…",true);
  try {
    const raw=await fetchBounded(HUB_API,MAX_INDEX_BYTES);
    const decoded=new TextDecoder("utf-8",{fatal:true}).decode(raw);
    const parsed=JSON.parse(decoded);
    index=indexedSource(parsed);
    get("source").textContent=index.revision;
    get("updated").textContent=index.updated||"UNAVAILABLE";
    get("observed").textContent=new Date().toISOString();
    get("scope").textContent=index.coverage;
    updateSummary();table();draw();
    say("Indexed "+index.fileCount.toLocaleString()+" source paths at SHA "+
      index.revision.slice(0,12)+"…. Parsed content: 0 until inspected. Full SZL estate coverage UNVERIFIED.");
  }catch(err){
    clear();say("PUBLIC INDEX UNAVAILABLE: "+safe(err?.message||"SOURCE_UNAVAILABLE",80)+
      ". Do not replace missing bytes with historical or synthetic receipts.",true);
  }finally{loading=false;get("refresh").disabled=false;updateSummary();}
}
get("refresh").addEventListener("click",refresh);
get("inspect-more").addEventListener("click",inspectBatch);
get("search").addEventListener("input",()=>{page=0;table();});
get("only-candidates").addEventListener("change",()=>{page=0;table();});
get("previous").addEventListener("click",()=>{page=Math.max(0,page-1);table();});
get("next").addEventListener("click",()=>{page++;table();});
window.addEventListener("resize",()=>{if(index)draw();});
refresh();