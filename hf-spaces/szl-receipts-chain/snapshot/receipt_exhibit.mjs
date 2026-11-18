/* SPDX-License-Identifier: Apache-2.0
 * Read-only public evidence index; hash of observed bytes is not signed provenance.
 */
export const DATASET = "SZLHOLDINGS/frontier-54-runs";
export const HUB_API = "https://huggingface.co/api/datasets/" + DATASET;
export const MAX_INDEX_BYTES = 8_000_000;
export const MAX_RECORD_BYTES = 256_000;
export const MAX_FILES = 25_000;
const SHA = /^[a-f0-9]{40}$/;
const PATH = /^[A-Za-z0-9_.@+-]+(?:\/[A-Za-z0-9_.@+-]+)*$/;
const RECEIPT_HINT = /(?:^|\/)(?:receipts?|officers)(?:\/|$)|(?:^|[\/_-])(?:receipt|attest|witness|proof|audit)(?:[\/_-]|\.)/i;

export class EvidenceUnavailable extends Error {
  constructor(code) { super(code); this.name = "EvidenceUnavailable"; }
}

export function indexedSource(payload) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload) ||
      payload.id !== DATASET || !SHA.test(payload.sha) ||
      !Array.isArray(payload.siblings) || payload.siblings.length > MAX_FILES)
    throw new EvidenceUnavailable("INDEX_CONTRACT_UNAVAILABLE");
  const seen = new Set(), entries = [];
  for (const candidate of payload.siblings) {
    const path = candidate?.rfilename;
    if (typeof path !== "string" || path.length > 280 ||
        !PATH.test(path) || path.includes("..") || seen.has(path))
      throw new EvidenceUnavailable("INDEX_PATH_OR_DUPLICATE_INVALID");
    seen.add(path);
    entries.push({path, category: path.split("/")[0],
      isJson: /\.jsonl?$/i.test(path),
      isReceiptCandidate: /\.jsonl?$/i.test(path) && RECEIPT_HINT.test(path)});
  }
  entries.sort((a, b) => a.path.localeCompare(b.path, "en"));
  const categories = {};
  for (const e of entries) categories[e.category] = (categories[e.category] || 0) + 1;
  return {
    schema:"szl.receipt-exhibit-index/v1", revision:payload.sha,
    updated: typeof payload.lastModified === "string" ? payload.lastModified : null,
    files: entries, fileCount:entries.length,
    jsonCount:entries.filter(e => e.isJson).length,
    candidateCount:entries.filter(e => e.isReceiptCandidate).length,
    coverage:"PUBLIC_DATASET_FILE_PATHS_ONLY_NOT_ESTATE_RECEIPT_COMPLETENESS",
    authenticityVerified:false, inclusionVerified:false, perLaneCompletionVerified:false,
    categories,
  };
}

export function sampledPaths(index, limit=72) {
  if (!index || !Array.isArray(index.files) || !Number.isInteger(limit) || limit<1 || limit>150)
    throw new EvidenceUnavailable("SAMPLE_CONTRACT_INVALID");
  const buckets=new Map(), result=[];
  for (const file of index.files.filter(e=>e.isReceiptCandidate)) {
    const list=buckets.get(file.category)||[];
    list.push(file.path); buckets.set(file.category,list);
  }
  for (let slot=0;result.length<limit;slot++) {
    let any=false;
    for (const key of [...buckets.keys()].sort()) {
      const values=buckets.get(key);
      if (slot<values.length) {
        result.push(values[values.length-1-slot]); any=true;
        if(result.length>=limit) break;
      }
    }
    if(!any)break;
  }
  return result;
}

export function nextBatchPaths(index, inspectedPaths, limit=12) {
  if (!index || !Array.isArray(index.files) || !(inspectedPaths instanceof Set) ||
      !Number.isInteger(limit) || limit<1 || limit>12)
    throw new EvidenceUnavailable("INSPECTION_BATCH_CONTRACT_INVALID");
  const priority=sampledPaths(index,150);
  const remaining=index.files.filter(e=>e.isReceiptCandidate).map(e=>e.path);
  const output=[],added=new Set();
  for(const path of [...priority,...remaining]){
    if(inspectedPaths.has(path)||added.has(path))continue;
    added.add(path);output.push(path);
    if(output.length>=limit)break;
  }
  return output;
}

export function immutableUrl(path, revision, blob=false) {
  if (!SHA.test(revision) || typeof path!=="string" || !PATH.test(path) ||
      path.includes("..")) throw new EvidenceUnavailable("IMMUTABLE_PATH_INVALID");
  const safe=path.split("/").map(encodeURIComponent).join("/");
  return "https://huggingface.co/datasets/"+DATASET+
    (blob?"/blob/":"/resolve/")+revision+"/"+safe;
}

export async function fetchBounded(url, limit, fetcher=fetch) {
  if (!Number.isInteger(limit) || limit<1 || limit>MAX_INDEX_BYTES ||
      typeof url!=="string" ||
      !(url===HUB_API || url.startsWith("https://huggingface.co/datasets/"+DATASET+"/resolve/")))
    throw new EvidenceUnavailable("UNAPPROVED_FETCH");
  let r;
  // Hugging Face's Git blobs issue a same-host 307 to /api/resolve-cache/
  // on an exact commit. Browser requests have NO credentials. Follow that
  // redirect but reject any unrecognized final origin or unbound revision.
  try {r=await fetcher(url,{cache:"no-store",redirect:"follow",
      credentials:"omit",referrerPolicy:"no-referrer"});}
  catch {throw new EvidenceUnavailable("TRANSPORT_UNAVAILABLE");}
  if(!r?.ok)throw new EvidenceUnavailable("SOURCE_HTTP_UNAVAILABLE");
  if(typeof r.url!=="string")throw new EvidenceUnavailable("SOURCE_LOCATION_UNAVAILABLE");
  if(url===HUB_API){
    if(r.url!==HUB_API)throw new EvidenceUnavailable("INDEX_SOURCE_REDIRECTED");
  } else {
    const prefix="https://huggingface.co/datasets/"+DATASET+"/resolve/";
    if(!url.startsWith(prefix))
      throw new EvidenceUnavailable("IMMUTABLE_SOURCE_REDIRECT_INVALID");
    const suffix=url.slice(prefix.length);
    const sha=suffix.slice(0,40);
    if(!SHA.test(sha)||suffix[40]!=="/")
      throw new EvidenceUnavailable("IMMUTABLE_SOURCE_REDIRECT_INVALID");
    const encodedPath=suffix.slice(41);
    let path;
    try {path=encodedPath.split("/").map(decodeURIComponent).join("/");}
    catch {throw new EvidenceUnavailable("IMMUTABLE_SOURCE_REDIRECT_INVALID");}
    if(!PATH.test(path)||path.includes("..")||
       encodedPath!==path.split("/").map(encodeURIComponent).join("/"))
      throw new EvidenceUnavailable("IMMUTABLE_SOURCE_REDIRECT_INVALID");
    const final=new URL(r.url);
    const originalPath=new URL(url).pathname;
    // The HF resolve-cache endpoint URL-encodes all path separators as %2F.
    const cachePath="/api/resolve-cache/datasets/"+DATASET+"/"+sha+"/"+
      encodeURIComponent(path);
    if(final.origin!=="https://huggingface.co" || final.hash ||
       !(final.pathname===originalPath && !final.search || final.pathname===cachePath))
      throw new EvidenceUnavailable("IMMUTABLE_SOURCE_REDIRECT_INVALID");
  }
  const declared=Number(r.headers?.get?.("content-length")||0);
  if(declared>limit)throw new EvidenceUnavailable("RECORD_TOO_LARGE");
  const stream=r.body?.getReader?.();
  if(!stream)throw new EvidenceUnavailable("STREAM_UNAVAILABLE");
  const chunks=[];let size=0;
  try {
    while(true){
      const {value,done}=await stream.read();if(done)break;
      if(!value?.byteLength)continue;
      size+=value.byteLength;
      if(size>limit)throw new EvidenceUnavailable("RECORD_TOO_LARGE");
      chunks.push(value);
    }
  } finally {stream.releaseLock();}
  const data=new Uint8Array(size);let offset=0;
  for(const c of chunks){data.set(c,offset);offset+=c.byteLength;}
  return data;
}

export async function byteDigest(bytes, cryptoApi=globalThis.crypto) {
  if(!(bytes instanceof Uint8Array)||!cryptoApi?.subtle)
    throw new EvidenceUnavailable("DIGEST_UNAVAILABLE");
  const hash=new Uint8Array(await cryptoApi.subtle.digest("SHA-256",bytes.slice()));
  return [...hash].map(x=>x.toString(16).padStart(2,"0")).join("");
}

export function boundedReceipt(bytes,path) {
  if(!(bytes instanceof Uint8Array)||bytes.length>MAX_RECORD_BYTES ||
     typeof path!=="string"||!PATH.test(path))throw new EvidenceUnavailable("RECEIPT_BYTE_BUDGET_OR_PATH");
  let plain;try {plain=new TextDecoder("utf-8",{fatal:true}).decode(bytes);}
  catch {throw new EvidenceUnavailable("RECEIPT_NOT_UTF8");}
  let data,records=1;
  try {
    if(path.endsWith(".jsonl")) {
      const lines=plain.split(/\r?\n/).filter(l=>l.trim());
      if(lines.length>5000)throw new EvidenceUnavailable("JSONL_LINE_BUDGET");
      records=lines.length;
      for(const l of lines){
        const item=JSON.parse(l);
        if(!item||typeof item!=="object"||Array.isArray(item))
          throw new EvidenceUnavailable("RECEIPT_RECORD_NOT_OBJECT");
      }
      data=lines.slice(0,12).map(l=>JSON.parse(l));
    } else {
      data=JSON.parse(plain);
      if(!data||typeof data!=="object"||Array.isArray(data))
        throw new EvidenceUnavailable("RECEIPT_RECORD_NOT_OBJECT");
    }
  }catch(e){if(e instanceof EvidenceUnavailable)throw e;
    throw new EvidenceUnavailable("RECEIPT_PARSE_UNAVAILABLE");}
  return {data,records,displayedRecords:Array.isArray(data)?data.length:1,
    state:"PARSED_BYTES_NOT_SIGNATURE_VERIFIED",
    signatureVerified:false,merkleInclusionVerified:false,subjectBound:false};
}

export function coverage(index, inspected) {
  const parsed=inspected.filter(x=>x.status==="PARSED_BYTES_NOT_SIGNATURE_VERIFIED").length;
  const failures=inspected.filter(x=>x.status!=="PARSED_BYTES_NOT_SIGNATURE_VERIFIED").length;
  return {indexFiles:index.fileCount,receiptCandidatePaths:index.candidateCount,
    inspectedCandidatePaths:inspected.length,parsedCandidatePaths:parsed,
    unavailableCandidatePaths:failures,
    uninspectedCandidatePaths:Math.max(0,index.candidateCount-inspected.length),
    inspectionRatio:index.candidateCount?parsed/index.candidateCount:null,
    trustedSignaturesVerified:0,merkleInclusionsVerified:0,
    interpretation:"OBSERVED_BYTES_ONLY_NO_AUTHORITY"};
}