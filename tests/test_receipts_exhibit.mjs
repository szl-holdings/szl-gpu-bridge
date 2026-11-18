/* SPDX-License-Identifier: Apache-2.0 */
import {test} from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {webcrypto} from "node:crypto";
import {
  DATASET,HUB_API,MAX_INDEX_BYTES,MAX_RECORD_BYTES,indexedSource,sampledPaths,nextBatchPaths,
  immutableUrl,fetchBounded,byteDigest,boundedReceipt,coverage,
} from "../hf-spaces/szl-receipts-chain/snapshot/receipt_exhibit.mjs";

const REV="c".repeat(40);
const info=(paths)=>({id:DATASET,sha:REV,lastModified:"2026-11-17T20:00:00.000Z",
  siblings:paths.map(rfilename=>({rfilename}))});
const utf=s=>new TextEncoder().encode(s);
function response(items,contentLength,url=HUB_API) {
  let offset=0;
  return {ok:true,url,headers:{get:()=>contentLength===undefined?null:String(contentLength)},
    body:{getReader:()=>({
      read:async()=>offset<items.length?{value:items[offset++],done:false}:{done:true},
      releaseLock:()=>{},
    })}};
}
test("full index counts paths separately from likely receipts and signatures",()=>{
  const paths=["README.md","receipts/b.json","mesh/lanelogs/worker.json",
    "receipts/a.jsonl","officers/attest.json","benchmark/result.json"];
  const v=indexedSource(info(paths));
  assert.equal(v.fileCount,6);assert.equal(v.jsonCount,5);
  assert.equal(v.candidateCount,3);
  assert.equal(v.authenticityVerified,false);
  assert.equal(v.inclusionVerified,false);
  assert.equal(v.perLaneCompletionVerified,false);
  const result=coverage(v,[]);
  assert.deepEqual([result.inspectedCandidatePaths,result.uninspectedCandidatePaths],[0,3]);
  assert.equal(result.trustedSignaturesVerified,0);
  assert.equal(result.inspectionRatio,0);
});
test("multiple source directories sampled deterministically without first-N traversal bias",()=>{
  const paths=["README.md"];
  for(let i=0;i<130;i++) paths.push("receipts/r"+i+".json");
  for(let i=0;i<15;i++) paths.push("officers/r"+i+".json");
  const a=indexedSource(info(paths));
  const sample=sampledPaths(a,20);
  assert.equal(sample.length,20);
  assert(sample.some(p=>p.startsWith("officers/")));
  assert(sample.some(p=>p.startsWith("receipts/")));
  assert.equal(new Set(sample).size,20);
  assert.deepEqual(sample,sampledPaths(indexedSource(info([...paths].reverse())),20));
});
test("bounded batches eventually expose every indexed receipt-candidate path",()=>{
  const paths=Array.from({length:440},(_,i)=>"receipts/path-"+i+".json");
  paths.push("README.md");
  const v=indexedSource(info(paths));
  const inspected=new Set();
  let loops=0;
  while(true){
    const batch=nextBatchPaths(v,inspected,12);
    if(!batch.length)break;
    assert(batch.length<=12);
    for(const file of batch)inspected.add(file);
    if(++loops>40)throw Error("Unbounded inspection loop");
  }
  assert.equal(inspected.size,440);
  assert.equal(nextBatchPaths(v,inspected,12).length,0);
  assert.throws(()=>nextBatchPaths(v,inspected,999),/BATCH_CONTRACT/);
});
test("untrusted metadata duplicate, path traversal, malformed SHA fail closed",()=>{
  assert.throws(()=>indexedSource(info(["receipts/a.json","receipts/a.json"])),/INDEX_PATH/);
  assert.throws(()=>indexedSource(info(["receipts/..\/secret.json"])),/INDEX_PATH/);
  assert.throws(()=>indexedSource({...info(["receipts/a.json"]),sha:"main"}),/INDEX_CONTRACT/);
  assert.throws(()=>indexedSource({...info(["receipts/a.json"]),siblings:null}),/INDEX_CONTRACT/);
  assert.throws(()=>sampledPaths(null),/SAMPLE_CONTRACT/);
});
test("immutable links have digest revision; never resolve/main or path traversal",()=>{
  const href=immutableUrl("receipts/model/v1.json",REV);
  assert.equal(href,"https://huggingface.co/datasets/"+DATASET+"/resolve/"+REV+"/receipts/model/v1.json");
  assert(immutableUrl("receipts/model/v1.json",REV,true).includes("/blob/"+REV+"/"));
  assert.throws(()=>immutableUrl("x.json","main"),/IMMUTABLE_PATH/);
  assert.throws(()=>immutableUrl("../secret.json",REV),/IMMUTABLE_PATH/);
});
test("bounded fetch enforces exact source, content length and streaming limit",async()=>{
  const good=HUB_API;
  const fake=async()=>response([utf('{"ok":true}')]);
  assert.equal(new TextDecoder().decode(await fetchBounded(good,100,fake)),'{"ok":true}');
  await assert.rejects(()=>fetchBounded("https://evil.example/secret",100,fake),/UNAPPROVED_FETCH/);
  await assert.rejects(()=>fetchBounded(good,10,async()=>response([],100)),/RECORD_TOO_LARGE/);
  await assert.rejects(()=>fetchBounded(good,5,async()=>response([utf("123456")])),/RECORD_TOO_LARGE/);
  await assert.rejects(()=>fetchBounded(good,10,async()=>{throw Error("Bearer private-key")}),/TRANSPORT_UNAVAILABLE/);
  await assert.rejects(()=>fetchBounded(good,MAX_INDEX_BYTES+1,fake),/UNAPPROVED_FETCH/);
});
test("exact Hub 307 cache path is allowed, unbound or cross-host final URL denied",async()=>{
  const path="receipts/a.json", request=immutableUrl(path,REV);
  const cached="https://huggingface.co/api/resolve-cache/datasets/"+DATASET+"/"+REV+"/receipts%2Fa.json";
  let options;
  const reader=async(url,opts)=>{
    options=opts;
    return response([utf('{"a":1}')],7,cached);
  };
  const bytes=await fetchBounded(request,MAX_RECORD_BYTES,reader);
  assert.equal(new TextDecoder().decode(bytes),'{"a":1}');
  assert.equal(options.redirect,"follow");
  assert.equal(options.credentials,"omit");
  assert.equal(options.referrerPolicy,"no-referrer");
  await assert.rejects(()=>fetchBounded(request,99,async()=>
    response([utf('{}')],2,"https://evil.example/file")),/IMMUTABLE_SOURCE_REDIRECT_INVALID/);
  await assert.rejects(()=>fetchBounded(request,99,async()=>
    response([utf('{}')],2,"https://huggingface.co/api/resolve-cache/datasets/"+DATASET+"/"+REV+"/receipts%2Fother.json")),/IMMUTABLE_SOURCE_REDIRECT_INVALID/);
  await assert.rejects(()=>fetchBounded(request,99,async()=>
    response([utf('{}')],2,"https://huggingface.co/datasets/"+DATASET+"/resolve/"+REV+"/receipts/another.json")),/IMMUTABLE_SOURCE_REDIRECT_INVALID/);
  await assert.rejects(()=>fetchBounded(request,99,async()=>
    response([utf('{}')],2,"https://huggingface.co/api/resolve-cache/datasets/"+DATASET+"/"+"d".repeat(40)+"/x")),/IMMUTABLE_SOURCE_REDIRECT_INVALID/);
  await assert.rejects(()=>fetchBounded(HUB_API,99,async()=>
    response([utf('{}')],2,cached)),/INDEX_SOURCE_REDIRECTED/);
});
test("parsed file does not magically verify embedded hash, signature or signer",async()=>{
  const bytes=utf('{"kind":"training","receipt_sha256":"fake","token":"secret"}');
  const v=boundedReceipt(bytes,"receipts/test.json");
  assert.equal(v.state,"PARSED_BYTES_NOT_SIGNATURE_VERIFIED");
  assert.equal(v.signatureVerified,false);assert.equal(v.merkleInclusionVerified,false);
  assert.equal(v.subjectBound,false);
  const digest=await byteDigest(bytes,webcrypto);
  assert.match(digest,/^[a-f0-9]{64}$/);
  assert.notEqual(digest,"fake");
});
test("JSONL counts every line while showing bounded rows; invalid data never passes",()=>{
  const many=Array.from({length:15},(_,i)=>JSON.stringify({i})).join("\n")+"\n";
  const r=boundedReceipt(utf(many),"receipts/a.jsonl");
  assert.equal(r.records,15);assert.equal(r.displayedRecords,12);
  assert.throws(()=>boundedReceipt(utf('{"ok":1}\nnot-json'),"receipts/a.jsonl"),/RECEIPT_PARSE/);
  assert.throws(()=>boundedReceipt(utf('{"ok":1}\n[]'),"receipts/a.jsonl"),/RECEIPT_RECORD/);
  assert.throws(()=>boundedReceipt(new Uint8Array(MAX_RECORD_BYTES+1),"receipts/a.json"),/BYTE_BUDGET/);
});
test("public page is read-only, has explicit authority holds and deploys JS assets",()=>{
  const dir=new URL("../hf-spaces/szl-receipts-chain/snapshot/",import.meta.url);
  const html=readFileSync(new URL("index.html",dir),"utf8");
  const app=readFileSync(new URL("exhibit_app.mjs",dir),"utf8");
  const docker=readFileSync(new URL("Dockerfile",dir),"utf8");
  assert(html.includes('receipt_exhibit')===false);
  assert(html.includes('id="coverage-label"'));
  assert(html.includes("Signatures")||html.includes("signature"));
  assert(html.includes("NOT INSPECTED"));
  assert(!html.includes("Every claim the estate makes, as a live tamper-evident chain"));
  assert(!html.includes("sampled live"));
  assert(app.includes('textContent')); assert(!app.includes('innerHTML'));
  assert(app.includes('selectionGeneration++')); // source-refresh and click race invalidation
  assert(app.includes('index.revision!==revision'));
  assert(app.includes("MAX_RECORD_BYTES")); assert(app.includes("MAX_INDEX_BYTES"));
  assert(docker.includes("COPY receipt_exhibit.mjs"));
  assert(docker.includes("COPY exhibit_app.mjs"));
  assert(docker.includes("COPY favicon.svg"));
  assert(html.includes('rel="icon" type="image/svg+xml"'));
});