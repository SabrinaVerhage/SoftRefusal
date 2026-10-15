const $ = id => document.getElementById(id);
let sessionListKey = '', instructionSession = null, instructionDirty = false;
let session = localStorage.getItem('gemma-session'), state, rendering = '', currentJob = null, sending = false, appClosed = false, downloading = false;
const stages = ['prepared','capturing','generating','saved'];
const autoTagStage = 'Reply saved · Gemma is tagging its behaviour';
const stageNames = {starting:'Preparing your conversation',loading:'Loading Gemma on your laptop',queued:'Waiting for the cloud GPU',prepared:'Conversation prepared',capturing:'Capturing internal snapshots',generating:'Gemma is writing',saving:'Saving the recording',saved:'Recording saved',stopped:'Reply stopped',error:'Recording needs attention'};
async function api(path, data) {
  const response = await fetch(path, data === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  const result = await response.json(); if(!response.ok) throw new Error(result.error || 'The request failed.'); return result;
}
function alertMessage(text){$('alert').hidden=!text;$('alert').textContent=text||'';}
function node(tag, className, text){const el=document.createElement(tag);if(className)el.className=className;if(text!==undefined)el.textContent=text;return el;}
async function newChat(){try{const r=await api('/api/new',{});session=r.session;localStorage.setItem('gemma-session',session);rendering='';alertMessage('');await refresh();$('prompt').focus();}catch(e){alertMessage(e.message);}}
function message(role,text,meta,warning=false){const box=node('article','message '+role);box.append(node('div','message-label',role==='user'?'YOU':'GEMMA'));box.append(node('div','message-body',text));if(meta)box.append(node('div','message-meta'+(warning?' warning':''),meta));return box;}
const outcomes=[['answered','Answered'],['redirected','Redirected'],['refused','Refused'],['unclear','Unclear']];
function labelBar(r){
  const bar=node('div','label-bar');
  const source=r.label_source==='human'?'Your label':r.label_source==='model_judge'?(r.label_status==='error'?'Auto-tag unavailable':'Auto-tagged by Gemma'):'Not tagged yet';
  const title=node('span','label-title',source+' · correct if needed');title.title=r.label_reason||'';bar.append(title);
  outcomes.forEach(([value,text])=>{const on=r.label===value,b=node('button','label-option'+(on?' active':''),text);b.type='button';b.setAttribute('aria-pressed',String(on));b.onclick=async()=>{try{await api('/api/label',{session,point_id:r.point_id,outcome:value});rendering='';await refresh();}catch(e){alertMessage(e.message);}};bar.append(b);});
  if(r.label_source==='human'&&state?.auto_label_enabled){const reset=node('button','label-option','Use auto tag');reset.type='button';reset.onclick=async()=>{try{await api('/api/label',{session,point_id:r.point_id,outcome:'auto'});rendering='';await refresh();}catch(e){alertMessage(e.message);}};bar.append(reset);}
  if(r.label_reason)bar.append(node('span','label-reason',r.label_reason));
  return bar;
}
function renderMessages(s){
  const job=s.job;const key=JSON.stringify([s.rows,job?.id,job?.status,job?.response]);if(key===rendering)return;rendering=key;
  const list=$('messages'),nearBottom=list.scrollHeight-list.scrollTop-list.clientHeight<100;
  const welcome=$('welcome');list.querySelectorAll('article').forEach(el=>el.remove());
  if(!s.rows.length&&!job){if(welcome)welcome.hidden=false;list.scrollTop=0;return;}
  if(welcome)welcome.hidden=true;
  s.rows.forEach(r=>{list.append(message('user',r.prompt));const reply=message('assistant',r.response,r.hit_token_limit?'Saved · reply reached its length limit; it may be unfinished':'Recording saved · four full snapshots',r.hit_token_limit);reply.append(labelBar(r));list.append(reply);});
  if(job&&job.status!=='complete'&&s.rows.length===job.saved_turns_before){list.append(message('user',job.prompt));list.append(message('assistant',job.response||'…',job.status==='running'?'In progress · not saved yet':'Unfinished attempt · not part of the recorded conversation',job.status!=='running'));}
  if(nearBottom||currentJob!==job?.id)list.scrollTop=list.scrollHeight;
  currentJob=job?.id;
}
function strip(snapshot, layer){
  const card=node('div','snapshot');const head=node('div','head');head.append(node('span','',`Layer ${layer}`),node('span','check',snapshot?'✓':'—'));card.append(head);
  const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 128 38');svg.setAttribute('role','img');svg.setAttribute('aria-label',snapshot?`Layer ${layer}: summary of ${snapshot.dimensions} recorded values`:`Layer ${layer}: waiting for a snapshot`);
  const line=document.createElementNS(ns,'path');line.setAttribute('d','M0 19 H128');line.setAttribute('stroke','#e2e6dc');svg.append(line);
  if(snapshot){const max=Math.max(...snapshot.preview.map(Math.abs),1e-12);snapshot.preview.forEach((v,i)=>{const h=Math.max(Math.abs(v)/max*16,.5),rect=document.createElementNS(ns,'rect');rect.setAttribute('x',i*4);rect.setAttribute('y',v>=0?19-h:19);rect.setAttribute('width',2.5);rect.setAttribute('height',h);rect.setAttribute('fill',v>=0?'#648875':'#be937a');svg.append(rect);});}
  card.append(svg,node('small','',snapshot?`${snapshot.dimensions.toLocaleString()} values captured`:'Waiting for a message'));return card;
}
function applyMode(s){if(s.mode!=='remote'||document.body.dataset.mode==='remote')return;document.body.dataset.mode='remote';document.title='Gemma 12B · Recording room';stageNames.loading='Connecting to the cloud GPU';$('where').textContent='Gemma runs on a cloud GPU';$('where-note').textContent='Gemma 3 12B runs on RunPod. Your conversations and recordings are saved on this computer.';$('guide-where').textContent='(Gemma 3 12B) running on a cloud GPU. The recordings are saved on this computer';$('snapshot-size').textContent='3,840';const long=$('reply-limit').querySelector('option[value="2048"]');long.disabled=true;long.textContent='Extra long · not available on the cloud GPU';}
function render(s){state=s;applyMode(s);const job=s.job,busy=s.busy,ownBusy=job?.status==='running';
  if(instructionSession!==session){instructionSession=session;instructionDirty=false;}if(!instructionDirty)$('instruction').value=s.instruction||'';$('instruction').disabled=busy||!('instruction' in s);$('save-instruction').disabled=busy||!('instruction' in s);$('brief-instruction').disabled=busy||!('instruction' in s);$('instruction-status').textContent=!('instruction' in s)?'· stop and reopen app to enable':instructionDirty?'· unsaved changes':s.instruction?'· active':'· none';
  $('new-compact').disabled=busy;$('session-compact').disabled=busy;
  const listKey=JSON.stringify(s.sessions.map(x=>[x.id,x.title]));if(listKey!==sessionListKey){sessionListKey=listKey;$('session-compact').replaceChildren(...s.sessions.map(item=>{const o=node('option','',item.title);o.value=item.id;return o;}));}$('session-compact').value=session;
  $('send').disabled=busy||sending;$('new-chat').disabled=busy;$('stop').hidden=!ownBusy;$('prompt').disabled=busy;
  $('model-status').textContent=s.mode==='remote'?(s.model_ready?'Gemma 3 12B · cloud GPU connected':'Gemma 3 12B · connects on first message'):(s.model_ready?'Gemma 3 · ready':'Gemma 3 · loads on first message');
  $('saved-count').textContent=`${s.rows.length} ${s.rows.length===1?'turn':'turns'} recorded · ${s.rows.length*4} snapshots`;
  $('auto-tag').hidden=!s.auto_label_enabled;$('auto-tag').disabled=busy||!s.rows.some(r=>!r.label_source||r.label_status==='error');
  $('export').disabled=!s.rows.length||downloading;$('reply-limit').disabled=busy;$('shutdown').disabled=s.closing;$('shutdown-compact').disabled=s.closing;$('recording-path').textContent=s.recording_path;
  $('settings').textContent=`${s.mode==='remote'?'Gemma 3 12B IT on the cloud GPU':'Gemma 3 4B IT'} · layers ${s.config.layers.join(', ')} (counting from zero). Choose a reply cap below the composer; the exact cap is recorded per turn; conversation input at ${s.config.max_input_tokens}. Full numbers are saved, not just these previews.`;
  $('sessions').replaceChildren(...s.sessions.map(item=>{const b=node('button','session'+(item.id===session?' active':''),item.title);b.title=item.title;b.disabled=busy;b.onclick=()=>{session=item.id;localStorage.setItem('gemma-session',session);rendering='';$('messages').querySelectorAll('article').forEach(x=>x.remove());$('welcome').hidden=false;alertMessage('');refresh();};return b;}));
  renderMessages(s);
  const stage=job?.stage; $('activity').textContent=stage==='labelling'?autoTagStage:stageNames[stage]||(s.rows.length?'Saved conversation':'Ready for your first message');
  if(busy&&!ownBusy)$('activity').textContent='Gemma is replying in another conversation';
  if(s.label_task?.status==='running'){$('activity').textContent=`Auto-tagging saved replies · ${s.label_task.done}/${s.label_task.total}`;$('stop').hidden=false;}
  if(s.label_task?.error)alertMessage(s.label_task.error);
  $('activity-dot').className='dot'+(ownBusy?' working':job?.status==='error'?' error':'');
  $('elapsed').textContent=ownBusy?`${Math.max(0,Math.floor(Date.now()/1000-job.started_seconds))}s`:'';
  let phaseIndex=stages.indexOf(stage);if(stage==='saving')phaseIndex=2;
  stages.forEach((p,i)=>$('phase-'+p).classList.toggle('done',phaseIndex>=i||(job?.status==='complete')));
  const snaps=job?job.snapshots:(s.rows.at(-1)?.snapshot_preview||[]);
  $('snapshots').replaceChildren(...s.config.layers.map(layer=>strip(snaps.find(x=>x.layer===layer),layer)));
  const events=(job?.events||[]).filter(x=>x.message).slice(-40),log=$('events');
  log.replaceChildren(...events.map(e=>{const li=node('li');li.append(node('time','',new Date(e.time).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'})),node('span','',e.message));return li;}));
  if(!events.length)log.append(node('li','empty-log',s.rows.length?'Saved snapshots are shown above. A new message starts the live log.':'Send a message to see the steps here.'));
  log.scrollTop=log.scrollHeight;
  $('tokens').textContent=job?.generated_tokens?`${job.generated_tokens} text pieces written`:'';
  renderInside(s);if(job?.error)alertMessage(job.error);
}
async function refresh(){const requestedSession=session;try{const s=await api('/api/state'+(session?'?session='+encodeURIComponent(session):''));if(requestedSession!==session)return;if(!session){if(s.sessions.length){session=s.sessions[0].id;localStorage.setItem('gemma-session',session);return refresh();}return newChat();}render(s);}catch(e){if(e.message==='Conversation not found'){session=null;localStorage.removeItem('gemma-session');return refresh();}alertMessage('The local app is not responding. Your completed recordings stay saved. Reopen the Gemma launcher to reconnect.');}}
$('composer').onsubmit=async event=>{event.preventDefault();const text=$('prompt').value.trim();if(!text||sending||state?.busy)return;if(instructionDirty){alertMessage('Save your conversation instructions before sending.');return;}sending=true;$('send').disabled=true;alertMessage('');try{await api('/api/send',{session,prompt:text,reply_limit:Number($('reply-limit').value)});$('prompt').value='';await refresh();}catch(e){alertMessage(e.message);}finally{sending=false;}};
$('prompt').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();$('composer').requestSubmit();}};
$('new-chat').onclick=newChat;$('new-compact').onclick=newChat;
$('session-compact').onchange=()=>{session=$('session-compact').value;localStorage.setItem('gemma-session',session);rendering='';alertMessage('');refresh();};
$('guide-compact').onclick=()=>$('guide').showModal();
$('stop').onclick=async()=>{try{await api('/api/stop',{});alertMessage('Stopping at the next safe point. Earlier completed turns remain saved.');}catch(e){alertMessage(e.message);}};
$('auto-tag').onclick=async()=>{try{await api('/api/auto-label',{session});alertMessage('Gemma is tagging saved replies. Your corrections will be kept.');await refresh();}catch(e){alertMessage(e.message);}};
$('export').onclick=async()=>{if(downloading)return;downloading=true;$('export').disabled=true;try{const r=await fetch('/api/export?session='+encodeURIComponent(session));if(!r.ok)throw new Error('The export could not be created.');const blob=await r.blob(),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`gemma-conversation-${session.slice(0,8)}.zip`;a.click();setTimeout(()=>URL.revokeObjectURL(url),10000);alertMessage('Recording exported once. Your conversation is also already saved in the project folder.');}catch(e){alertMessage(e.message);}finally{downloading=false;$('export').disabled=false;}};
$('explain-open').onclick=()=>$('guide').showModal();$('explain-close').onclick=()=>$('guide').close();
document.querySelectorAll('[data-prompt]').forEach(b=>b.onclick=()=>{$('prompt').value=b.dataset.prompt;$('prompt').focus();});
async function loop(){if(appClosed)return;await refresh();if(!appClosed)setTimeout(loop,650);}loop();

$('inside-open').onclick=()=>$('inside').showModal();$('inside-close').onclick=()=>$('inside').close();
$('candidate-step').oninput=()=>{$('follow-candidates').checked=false;if(state)renderInside(state);};
function matrix(target, rows, labels){const root=$(target);root.replaceChildren();if(!rows?.length){root.append(node('p','','Waiting for a measured pass.'));return;}const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg'),height=rows.length*13+10;svg.setAttribute('viewBox',`0 0 380 ${height}`);svg.setAttribute('role','img');svg.setAttribute('aria-label',`${rows.length} rows of measured vector group averages`);const max=Math.max(...rows.flat().map(Math.abs),1e-12);rows.forEach((row,i)=>{const text=document.createElementNS(ns,'text');text.setAttribute('x',0);text.setAttribute('y',i*13+10);text.textContent=labels[i];svg.append(text);row.forEach((v,j)=>{const rect=document.createElementNS(ns,'rect');rect.setAttribute('x',60+j*20);rect.setAttribute('y',i*13);rect.setAttribute('width',18);rect.setAttribute('height',11);rect.setAttribute('rx',1);rect.setAttribute('fill',v>=0?'#4b765b':'#b78468');rect.setAttribute('fill-opacity',.08+.92*Math.abs(v)/max);const title=document.createElementNS(ns,'title');title.textContent=`${labels[i]}, coordinate group ${j}: ${v.toPrecision(4)}`;rect.append(title);svg.append(rect);});});root.append(svg);}
function renderInside(s){const trace=s.job?.education||s.rows.at(-1)?.education_trace||{},tokens=trace.tokens?.shown||[],emb=trace.embeddings,layers=trace.layers||[],candidates=trace.candidates||[];$('trace-status').textContent=layers.length?`${trace.tokens?.total||'?'} prompt tokens · ${layers.length} layers observed · ${candidates.length} next-token decisions recorded`:'Send a new message to capture these steps. Earlier recordings may not contain this additional view.';$('token-pieces').replaceChildren(...tokens.map(t=>{const span=node('span','token-piece',t.piece);span.title=`Position ${t.position}; token ID ${t.id}`;span.append(node('small','',`#${t.id}`));return span;}));matrix('embedding-matrix',emb?.values,emb?.values.map((_,i)=>`Token ${emb.first_position+i}`)||[]);matrix('layer-matrix',layers.map(x=>x.preview),layers.map(x=>`Layer ${x.layer}`));const slider=$('candidate-step');slider.max=Math.max(0,candidates.length-1);if($('follow-candidates').checked)slider.value=slider.max;const index=Math.min(Number(slider.value),Math.max(0,candidates.length-1)),current=candidates[index];$('candidate-label').textContent=current?`Decision ${index+1} of ${candidates.length}`:'Waiting for a reply';$('candidate-bars').replaceChildren(...(current?.top||[]).map((t,i)=>{const row=node('div','candidate'),track=node('div','candidate-track'),bar=node('div','candidate-fill');bar.style.width=`${Math.max(0,Math.min(100,t.probability*100))}%`;track.append(bar);row.append(node('code','',t.piece),track,node('span','candidate-percent',(100*t.probability).toFixed(2)+'%'));row.title=`Token ${t.id}${i===0?' · highest displayed probability':''}`;return row;}));}
async function shutdown(){try{await api('/api/shutdown',{});appClosed=true;$('send').disabled=true;$('prompt').disabled=true;$('shutdown').disabled=true;$('shutdown-compact').disabled=true;$('activity').textContent='Stopping Gemma…';alertMessage('Stopping at the next layer or token boundary, then closing the app. Earlier completed recordings remain saved.');const check=async()=>{try{await api('/api/state');setTimeout(check,1000);}catch(e){$('activity').textContent='App stopped';alertMessage('App stopped. Your completed recordings remain saved. Use Open Gemma Recording Room to start again.');}};setTimeout(check,1000);}catch(e){alertMessage(e.message);}}
$('shutdown').onclick=shutdown;$('shutdown-compact').onclick=shutdown;

$('instruction').oninput=()=>{instructionDirty=true;$('instruction-status').textContent='· unsaved changes';};
async function saveInstruction(){try{await api('/api/instruction',{session,instruction:$('instruction').value});instructionDirty=false;await refresh();alertMessage('Instructions saved. They will be included automatically from the next reply.');}catch(e){alertMessage(e.message);}}
$('save-instruction').onclick=saveInstruction;
$('brief-instruction').onclick=()=>{$('instruction').value='Keep your reply under 50 words, in two or three sentences.';instructionDirty=true;saveInstruction();};
