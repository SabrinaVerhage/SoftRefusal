'use strict';
const $ = id => document.getElementById(id);
let applicationToken = '', connected = false, submitting = false, imageURL = null, selectedJob = null;
let limits = {per_credential: 3, chat_input_tokens: 8192, chat_messages: 64, chat_characters: 24000};
const jobs = new Map(), polling = new Set();
const newThread = () => ({id: crypto.randomUUID(), messages: [], lastJobId: null, lastSuccessId: null, usage: null, lastFinishedAt: 0});
const threads = {gemma: newThread(), hermes: newThread()};
const isActive = job => job.status === 'queued' || job.status === 'running';
const activeCount = () => [...jobs.values()].filter(isActive).length;

function enableControls() {
  document.querySelectorAll('.submit').forEach(button => {
    button.disabled = !connected || submitting || activeCount() >= limits.per_credential;
  });
  $('model').disabled = submitting; $('new-chat').disabled = submitting;
}
function friendlyIssue(code, data = {}) {
  if (code === 'input_too_long') return data.limit_tokens == null ?
    'This saved request exceeded the input limit used at the time. It is a length error, not a refusal.' :
    'Input has ' + data.input_tokens + ' tokens; the limit is ' + data.limit_tokens +
    '. Shorten the prompt, or start a new chat if it includes long history. This is a length limit, not a refusal.';
  if (code === 'conversation_limit_exceeded') return 'This conversation reached the message or text limit. Start a new conversation; previous jobs remain saved.';
  if (code === 'chat_dependency_failed') return 'An earlier queued message failed, so this follow-up could not run. Retry from the last successful reply or start a new conversation.';
  if (code === 'per_credential_queue_full') return 'Your ' + limits.per_credential + ' active queue slots are full. Wait for one job to finish.';
  if (code === 'queue_full') return 'The shared GPU queue is full. Try again after a job finishes.';
  if (code === 'submission_rate_limit') return 'Six requests were submitted within a minute. Wait briefly before adding another.';
  if (code === 'submissions_paused') return 'Submissions are temporarily paused while the server is updated.';
  return code;
}
async function request(path, options = {}) {
  const response = await fetch(path, {...options, headers: {
    Authorization: 'Bearer ' + applicationToken,
    ...(options.body ? {'Content-Type': 'application/json'} : {}), ...options.headers
  }, cache: 'no-store'});
  if (!response.ok) {
    const data = await response.json().catch(() => ({})), error = data.detail || data.error || {};
    throw new Error(friendlyIssue(error.code || 'Request failed (' + response.status + ')', error));
  }
  return response;
}
function message(role, content, label) {
  const element = document.createElement('div'); element.className = 'message ' + role;
  const heading = document.createElement('strong'); heading.textContent = label;
  element.append(heading, document.createTextNode(content)); return element;
}
function renderConversation() {
  const model = $('model').value, thread = threads[model]; $('conversation').replaceChildren();
  for (const entry of thread.messages) $('conversation').append(message(entry.role, entry.content,
    entry.role === 'user' ? 'YOU' : model.toUpperCase()));
  for (const job of [...jobs.values()].filter(j => j.thread === thread && isActive(j)).sort((a,b) => a.created_at-b.created_at)) {
    if (job.prompt) $('conversation').append(message('user', job.prompt, 'YOU · ' + job.status.toUpperCase()));
    if (job.progress?.text) $('conversation').append(message('assistant', job.progress.text, model.toUpperCase() + ' · LIVE'));
  }
  $('conversation').scrollTop = $('conversation').scrollHeight;
  $('context-hint').textContent = 'Input limit: ' + limits.chat_input_tokens.toLocaleString() + ' tokens, including conversation history.' +
    (thread.usage ? ' Last completed request: ' + thread.usage.prompt_tokens.toLocaleString() + ' input tokens.' : '') +
    ' Up to ' + limits.chat_messages + ' messages. Nothing is automatically removed from history.';
}
function phaseText(job) {
  if (job.status === 'queued') return 'Queued · position ' + (job.queue_position ?? 'pending');
  if (job.status === 'running') {
    const phase = job.progress?.phase, elapsed = Math.max(0, Date.now()/1000 - (job.started_at || job.created_at));
    return (phase === 'loading' ? 'Loading ' + job.model + ' into GPU memory' : phase === 'generating' ?
      'Generating with ' + job.model + (job.progress.output_tokens ? ' · ' + job.progress.output_tokens + ' tokens' : '') :
      'Preparing ' + job.model) + ' · ' + elapsed.toFixed(0) + ' s';
  }
  return job.result?.error ? friendlyIssue(job.result.error.code, job.result.error) : job.status === 'interrupted' ? 'Interrupted' :
    job.result?.outcome === 'pipeline_blocked' ? 'Blocked by image pipeline' :
    job.result?.finish_reason === 'length' ? 'Completed · response length limit reached' : 'Completed';
}
function renderJobs() {
  $('queue-summary').textContent = activeCount() + ' of ' + limits.per_credential + ' active slots used. One GPU executes jobs in order; you can queue text and images while it works.';
  $('jobs').replaceChildren();
  for (const job of [...jobs.values()].sort((a,b)=>b.created_at-a.created_at).slice(0,50)) {
    const row = document.createElement('li'), button = document.createElement('button'); button.type = 'button';
    button.textContent = job.model.toUpperCase() + ' · ' + job.job_id.slice(0,8) + ' · ' + phaseText(job);
    if (job.job_id === selectedJob) button.className = 'selected';
    button.addEventListener('click', () => {selectedJob = job.job_id; renderJobs(); showJob(job);});
    row.append(button); $('jobs').append(row);
  }
  enableControls();
}
async function displayImage(job) {
  const id = job.job_id; if (!job.result?.image_available || job.status !== 'succeeded') return;
  try {
    const response = await request(job.result_url);
    if (!response.headers.get('content-type')?.startsWith('image/png')) throw new Error('Expected a PNG result');
    const blob = await response.blob(); if (selectedJob !== id) return;
    if (imageURL) URL.revokeObjectURL(imageURL);
    imageURL = URL.createObjectURL(blob); $('image').src = imageURL; $('image').hidden = false;
    $('image-caption').textContent = 'Image job ' + id.slice(0,8);
  } catch (error) {$('image-caption').textContent = 'Image retrieval issue · ' + error.message;}
}
function showJob(job) {
  $('job-status').textContent = phaseText(job);
  $('live-output').hidden = !job.progress?.text || !isActive(job); $('live-output').textContent = job.progress?.text || '';
  const result = job.result || {}, metrics = result.metrics || {}, memory = metrics.inference_memory;
  $('raw').textContent = JSON.stringify(job, (key,value)=>key === 'thread' ? undefined : value, 2);
  const parts = [];
  for (const [key,label] of [['queue_wait_seconds','Queue'],['model_load_seconds','Load'],['inference_seconds','Inference']])
    if (Number.isFinite(metrics[key])) parts.push(label + ' ' + metrics[key].toFixed(1) + ' s');
  if (memory?.peak_allocated_bytes != null) parts.push('Allocator peak ' + (memory.peak_allocated_bytes/2**30).toFixed(2) + ' GiB');
  if (memory?.sampled_device_peak_bytes != null) parts.push('Device peak ' + (memory.sampled_device_peak_bytes/2**30).toFixed(2) + ' GiB');
  $('measurement').textContent = parts.join(' · ') || 'Measurements appear when the job finishes.';
  $('outcome').textContent = result.error ? 'Execution or input error; no model refusal is inferred.' :
    result.outcome === 'pipeline_blocked' ? 'Pipeline filter signal recorded separately from model behaviour.' :
    'Outcome: unlabelled. Raw generated output is preserved for review.';
  if (job.kind === 'image') {
    if (job.status === 'succeeded' && result.image_available) displayImage(job);
    else {$('image').hidden = true; $('image-caption').textContent = phaseText(job);}
  }
}
function acceptCompleted(job) {
  if (job.kind !== 'chat' || !job.thread) return; const thread = job.thread;
  if (job.status === 'succeeded' && typeof job.result?.text === 'string' && job.finished_at >= thread.lastFinishedAt) {
    const messages = job.result.generation_settings?.messages;
    if (messages) thread.messages = [...messages, {role:'assistant',content:job.result.text}];
    thread.lastSuccessId = job.job_id; thread.usage = job.result.usage; thread.lastFinishedAt = job.finished_at;
  } else if (thread.lastJobId === job.job_id) thread.lastJobId = thread.lastSuccessId;
  renderConversation();
}
async function observeJob(id) {
  if (polling.has(id)) return; polling.add(id); const deadline = Date.now()+3600*1000;
  try {
    while (connected && isActive(jobs.get(id))) {
      await new Promise(resolve=>setTimeout(resolve,750));
      if (Date.now()>deadline) throw new Error('Polling expired; job ' + id + ' remains saved on the server.');
      const previous = jobs.get(id), incoming = await (await request(previous.status_url)).json();
      const job = {...previous,...incoming}; jobs.set(id,job);
      if (!isActive(job)) acceptCompleted(job);
      renderJobs(); renderConversation(); if (selectedJob===id) showJob(job);
    }
  } catch (error) {$('connection').textContent = 'Connection issue · ' + error.message + '. Reconnect to recover saved jobs.';}
  finally {polling.delete(id);}
}
async function submit(kind,payload,thread=null,prompt='') {
  if (!connected || submitting) return null; submitting=true; enableControls();
  try {
    const job=await (await request('/v1/'+kind+'/jobs',{method:'POST',body:JSON.stringify(payload),headers:{'Idempotency-Key':crypto.randomUUID()}})).json();
    job.thread=thread; job.prompt=prompt; jobs.set(job.job_id,job); selectedJob=job.job_id;
    if (thread) thread.lastJobId=job.job_id;
    renderJobs(); renderConversation(); showJob(job); observeJob(job.job_id); return job;
  } catch(error) {$('job-status').textContent=error.message; $('outcome').textContent='No model outcome inferred from this submission issue.';return null;}
  finally {submitting=false;enableControls();}
}
$('access-form').addEventListener('submit',async event=>{
  event.preventDefault();applicationToken=$('access').value.trim();$('access').value='';
  try {
    const health=await (await request('/v1/health')).json(); connected=health.status==='accepting'; limits={...limits,...health.limits};
    $('connection').textContent=connected?'Connected · shared GPU queue · BF16':'Connected · submissions paused';
    const hermes=$('model').querySelector('option[value=hermes]');hermes.disabled=!health.models.hermes.enabled;
    hermes.textContent=health.models.hermes.enabled?'Hermes 4 14B · BF16':'Hermes 4 14B · comparison pending';
    const recent=await (await request('/v1/jobs?limit=30')).json();jobs.clear();
    for (const job of recent.jobs) jobs.set(job.job_id,job);
    for (const model of ['gemma','hermes']) {
      const latest=recent.jobs.find(j=>j.model===model&&j.kind==='chat'&&j.status==='succeeded');
      if (latest) {const thread=threads[model];thread.lastJobId=thread.lastSuccessId=latest.job_id;
        thread.messages=[...(latest.result.generation_settings?.messages||[]),{role:'assistant',content:latest.result.text}];thread.usage=latest.result.usage;thread.lastFinishedAt=latest.finished_at;}
      const pending=recent.jobs.filter(j=>j.model===model&&j.kind==='chat'&&isActive(j));
      if (pending.length) threads[model].lastJobId=pending[0].job_id;
      for (const job of pending) job.thread=threads[model];
    }
    $('access-form').hidden=true;$('disconnect').hidden=false;selectedJob=recent.jobs[0]?.job_id||null;
    renderJobs();renderConversation();if(selectedJob)showJob(jobs.get(selectedJob));
    for (const job of recent.jobs.filter(isActive)) observeJob(job.job_id);
  } catch(error){applicationToken='';connected=false;$('connection').textContent=error.message;}
  enableControls();
});
$('disconnect').addEventListener('click',()=>{applicationToken='';connected=false;$('access-form').hidden=false;$('disconnect').hidden=true;$('connection').textContent='Disconnected';enableControls();});
$('model').addEventListener('change',renderConversation);
$('new-chat').addEventListener('click',()=>{threads[$('model').value]=newThread();renderConversation();});
$('chat-form').addEventListener('submit',async event=>{
  event.preventDefault();const model=$('model').value,content=$('chat-prompt').value.trim(),thread=threads[model];if(!content)return;
  const payload={model,messages:[{role:'user',content}],max_tokens:256,temperature:0,seed:0};
  if(thread.lastJobId)payload.after_job_id=thread.lastJobId;
  const job=await submit('chat',payload,thread,content);if(job&&$('chat-prompt').value.trim()===content)$('chat-prompt').value='';
});
$('image-form').addEventListener('submit',async event=>{
  event.preventDefault();const size=Number($('size').value),prompt=$('image-prompt').value.trim();if(!prompt)return;
  await submit('images',{model:'flux',prompt,width:size,height:size,steps:4,seed:0},null,prompt);
});
setInterval(()=>{if(connected){renderJobs();if(selectedJob&&isActive(jobs.get(selectedJob)))showJob(jobs.get(selectedJob));}},1000);
