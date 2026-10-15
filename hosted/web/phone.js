(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  let sid = sessionStorage.getItem('softrefusal.conversation'), busy = false, online = false, posting = false, signature = '';
  let pending = null, fresh = false;
  try { pending = JSON.parse(sessionStorage.getItem('softrefusal.pending') || 'null'); } catch {}
  function error(text) { $('error').textContent = text || ''; $('error').hidden = !text; }
  async function api(path, body) {
    const response = await fetch(path, {cache:'no-store', ...(body ? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)} : {})});
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'The request could not be completed.');
    return result;
  }
  function controls() {
    $('send').disabled = busy || !online || posting || !$('prompt').value.trim();
    $('new').disabled = busy || posting;
    $('stop').hidden = !busy;
    $('start').disabled = !$('consent').checked || posting;
  }
  function bubble(role, text, label, capped) {
    const node = document.createElement('div'); node.className = `bubble ${role}`;
    const who = document.createElement('span'); who.className = 'speaker'; who.textContent = role === 'user' ? 'YOU' : 'GEMMA';
    node.append(who, document.createTextNode(text));
    if (label || capped) { const tag = document.createElement('span'); tag.className = 'tag'; tag.textContent = [label, capped ? 'Reply reached its length limit' : ''].filter(Boolean).join(' · '); node.append(tag); }
    return node;
  }
  function render(state) {
    online = state.model.online; $('connection').textContent = state.model.message; $('connection').classList.toggle('online', online);
    if (state.conversation && !fresh) { sid = state.conversation; sessionStorage.setItem('softrefusal.conversation', sid); }
    $('welcome').hidden = !!sid; $('conversation').hidden = !sid;
    const job = state.job; busy = ['queued','running'].includes(job?.status);
    let status = job?.status === 'queued' ? `Waiting for the model · queue position ${job.position}` : job?.status === 'running' ? (job.stage === 'labelling' ? 'Reply saved · tagging response behaviour…' : job.response ? 'Gemma is replying…' : 'Preparing the model…') : state.rows.length ? state.map.message : '';
    $('status').textContent = status;
    if (job?.error) error(job.error);
    const next = JSON.stringify([state.rows, job]);
    if (next !== signature) {
      const nearBottom = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 140;
      signature = next; const nodes = [];
      for (const row of state.rows) {
        nodes.push(bubble('user',row.prompt));
        nodes.push(bubble('assistant',row.response, row.label && row.label !== 'unlabeled' ? `${row.label} · ${row.label_source === 'human' ? 'human label' : 'automatic label'}` : '',row.hit_token_limit));
      }
      // Saved turns are visible while the separate automatic tag is still running.
      const alreadySaved = job && state.rows.some(row => row.prompt === job.prompt && row.response === job.response) && job.stage === 'labelling';
      if (busy && !alreadySaved) { nodes.push(bubble('user',job.prompt)); if (job.response) nodes.push(bubble('assistant',job.response)); }
      if (!nodes.length) { const intro = document.createElement('p'); intro.className='fine'; intro.textContent='Your conversation starts here. Each completed turn becomes a point on the map.'; nodes.push(intro); }
      $('messages').replaceChildren(...nodes);
      if (nearBottom && nodes.length) $('composer').scrollIntoView({block:'end',behavior:'instant'});
    }
    $('path-link').href = '/map/?conversation=' + encodeURIComponent(sid || '');
    controls();
  }
  async function refresh() {
    try { render(await api('/api/state' + (sid ? '?conversation='+encodeURIComponent(sid) : ''))); }
    catch (e) { if (e.message === 'Conversation not found.') { sid=null; sessionStorage.removeItem('softrefusal.conversation'); } else { $('connection').textContent='Connection interrupted · retrying'; online=false; controls(); } }
  }
  $('consent').onchange=controls;
  $('prompt').oninput=()=>{ $('count').textContent=`${$('prompt').value.length.toLocaleString()} / 4,000`; controls(); };
  $('start').onclick=async()=>{ posting=true;controls();error('');try { const result=await api('/api/start',{consent:true});fresh=false;sid=result.conversation;await refresh();$('prompt').focus(); }catch(e){error(e.message);}finally{posting=false;controls();} };
  $('new').onclick=()=>{ fresh=true;sid=null;sessionStorage.removeItem('softrefusal.conversation');$('welcome').hidden=false;$('conversation').hidden=true;$('consent').checked=false;signature='';controls();window.scrollTo(0,0); };
  $('composer').onsubmit=async event=>{
    event.preventDefault();const prompt=$('prompt').value.trim();if(!prompt||busy||posting||!online)return;
    posting=true;controls();error('');
    if(!pending||pending.prompt!==prompt||pending.conversation!==sid)pending={conversation:sid,prompt,request_key:crypto.randomUUID()};
    sessionStorage.setItem('softrefusal.pending',JSON.stringify(pending));
    try{await api('/api/send',pending);pending=null;sessionStorage.removeItem('softrefusal.pending');$('prompt').value='';$('count').textContent='0 / 4,000';await refresh();}
    catch(e){error(e.message);}finally{posting=false;controls();}
  };
  $('stop').onclick=async()=>{error('');try{await api('/api/stop',{conversation:sid});await refresh();}catch(e){error(e.message);}};
  async function poll(){await refresh();setTimeout(poll,document.hidden?5000:1200);}poll();
})();
