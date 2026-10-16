(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  let sid = sessionStorage.getItem('softrefusal.conversation'), busy = false, online = false, posting = false, signature = '';
  let pending = null, fresh = false;
  // Thinking panels the visitor opened or closed themselves, keyed by turn.
  const toggled = new Map();
  try { pending = JSON.parse(sessionStorage.getItem('softrefusal.pending') || 'null'); } catch {}
  function error(text) { $('error').textContent = text || ''; $('error').hidden = !text; }
  async function api(path, body) {
    const response = await fetch(path, {cache:'no-store', ...(body ? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)} : {})});
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'The request could not be completed.');
    return result;
  }
  function node(tag, className, text) { const el = document.createElement(tag); if (className) el.className = className; if (text !== undefined) el.textContent = text; return el; }
  // Thinking models write <think>…</think> before the answer. The stored reply stays raw.
  function splitThinking(text) {
    text = text || '';
    const start = text.indexOf('<think>'), end = text.indexOf('</think>');
    if (start < 0 && end < 0) return {thinking: null, answer: text, pending: false};
    const from = start < 0 ? 0 : start + 7;
    if (end < 0) return {thinking: text.slice(from).trim(), answer: '', pending: true};
    return {thinking: text.slice(from, end).trim(), answer: ((start < 0 ? '' : text.slice(0, start)) + text.slice(end + 8)).trim(), pending: false};
  }
  function controls() {
    $('send').disabled = busy || !online || posting || !$('prompt').value.trim();
    $('new').disabled = busy || posting;
    $('stop').hidden = !busy;
    $('start').disabled = !$('consent').checked || posting;
  }
  function userBubble(text) {
    const box = node('div', 'bubble user'); box.append(node('span', 'speaker', 'YOU'), node('div', 'text', text)); return box;
  }
  function aiBubble(key, text, {label, source, capped, live, latest}) {
    const box = node('div', 'bubble assistant'), parts = splitThinking(text);
    box.append(node('span', 'speaker', 'AI'));
    if (parts.thinking !== null) {
      const details = node('details', 'thinking' + (parts.pending ? ' pending' : ''));
      details.open = toggled.has(key) ? toggled.get(key) : latest;
      details.ontoggle = () => toggled.set(key, details.open);
      const words = parts.thinking ? parts.thinking.split(/\s+/).length : 0;
      details.append(node('summary', '', parts.pending ? 'Thinking' : `Thinking · ${words} words`), node('div', 'thinking-text', parts.thinking || '…'));
      box.append(details);
    }
    const answer = parts.answer || (parts.pending ? '' : live ? '…' : '(No reply text.)');
    if (answer) box.append(node('div', 'text', answer));
    if (label && label !== 'unlabeled') {
      const tag = node('span', 'outcome'); tag.append(node('i', `dot ${label}`), document.createTextNode(label[0].toUpperCase() + label.slice(1)));
      tag.append(node('small', '', source === 'human' ? 'labelled by hand' : 'auto-tagged'));
      box.append(tag);
    }
    if (capped) box.append(node('span', 'note', 'Reply reached its length limit'));
    return box;
  }
  function render(state) {
    online = state.model.online; $('connection').textContent = online ? 'AI online' : state.model.message; $('connection').classList.toggle('online', online);
    if (state.conversation && !fresh) { sid = state.conversation; sessionStorage.setItem('softrefusal.conversation', sid); }
    $('welcome').hidden = !!sid; $('conversation').hidden = !sid;
    const job = state.job; busy = ['queued','running'].includes(job?.status);
    const thinking = job?.response && splitThinking(job.response).pending;
    $('status').textContent = job?.status === 'queued' ? `Waiting for the AI · you are number ${job.position} in line` : job?.status === 'running' ? (job.stage === 'labelling' ? 'Reply saved · marking how the AI responded…' : thinking ? 'The AI is thinking…' : job.response ? 'The AI is replying…' : 'The AI is reading your message…') : '';
    $('status').classList.toggle('working', busy);
    if (job?.error) error(job.error);
    const turns = state.rows.length;
    $('turns').textContent = turns ? `YOUR CONVERSATION · ${turns} ${turns === 1 ? 'TURN' : 'TURNS'}` : 'YOUR CONVERSATION';
    $('prompt').placeholder = turns ? 'Push a little further…' : 'Try to get the AI to talk spicy…';
    const next = JSON.stringify([state.rows, job]);
    if (next !== signature) {
      const nearBottom = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 140;
      signature = next; const nodes = [];
      // Saved turns are visible while the separate automatic tag is still running.
      const alreadySaved = job && state.rows.some(row => row.prompt === job.prompt && row.response === job.response) && job.stage === 'labelling';
      const live = busy && !alreadySaved;
      state.rows.forEach((row, i) => {
        nodes.push(userBubble(row.prompt));
        nodes.push(aiBubble(row.point_id || i, row.response, {label: row.label, source: row.label_source, capped: row.hit_token_limit, latest: !live && i === turns - 1}));
      });
      if (live) { nodes.push(userBubble(job.prompt)); if (job.response) nodes.push(aiBubble('live', job.response, {live: true, latest: true})); }
      $('messages').replaceChildren(...nodes);
      $('starters').hidden = nodes.length > 0;
      if (nearBottom && nodes.length) $('composer').scrollIntoView({block:'end',behavior:'instant'});
    }
    controls();
  }
  async function refresh() {
    try { render(await api('/api/state' + (sid ? '?conversation='+encodeURIComponent(sid) : ''))); }
    catch (e) { if (e.message === 'Conversation not found.') { sid=null; sessionStorage.removeItem('softrefusal.conversation'); } else { $('connection').textContent='Connection interrupted · retrying'; online=false; controls(); } }
  }
  function updateCount() { const n = $('prompt').value.length; $('count').textContent = n > 3000 ? `${n.toLocaleString()} / 4,000` : ''; controls(); }
  $('consent').onchange=controls;
  $('prompt').oninput=updateCount;
  document.querySelectorAll('.chip').forEach(chip => { chip.onclick = () => { $('prompt').value = chip.textContent; updateCount(); $('prompt').focus(); }; });
  $('start').onclick=async()=>{ posting=true;controls();error('');try { const result=await api('/api/start',{consent:true});fresh=false;sid=result.conversation;toggled.clear();await refresh();$('prompt').focus(); }catch(e){error(e.message);}finally{posting=false;controls();} };
  $('new').onclick=()=>{ fresh=true;sid=null;sessionStorage.removeItem('softrefusal.conversation');$('welcome').hidden=false;$('conversation').hidden=true;$('consent').checked=false;signature='';controls();window.scrollTo(0,0); };
  $('composer').onsubmit=async event=>{
    event.preventDefault();const prompt=$('prompt').value.trim();if(!prompt||busy||posting||!online)return;
    posting=true;controls();error('');
    if(!pending||pending.prompt!==prompt||pending.conversation!==sid)pending={conversation:sid,prompt,request_key:crypto.randomUUID()};
    sessionStorage.setItem('softrefusal.pending',JSON.stringify(pending));
    try{await api('/api/send',pending);pending=null;sessionStorage.removeItem('softrefusal.pending');$('prompt').value='';updateCount();await refresh();}
    catch(e){error(e.message);}finally{posting=false;controls();}
  };
  $('stop').onclick=async()=>{error('');try{await api('/api/stop',{conversation:sid});await refresh();}catch(e){error(e.message);}};
  async function poll(){await refresh();setTimeout(poll,document.hidden?5000:1200);}poll();
})();
