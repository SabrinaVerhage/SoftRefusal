(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  let state = {online:false,busy:false,signed_in:false,jobs:[]}, posting = false, pending = null, signature = '';
  try { pending = JSON.parse(sessionStorage.getItem('softrefusal.image-pending') || 'null'); } catch {}
  function error(message) { $('error').textContent=message || ''; $('error').hidden=!message; }
  async function api(path, body) {
    const response=await fetch(path,{cache:'no-store',...(body ? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)} : {})});
    const result=await response.json(); if(!response.ok) throw Error(typeof result.detail==='string'?result.detail:'Technical error: request failed.'); return result;
  }
  function controls() {
    $('start').disabled=posting || !$('consent').checked;
    $('generate').disabled=posting || !state.online || state.busy || !$('prompt').value.trim();
    $('queue-note').textContent=state.busy?'Your image or chat request is in the shared queue. You can submit again when it finishes.':'';
  }
  function render() {
    $('welcome').hidden=state.signed_in; $('image-form').hidden=!state.signed_in;
    $('connection').textContent=state.online?'Image model online':'Image model offline'; $('connection').classList.toggle('online',state.online);
    const next=JSON.stringify(state.jobs);
    if(signature!==next) {
      signature=next;
      const cards=state.jobs.map(job=>{
        const card=document.createElement('article');card.className='image-card';
        const status=document.createElement('span');status.className='outcome';
        status.textContent=job.status==='queued'?`Queued · position ${job.position}`:job.status==='running'?(job.upstream_status==='queued'?`Waiting on GPU · position ${job.upstream_position ?? 'pending'}`:`Generating · ${job.stage || 'preparing'}`):job.status==='complete'?'Image result':job.status==='no_image'?'No image returned':'Technical interruption';
        const prompt=document.createElement('p');prompt.textContent=job.prompt;
        const settings=document.createElement('p');settings.className='fine';settings.textContent=`${job.settings.width} × ${job.settings.height} · Seed ${job.settings.seed} · 4 steps · FLUX.2 klein 4B · BF16`;
        card.append(status,prompt,settings);
        if(job.error){const note=document.createElement('p');note.className='error';note.textContent=job.error;card.append(note);}
        if(job.result_url){const img=document.createElement('img');img.alt=job.prompt;img.src=job.result_url;img.loading='lazy';img.onerror=()=>{const note=document.createElement('p');note.textContent='The saved image could not be loaded. Reload to retry.';img.replaceWith(note);};const download=document.createElement('a');download.href=job.result_url;download.download=`soft-refusal-${job.id}.png`;download.textContent='Download PNG ↧';card.append(img,download);}
        return card;
      });
      if(!cards.length){const empty=document.createElement('p');empty.className='fine';empty.textContent='Your first image test will appear here. Results are not automatically labelled as answers or refusals.';cards.push(empty);}
      $('results').replaceChildren(...cards);
    }
    controls();
  }
  async function refresh(){try{state=await api('/api/images');render();}catch{state.online=false;$('connection').textContent='Connection interrupted · retrying';controls();}}
  $('consent').onchange=controls;$('prompt').oninput=controls;
  $('start').onclick=async()=>{posting=true;controls();error('');try{await api('/api/images/session',{consent:true});await refresh();$('prompt').focus();}catch(e){error(e.message);}finally{posting=false;controls();}};
  $('image-form').onsubmit=async event=>{
    event.preventDefault();if(posting || state.busy || !state.online)return;
    const body={prompt:$('prompt').value.trim(),width:Number($('width').value),height:Number($('height').value),seed:Number($('seed').value)};
    if(!pending || JSON.stringify(pending.body)!==JSON.stringify(body))pending={body,key:crypto.randomUUID()};
    sessionStorage.setItem('softrefusal.image-pending',JSON.stringify(pending));posting=true;controls();error('');
    try{await api('/api/images/jobs',{...body,request_key:pending.key});pending=null;sessionStorage.removeItem('softrefusal.image-pending');await refresh();}catch(e){error(e.message);}finally{posting=false;controls();}
  };
  async function poll(){await refresh();setTimeout(poll,document.hidden?5000:1500);}poll();
})();
