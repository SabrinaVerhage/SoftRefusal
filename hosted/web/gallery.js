(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  let nextOffset = null, loading = false;
  const seen = new Set();
  function openImage(item) {
    $('detail-image').src = item.image_url;
    $('detail-image').alt = item.prompt;
    $('detail-prompt').textContent = item.prompt;
    $('detail-settings').textContent = `${item.width} × ${item.height} · Seed ${item.seed} · FLUX.2 klein 4B`;
    $('detail-settings').textContent += item.lora ? ` · ${item.lora.name} · Strength ${item.lora.strength}` : ' · Original model';
    $('download').href = item.image_url;
    $('download').download = `soft-refusal-${item.id}.png`;
    $('detail').showModal();
  }
  function card(item) {
    const figure = document.createElement('figure'); figure.className = 'gallery-card';
    const button = document.createElement('button'); button.type = 'button'; button.className = 'image-open';
    button.setAttribute('aria-label', `Open image: ${item.prompt}`);
    const image = document.createElement('img'); image.src = item.image_url; image.alt = item.prompt;
    image.loading = 'lazy'; image.width = item.width; image.height = item.height;
    image.onerror = () => { const note = document.createElement('span'); note.className = 'unavailable'; note.textContent = 'Image unavailable. Refresh to try again.'; image.replaceWith(note); };
    button.append(image); button.onclick = () => openImage(item);
    const caption = document.createElement('figcaption'); caption.textContent = item.prompt;
    const meta = document.createElement('p'); meta.className = 'fine';
    meta.textContent = `${item.width} × ${item.height} · ${new Date(item.created * 1000).toLocaleDateString(undefined, {day:'numeric', month:'short', year:'numeric'})}`;
    figure.append(button, caption, meta); return figure;
  }
  async function load(reset = false) {
    if (loading) return;
    loading = true; $('refresh').disabled = true; $('more').disabled = true;
    $('gallery-status').textContent = 'Loading the collection…';
    try {
      const response = await fetch(`/api/gallery?offset=${reset ? 0 : nextOffset || 0}`, {cache:'no-store'});
      if (!response.ok) throw Error('Could not load the collection. Try Refresh in a moment.');
      const result = await response.json();
      if (reset) { $('gallery').replaceChildren(); seen.clear(); }
      for (const item of result.images) {
        if (!seen.has(item.id)) { $('gallery').append(card(item)); seen.add(item.id); }
      }
      nextOffset = result.next_offset;
      $('empty').hidden = seen.size !== 0;
      $('more').hidden = nextOffset === null;
      $('gallery-status').textContent = seen.size ? `${seen.size} ${seen.size === 1 ? 'image' : 'images'} in view` : 'Ready for your imagination.';
    } catch (error) { $('gallery-status').textContent = error.message; }
    finally { loading = false; $('refresh').disabled = false; $('more').disabled = false; }
  }
  $('refresh').onclick = () => load(true);
  $('more').onclick = () => load();
  $('close-detail').onclick = () => $('detail').close();
  $('detail').onclick = event => { if(event.target === $('detail')) { const box = $('detail').getBoundingClientRect(); if(event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) $('detail').close(); } };
  load(true);
})();
