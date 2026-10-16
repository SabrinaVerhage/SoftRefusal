(() => {
  // Live status sits quietly in the masthead; the map and recording room are separate pages.
  const label = document.getElementById('edition-text');
  let version = '';
  label.textContent = 'CONNECTING';
  async function refresh() {
    try {
      const response = await fetch('/api/map', {cache: 'no-store'}); if (!response.ok) throw Error('offline'); const payload = await response.json();
      if (payload.points?.length) {
        const conversations = new Set(payload.points.map(p => p.conversation_id)).size;
        label.textContent = `LIVE · ${conversations} ${conversations === 1 ? 'CONVERSATION' : 'CONVERSATIONS'}`;
        if (version !== payload.created_utc) { version = payload.created_utc; window.dispatchEvent(new CustomEvent('softrefusal:recordings', {detail: payload})); }
      } else label.textContent = 'WAITING FOR THE FIRST CONVERSATION';
    } catch { label.textContent = 'OFFLINE · LAST MAP SHOWN'; }
    setTimeout(refresh, document.hidden ? 15000 : 4000);
  }
  refresh();
})();
