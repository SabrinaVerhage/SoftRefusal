/* Preserve raw output in storage; separate only the text the model actually emits. */
(function(root) {
  function splitReasoning(text) {
    text = String(text || '');
    const start = text.indexOf('<think>');
    if (start < 0) return {thinking:'', answer:text, pending:false};
    const end = text.indexOf('</think>', start + 7);
    return {thinking:text.slice(start + 7, end < 0 ? undefined : end).trim(),
      answer:(text.slice(0, start) + (end < 0 ? '' : text.slice(end + 8))).trim(), pending:end < 0};
  }
  root.splitReasoning = splitReasoning;
  if (typeof module !== 'undefined') module.exports = splitReasoning;
})(typeof window === 'undefined' ? globalThis : window);
