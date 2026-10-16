/* Preserve raw output in storage; separate only the text the model actually emits. */
(function(root) {
  function splitReasoning(text) {
    text = String(text || '');
    const start = text.indexOf('<think>');
    const from = start < 0 ? 0 : start + 7;
    const end = text.indexOf('</think>', from);
    if (start < 0 && end < 0) return {thinking:'', answer:text, pending:false};
    return {thinking:text.slice(from, end < 0 ? undefined : end).trim(),
      answer:((start < 0 ? '' : text.slice(0, start)) + (end < 0 ? '' : text.slice(end + 8))).trim(), pending:end < 0};
  }
  root.splitReasoning = splitReasoning;
  if (typeof module !== 'undefined') module.exports = splitReasoning;
})(typeof window === 'undefined' ? globalThis : window);
