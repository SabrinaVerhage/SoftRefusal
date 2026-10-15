/* Browser regression: run against local files (or MAP_TEST_URL), with Playwright installed.
   No model/API requests. All conversation data is explicitly synthetic. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true, channel: 'msedge' });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, reducedMotion: 'reduce' });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const base = process.env.MAP_TEST_URL || require('node:url').pathToFileURL(path.join(__dirname, 'index.html')).href;
    await page.goto(new URL('?study', base).href);
    await page.locator('#replay').waitFor({ state: 'visible' });
    assert.match(await page.locator('#source-summary').textContent(), /no model measurements/);
    await page.locator('#conversation-select').selectOption('study-return');
    assert.match(await page.locator('#replay-status').textContent(), /Turn 1 \/ 5/);
    await page.locator('#replay-next').click();
    assert.match(await page.locator('#point-prompt').textContent(), /how dancing feels/);
    await page.locator('#replay-prev').click();
    assert.equal(await page.locator('#replay-prev').isDisabled(), true);
    await page.locator('#replay-turn').fill('3');
    assert.match(await page.locator('#replay-status').textContent(), /Turn 4 \/ 5 · refused/);
    await page.locator('[data-outcome="refused"]').click();
    assert.match(await page.locator('#replay-status').textContent(), /hidden by filter/);
    await page.locator('[data-outcome="refused"]').click();
    await page.locator('#replay-play').click();
    await page.waitForFunction(() => document.getElementById('replay-status').textContent.includes('Turn 5 / 5'));
    await page.waitForFunction(() => document.getElementById('replay-play').getAttribute('aria-pressed') === 'false');
    assert.equal(await page.locator('#replay-next').isDisabled(), true);
    await page.locator('#replay-play').click();
    assert.match(await page.locator('#replay-status').textContent(), /Turn 1 \/ 5/);
    await page.locator('#replay-play').click();
    assert.equal(await page.locator('#tour').getAttribute('aria-pressed'), 'false');
    await page.locator('#replay-full').click();
    await page.locator('#conversation-select').selectOption('study-nearby');
    assert.match(await page.locator('#point-prompt').textContent(), /romantic scene/);
    await page.locator('#replay-turn').fill('3');
    const out = path.join(__dirname, 'runtime'); fs.mkdirSync(out, { recursive: true });
    await page.screenshot({ path: path.join(out, 'replay-desktop.png') });
    await page.locator('#dataset').selectOption('umap');
    assert.equal(await page.locator('#replay').isVisible(), false);
    await page.locator('#dataset').selectOption('study');
    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator('#inspector-mobile').click();
    await page.locator('#conversation-select').selectOption('study-boundary');
    await page.locator('#replay-next').click();
    assert.match(await page.locator('#replay-status').textContent(), /Turn 2 \/ 5/);
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({ path: path.join(out, 'replay-mobile.png') });
    assert.deepEqual(errors, []);
    console.log('Passed: conversation selection, stepping, scrubbing, filtering, timed replay, completion, pause/restart, dataset switch, reduced motion and mobile controls.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
