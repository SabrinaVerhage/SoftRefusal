/* Browser checks with authored fixtures only; no GPU or live visitor requests. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium} = require('playwright');

(async () => {
  const browser = await chromium.launch({headless:true, channel:'msedge'});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1000}, reducedMotion:'reduce'});
    const errors = []; page.on('pageerror', e => errors.push(e.message));
    // Count the renderer's two-arm arrowheads, independently of selection.
    await page.addInitScript(() => {
      const proto = CanvasRenderingContext2D.prototype;
      for (const name of ['setTransform','beginPath','moveTo','lineTo','stroke']) {
        const original = proto[name];
        proto[name] = function(...args) {
          if (name === 'setTransform') window.testArrows = 0;
          if (name === 'beginPath') this.testPath = [];
          if (name === 'moveTo' || name === 'lineTo') (this.testPath ||= []).push(args);
          if (name === 'stroke' && this.testPath?.length === 3) {
            const [a,b,c] = this.testPath;
            if (Math.abs(Math.hypot(a[0]-b[0],a[1]-b[1])-6)<.01 && Math.abs(Math.hypot(c[0]-b[0],c[1]-b[1])-6)<.01) window.testArrows++;
          }
          return original.apply(this,args);
        };
      }
    });
    await page.goto(pathToFileURL(path.join(__dirname,'index.html')).href);
    await page.evaluate(() => {
      // Mimics the live payload shape; every point is an authored test fixture.
      const points = Array.from({length:24},(_,i) => ({point_id:`fixture-${i}`,conversation_id:`c${Math.floor(i/2)}`,conversation_number:Math.floor(i/2)+1,
        turn_id:i%2,previous_point_id:i%2?`fixture-${i-1}`:null,core_score:i%2?.65:-.2,angle:Math.floor(i/2)/12*Math.PI*2,lift:0,
        completed_utc:i>=22?new Date().toISOString():null,prompt:'A romantic kiss',response:'<think>Fixture reasoning.</think>Fixture answer.',outcome:i%2?'refused':'answered',topic:'romance',layers:{20:[i,0,0]}}));
      window.fixturePayload = {kind:'recordings',synthetic:false,model_id:'fixture',created_utc:'fixture-1',layers:[20],default_layer:20,conversations:12,points,
        semantic:{layer:20,radial_mapping:'smooth_v1',frozen:true,reference_id:'fixture-only',labels:{answered:12,refused:12,redirected:0,unclear:0,unlabeled:0}}};
      window.dispatchEvent(new CustomEvent('softrefusal:recordings',{detail:window.fixturePayload}));
    });
    await page.waitForFunction(() => window.testArrows === 10);
    assert(await page.locator('#core-label').isVisible());
    assert(await page.locator('.theme-cloud').count() > 0);
    assert.match(await page.locator('#projection-note').textContent(),/Fixed refusal reference/);
    const out=path.join(__dirname,'runtime');fs.mkdirSync(out,{recursive:true});
    await page.screenshot({path:path.join(out,'final-map-desktop.png')});
    await page.evaluate(() => {
      const p=structuredClone(window.fixturePayload);p.semantic={unavailable:'Fixture missing labels'};
      window.dispatchEvent(new CustomEvent('softrefusal:recordings',{detail:p}));
    });
    assert(await page.locator('#core-label').isVisible());
    assert.match(await page.locator('#projection-note').textContent(),/illustrative core/);
    await page.setViewportSize({width:390,height:844});
    assert(await page.evaluate(() => document.documentElement.scrollWidth<=innerWidth));
    await page.screenshot({path:path.join(out,'final-map-mobile.png')});
    await page.evaluate(() => window.dispatchEvent(new CustomEvent('softrefusal:recordings',{detail:window.fixturePayload})));
    assert.match(await page.locator('#projection-note').textContent(),/Fixed refusal reference/);

    const phone = await browser.newPage({viewport:{width:390,height:844},isMobile:true,hasTouch:true});
    phone.on('pageerror',e=>errors.push(e.message));
    let state={conversation:'fixture',rows:Array.from({length:10},(_,i)=>({point_id:`r${i}`,prompt:'Earlier message '.repeat(8),response:'Earlier reply '.repeat(8),label:'answered'})),job:null,model:{name:'Fixture',online:true,auto_label:true}};
    await phone.addInitScript(()=>sessionStorage.setItem('softrefusal.conversation','fixture'));
    await phone.route('https://softrefusal.test/**', async route => {
      const url=new URL(route.request().url());
      if(url.pathname==='/api/state')return route.fulfill({json:state});
      if(url.pathname==='/api/send'){
        state.job={id:'new',status:'running',prompt:route.request().postDataJSON().prompt,response:''};return route.fulfill({json:{id:'new'}});
      }
      const file=url.pathname==='/'?'index.html':path.basename(url.pathname);
      const allowed=['index.html','phone.css','phone.js','reasoning.js'];
      if(!allowed.includes(file))return route.fulfill({status:404,body:''});
      return route.fulfill({body:fs.readFileSync(path.join(__dirname,'../hosted/web',file)),contentType:file.endsWith('.html')?'text/html':file.endsWith('.css')?'text/css':'application/javascript'});
    });
    await phone.goto('https://softrefusal.test/');
    await phone.locator('.bubble').last().waitFor();
    assert.equal(await phone.locator('a[href*="map"],a[href*="images"]').count(),0);
    await phone.locator('#prompt').fill('My latest message'); await phone.locator('#send').click();
    await phone.waitForFunction(()=>document.querySelector('.bubble:last-child .text')?.textContent==='My latest message');
    await phone.evaluate(()=>scrollTo(0,0));
    state.job.response='The newest AI answer. '.repeat(12);
    await phone.waitForFunction(()=>document.querySelector('.bubble:last-child .text')?.textContent.startsWith('The newest AI answer'));
    await phone.waitForFunction(()=>scrollY>500 && document.querySelector('#composer').getBoundingClientRect().bottom<=innerHeight+2);
    assert(await phone.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    await phone.screenshot({path:path.join(out,'final-phone-mobile.png')});
    assert.deepEqual(errors,[]);
    console.log('Passed: ten newest paths, unselected arrows, theme labels, donut fallback, mobile layout, removed links, sent-message display and automatic reply scrolling.');
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
