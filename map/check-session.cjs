/* Accelerated visitor-idle checks; local fixtures, no live requests. */
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {chromium}=require('playwright');
(async()=>{
  const browser=await chromium.launch({channel:'msedge',headless:true});
  try {
    const page=await browser.newPage({viewport:{width:390,height:844},isMobile:true,hasTouch:true});
    const errors=[];page.on('pageerror',e=>errors.push(e.message));
    await page.clock.install({time:new Date('2026-10-18T10:00:00Z')});
    await page.addInitScript(()=>{
      if(!sessionStorage.getItem('fixture.initialized')){
        sessionStorage.setItem('fixture.initialized','true');
        sessionStorage.setItem('softrefusal.conversation','old');
        sessionStorage.setItem('softrefusal.activity',String(Date.now()));
      }
    });
    const state={conversation:'old',rows:[{point_id:'saved',prompt:'A fixture prompt',response:'A saved answer',label:'answered'}],job:null,model:{name:'Fixture',online:true,auto_label:true}};
    let starts=0;
    await page.route('https://session.test/**',route=>{
      const url=new URL(route.request().url());
      if(url.pathname==='/api/state')return route.fulfill({json:url.searchParams.get('resume')==='false'?{...state,conversation:null,rows:[],job:null}:state});
      if(url.pathname==='/api/activity')return route.fulfill({json:{ok:true}});
      if(url.pathname==='/api/start'){state.conversation='new-'+(++starts);state.rows=[];return route.fulfill({json:{conversation:state.conversation}});}
      const file=url.pathname==='/'?'index.html':path.basename(url.pathname);
      if(!['index.html','phone.js','phone.css','reasoning.js'].includes(file))return route.fulfill({status:404,body:''});
      return route.fulfill({body:fs.readFileSync(path.join(__dirname,'../hosted/web',file)),contentType:file.endsWith('.html')?'text/html':file.endsWith('.css')?'text/css':'application/javascript'});
    });
    await page.goto('https://session.test/');
    await page.locator('.bubble').last().waitFor();
    await page.clock.fastForward(19*60*1000);
    assert(await page.locator('#conversation').isVisible());
    await page.locator('#prompt').fill('Still typing');
    await page.clock.fastForward(2*60*1000);
    assert(await page.locator('#conversation').isVisible());
    await page.clock.fastForward(18*60*1000+1000);
    await page.waitForFunction(()=>!document.getElementById('welcome').hidden);
    assert(await page.locator('#session-notice').isVisible());
    assert.equal(await page.locator('#messages').textContent(),'');
    assert.equal(await page.locator('#prompt').inputValue(),'');
    assert.equal(await page.evaluate(()=>sessionStorage.getItem('softrefusal.conversation')),null);
    await page.reload();
    assert(await page.locator('#welcome').isVisible());
    assert.equal(await page.evaluate(()=>sessionStorage.getItem('softrefusal.conversation')),null);
    await page.locator('#consent').check();await page.locator('#start').click();
    await page.waitForFunction(()=>!document.getElementById('conversation').hidden);
    assert.equal(starts,1);
    state.job={id:'working',status:'running',prompt:'Fixture',response:'A reply in progress'};
    await page.clock.fastForward(2000);
    await page.waitForFunction(()=>!document.getElementById('stop').hidden);
    await page.clock.fastForward(21*60*1000);
    assert(await page.locator('#conversation').isVisible());
    await page.reload();
    await page.waitForFunction(()=>!document.getElementById('stop').hidden);
    assert(await page.locator('#conversation').isVisible());
    state.job={...state.job,status:'complete'};
    await page.clock.fastForward(2000);
    await page.waitForFunction(()=>document.getElementById('stop').hidden);
    await page.clock.fastForward(19*60*1000);
    assert(await page.locator('#conversation').isVisible());
    await page.clock.fastForward(61*1000);
    await page.waitForFunction(()=>!document.getElementById('welcome').hidden);
    assert.deepEqual(errors,[]);
    console.log('Passed: 20-minute idle reset, typing renewal, cleared draft/history, reload without resuming, new consent/session and active-reply protection.');
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
