import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import { chromium } from 'playwright';

const targets = JSON.parse(await fs.readFile('targets.json','utf8'));
await fs.rm('output',{recursive:true,force:true});
await fs.mkdir('output',{recursive:true});

function parseClock(s) {
  const m=String(s).match(/(?:(\d+):)?(\d+):(\d+)/);
  if (!m) return null;
  return Number(m[1]||0)*3600+Number(m[2])*60+Number(m[3]);
}
function extractProgress(text) {
  const all=[...String(text).matchAll(/(\d{1,2}:\d{2})\s*\/\s*(\d{1,2}:\d{2})/g)];
  if (!all.length) return null;
  const [cur,total]=all[all.length-1].slice(1);
  return {text:`${cur}/${total}`,current_seconds:parseClock(cur),duration_seconds:parseClock(total)};
}
function dhash(buf){ return crypto.createHash('sha256').update(buf).digest('hex'); }

const browser = await chromium.launch({
  headless: true,
  args: ['--disable-dev-shm-usage','--no-sandbox']
});

for (const target of targets) {
  const safeKey = String(target.key || 'target').replace(/[^a-zA-Z0-9._-]/g,'_');
  const dir = `output/${safeKey}`;
  await fs.mkdir(dir,{recursive:true});
  const meta = {
    key: target.key,
    platform: target.platform,
    original_url: target.url,
    started_at: new Date().toISOString(),
    capture_scope: 'inaccessible',
    capture_method: null,
    status: 'not_started'
  };

  const m = String(target.url).match(/\/video\/(\d+)/);
  if (String(target.platform).toLowerCase() !== 'tiktok' || !m) {
    meta.status='unsupported_target';
    await fs.writeFile(`${dir}/metadata.json`, JSON.stringify(meta,null,2));
    continue;
  }

  const videoId=m[1];
  meta.video_id=videoId;
  meta.player_url=`https://www.tiktok.com/player/v1/${videoId}?autoplay=0&loop=0&controls=1&timestamp=1&muted=1&description=1`;

  const context = await browser.newContext({
    viewport: {width: 576, height: 1024},
    recordVideo: {dir:`${dir}/recordings`, size:{width:576,height:1024}},
    userAgent: 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36'
  });
  const page = await context.newPage();
  const recording = page.video();

  try {
    const nav = await page.goto(meta.player_url,{waitUntil:'domcontentloaded',timeout:60000});
    meta.player_http_status=nav?.status() ?? null;
    await page.waitForTimeout(3500);

    const initialText=await page.locator('body').innerText().catch(()=> '');
    meta.initial_progress=extractProgress(initialText);
    meta.body_has_ai_label=/AI[- ]generated/i.test(initialText);

    await page.screenshot({path:`${dir}/frame-000.png`,fullPage:true});
    const hashes=[dhash(await fs.readFile(`${dir}/frame-000.png`))];

    // Official TikTok player is visually rendered but its media element can live outside
    // the ordinary DOM tree. Click the visible central play control; no login/CAPTCHA bypass.
    await page.mouse.click(288,512);
    await page.waitForTimeout(900);

    const observations=[];
    const expected=meta.initial_progress?.duration_seconds || 15;
    const watchSeconds=Math.min(Math.max(expected+3,8),45);

    for (let sec=1; sec<=watchSeconds; sec++) {
      await page.waitForTimeout(1000);
      const text=await page.locator('body').innerText().catch(()=> '');
      const p=extractProgress(text);
      const shot=`${dir}/frame-${String(sec).padStart(3,'0')}.png`;
      await page.screenshot({path:shot,fullPage:true});
      const h=dhash(await fs.readFile(shot));
      hashes.push(h);
      observations.push({elapsed_seconds:sec,progress:p,frame_sha256:h});
      if (p?.duration_seconds && p.current_seconds >= p.duration_seconds-1) {
        // give the end frame a moment to settle
        await page.waitForTimeout(700);
        break;
      }
    }

    meta.progress_observations=observations;
    meta.distinct_frame_hashes=new Set(hashes).size;
    const progress=observations.map(x=>x.progress).filter(Boolean);
    if (!meta.initial_progress && progress.length) meta.initial_progress=progress[0];
    const total=meta.initial_progress?.duration_seconds || progress.find(x=>x.duration_seconds)?.duration_seconds || null;
    const maxCurrent=progress.reduce((a,x)=>Math.max(a,x.current_seconds ?? 0),0);
    meta.duration_seconds=total;
    meta.max_observed_seconds=maxCurrent;

    // Full-video means the official public player was actually rendered sequentially
    // from the start through the end. It does NOT mean a raw platform MP4 was downloaded.
    const progressed=maxCurrent >= 2;
    const reachedEnd=total !== null && maxCurrent >= Math.max(1,total-1);
    const visuallyChanged=meta.distinct_frame_hashes >= Math.min(5, Math.max(3, Math.floor((total||8)/3)));

    if (progressed && reachedEnd && visuallyChanged) {
      meta.capture_scope='full_video';
      meta.capture_method='official_tiktok_player_full_playback';
      meta.status='full_playback_verified';
      meta.raw_platform_media_downloaded=false;
    } else if (visuallyChanged) {
      meta.capture_scope='partial_frames';
      meta.capture_method='official_tiktok_player_timed_frames';
      meta.status='partial_frames_captured';
    } else {
      meta.capture_scope='inaccessible';
      meta.status='player_loaded_no_verified_playback';
    }
  } catch (e) {
    meta.status='capture_error';
    meta.error=String(e?.stack || e);
    try { await page.screenshot({path:`${dir}/failure.png`,fullPage:true}); } catch {}
  } finally {
    meta.finished_at=new Date().toISOString();
    await fs.writeFile(`${dir}/metadata.json`,JSON.stringify(meta,null,2));
    await context.close();
    if (recording) {
      try {
        const p=await recording.path();
        await fs.copyFile(p,`${dir}/player-session.webm`);
      } catch {}
    }
  }
}
await browser.close();
