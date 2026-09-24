import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { chromium } from 'playwright';

const targets = JSON.parse(await fs.readFile('targets.json','utf8'));
await fs.rm('output',{recursive:true,force:true});
await fs.mkdir('output',{recursive:true});

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
    userAgent: 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36'
  });
  const page = await context.newPage();

  try {
    const nav = await page.goto(meta.player_url,{waitUntil:'domcontentloaded',timeout:60000});
    meta.player_http_status=nav?.status() ?? null;
    await page.waitForTimeout(3500);
    await page.waitForSelector('video',{timeout:30000});
    const video=page.locator('video').first();

    await video.evaluate(async v => {
      v.muted=true;
      try { await v.play(); } catch {}
    });
    await page.waitForTimeout(2500);

    let info=await video.evaluate(v => ({
      duration: Number.isFinite(v.duration) ? v.duration : null,
      video_width: v.videoWidth || null,
      video_height: v.videoHeight || null,
      ready_state: v.readyState,
      current_time: v.currentTime,
      paused: v.paused,
      current_src: v.currentSrc || v.src || null
    }));
    if (!info.duration) {
      await page.waitForTimeout(4000);
      info=await video.evaluate(v => ({
        duration: Number.isFinite(v.duration) ? v.duration : null,
        video_width: v.videoWidth || null,
        video_height: v.videoHeight || null,
        ready_state: v.readyState,
        current_time: v.currentTime,
        paused: v.paused,
        current_src: v.currentSrc || v.src || null
      }));
    }
    Object.assign(meta,info);

    // Capture representative frames using the official player. These alone count only as partial frames.
    if (info.duration && info.duration > 0) {
      const points=[0.05, info.duration*0.25, info.duration*0.5, info.duration*0.75, Math.max(0.05,info.duration-0.15)];
      meta.sample_timestamps=points;
      for (let i=0;i<points.length;i++) {
        const t=Math.min(points[i],Math.max(0.05,info.duration-0.05));
        await video.evaluate((v,t)=>new Promise(resolve=>{
          const done=()=>{v.removeEventListener('seeked',done);resolve();};
          v.addEventListener('seeked',done,{once:true});
          try { v.currentTime=t; } catch { resolve(); }
          setTimeout(resolve,4000);
        }),t);
        await page.waitForTimeout(250);
        await video.screenshot({path:`${dir}/frame-${i}.png`});
      }
      meta.capture_scope='partial_frames';
    }

    // If the official player exposes a signed currentSrc, fetch the complete media without bypassing auth/CAPTCHA.
    if (info.current_src && /^https:\/\//.test(info.current_src)) {
      try {
        const r=await context.request.get(info.current_src,{
          headers:{'referer':'https://www.tiktok.com/'},
          timeout:60000
        });
        meta.media_http_status=r.status();
        meta.media_content_type=r.headers()['content-type'] || null;
        if (r.ok()) {
          const body=await r.body();
          if (body.byteLength > 100000) {
            const mp4=`${dir}/video.mp4`;
            await fs.writeFile(mp4,body);
            meta.media_bytes=body.byteLength;
            meta.media_sha256=crypto.createHash('sha256').update(body).digest('hex');
            try {
              const probe=execFileSync('ffprobe',[
                '-v','error','-show_entries','format=duration,size',
                '-of','json',mp4
              ],{encoding:'utf8'});
              meta.ffprobe=JSON.parse(probe);
              const dur=Number(meta.ffprobe?.format?.duration);
              if (dur > 0) {
                meta.capture_scope='full_video';
                meta.status='full_video_captured';
              }
            } catch (e) {
              meta.ffprobe_error=String(e?.message || e);
            }
          }
        }
      } catch (e) {
        meta.media_fetch_error=String(e?.message || e);
      }
    }

    if (meta.status !== 'full_video_captured') {
      meta.status=meta.capture_scope === 'partial_frames' ? 'partial_frames_captured' : 'player_loaded_no_media';
    }
  } catch (e) {
    meta.status='capture_error';
    meta.error=String(e?.stack || e);
    try { await page.screenshot({path:`${dir}/failure.png`,fullPage:true}); } catch {}
  } finally {
    meta.finished_at=new Date().toISOString();
    await fs.writeFile(`${dir}/metadata.json`,JSON.stringify(meta,null,2));
    await context.close();
  }
}
await browser.close();
