import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { chromium } from 'playwright';

const targets = JSON.parse(await fs.readFile('targets.json','utf8'));
await fs.rm('output',{recursive:true,force:true});
await fs.mkdir('output',{recursive:true});

const sha256 = (buf) => crypto.createHash('sha256').update(buf).digest('hex');

function probeMedia(path) {
  const out=execFileSync('ffprobe',[
    '-v','error',
    '-show_entries','format=duration,size,format_name:stream=codec_type,codec_name,width,height,avg_frame_rate',
    '-of','json',path
  ],{encoding:'utf8'});
  return JSON.parse(out);
}

async function sampleMedia(path,dir,duration) {
  const points=[0, duration*0.25, duration*0.5, duration*0.75, Math.max(0,duration-0.1)];
  const frames=[];
  for (let i=0;i<points.length;i++) {
    const p=`${dir}/media-frame-${i}.jpg`;
    execFileSync('ffmpeg',['-y','-ss',String(points[i]),'-i',path,'-frames:v','1','-q:v','3',p],{stdio:'ignore'});
    const b=await fs.readFile(p);
    frames.push({t:points[i],sha256:sha256(b),bytes:b.byteLength});
  }
  return frames;
}

async function fetchMedia(url,path) {
  const r=await fetch(url,{
    redirect:'follow',
    headers:{
      'user-agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/140 Safari/537.36',
      'accept':'video/*,*/*;q=0.8'
    }
  });
  const type=r.headers.get('content-type');
  const buf=Buffer.from(await r.arrayBuffer());
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  if (buf.byteLength < 100000) throw new Error(`response too small: ${buf.byteLength}`);
  await fs.writeFile(path,buf);
  const probe=probeMedia(path);
  const duration=Number(probe?.format?.duration || 0);
  const hasVideo=(probe?.streams||[]).some(s=>s.codec_type==='video');
  if (!(duration>0) || !hasVideo) throw new Error('downloaded response is not a decodable complete video container');
  return {
    final_url:r.url,
    status:r.status,
    content_type:type,
    bytes:buf.byteLength,
    sha256:sha256(buf),
    probe,
    duration
  };
}

async function tryTikTokOfficialPlayer(target,dir,meta,browser) {
  const m=String(target.url).match(/\/video\/(\d+)/);
  if (!m) return {ok:false,reason:'bad_tiktok_url'};
  const videoId=m[1];
  meta.video_id=videoId;
  meta.player_url=`https://www.tiktok.com/player/v1/${videoId}?autoplay=0&loop=0&controls=1&timestamp=1&muted=1&description=1`;

  const context=await browser.newContext({
    viewport:{width:576,height:1024},
    recordVideo:{dir:`${dir}/recordings`,size:{width:576,height:1024}},
    userAgent:'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140 Safari/537.36'
  });
  const page=await context.newPage();
  const recording=page.video();
  try {
    const nav=await page.goto(meta.player_url,{waitUntil:'domcontentloaded',timeout:60000});
    meta.official_player_http_status=nav?.status() ?? null;
    await page.waitForTimeout(3500);
    const body=await page.locator('body').innerText().catch(()=> '');
    const mm=body.match(/(\d{1,2}):(\d{2})\s*\/\s*(\d{1,2}):(\d{2})/);
    if (mm) {
      meta.official_player_duration_seconds=Number(mm[3])*60+Number(mm[4]);
    }
    meta.official_player_ai_label=/AI[- ]generated/i.test(body);
    await page.screenshot({path:`${dir}/official-player-before.png`,fullPage:true});
    await page.mouse.click(288,512);
    await page.waitForTimeout(2200);
    await page.screenshot({path:`${dir}/official-player-after.png`,fullPage:true});
    const after=await page.locator('body').innerText().catch(()=> '');
    if (/player error|error occurred with the video player/i.test(after)) {
      meta.official_player_playback='error_after_play';
      return {ok:false,reason:'official_player_stream_error'};
    }
    meta.official_player_playback='loaded_no_proven_full_playback';
    return {ok:false,reason:'official_player_full_playback_not_proven'};
  } catch(e) {
    meta.official_player_playback='capture_error';
    meta.official_player_error=String(e?.message||e);
    return {ok:false,reason:'official_player_capture_error'};
  } finally {
    await context.close();
    if (recording) {
      try {
        const p=await recording.path();
        await fs.copyFile(p,`${dir}/official-player-session.webm`);
      } catch {}
    }
  }
}

const browser=await chromium.launch({headless:true,args:['--disable-dev-shm-usage','--no-sandbox']});

for (const target of targets) {
  const safeKey=String(target.key||'target').replace(/[^a-zA-Z0-9._-]/g,'_');
  const dir=`output/${safeKey}`;
  await fs.mkdir(dir,{recursive:true});
  const meta={
    key:target.key,
    platform:String(target.platform||'').toLowerCase(),
    original_url:target.url,
    started_at:new Date().toISOString(),
    capture_scope:'inaccessible',
    status:'not_started',
    resolver:null
  };

  try {
    if (meta.platform==='tiktok') {
      await tryTikTokOfficialPlayer(target,dir,meta,browser);
      const u=new URL(target.url);
      const resolver=`https://d.tiktokfix.com${u.pathname}`;
      meta.resolver={provider:'TikTokFix',url:resolver,role:'public third-party media resolver; not a native metrics source'};
      try {
        const p=`${dir}/resolved-video.mp4`;
        const got=await fetchMedia(resolver,p);
        meta.resolver_result=got;
        meta.media_frames=await sampleMedia(p,dir,got.duration);
        const durationMatches=!meta.official_player_duration_seconds || Math.abs(got.duration-meta.official_player_duration_seconds)<=1.5;
        if (!durationMatches) throw new Error(`duration mismatch official=${meta.official_player_duration_seconds} resolved=${got.duration}`);
        meta.capture_scope='full_video';
        meta.capture_method='third_party_public_resolver_download_verified_against_original_and_ffprobe';
        meta.status='full_video_captured';
        meta.full_video_verified=true;
      } catch(e) {
        meta.resolver_error=String(e?.message||e);
      }
    } else if (meta.platform==='instagram') {
      const m=String(target.url).match(/instagram\.com\/(?:[^/]+\/)?(?:reel|reels)\/([A-Za-z0-9_-]+)/i);
      if (!m) throw new Error('bad_instagram_reel_url');
      const code=m[1];
      meta.shortcode=code;
      const embed=`https://www.instagram.com/reel/${code}/embed/`;
      const c=await browser.newContext({viewport:{width:576,height:1024}});
      const p=await c.newPage();
      try {
        const nav=await p.goto(embed,{waitUntil:'domcontentloaded',timeout:60000});
        meta.official_embed_http_status=nav?.status()??null;
        await p.waitForTimeout(3500);
        await p.screenshot({path:`${dir}/official-instagram-embed.png`,fullPage:true});
        const txt=await p.locator('body').innerText().catch(()=> '');
        meta.official_embed_login_wall=/log in|login|로그인/i.test(txt);
        meta.official_embed_loaded=meta.official_embed_http_status===200;
      } finally { await c.close(); }

      const resolver=`https://d.instagram7.com/reel/${code}/`;
      meta.resolver={provider:'Instagram7',url:resolver,role:'public third-party media resolver; not a native metrics source'};
      try {
        const media=`${dir}/resolved-video.mp4`;
        const got=await fetchMedia(resolver,media);
        meta.resolver_result=got;
        meta.media_frames=await sampleMedia(media,dir,got.duration);
        meta.capture_scope='full_video';
        meta.capture_method='third_party_public_resolver_download_ffprobe_verified';
        meta.status='full_video_captured';
        meta.full_video_verified=true;
      } catch(e) {
        meta.resolver_error=String(e?.message||e);
      }
    } else {
      meta.status='unsupported_platform';
    }
  } catch(e) {
    meta.status='capture_error';
    meta.error=String(e?.stack||e);
  } finally {
    meta.finished_at=new Date().toISOString();
    await fs.writeFile(`${dir}/metadata.json`,JSON.stringify(meta,null,2));
  }
}

await browser.close();
