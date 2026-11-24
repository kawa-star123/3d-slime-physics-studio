/*
 *構成
 *1. 純粋ロジック: 定数設定, 画像解析, 3D立体化, 3Dモデル読込, ソフトボディ物理
 *2. 起動・UI: 初期パラメータと状態管理, 多言語辞書, UI操作とファイル読込
 *3. 3D エンジン: Three.js動的読込, シェーダーと床・星空描画, マウス操作とメインループ
 */
window.__studioBooted = true;

const VIEW_KEYS = ['front', 'right', 'back', 'left', 'top'];
const MAX_DIM = 512;            // 解析・テクスチャ用の最大辺 (px)
const CARVE_FRACTION = 0.3;     // 多視点時の深度彫刻の最大深さ（軸長に対する比）
const SMOOTH_ITERS = 2;         // Taubin スムージング反復数
const SPRING_WEIGHT = [1.0, 0.45, 0.2, 0.3]; // 構造 / せん断 / 曲げ / 内部支持 の剛性倍率
const MODEL_WARN_VERTS = 5000;   // これを超えたら注意表示（推奨上限）
const MODEL_HARD_VERTS = 12000;  // これを超えたら頂点クラスタリングで間引く
const MODEL_TARGET_VERTS = 8000; // 間引き後の目標頂点数

const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);

// 画像解析：シルエット（アルファ or 背景しきい値, 色ブリード, 深度, 距離変換

function boxBlur2D(data, w, h, r) {
  if (r < 1) return;
  const tmp = new Float32Array(w * h);
  const inv = 1 / (2 * r + 1);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    let sum = 0;
    for (let i = -r; i <= r; i++) sum += data[row + clamp(i, 0, w - 1)];
    for (let x = 0; x < w; x++) {
      tmp[row + x] = sum * inv;
      sum += data[row + clamp(x + r + 1, 0, w - 1)] - data[row + clamp(x - r, 0, w - 1)];
    }
  }
  for (let x = 0; x < w; x++) {
    let sum = 0;
    for (let i = -r; i <= r; i++) sum += tmp[clamp(i, 0, h - 1) * w + x];
    for (let y = 0; y < h; y++) {
      data[y * w + x] = sum * inv;
      sum += tmp[clamp(y + r + 1, 0, h - 1) * w + x] - tmp[clamp(y - r, 0, h - 1) * w + x];
    }
  }
}

// 3-4 チャンファー距離変換
function distanceTransform(mask, w, h) {
  const W = w + 2, H = h + 2, INF = 1e9;
  const d = new Float32Array(W * H);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) d[(y + 1) * W + x + 1] = mask[y * w + x] ? INF : 0;
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      if (d[i] === 0) continue;
      d[i] = Math.min(d[i - 1] + 3, d[i - W] + 3, d[i - W - 1] + 4, d[i - W + 1] + 4);
    }
  }
  for (let y = H - 2; y >= 1; y--) {
    for (let x = W - 2; x >= 1; x--) {
      const i = y * W + x;
      if (d[i] === 0) continue;
      d[i] = Math.min(d[i], d[i + 1] + 3, d[i + W] + 3, d[i + W + 1] + 4, d[i + W - 1] + 4);
    }
  }
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = d[(y + 1) * W + x + 1] / 3;
  return out;
}

/**
 * RGBA 画像を解析して視点データを返す。
 *  - mask   : シルエット (1=物体)。アルファがあればアルファ、無ければ外周色との差 + 外周からの塗りつぶし
 *  - tex    : シルエット外へ色をにじませたテクスチャ（白縁防止）
 *  - round  : 距離変換 / 最大距離（ふくらみ用）
 *  - height : 輝度とコントラストから推定した擬似深度（computeHeight で更新）
 */
function analyzeImage(rgba, w, h, bgThreshold) {
  const n = w * h;

  let transparent = 0;
  for (let i = 0; i < n; i++) if (rgba[i * 4 + 3] < 250) transparent++;
  const hasAlpha = transparent > 0; // アルファ < 250 を 1 画素でも含む = 透過画像

  const mask = new Uint8Array(n);
  if (hasAlpha) {
    for (let i = 0; i < n; i++) mask[i] = rgba[i * 4 + 3] >= 128 ? 1 : 0;
  } else {
    // 不透明画像：四隅の色を背景色とし、四隅からの塗りつぶし（Flood Fill）で背景だけを除く
    const rs = [], gs = [], bs = [];
    const addPx = (x, y) => {
      const o = (y * w + x) * 4;
      rs.push(rgba[o]); gs.push(rgba[o + 1]); bs.push(rgba[o + 2]);
    };
    const cs = Math.min(3, w, h);
    for (let dy = 0; dy < cs; dy++) for (let dx = 0; dx < cs; dx++) {
      addPx(dx, dy); addPx(w - 1 - dx, dy); addPx(dx, h - 1 - dy); addPx(w - 1 - dx, h - 1 - dy);
    }
    const med = (a) => { a.sort((p, q) => p - q); return a[a.length >> 1]; };
    const br = med(rs), bg = med(gs), bb = med(bs);
    const thr = bgThreshold * 441.67;
    const bgLike = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const o = i * 4;
      const dr = rgba[o] - br, dg = rgba[o + 1] - bg, db = rgba[o + 2] - bb;
      bgLike[i] = Math.sqrt(dr * dr + dg * dg + db * db) < thr ? 1 : 0;
    }
    const reach = new Uint8Array(n);
    const stack = new Int32Array(n);
    let sp = 0;
    const push = (i) => { if (bgLike[i] && !reach[i]) { reach[i] = 1; stack[sp++] = i; } };
    push(0); push(w - 1); push((h - 1) * w); push(h * w - 1); // 四隅だけが探索の起点
    while (sp > 0) {
      const i = stack[--sp];
      const x = i % w, y = (i / w) | 0;
      if (x > 0) push(i - 1);
      if (x < w - 1) push(i + 1);
      if (y > 0) push(i - w);
      if (y < h - 1) push(i + w);
    }
    for (let i = 0; i < n; i++) mask[i] = reach[i] ? 0 : 1;
  }

  let area = 0, minx = w, maxx = -1, miny = h, maxy = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!mask[y * w + x]) continue;
      area++;
      if (x < minx) minx = x;
      if (x > maxx) maxx = x;
      if (y < miny) miny = y;
      if (y > maxy) maxy = y;
    }
  }
  let fallback = false;
  if (area < n * 0.005) {
    mask.fill(1); minx = 0; maxx = w - 1; miny = 0; maxy = h - 1; area = n; fallback = true;
  }
  const bbox = { x0: minx, y0: miny, x1: maxx + 1, y1: maxy + 1 };
  bbox.w = bbox.x1 - bbox.x0;
  bbox.h = bbox.y1 - bbox.y0;
  bbox.cx = (bbox.x0 + bbox.x1) / 2;
  bbox.cy = (bbox.y0 + bbox.y1) / 2;

  // シルエット外へ最寄りの内部色をにじませる
  const tex = new Uint8ClampedArray(n * 4);
  const queue = new Int32Array(n);
  const done = new Uint8Array(n);
  let qh = 0, qt = 0;
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    tex[o] = rgba[o]; tex[o + 1] = rgba[o + 1]; tex[o + 2] = rgba[o + 2]; tex[o + 3] = 255;
    if (mask[i]) { done[i] = 1; queue[qt++] = i; }
  }
  while (qh < qt) {
    const i = queue[qh++];
    const x = i % w, y = (i / w) | 0;
    const o = i * 4;
    const visit = (j) => {
      if (done[j]) return;
      done[j] = 1;
      const p = j * 4;
      tex[p] = tex[o]; tex[p + 1] = tex[o + 1]; tex[p + 2] = tex[o + 2];
      queue[qt++] = j;
    };
    if (x > 0) visit(i - 1);
    if (x < w - 1) visit(i + 1);
    if (y > 0) visit(i - w);
    if (y < h - 1) visit(i + w);
  }

  const lum = new Float32Array(n);
  for (let i = 0; i < n; i++) lum[i] = (0.299 * tex[i * 4] + 0.587 * tex[i * 4 + 1] + 0.114 * tex[i * 4 + 2]) / 255;

  const dist = distanceTransform(mask, w, h);
  let dmax = 1;
  for (let i = 0; i < n; i++) if (mask[i] && dist[i] > dmax) dmax = dist[i];
  const round = new Float32Array(n);
  for (let i = 0; i < n; i++) round[i] = mask[i] ? Math.min(1, dist[i] / dmax) : 0;

  return { w, h, mask, tex, lum, round, height: new Float32Array(n), bbox, hasAlpha, fallback, area };
}

// 輝度 -> 擬似深度 (0..1)。2-98 パーセンタイルで正規化し、コントラストを掛けてぼかす。
function computeHeight(view, contrast) {
  const { w, h, lum, mask } = view;
  const n = w * h;
  const hist = new Float32Array(256);
  let cnt = 0;
  for (let i = 0; i < n; i++) {
    if (!mask[i]) continue;
    hist[Math.min(255, (lum[i] * 255) | 0)]++;
    cnt++;
  }
  let acc = 0, lo = 0, hi = 255, gotLo = false;
  const tLo = cnt * 0.02, tHi = cnt * 0.98;
  for (let b = 0; b < 256; b++) {
    acc += hist[b];
    if (!gotLo && acc >= tLo) { lo = b; gotLo = true; }
    if (acc >= tHi) { hi = b; break; }
  }
  const range = Math.max(8, hi - lo);
  const out = view.height;
  for (let i = 0; i < n; i++) {
    const t = clamp((lum[i] * 255 - lo) / range, 0, 1);
    out[i] = clamp(0.5 + (t - 0.5) * contrast, 0, 1);
  }
  boxBlur2D(out, w, h, Math.max(1, Math.round(Math.max(w, h) / 128)));
}

/* 
 * 視点の幾何：正射影の割り当てとワールド寸法の推定
 *   world: +X 右, +Y 上, +Z 手前（Front カメラは +Z 側から -Z を見る）
 *   (a, b) は画像の「右」「下」方向に対応するワールド座標
 */

let _pa = 0, _pb = 0;
function projAB(key, x, y, z) {
  switch (key) {
    case 'front': _pa = x; _pb = -y; break;
    case 'right': _pa = -z; _pb = -y; break;
    case 'back': _pa = -x; _pb = -y; break;
    case 'left': _pa = z; _pb = -y; break;
    default: _pa = x; _pb = z; break; // top（画像の上 = 奥 -Z）
  }
}

// その視点から見て「手前」を正とする座標
function outwardCoord(key, x, y, z) {
  switch (key) {
    case 'front': return z;
    case 'right': return x;
    case 'back': return -z;
    case 'left': return -x;
    default: return y;
  }
}

/**
 * 与えられた視点群から物体の半寸法 (hx, hy, hz) を決定し、各視点に ppu（px/ワールド単位）を設定する。
 * 視点が1つの平面（XY / ZY / XZ）にしか無い場合は、その法線軸が「未指定軸」となり、
 * 押し出し厚み（reliefDepth）とふくらみで厚みを与える。
 */
function configureViews(views, reliefDepth) {
  const f = views.front, r = views.right, b = views.back, l = views.left, t = views.top;
  if (!VIEW_KEYS.some((k) => views[k])) return null;

  const ratio = (v) => v.bbox.w / v.bbox.h;
  const avg = (a) => a.reduce((s, x) => s + x, 0) / a.length;
  const hasXY = !!(f || b), hasZY = !!(r || l), hasXZ = !!t;
  const planes = (hasXY ? 1 : 0) + (hasZY ? 1 : 0) + (hasXZ ? 1 : 0);
  const ws = [f, b].filter(Boolean).map(ratio);
  const ds = [r, l].filter(Boolean).map(ratio);

  let freeAxis = null, W = 1, H = 1, D = 1;
  if (planes === 1) {
    if (hasXY) { freeAxis = 'z'; W = avg(ws); H = 1; D = reliefDepth * Math.min(W, H); }
    else if (hasZY) { freeAxis = 'x'; D = avg(ds); H = 1; W = reliefDepth * Math.min(D, H); }
    else {
      freeAxis = 'y';
      const tr = ratio(t);
      W = tr >= 1 ? 1 : tr;
      D = tr >= 1 ? 1 / tr : 1;
      H = reliefDepth * Math.min(W, D);
    }
  } else {
    if (ws.length) W = avg(ws);
    if (ds.length) D = avg(ds);
    if (t) {
      const tr = ratio(t);
      if (!ws.length && ds.length) W = D * tr;
      else if (ws.length && !ds.length) D = W / tr;
    }
  }

  const m = Math.max(W, H, D);
  const hx = W / m, hy = H / m, hz = D / m; // 最長軸の全長が 2 になる

  // 視点ごとの自動ズーム補正はしない：すべての画像で同じスケール（px/単位）を共有する
  const ppus = [];
  for (const k of ['front', 'back', 'right', 'left']) if (views[k]) ppus.push(views[k].bbox.h / (2 * hy));
  if (t) ppus.push(t.bbox.w / (2 * hx));
  const ppu = avg(ppus);
  for (const k of VIEW_KEYS) if (views[k]) views[k].ppu = ppu;

  return { hx, hy, hz, freeAxis };
}

// 立体化：ボクセル占有場

const EDGE_A = [], EDGE_B = [];
for (let a = 0; a < 8; a++) {
  for (let bit = 1; bit <= 4; bit <<= 1) {
    if (!(a & bit)) { EDGE_A.push(a); EDGE_B.push(a | bit); }
  }
}

function blurAxis(src, dst, nx, ny, nz, axis) {
  const sx = 1, sy = nx, sz = nx * ny;
  const stride = axis === 0 ? sx : axis === 1 ? sy : sz;
  const len = axis === 0 ? nx : axis === 1 ? ny : nz;
  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const idx = i + nx * (j + ny * k);
        const p = axis === 0 ? i : axis === 1 ? j : k;
        const a = p > 0 ? src[idx - stride] : 0;
        const c = p < len - 1 ? src[idx + stride] : 0;
        dst[idx] = (a + src[idx] + c) / 3;
      }
    }
  }
}

function surfaceNets(F, nx, ny, nz, h) {
  const cellV = new Int32Array(nx * ny * nz).fill(-1);
  const pos = [];
  const dims = [nx, ny, nz];
  const strides = [1, nx, nx * ny];
  const cx = (nx - 1) / 2, cy = (ny - 1) / 2, cz = (nz - 1) / 2;
  const cv = new Float64Array(8);

  for (let k = 0; k < nz - 1; k++) {
    for (let j = 0; j < ny - 1; j++) {
      for (let i = 0; i < nx - 1; i++) {
        const base = i + nx * (j + ny * k);
        let mask = 0;
        for (let c = 0; c < 8; c++) {
          const val = F[base + (c & 1 ? 1 : 0) + (c & 2 ? nx : 0) + (c & 4 ? nx * ny : 0)];
          cv[c] = val;
          if (val > 0) mask |= 1 << c;
        }
        if (mask === 0 || mask === 255) continue;
        let px = 0, py = 0, pz = 0, cnt = 0;
        for (let e = 0; e < 12; e++) {
          const a = EDGE_A[e], b = EDGE_B[e];
          if (((mask >> a) & 1) === ((mask >> b) & 1)) continue;
          const t = cv[a] / (cv[a] - cv[b]);
          px += (a & 1) + t * ((b & 1) - (a & 1));
          py += ((a >> 1) & 1) + t * (((b >> 1) & 1) - ((a >> 1) & 1));
          pz += ((a >> 2) & 1) + t * (((b >> 2) & 1) - ((a >> 2) & 1));
          cnt++;
        }
        cellV[base] = pos.length / 3;
        pos.push((i + px / cnt - cx) * h, (j + py / cnt - cy) * h, (k + pz / cnt - cz) * h);
      }
    }
  }

  const quads = [];
  const p = [0, 0, 0];
  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const idx = i + nx * (j + ny * k);
        const f0 = F[idx];
        p[0] = i; p[1] = j; p[2] = k;
        for (let a = 0; a < 3; a++) {
          if (p[a] >= dims[a] - 1) continue;
          const f1 = F[idx + strides[a]];
          if (f0 > 0 === f1 > 0) continue;
          const u = (a + 1) % 3, v = (a + 2) % 3;
          if (p[u] < 1 || p[v] < 1 || p[u] > dims[u] - 2 || p[v] > dims[v] - 2) continue;
          const v0 = cellV[idx];
          const v1 = cellV[idx - strides[u]];
          const v2 = cellV[idx - strides[u] - strides[v]];
          const v3 = cellV[idx - strides[v]];
          if (v0 < 0 || v1 < 0 || v2 < 0 || v3 < 0) continue;
          if (f0 > 0) quads.push(v0, v1, v2, v3);
          else quads.push(v0, v3, v2, v1);
        }
      }
    }
  }
  return { pos, quads };
}

/** Surface Nets の四角形メッシュからスムージング・三角形化・バネ網を構築 */
function finalizeMesh(posFlat, quadFlat) {
  const n = posFlat.length / 3;
  let pos = Float64Array.from(posFlat);
  const nq = quadFlat.length / 4;
  const keyOf = (a, b) => (a < b ? a * n + b : b * n + a);

  const structMap = new Set(), shearMap = new Set(), bendMap = new Set();
  const stA = [], stB = [], shA = [], shB = [];
  const add = (set, la, lb, a, b) => {
    const key = keyOf(a, b);
    if (set.has(key)) return;
    set.add(key); la.push(a); lb.push(b);
  };
  for (let q = 0; q < nq; q++) {
    const q0 = quadFlat[4 * q], q1 = quadFlat[4 * q + 1], q2 = quadFlat[4 * q + 2], q3 = quadFlat[4 * q + 3];
    add(structMap, stA, stB, q0, q1);
    add(structMap, stA, stB, q1, q2);
    add(structMap, stA, stB, q2, q3);
    add(structMap, stA, stB, q3, q0);
    add(shearMap, shA, shB, q0, q2);
    add(shearMap, shA, shB, q1, q3);
  }

  // 構造エッジの隣接リスト (CSR)
  const deg = new Int32Array(n);
  for (let i = 0; i < stA.length; i++) { deg[stA[i]]++; deg[stB[i]]++; }
  const off = new Int32Array(n + 1);
  for (let i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
  const adj = new Int32Array(off[n]);
  const fillp = off.slice(0, n);
  for (let i = 0; i < stA.length; i++) {
    adj[fillp[stA[i]]++] = stB[i];
    adj[fillp[stB[i]]++] = stA[i];
  }

  // Taubin スムージング（収縮を抑えた平滑化）
  const smoothPass = (lambda) => {
    const out = new Float64Array(pos.length);
    for (let i = 0; i < n; i++) {
      const s = off[i], e = off[i + 1], o = 3 * i;
      if (e === s) { out[o] = pos[o]; out[o + 1] = pos[o + 1]; out[o + 2] = pos[o + 2]; continue; }
      let ax = 0, ay = 0, az = 0;
      for (let t = s; t < e; t++) { const j = 3 * adj[t]; ax += pos[j]; ay += pos[j + 1]; az += pos[j + 2]; }
      const inv = 1 / (e - s);
      out[o] = pos[o] + lambda * (ax * inv - pos[o]);
      out[o + 1] = pos[o + 1] + lambda * (ay * inv - pos[o + 1]);
      out[o + 2] = pos[o + 2] + lambda * (az * inv - pos[o + 2]);
    }
    pos = out;
  };
  for (let it = 0; it < SMOOTH_ITERS; it++) { smoothPass(0.5); smoothPass(-0.53); }

  // 三角形化（短い対角線で分割）
  const d2 = (a, b) => {
    const dx = pos[3 * a] - pos[3 * b], dy = pos[3 * a + 1] - pos[3 * b + 1], dz = pos[3 * a + 2] - pos[3 * b + 2];
    return dx * dx + dy * dy + dz * dz;
  };
  const tris = new Int32Array(nq * 6);
  for (let q = 0; q < nq; q++) {
    const q0 = quadFlat[4 * q], q1 = quadFlat[4 * q + 1], q2 = quadFlat[4 * q + 2], q3 = quadFlat[4 * q + 3];
    const o = 6 * q;
    if (d2(q0, q2) <= d2(q1, q3)) {
      tris[o] = q0; tris[o + 1] = q1; tris[o + 2] = q2; tris[o + 3] = q0; tris[o + 4] = q2; tris[o + 5] = q3;
    } else {
      tris[o] = q1; tris[o + 1] = q2; tris[o + 2] = q3; tris[o + 3] = q1; tris[o + 4] = q3; tris[o + 5] = q0;
    }
  }

  // 符号付き体積と表面積。負なら巻き順を反転して外向きにそろえる
  const measure = () => {
    let vol = 0, area = 0;
    for (let t = 0; t < tris.length; t += 3) {
      const a = 3 * tris[t], b = 3 * tris[t + 1], c = 3 * tris[t + 2];
      const bcx = pos[b + 1] * pos[c + 2] - pos[b + 2] * pos[c + 1];
      const bcy = pos[b + 2] * pos[c] - pos[b] * pos[c + 2];
      const bcz = pos[b] * pos[c + 1] - pos[b + 1] * pos[c];
      vol += (pos[a] * bcx + pos[a + 1] * bcy + pos[a + 2] * bcz) / 6;
      const e1x = pos[b] - pos[a], e1y = pos[b + 1] - pos[a + 1], e1z = pos[b + 2] - pos[a + 2];
      const e2x = pos[c] - pos[a], e2y = pos[c + 1] - pos[a + 1], e2z = pos[c + 2] - pos[a + 2];
      const cx = e1y * e2z - e1z * e2y, cy = e1z * e2x - e1x * e2z, cz = e1x * e2y - e1y * e2x;
      area += 0.5 * Math.sqrt(cx * cx + cy * cy + cz * cz);
    }
    return { vol, area };
  };
  let { vol, area } = measure();
  if (vol < 0) {
    for (let t = 0; t < tris.length; t += 3) { const s = tris[t + 1]; tris[t + 1] = tris[t + 2]; tris[t + 2] = s; }
    vol = -vol;
  }

  // 頂点法線
  const normals = new Float64Array(3 * n);
  for (let t = 0; t < tris.length; t += 3) {
    const a = 3 * tris[t], b = 3 * tris[t + 1], c = 3 * tris[t + 2];
    const e1x = pos[b] - pos[a], e1y = pos[b + 1] - pos[a + 1], e1z = pos[b + 2] - pos[a + 2];
    const e2x = pos[c] - pos[a], e2y = pos[c + 1] - pos[a + 1], e2z = pos[c + 2] - pos[a + 2];
    const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
    for (const o of [a, b, c]) { normals[o] += nx; normals[o + 1] += ny; normals[o + 2] += nz; }
  }
  for (let i = 0; i < n; i++) {
    const o = 3 * i;
    const l = Math.hypot(normals[o], normals[o + 1], normals[o + 2]) || 1;
    normals[o] /= l; normals[o + 1] /= l; normals[o + 2] /= l;
  }

  // バネ網：構造(0) / せん断(1) / 曲げ(2)
  const sa = [], sb = [], st = [];
  for (let i = 0; i < stA.length; i++) { sa.push(stA[i]); sb.push(stB[i]); st.push(0); }
  for (let i = 0; i < shA.length; i++) { sa.push(shA[i]); sb.push(shB[i]); st.push(1); }
  for (let b = 0; b < n; b++) {
    const s = off[b], e = off[b + 1];
    for (let p = s; p < e; p++) {
      for (let q = p + 1; q < e; q++) {
        const i = adj[p], j = adj[q];
        const ux = pos[3 * i] - pos[3 * b], uy = pos[3 * i + 1] - pos[3 * b + 1], uz = pos[3 * i + 2] - pos[3 * b + 2];
        const vx = pos[3 * j] - pos[3 * b], vy = pos[3 * j + 1] - pos[3 * b + 1], vz = pos[3 * j + 2] - pos[3 * b + 2];
        const cosA = (ux * vx + uy * vy + uz * vz) / ((Math.hypot(ux, uy, uz) * Math.hypot(vx, vy, vz)) + 1e-12);
        if (cosA >= -0.75) continue;
        const key = keyOf(i, j);
        if (structMap.has(key) || shearMap.has(key) || bendMap.has(key)) continue;
        bendMap.add(key);
        sa.push(i); sb.push(j); st.push(2);
      }
    }
  }
  const ns = sa.length;
  const L0 = new Float64Array(ns), W = new Float64Array(ns);
  for (let s = 0; s < ns; s++) {
    L0[s] = Math.sqrt(d2(sa[s], sb[s]));
    W[s] = SPRING_WEIGHT[st[s]];
  }

  const structEdges = new Uint32Array(stA.length * 2);
  for (let i = 0; i < stA.length; i++) { structEdges[2 * i] = stA[i]; structEdges[2 * i + 1] = stB[i]; }

  // プレクサス表示用：三角形メッシュの一意なエッジ
  const triEdgeSet = new Set();
  const te = [];
  for (let t = 0; t < tris.length; t += 3) {
    for (let e = 0; e < 3; e++) {
      const p = tris[t + e], q = tris[t + ((e + 1) % 3)];
      const key = keyOf(p, q);
      if (!triEdgeSet.has(key)) { triEdgeSet.add(key); te.push(p, q); }
    }
  }
  const triEdges = Uint32Array.from(te);

  return {
    n, pos, normals, tris, structEdges, triEdges, structCount: stA.length,
    spring: { a: Int32Array.from(sa), b: Int32Array.from(sb), L0, w: W, count: ns },
    volume: vol, area,
  };
}

/** 視点群から 3D ソフトボディ用メッシュを生成する */
function buildMesh(views, params) {
  const ext = configureViews(views, params.reliefDepth);
  if (!ext) return null;
  const { hx, hy, hz, freeAxis } = ext;
  const alpha = params.relief;
  const h = 2 / params.resolution;
  const pad = 3;
  const nx = Math.max(3, Math.ceil((2 * hx) / h)) + 2 * pad;
  const ny = Math.max(3, Math.ceil((2 * hy) / h)) + 2 * pad;
  const nz = Math.max(3, Math.ceil((2 * hz) / h)) + 2 * pad;
  const cx = (nx - 1) / 2, cy = (ny - 1) / 2, cz = (nz - 1) / 2;
  const occ = new Float32Array(nx * ny * nz);

  const vs = VIEW_KEYS.filter((k) => views[k]).map((key) => {
    const v = views[key];
    const hAxis = key === 'front' || key === 'back' ? hz : key === 'right' || key === 'left' ? hx : hy;
    let prof = null;
    if (freeAxis) {
      prof = new Float32Array(v.w * v.h);
      for (let i = 0; i < prof.length; i++) {
        const r = v.round[i];
        const pr = Math.sqrt(Math.max(0, 1 - (1 - r) * (1 - r))); // 円弧プロファイル（ピローの膨らみ）
        prof[i] = clamp((1 - alpha) * pr + alpha * v.height[i], 0, 1);
      }
    }
    return { key, v, hAxis, prof, pix: 0, nn: 0 };
  });

  let posVi = null, negVi = null;
  if (freeAxis) {
    const posKey = { z: 'front', x: 'right', y: 'top' }[freeAxis];
    const negKey = { z: 'back', x: 'left', y: null }[freeAxis];
    posVi = vs.find((vi) => vi.key === posKey) || null;
    negVi = negKey ? vs.find((vi) => vi.key === negKey) || null : null;
  }

  const hx2 = hx + h * 0.5, hy2 = hy + h * 0.5, hz2 = hz + h * 0.5;
  let count = 0;
  for (let k = 0; k < nz; k++) {
    const z = (k - cz) * h;
    if (z < -hz2 || z > hz2) continue;
    for (let j = 0; j < ny; j++) {
      const y = (j - cy) * h;
      if (y < -hy2 || y > hy2) continue;
      for (let i = 0; i < nx; i++) {
        const x = (i - cx) * h;
        if (x < -hx2 || x > hx2) continue;

        // 1) 各視点のシルエットの交差（Visual Hull）
        let inside = true;
        for (let q = 0; q < vs.length; q++) {
          const vi = vs[q], v = vi.v;
          projAB(vi.key, x, y, z);
          const ix = Math.floor(v.bbox.cx + _pa * v.ppu);
          const iy = Math.floor(v.bbox.cy + _pb * v.ppu);
          if (ix < 0 || iy < 0 || ix >= v.w || iy >= v.h) { inside = false; break; }
          const pix = iy * v.w + ix;
          if (!v.mask[pix]) { inside = false; break; }
          vi.pix = pix;
          vi.nn = outwardCoord(vi.key, x, y, z);
        }
        if (!inside) continue;

        if (freeAxis) {
          // 2a) 未指定軸：輝度深度 + 距離変換のふくらみで前後へ押し出し
          const thick = (vi) => vi.hAxis * (0.2 + 0.8 * vi.prof[vi.pix]);
          if (posVi && negVi) {
            if (posVi.nn > thick(posVi) || negVi.nn > thick(negVi)) inside = false;
          } else {
            const vi = posVi || negVi;
            if (Math.abs(vi.nn) > thick(vi)) inside = false;
          }
        } else {
          // 2b) 多視点：各視点の擬似深度で手前側を浅く彫刻
          for (let q = 0; q < vs.length; q++) {
            const vi = vs[q];
            const t = vi.hAxis - vi.nn;
            const cd = alpha * CARVE_FRACTION * 2 * vi.hAxis * (1 - vi.v.height[vi.pix]);
            if (t < cd) { inside = false; break; }
          }
        }
        if (inside) { occ[i + nx * (j + ny * k)] = 1; count++; }
      }
    }
  }
  if (count < 8) return null;

  // 占有場を 3x3x3 ボックスフィルタでなめらかな密度場へ
  const bufA = new Float32Array(occ.length), bufB = new Float32Array(occ.length);
  blurAxis(occ, bufA, nx, ny, nz, 0);
  blurAxis(bufA, bufB, nx, ny, nz, 1);
  blurAxis(bufB, bufA, nx, ny, nz, 2);
  for (let i = 0; i < bufA.length; i++) {
    let f = bufA[i] - 0.5;
    if (Math.abs(f) < 1e-6) f = 1e-6;
    bufA[i] = f;
  }

  const nets = surfaceNets(bufA, nx, ny, nz, h);
  if (nets.quads.length === 0) return null;
  const mesh = finalizeMesh(nets.pos, nets.quads);
  mesh.ext = ext;
  mesh.voxels = count;
  return mesh;
}

/** 起動時・画像なしのときに表示する見本モデル（角丸キューブ型のスライム）。符号付き距離場を Surface Nets で抽出。 */
function buildSampleMesh(resolution) {
  const bx = 0.9, by = 0.8, bz = 0.9, rad = 0.34;
  const h = 2 / resolution;
  const pad = 3;
  const nx = Math.ceil((2 * (bx + 0.05)) / h) + 2 * pad;
  const ny = Math.ceil((2 * (by + 0.05)) / h) + 2 * pad;
  const nz = Math.ceil((2 * (bz + 0.05)) / h) + 2 * pad;
  const cx = (nx - 1) / 2, cy = (ny - 1) / 2, cz = (nz - 1) / 2;
  const F = new Float32Array(nx * ny * nz);
  for (let k = 0; k < nz; k++) {
    const z = (k - cz) * h;
    for (let j = 0; j < ny; j++) {
      const y = (j - cy) * h;
      for (let i = 0; i < nx; i++) {
        const x = (i - cx) * h;
        const qx = Math.abs(x) - (bx - rad), qy = Math.abs(y) - (by - rad), qz = Math.abs(z) - (bz - rad);
        const d = Math.hypot(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0))
          + Math.min(Math.max(qx, Math.max(qy, qz)), 0) - rad;
        let f = -d;
        if (Math.abs(f) < 1e-6) f = 1e-6;
        F[i + nx * (j + ny * k)] = f;
      }
    }
  }
  const nets = surfaceNets(F, nx, ny, nz, h);
  const mesh = finalizeMesh(nets.pos, nets.quads);
  mesh.ext = { hx: bx, hy: by, hz: bz, freeAxis: null };
  mesh.isSample = true;
  return mesh;
}

// Wavefront .obj のインポート（自作の軽量パーサー）とソフトボディ化

/**
 * v / vn / vt / f / mtllib / usemtl を読む。
 *  - v x y z r g b の頂点カラーに対応
 *  - f は v, v/vt, v//vn, v/vt/vn と負のインデックスに対応し、多角形は扇状に三角形化する
 */
function parseOBJ(text) {
  const pos = [], vcol = [], vt = [], faces = [], faceVt = [], faceMat = [], mtllibs = [];
  let vn = 0, mat = '', anyVC = false;
  const lines = text.split(/\r\n|\n|\r/);
  for (let li = 0; li < lines.length; li++) {
    const line = lines[li].trim();
    if (!line || line.charCodeAt(0) === 35) continue; // 空行 / コメント
    const tok = line.split(/\s+/);
    const head = tok[0];
    if (head === 'v') {
      const x = parseFloat(tok[1]), y = parseFloat(tok[2]), z = parseFloat(tok[3]);
      pos.push(isFinite(x) ? x : 0, isFinite(y) ? y : 0, isFinite(z) ? z : 0);
      const r = parseFloat(tok[4]), g = parseFloat(tok[5]), b = parseFloat(tok[6]);
      if (tok.length >= 7 && isFinite(r) && isFinite(g) && isFinite(b)) {
        const sc = Math.max(r, g, b) > 1.0001 ? 1 / 255 : 1; // 0-255 表記にも対応
        vcol.push(clamp(r * sc, 0, 1), clamp(g * sc, 0, 1), clamp(b * sc, 0, 1));
        anyVC = true;
      } else vcol.push(NaN, NaN, NaN);
    } else if (head === 'vt') {
      vt.push(parseFloat(tok[1]) || 0, parseFloat(tok[2]) || 0);
    } else if (head === 'vn') {
      vn++; // 法線は読み飛ばし、読み込み後に平滑法線を再計算する
    } else if (head === 'mtllib') {
      for (let k = 1; k < tok.length; k++) mtllibs.push(tok.slice(k).join(' ').split(/[\\/]/).pop());
    } else if (head === 'usemtl') {
      mat = tok.slice(1).join(' ');
    } else if (head === 'f') {
      const idx = [], tix = [];
      const nv = pos.length / 3, nt = vt.length / 2;
      for (let k = 1; k < tok.length; k++) {
        const parts = tok[k].split('/');
        const i = parseInt(parts[0], 10);
        if (!i) continue;
        idx.push(i > 0 ? i - 1 : nv + i);
        const ti = parts[1] ? parseInt(parts[1], 10) : 0;
        tix.push(ti ? (ti > 0 ? ti - 1 : nt + ti) : -1);
      }
      if (idx.length >= 3) { faces.push(idx); faceVt.push(tix); faceMat.push(mat); }
    }
  }
  return { pos: Float64Array.from(pos), vcol: anyVC ? Float32Array.from(vcol) : null, vt: Float64Array.from(vt), faces, faceVt, faceMat, mtllibs, normalsInFile: vn };
}

/** .mtl を読む：newmtl / Kd / map_Kd */
function parseMTL(text) {
  const out = {};
  let cur = null;
  for (const raw of text.split(/\r\n|\n|\r/)) {
    const line = raw.trim();
    if (!line || line.charCodeAt(0) === 35) continue;
    const tok = line.split(/\s+/);
    const key = tok[0].toLowerCase();
    if (key === 'newmtl') { cur = out[tok.slice(1).join(' ')] = { kd: null, map: null }; }
    else if (!cur) continue;
    else if (key === 'kd') {
      const r = parseFloat(tok[1]), g = parseFloat(tok[2]), b = parseFloat(tok[3]);
      if (isFinite(r) && isFinite(g) && isFinite(b)) cur.kd = [clamp(r, 0, 1), clamp(g, 0, 1), clamp(b, 0, 1)];
    } else if (key === 'map_kd') {
      cur.map = tok[tok.length - 1].split(/[\\/]/).pop();
    }
  }
  return out;
}

/*
 * 多形式の3Dモデル読み込み：拡張子から自動判別する。
 * obj（mtl・テクスチャ画像と同時選択） / glb / gltf / stl / ply は自作の軽量パーサー、
 * fbx / dae は Three.js のローダーを必要なときだけ読み込む。
 * どの形式も OBJ と同じ中間表現（pos / vcol / vt / faces / faceVt / faceMat）へそろえる。
 */

const MODEL_EXTS = ['obj', 'glb', 'gltf', 'stl', 'ply', 'fbx', 'dae'];

function fileExt(name) {
  const m = /\.([A-Za-z0-9]+)$/.exec(name || '');
  return m ? m[1].toLowerCase() : '';
}

const toSRGB = (c) => (c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055);

function emptyParsed() {
  return { pos: new Float64Array(0), vcol: null, vt: new Float64Array(0), faces: [], faceVt: [], faceMat: [], mtllibs: [], normalsInFile: 0 };
}

/** STL（バイナリ / ASCII）。CAD 系は Z 上向きが多いため Y 上向きへ回す */
function parseSTL(buf) {
  const dv = new DataView(buf);
  let pos;
  const bin = buf.byteLength >= 84 && 84 + dv.getUint32(80, true) * 50 === buf.byteLength;
  if (bin) {
    const n = dv.getUint32(80, true);
    pos = new Float64Array(n * 9);
    for (let i = 0; i < n; i++) {
      const o = 84 + i * 50 + 12;
      for (let k = 0; k < 9; k++) pos[i * 9 + k] = dv.getFloat32(o + k * 4, true);
    }
  } else {
    const txt = new TextDecoder().decode(buf), re = /vertex\s+(\S+)\s+(\S+)\s+(\S+)/g, a = [];
    let m;
    while ((m = re.exec(txt))) a.push(parseFloat(m[1]), parseFloat(m[2]), parseFloat(m[3]));
    pos = Float64Array.from(a);
  }
  const nt = Math.floor(pos.length / 9);
  pos = pos.subarray(0, nt * 9);
  for (let i = 0; i < nt * 3; i++) { const y = pos[3 * i + 1], z = pos[3 * i + 2]; pos[3 * i + 1] = z; pos[3 * i + 2] = -y; }
  const out = emptyParsed();
  out.pos = Float64Array.from(pos);
  for (let i = 0; i < nt; i++) { out.faces.push([3 * i, 3 * i + 1, 3 * i + 2]); out.faceVt.push([-1, -1, -1]); out.faceMat.push(''); }
  return out;
}

/** PLY（ASCII / バイナリ）。頂点カラー red green blue に対応 */
function parsePLY(buf) {
  const u8 = new Uint8Array(buf);
  const headTxt = new TextDecoder('latin1').decode(u8.subarray(0, Math.min(u8.length, 16384)));
  const em = /end_header\r?\n/.exec(headTxt);
  if (!em) throw new Error('ply header');
  const hdrLen = em.index + em[0].length;
  const els = [];
  let format = 'ascii';
  for (const line of headTxt.slice(0, em.index).split(/\r?\n/)) {
    const t = line.trim().split(/\s+/);
    if (t[0] === 'format') format = t[1];
    else if (t[0] === 'element') els.push({ name: t[1], count: parseInt(t[2], 10), props: [] });
    else if (t[0] === 'property' && els.length) {
      if (t[1] === 'list') els[els.length - 1].props.push({ name: t[4], list: true, ct: t[2], it: t[3] });
      else els[els.length - 1].props.push({ name: t[2], type: t[1] });
    }
  }
  const SZ = { char: 1, int8: 1, uchar: 1, uint8: 1, short: 2, int16: 2, ushort: 2, uint16: 2, int: 4, int32: 4, uint: 4, uint32: 4, float: 4, float32: 4, double: 8, float64: 8 };
  const le = format === 'binary_little_endian';
  const dv = new DataView(buf);
  let off = hdrLen, tokens = null, ti = 0;
  if (format === 'ascii') tokens = new TextDecoder('latin1').decode(u8.subarray(hdrLen)).split(/\s+/).filter(Boolean);
  const read = (type) => {
    if (tokens) return parseFloat(tokens[ti++]);
    let v;
    switch (type) {
      case 'char': case 'int8': v = dv.getInt8(off); break;
      case 'uchar': case 'uint8': v = dv.getUint8(off); break;
      case 'short': case 'int16': v = dv.getInt16(off, le); break;
      case 'ushort': case 'uint16': v = dv.getUint16(off, le); break;
      case 'int': case 'int32': v = dv.getInt32(off, le); break;
      case 'uint': case 'uint32': v = dv.getUint32(off, le); break;
      case 'float': case 'float32': v = dv.getFloat32(off, le); break;
      default: v = dv.getFloat64(off, le);
    }
    off += SZ[type] || 4;
    return v;
  };
  const out = emptyParsed();
  for (const el of els) {
    if (el.name === 'vertex') {
      const n = el.count, pos = new Float64Array(3 * n);
      const idx = (names) => el.props.findIndex((p) => names.indexOf(p.name) >= 0);
      const ix = idx(['x']), iy = idx(['y']), iz = idx(['z']);
      const ir = idx(['red', 'r', 'diffuse_red']), ig = idx(['green', 'g', 'diffuse_green']), ib = idx(['blue', 'b', 'diffuse_blue']);
      const hasC = ir >= 0 && ig >= 0 && ib >= 0;
      const col = hasC ? new Float32Array(3 * n) : null;
      for (let i = 0; i < n; i++) {
        for (let k = 0; k < el.props.length; k++) {
          const pr = el.props[k];
          if (pr.list) { const c = read(pr.ct); for (let q = 0; q < c; q++) read(pr.it); continue; }
          const v = read(pr.type);
          if (k === ix) pos[3 * i] = v; else if (k === iy) pos[3 * i + 1] = v; else if (k === iz) pos[3 * i + 2] = v;
          else if (hasC) {
            const cs = (pr.type === 'float' || pr.type === 'float32' || pr.type === 'double' || pr.type === 'float64') && v <= 1.0001 ? 1 : 1 / 255;
            if (k === ir) col[3 * i] = v * cs; else if (k === ig) col[3 * i + 1] = v * cs; else if (k === ib) col[3 * i + 2] = v * cs;
          }
        }
      }
      out.pos = pos; out.vcol = col;
    } else if (el.name === 'face') {
      for (let i = 0; i < el.count; i++) {
        for (const pr of el.props) {
          if (!pr.list) { read(pr.type); continue; }
          const c = read(pr.ct), f = [];
          for (let q = 0; q < c; q++) f.push(read(pr.it));
          if ((pr.name === 'vertex_indices' || pr.name === 'vertex_index') && f.length >= 3) { out.faces.push(f); out.faceVt.push(f.map(() => -1)); out.faceMat.push(''); }
        }
      }
    } else {
      for (let i = 0; i < el.count; i++) for (const pr of el.props) { if (pr.list) { const c = read(pr.ct); for (let q = 0; q < c; q++) read(pr.it); } else read(pr.type); }
    }
  }
  return out;
}

/* glTF / GLB */

function mat4Mul(a, b) {
  const o = new Float64Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
    o[c * 4 + r] = s;
  }
  return o;
}

function nodeMatrix(n) {
  if (n.matrix && n.matrix.length === 16) return Float64Array.from(n.matrix);
  const t = n.translation || [0, 0, 0], q = n.rotation || [0, 0, 0, 1], s = n.scale || [1, 1, 1];
  const [x, y, z, w] = q;
  return Float64Array.from([
    (1 - 2 * (y * y + z * z)) * s[0], 2 * (x * y + z * w) * s[0], 2 * (x * z - y * w) * s[0], 0,
    2 * (x * y - z * w) * s[1], (1 - 2 * (x * x + z * z)) * s[1], 2 * (y * z + x * w) * s[1], 0,
    2 * (x * z + y * w) * s[2], 2 * (y * z - x * w) * s[2], (1 - 2 * (x * x + y * y)) * s[2], 0,
    t[0], t[1], t[2], 1,
  ]);
}

function readAccessor(json, bufs, idx) {
  const a = json.accessors[idx];
  const nc = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT2: 4, MAT3: 9, MAT4: 16 }[a.type] || 1;
  const out = new Float64Array(a.count * nc);
  if (a.bufferView === undefined) return { data: out, nc };
  const bv = json.bufferViews[a.bufferView];
  const csz = { 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 }[a.componentType];
  const stride = bv.byteStride || csz * nc;
  const base = (bv.byteOffset || 0) + (a.byteOffset || 0);
  const dv = new DataView(bufs[bv.buffer]);
  const norm = a.normalized ? { 5120: 127, 5121: 255, 5122: 32767, 5123: 65535 }[a.componentType] : 0;
  for (let i = 0; i < a.count; i++) {
    for (let k = 0; k < nc; k++) {
      const o = base + i * stride + k * csz;
      let v;
      switch (a.componentType) {
        case 5120: v = dv.getInt8(o); break;
        case 5121: v = dv.getUint8(o); break;
        case 5122: v = dv.getInt16(o, true); break;
        case 5123: v = dv.getUint16(o, true); break;
        case 5125: v = dv.getUint32(o, true); break;
        default: v = dv.getFloat32(o, true);
      }
      out[i * nc + k] = norm ? Math.max(v / norm, -1) : v;
    }
  }
  return { data: out, nc };
}

function dataUriToBytes(uri) {
  const m = /^data:([^;,]*)(;base64)?,(.*)$/s.exec(uri);
  if (!m) return null;
  if (m[2]) {
    const bin = atob(m[3]), u = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return { bytes: u, mime: m[1] };
  }
  return { bytes: new TextEncoder().encode(decodeURIComponent(m[3])), mime: m[1] };
}

/** glTF / GLB。外部の .bin・画像は aux（小文字のファイル名 → File）から探す。足りなければ missing を付けて投げる */
async function parseGLTF(file, aux) {
  const buf = await file.arrayBuffer();
  let json, glbBin = null;
  const magic = buf.byteLength > 12 ? new DataView(buf).getUint32(0, true) : 0;
  if (magic === 0x46546C67) {
    const dv = new DataView(buf);
    let o = 12;
    while (o + 8 <= buf.byteLength) {
      const len = dv.getUint32(o, true), type = dv.getUint32(o + 4, true);
      const body = buf.slice(o + 8, o + 8 + len);
      if (type === 0x4E4F534A) json = JSON.parse(new TextDecoder().decode(body));
      else if (type === 0x004E4942 && !glbBin) glbBin = body;
      o += 8 + len + ((4 - (len % 4)) % 4);
    }
  } else json = JSON.parse(new TextDecoder().decode(buf));
  if (!json || !json.meshes) throw new Error('gltf');
  if ((json.extensionsRequired || []).some((e) => /draco|meshopt/i.test(e))) throw new Error('gltf compressed');

  const fetchUri = async (uri) => {
    const d = dataUriToBytes(uri);
    if (d) return d.bytes.buffer.slice(d.bytes.byteOffset, d.bytes.byteOffset + d.bytes.byteLength);
    const name = decodeURIComponent(uri).split(/[\\/]/).pop();
    const f = aux.get(name.toLowerCase());
    if (!f) { const err = new Error('missing'); err.missing = name; throw err; }
    return f.arrayBuffer();
  };
  const bufs = [];
  for (let i = 0; i < (json.buffers || []).length; i++) {
    const b = json.buffers[i];
    bufs.push(b.uri ? await fetchUri(b.uri) : glbBin);
  }

  // マテリアルとテクスチャ
  const textures = {}, mtls = {};
  const imgName = (i) => 'gltf-img-' + i;
  const imgDone = {};
  const loadImg = async (ti) => {
    const tex = json.textures && json.textures[ti];
    if (!tex || tex.source === undefined) return null;
    const ii = tex.source;
    if (imgDone[ii] !== undefined) return imgDone[ii];
    imgDone[ii] = null;
    const im = json.images[ii];
    let bytes = null, mime = im.mimeType || 'image/png';
    if (im.bufferView !== undefined) {
      const bv = json.bufferViews[im.bufferView];
      bytes = new Uint8Array(bufs[bv.buffer], bv.byteOffset || 0, bv.byteLength);
    } else if (im.uri) {
      const ab = await fetchUri(im.uri);
      bytes = new Uint8Array(ab);
    }
    if (bytes) {
      try { textures[imgName(ii)] = await decodeTexture(new Blob([bytes], { type: mime })); imgDone[ii] = imgName(ii); } catch (e) { /* 色は Kd で代用 */ }
    }
    return imgDone[ii];
  };
  for (let mi = 0; mi < (json.materials || []).length; mi++) {
    const pbr = json.materials[mi].pbrMetallicRoughness || {};
    const f = pbr.baseColorFactor || [1, 1, 1, 1];
    mtls['m' + mi] = { kd: [toSRGB(f[0]), toSRGB(f[1]), toSRGB(f[2])], map: null };
    if (pbr.baseColorTexture) mtls['m' + mi].map = await loadImg(pbr.baseColorTexture.index);
  }

  // シーンのノードを辿って、ワールド変換つきでプリミティブを集める
  const pos = [], col = [], vt = [], faces = [], faceVt = [], faceMat = [];
  let anyCol = false;
  const addPrim = (prim, M) => {
    const mode = prim.mode === undefined ? 4 : prim.mode;
    if (mode < 4 || prim.attributes.POSITION === undefined) return;
    const P = readAccessor(json, bufs, prim.attributes.POSITION).data;
    const n = P.length / 3, base = pos.length / 3;
    for (let i = 0; i < n; i++) {
      const x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2];
      pos.push(M[0] * x + M[4] * y + M[8] * z + M[12], M[1] * x + M[5] * y + M[9] * z + M[13], M[2] * x + M[6] * y + M[10] * z + M[14]);
    }
    let hasUV = false;
    if (prim.attributes.TEXCOORD_0 !== undefined) {
      const T = readAccessor(json, bufs, prim.attributes.TEXCOORD_0).data;
      for (let i = 0; i < n; i++) vt.push(T[2 * i], 1 - T[2 * i + 1]);   // glTF は左上原点なので OBJ の向きへそろえる
      hasUV = true;
    } else for (let i = 0; i < n; i++) vt.push(0, 0);
    if (prim.attributes.COLOR_0 !== undefined) {
      const C = readAccessor(json, bufs, prim.attributes.COLOR_0);
      for (let i = 0; i < n; i++) col.push(toSRGB(C.data[C.nc * i]), toSRGB(C.data[C.nc * i + 1]), toSRGB(C.data[C.nc * i + 2]));
      anyCol = true;
    } else for (let i = 0; i < n; i++) col.push(NaN, NaN, NaN);
    let idx;
    if (prim.indices !== undefined) idx = Array.from(readAccessor(json, bufs, prim.indices).data);
    else { idx = new Array(n); for (let i = 0; i < n; i++) idx[i] = i; }
    const tri = [];
    if (mode === 4) for (let i = 0; i + 2 < idx.length; i += 3) tri.push([idx[i], idx[i + 1], idx[i + 2]]);
    else if (mode === 5) for (let i = 0; i + 2 < idx.length; i++) tri.push(i % 2 ? [idx[i + 1], idx[i], idx[i + 2]] : [idx[i], idx[i + 1], idx[i + 2]]);
    else if (mode === 6) for (let i = 1; i + 1 < idx.length; i++) tri.push([idx[0], idx[i], idx[i + 1]]);
    const mname = prim.material !== undefined && mtls['m' + prim.material] ? 'm' + prim.material : '';
    for (const t of tri) {
      const f = [base + t[0], base + t[1], base + t[2]];
      faces.push(f); faceVt.push(hasUV ? f.slice() : [-1, -1, -1]); faceMat.push(mname);
    }
  };
  const visit = (ni, parent) => {
    const nd = json.nodes[ni];
    const M = mat4Mul(parent, nodeMatrix(nd));
    if (nd.mesh !== undefined) for (const pr of json.meshes[nd.mesh].primitives) addPrim(pr, M);
    for (const c of nd.children || []) visit(c, M);
  };
  const I = Float64Array.from([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  if (json.nodes && json.nodes.length) {
    let roots = [];
    const sc = json.scenes && json.scenes[json.scene || 0];
    if (sc && sc.nodes) roots = sc.nodes;
    else { const ch = new Set(); json.nodes.forEach((n) => (n.children || []).forEach((c) => ch.add(c))); json.nodes.forEach((_, i) => { if (!ch.has(i)) roots.push(i); }); }
    for (const r of roots) visit(r, I);
  } else for (const m of json.meshes) for (const pr of m.primitives) addPrim(pr, I);

  const out = emptyParsed();
  out.pos = Float64Array.from(pos);
  out.vcol = anyCol ? Float32Array.from(col) : null;
  out.vt = Float64Array.from(vt);
  out.faces = faces; out.faceVt = faceVt; out.faceMat = faceMat;
  return { parsed: out, mtls, textures };
}

/* fbx / dae：Three.js のローダーを必要なときだけ読み込む */

async function parseViaThree(file, ext) {
  const buf = await file.arrayBuffer();
  let root;
  if (ext === 'fbx') {
    const mod = await import('three/addons/loaders/FBXLoader.js');
    root = new mod.FBXLoader().parse(buf, '');
  } else {
    const mod = await import('three/addons/loaders/ColladaLoader.js');
    root = new mod.ColladaLoader().parse(new TextDecoder().decode(buf), '').scene;
  }
  root.updateMatrixWorld(true);
  const pos = [], col = [], vt = [], faces = [], faceVt = [], faceMat = [], mtls = {}, textures = {};
  let anyCol = false, mc = 0;
  const imgToTexture = (img) => {
    const w = img.width || img.videoWidth, h = img.height || img.videoHeight;
    if (!w || !h) return null;
    const sc = Math.min(1, 1024 / Math.max(w, h));
    const c = document.createElement('canvas');
    c.width = Math.max(1, Math.round(w * sc)); c.height = Math.max(1, Math.round(h * sc));
    const g = c.getContext('2d', { willReadFrequently: true });
    g.drawImage(img, 0, 0, c.width, c.height);
    return { w: c.width, h: c.height, data: g.getImageData(0, 0, c.width, c.height).data };
  };
  const meshes = [];
  root.traverse((o) => { if (o.isMesh && o.geometry && o.geometry.attributes.position) meshes.push(o); });
  for (const mesh of meshes) {
    const g = mesh.geometry, M = mesh.matrixWorld.elements;
    const P = g.attributes.position, n = P.count, base = pos.length / 3;
    for (let i = 0; i < n; i++) {
      const x = P.getX(i), y = P.getY(i), z = P.getZ(i);
      pos.push(M[0] * x + M[4] * y + M[8] * z + M[12], M[1] * x + M[5] * y + M[9] * z + M[13], M[2] * x + M[6] * y + M[10] * z + M[14]);
    }
    const UV = g.attributes.uv, C = g.attributes.color;
    for (let i = 0; i < n; i++) vt.push(UV ? UV.getX(i) : 0, UV ? UV.getY(i) : 0);
    for (let i = 0; i < n; i++) col.push(C ? toSRGB(C.getX(i)) : NaN, C ? toSRGB(C.getY(i)) : NaN, C ? toSRGB(C.getZ(i)) : NaN);
    if (C) anyCol = true;
    const idx = g.index ? Array.from(g.index.array) : Array.from({ length: n }, (_, i) => i);
    const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    const names = mats.map((m) => {
      const name = 'mat' + (mc++);
      const e = { kd: m && m.color ? [toSRGB(m.color.r), toSRGB(m.color.g), toSRGB(m.color.b)] : null, map: null };
      if (m && m.map && m.map.image) {
        const t = imgToTexture(m.map.image);
        if (t) { textures['tex-' + name] = t; e.map = 'tex-' + name; }
      }
      mtls[name] = e;
      return name;
    });
    const groups = g.groups && g.groups.length ? g.groups : [{ start: 0, count: idx.length, materialIndex: 0 }];
    for (const gr of groups) {
      const mname = names[Math.min(gr.materialIndex || 0, names.length - 1)];
      const end = Math.min(idx.length, gr.start + gr.count);
      for (let i = gr.start; i + 2 < end; i += 3) {
        const f = [base + idx[i], base + idx[i + 1], base + idx[i + 2]];
        faces.push(f); faceVt.push(UV ? f.slice() : [-1, -1, -1]); faceMat.push(mname);
      }
    }
  }
  const out = emptyParsed();
  out.pos = Float64Array.from(pos);
  out.vcol = anyCol ? Float32Array.from(col) : null;
  out.vt = Float64Array.from(vt);
  out.faces = faces; out.faceVt = faceVt; out.faceMat = faceMat;
  return { parsed: out, mtls, textures };
}

/** 拡張子から形式を判別して読み込む。obj 以外は { parsed, mtls, textures } を返す */
async function parseModelFile(file, aux) {
  const ext = fileExt(file.name);
  if (ext === 'obj') return { parsed: parseOBJ(await file.text()), mtls: null, textures: null };
  if (ext === 'glb' || ext === 'gltf') return parseGLTF(file, aux);
  if (ext === 'stl') return { parsed: parseSTL(await file.arrayBuffer()), mtls: {}, textures: {} };
  if (ext === 'ply') return { parsed: parsePLY(await file.arrayBuffer()), mtls: {}, textures: {} };
  return parseViaThree(file, ext);
}

/** 三角形リスト（flat 配列）から未使用頂点を除いて詰める */
function compactMesh(pos, tris, cols) {
  const n = pos.length / 3;
  const remap = new Int32Array(n).fill(-1);
  let m = 0;
  for (let i = 0; i < tris.length; i++) if (remap[tris[i]] < 0) remap[tris[i]] = m++;
  const out = new Float64Array(3 * m);
  for (let i = 0; i < n; i++) {
    const j = remap[i];
    if (j >= 0) { out[3 * j] = pos[3 * i]; out[3 * j + 1] = pos[3 * i + 1]; out[3 * j + 2] = pos[3 * i + 2]; }
  }
  let cout = null;
  if (cols) {
    cout = new Float64Array(3 * m);
    for (let i = 0; i < n; i++) {
      const j = remap[i];
      if (j >= 0) { cout[3 * j] = cols[3 * i]; cout[3 * j + 1] = cols[3 * i + 1]; cout[3 * j + 2] = cols[3 * i + 2]; }
    }
  }
  const t2 = new Int32Array(tris.length);
  for (let i = 0; i < tris.length; i++) t2[i] = remap[tris[i]];
  return { pos: out, tris: t2, cols: cout };
}

/** 頂点クラスタリングによる間引き（正規化済み座標 [-1,1] 用） */
function decimateByClustering(pos, tris, target, cols) {
  const n = pos.length / 3;
  const cellOf = (R) => {
    const ids = new Int32Array(n);
    const dict = new Map();
    for (let i = 0; i < n; i++) {
      const ix = clamp(Math.floor(((pos[3 * i] + 1) / 2) * R), 0, R - 1);
      const iy = clamp(Math.floor(((pos[3 * i + 1] + 1) / 2) * R), 0, R - 1);
      const iz = clamp(Math.floor(((pos[3 * i + 2] + 1) / 2) * R), 0, R - 1);
      const key = (ix * R + iy) * R + iz;
      let id = dict.get(key);
      if (id === undefined) { id = dict.size; dict.set(key, id); }
      ids[i] = id;
    }
    return { ids, count: dict.size };
  };
  let best = cellOf(16);
  for (const R of [24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 256]) {
    const c = cellOf(R);
    if (c.count > target) break;
    best = c;
  }
  const { ids, count } = best;
  const acc = new Float64Array(3 * count), cnt = new Int32Array(count);
  const cacc = cols ? new Float64Array(3 * count) : null;
  for (let i = 0; i < n; i++) {
    const c = ids[i];
    acc[3 * c] += pos[3 * i]; acc[3 * c + 1] += pos[3 * i + 1]; acc[3 * c + 2] += pos[3 * i + 2];
    if (cols) { cacc[3 * c] += cols[3 * i]; cacc[3 * c + 1] += cols[3 * i + 1]; cacc[3 * c + 2] += cols[3 * i + 2]; }
    cnt[c]++;
  }
  for (let c = 0; c < count; c++) {
    acc[3 * c] /= cnt[c]; acc[3 * c + 1] /= cnt[c]; acc[3 * c + 2] /= cnt[c];
    if (cols) { cacc[3 * c] /= cnt[c]; cacc[3 * c + 1] /= cnt[c]; cacc[3 * c + 2] /= cnt[c]; }
  }
  const seen = new Set(), out = [];
  for (let t = 0; t < tris.length; t += 3) {
    const a = ids[tris[t]], b = ids[tris[t + 1]], c = ids[tris[t + 2]];
    if (a === b || b === c || a === c) continue;
    const lo = Math.min(a, b, c), hi = Math.max(a, b, c), mid = a + b + c - lo - hi;
    const key = `${lo},${mid},${hi}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(a, b, c);
  }
  return compactMesh(acc, Int32Array.from(out), cacc);
}

/**
 * パース済み OBJ からソフトボディ用メッシュを作る。
 *  1) 同一位置の頂点を溶接 -> 中心を原点へ・最大幅を 2.0 に正規化
 *  2) 頂点が多すぎる場合は間引き
 *  3) バネ網：構造（全エッジ）/ せん断（隣接三角形の対向頂点）/ 内部支持（内側へレイを飛ばし反対側の頂点を結ぶ）
 */
function buildObjMesh(parsed, material) {
  const mtls = (material && material.mtls) || {}, textures = (material && material.textures) || {};
  const raw = parsed.pos, nRaw = raw.length / 3;
  if (nRaw < 3 || !parsed.faces.length) return null;

  const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < nRaw; i++) for (let a = 0; a < 3; a++) {
    const v = raw[3 * i + a];
    if (v < mn[a]) mn[a] = v;
    if (v > mx[a]) mx[a] = v;
  }
  const extent = Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]);
  if (!(extent > 0)) return null;
  const scale = 2 / extent;
  const cen = [(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2];

  // 溶接 + 正規化
  const tol = extent * 1e-5;
  const weldMap = new Int32Array(nRaw);
  const dict = new Map();
  const wp = [];
  for (let i = 0; i < nRaw; i++) {
    const key = `${Math.round((raw[3 * i] - mn[0]) / tol)},${Math.round((raw[3 * i + 1] - mn[1]) / tol)},${Math.round((raw[3 * i + 2] - mn[2]) / tol)}`;
    let id = dict.get(key);
    if (id === undefined) {
      id = wp.length / 3;
      dict.set(key, id);
      wp.push((raw[3 * i] - cen[0]) * scale, (raw[3 * i + 1] - cen[1]) * scale, (raw[3 * i + 2] - cen[2]) * scale);
    }
    weldMap[i] = id;
  }
  // 色：頂点カラー / マテリアルの Kd / map_Kd テクスチャ（頂点位置の UV で取得）を、溶接後の頂点へ平均して集める
  const nW = wp.length / 3;
  const csum = new Float64Array(3 * nW), ccnt = new Int32Array(nW);
  const cornerColor = (fi, f, ci, out) => {
    const m = mtls[parsed.faceMat[fi]];
    if (m && m.map) {
      const tex = textures[m.map.toLowerCase()];
      const ti = parsed.faceVt[fi][ci];
      if (tex && ti >= 0 && ti * 2 + 1 < parsed.vt.length) {
        const u = parsed.vt[2 * ti], v = parsed.vt[2 * ti + 1];
        const px = clamp(Math.floor((u - Math.floor(u)) * tex.w), 0, tex.w - 1);
        const py = clamp(Math.floor((1 - (v - Math.floor(v))) * tex.h), 0, tex.h - 1);
        const o = (py * tex.w + px) * 4;
        if (tex.data[o + 3] >= 128) {   // 透明なテクセルは使わず、頂点カラー / Kd へ回す
          out[0] = tex.data[o] / 255; out[1] = tex.data[o + 1] / 255; out[2] = tex.data[o + 2] / 255;
          return true;
        }
      }
    }
    if (parsed.vcol) { // 頂点カラー（v x y z r g b）
      const o = 3 * f[ci];
      if (!isNaN(parsed.vcol[o])) { out[0] = parsed.vcol[o]; out[1] = parsed.vcol[o + 1]; out[2] = parsed.vcol[o + 2]; return true; }
    }
    if (m && m.kd) { out[0] = m.kd[0]; out[1] = m.kd[1]; out[2] = m.kd[2]; return true; } // .mtl の拡散色 Kd
    return false;
  };
  // 三角形化（扇状）
  const tl = [];
  const cc = [0, 0, 0];
  let anyColor = false;
  for (let fi = 0; fi < parsed.faces.length; fi++) {
    const f = parsed.faces[fi];
    let ok = true;
    for (const i of f) if (i < 0 || i >= nRaw) { ok = false; break; }
    if (!ok) continue;
    for (let k = 1; k < f.length - 1; k++) {
      const a = weldMap[f[0]], b = weldMap[f[k]], c = weldMap[f[k + 1]];
      if (a === b || b === c || a === c) continue;
      tl.push(a, b, c);
      const cors = [[a, 0], [b, k], [c, k + 1]];
      for (const [vid, ci] of cors) {
        if (cornerColor(fi, f, ci, cc)) { csum[3 * vid] += cc[0]; csum[3 * vid + 1] += cc[1]; csum[3 * vid + 2] += cc[2]; ccnt[vid]++; anyColor = true; }
      }
    }
  }
  if (!tl.length) return null;
  let wcols = null;
  if (anyColor) {
    wcols = new Float64Array(3 * nW);
    for (let i = 0; i < nW; i++) {
      if (ccnt[i]) { wcols[3 * i] = csum[3 * i] / ccnt[i]; wcols[3 * i + 1] = csum[3 * i + 1] / ccnt[i]; wcols[3 * i + 2] = csum[3 * i + 2] / ccnt[i]; }
      else { wcols[3 * i] = 1.0; wcols[3 * i + 1] = 0.28; wcols[3 * i + 2] = 0.65; } // 色情報の無い頂点はピンク色
    }
  }
  let { pos, tris, cols } = compactMesh(Float64Array.from(wp), Int32Array.from(tl), wcols);
  const vertsIn = pos.length / 3;
  let decimated = false;
  if (vertsIn > MODEL_HARD_VERTS) {
    ({ pos, tris, cols } = decimateByClustering(pos, tris, MODEL_TARGET_VERTS, cols));
    decimated = true;
    if (!tris.length) return null;
  }
  const n = pos.length / 3;
  const keyOf = (a, b) => (a < b ? a * n + b : b * n + a);

  // エッジ表と閉曲面判定
  const edgeCount = new Map();
  const stA = [], stB = [];
  for (let t = 0; t < tris.length; t += 3) {
    for (let e = 0; e < 3; e++) {
      const a = tris[t + e], b = tris[t + (e + 1) % 3];
      const key = keyOf(a, b);
      const c = edgeCount.get(key);
      if (c === undefined) { edgeCount.set(key, 1); stA.push(a); stB.push(b); } else edgeCount.set(key, c + 1);
    }
  }
  let closed = true;
  for (const c of edgeCount.values()) if (c !== 2) { closed = false; break; }

  // 符号付き体積 -> 外向きにそろえる
  const measure = () => {
    let vol = 0, area = 0;
    for (let t = 0; t < tris.length; t += 3) {
      const a = 3 * tris[t], b = 3 * tris[t + 1], c = 3 * tris[t + 2];
      const bcx = pos[b + 1] * pos[c + 2] - pos[b + 2] * pos[c + 1];
      const bcy = pos[b + 2] * pos[c] - pos[b] * pos[c + 2];
      const bcz = pos[b] * pos[c + 1] - pos[b + 1] * pos[c];
      vol += (pos[a] * bcx + pos[a + 1] * bcy + pos[a + 2] * bcz) / 6;
      const e1x = pos[b] - pos[a], e1y = pos[b + 1] - pos[a + 1], e1z = pos[b + 2] - pos[a + 2];
      const e2x = pos[c] - pos[a], e2y = pos[c + 1] - pos[a + 1], e2z = pos[c + 2] - pos[a + 2];
      const cx = e1y * e2z - e1z * e2y, cy = e1z * e2x - e1x * e2z, cz = e1x * e2y - e1y * e2x;
      area += 0.5 * Math.sqrt(cx * cx + cy * cy + cz * cz);
    }
    return { vol, area };
  };
  let { vol, area } = measure();
  if (vol < 0) {
    for (let t = 0; t < tris.length; t += 3) { const s = tris[t + 1]; tris[t + 1] = tris[t + 2]; tris[t + 2] = s; }
    vol = -vol;
  }
  if (!closed || vol < 1e-6) vol = 0; // 開いたメッシュでは体積保持を行わない

  // 頂点法線
  const normals = new Float64Array(3 * n);
  for (let t = 0; t < tris.length; t += 3) {
    const a = 3 * tris[t], b = 3 * tris[t + 1], c = 3 * tris[t + 2];
    const e1x = pos[b] - pos[a], e1y = pos[b + 1] - pos[a + 1], e1z = pos[b + 2] - pos[a + 2];
    const e2x = pos[c] - pos[a], e2y = pos[c + 1] - pos[a + 1], e2z = pos[c + 2] - pos[a + 2];
    const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
    for (const o of [a, b, c]) { normals[o] += nx; normals[o + 1] += ny; normals[o + 2] += nz; }
  }
  for (let i = 0; i < n; i++) {
    const o = 3 * i;
    const l = Math.hypot(normals[o], normals[o + 1], normals[o + 2]) || 1;
    normals[o] /= l; normals[o + 1] /= l; normals[o + 2] /= l;
  }

  // バネ網
  const sa = [], sb = [], st = [];
  for (let i = 0; i < stA.length; i++) { sa.push(stA[i]); sb.push(stB[i]); st.push(0); }
  const have = new Set(edgeCount.keys());
  // せん断：エッジを共有する 2 枚の三角形の対向頂点どうし
  const opp = new Map();
  for (let t = 0; t < tris.length; t += 3) {
    for (let e = 0; e < 3; e++) {
      const a = tris[t + e], b = tris[t + (e + 1) % 3], o = tris[t + (e + 2) % 3];
      const key = keyOf(a, b);
      const prev = opp.get(key);
      if (prev === undefined) opp.set(key, o);
      else {
        const k2 = keyOf(prev, o);
        if (prev !== o && !have.has(k2)) { have.add(k2); sa.push(prev); sb.push(o); st.push(1); }
      }
    }
  }
  // 内部支持：各頂点から内向き法線方向にレイを飛ばし、反対側の表面の最寄り頂点と結ぶ（厚みと体積を保つ）
  let supports = 0;
  const M = Math.min(n, 1500), stride = n / M;
  const nt = tris.length / 3;
  for (let m = 0; m < M; m++) {
    const i = Math.floor(m * stride);
    const ox = pos[3 * i], oy = pos[3 * i + 1], oz = pos[3 * i + 2];
    const dx = -normals[3 * i], dy = -normals[3 * i + 1], dz = -normals[3 * i + 2];
    let best = Infinity, hit = -1;
    for (let t = 0; t < nt; t++) {
      const ia = tris[3 * t], ib = tris[3 * t + 1], ic = tris[3 * t + 2];
      if (ia === i || ib === i || ic === i) continue;
      const ax = pos[3 * ia], ay = pos[3 * ia + 1], az = pos[3 * ia + 2];
      const e1x = pos[3 * ib] - ax, e1y = pos[3 * ib + 1] - ay, e1z = pos[3 * ib + 2] - az;
      const e2x = pos[3 * ic] - ax, e2y = pos[3 * ic + 1] - ay, e2z = pos[3 * ic + 2] - az;
      const px = dy * e2z - dz * e2y, py = dz * e2x - dx * e2z, pz = dx * e2y - dy * e2x;
      const det = e1x * px + e1y * py + e1z * pz;
      if (det > -1e-12 && det < 1e-12) continue;
      const inv = 1 / det;
      const tx = ox - ax, ty = oy - ay, tz = oz - az;
      const u = (tx * px + ty * py + tz * pz) * inv;
      if (u < 0 || u > 1) continue;
      const qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x;
      const v = (dx * qx + dy * qy + dz * qz) * inv;
      if (v < 0 || u + v > 1) continue;
      const tt = (e2x * qx + e2y * qy + e2z * qz) * inv;
      if (tt > 1e-4 && tt < best) { best = tt; hit = t; }
    }
    if (hit < 0) continue;
    const hx = ox + dx * best, hy = oy + dy * best, hz = oz + dz * best;
    let j = -1, bd = Infinity;
    for (let k = 0; k < 3; k++) {
      const v = tris[3 * hit + k];
      const d = (pos[3 * v] - hx) ** 2 + (pos[3 * v + 1] - hy) ** 2 + (pos[3 * v + 2] - hz) ** 2;
      if (d < bd) { bd = d; j = v; }
    }
    if (j < 0 || j === i) continue;
    const key = keyOf(i, j);
    if (have.has(key)) continue;
    if (Math.hypot(pos[3 * i] - pos[3 * j], pos[3 * i + 1] - pos[3 * j + 1], pos[3 * i + 2] - pos[3 * j + 2]) < 0.05) continue;
    have.add(key);
    sa.push(i); sb.push(j); st.push(3);
    supports++;
  }

  const ns = sa.length;
  const L0 = new Float64Array(ns), W = new Float64Array(ns);
  for (let s = 0; s < ns; s++) {
    const a = 3 * sa[s], b = 3 * sb[s];
    L0[s] = Math.hypot(pos[a] - pos[b], pos[a + 1] - pos[b + 1], pos[a + 2] - pos[b + 2]);
    W[s] = SPRING_WEIGHT[st[s]];
  }
  const structEdges = new Uint32Array(stA.length * 2);
  for (let i = 0; i < stA.length; i++) { structEdges[2 * i] = stA[i]; structEdges[2 * i + 1] = stB[i]; }

  return {
    n, pos, normals, tris, structEdges, triEdges: structEdges, structCount: stA.length,
    spring: { a: Int32Array.from(sa), b: Int32Array.from(sb), L0, w: W, count: ns },
    volume: vol, area,
    ext: { hx: (mx[0] - mn[0]) * scale / 2, hy: (mx[1] - mn[1]) * scale / 2, hz: (mx[2] - mn[2]) * scale / 2, freeAxis: null },
    isModel: true,
    colors: cols ? Float32Array.from(cols) : null,
    info: { vertsIn, verts: n, tris: tris.length / 3, springs: ns, supports, closed, decimated, colored: !!cols },
  };
}

/** 各頂点の初期位置・法線から、投影テクスチャの色を CPU で評価（シェーダーと同一の重み） */
function sampleColor(views, px, py, pz, nx, ny, nz, out) {
  let wr = 0, wg = 0, wb = 0, ws = 0;
  for (const key of VIEW_KEYS) {
    const v = views[key];
    if (!v) continue;
    const nd = outwardCoord(key, nx, ny, nz);
    const w = Math.pow(Math.max(nd, 0), 3) + 0.01;
    projAB(key, px, py, pz);
    const ix = clamp(Math.floor(v.bbox.cx + _pa * v.ppu), 0, v.w - 1);
    const iy = clamp(Math.floor(v.bbox.cy + _pb * v.ppu), 0, v.h - 1);
    const o = (iy * v.w + ix) * 4;
    wr += w * v.tex[o]; wg += w * v.tex[o + 1]; wb += w * v.tex[o + 2];
    ws += w;
  }
  if (ws <= 0) { out[0] = out[1] = out[2] = 128; return out; }
  out[0] = wr / ws; out[1] = wg / ws; out[2] = wb / ws;
  return out;
}

/* ---------------------------------------------------------------------
 * ソフトボディ物理エンジン (SoftBody クラス)
 *
 * 【全体の支配方程式】
 *   全ポテンシャルエネルギー E = E_spring（バネ） + E_restore（復元） + U_vol（体積保持）
 *   保存力 F_conservative = -∇E （エネルギーが低くなる方向へ働く力）
 *   ニュートンの運動方程式: m * x'' = -∇E - c * x' + m * g + F_grab
 *   ※計算をシンプルにするため、各頂点の質量は m = 1 と定義している。
 *   ※時間発展には「半陰的オイラー法 (Semi-Implicit Euler)」を使用し、数値爆発を防ぐ。
 * ------------------------------------------------------------------- */

class SoftBody {
  constructor(mesh, params) {
    this.P = params;
    const n = (this.n = mesh.n);           // 頂点（質点）の総数
    this.rest = Float64Array.from(mesh.pos); // 初期形状の座標 x0（復元バネの基準）
    this.x = Float64Array.from(mesh.pos);    // 現在の頂点座標 x(t) = (x, y, z)
    this.v = new Float64Array(3 * n);        // 現在の頂点速度 v(t) = (vx, vy, vz)
    this.grad = new Float64Array(3 * n);     // エネルギー勾配 ∇E（力 F の逆ベクトル：F = -∇E）

    // バネ網のデータ（sa[s] と sb[s] の頂点間を自然長 sL[s]・倍率 sW[s] のバネで結ぶ）
    this.sa = mesh.spring.a;
    this.sb = mesh.spring.b;
    this.sL = mesh.spring.L0;
    this.sW = mesh.spring.w;

    this.tris = mesh.tris;                 // 三角形ポリゴンの頂点インデックス（体積計算用）
    this.V0 = mesh.volume;                 // 初期体積 V0
    this.V = mesh.volume;                  // 現在の体積 V(t)
    this.aMean = mesh.area / n;            // 1頂点あたりの平均表面積（体積圧力のスケーリング用）
    this.grabForce = [0, 0, 0];            // マウス把持力 F_grab のベクトル
    this.floorY = -Infinity;               // 床の Y 座標
    this.free = false;                     // false: 固定モード / true: 重力モード（平行移動を許可）
    this.restitution = 0.35;               // 床との反発係数 e (0: 跳ねない 〜 1: 完全弾性衝突)

    // ネットワーク表示用のひずみ（伸び率）計算のための隣接バネ数カウント
    this.structCount = mesh.structCount;
    this.strainCnt = new Float32Array(n);
    for (let s = 0; s < this.structCount; s++) { this.strainCnt[this.sa[s]]++; this.strainCnt[this.sb[s]]++; }
    this.initSpectrum();
  }

  /** 数値発散などで形状が壊れたときに、初期座標 x0 と速度 0 にリセットする */
  reset() {
    this.x.set(this.rest);
    this.v.fill(0);
  }

  /**
   * 頂点ごとのひずみ（伸び率）を計算 (0.0 〜 1.0)
   * 【数式】ひずみ ε = |L - L0| / L0 = |L / L0 - 1|
   */
  computeStrain(out) {
    const { x, sa, sb, sL, structCount, strainCnt } = this;
    out.fill(0);
    for (let s = 0; s < structCount; s++) {
      const i = sa[s] * 3, j = sb[s] * 3;
      const dx = x[i] - x[j], dy = x[i + 1] - x[j + 1], dz = x[i + 2] - x[j + 2];
      // 現在の長さ L = √(dx² + dy² + dz²) と自然長 L0 の比率からひずみを算出
      const a = Math.abs(Math.sqrt(dx * dx + dy * dy + dz * dz) / sL[s] - 1);
      out[sa[s]] += a;
      out[sb[s]] += a;
    }
    for (let i = 0; i < out.length; i++) out[i] = clamp((out[i] / (strainCnt[i] || 1)) * 5, 0, 1);
  }

  /**
   * エネルギー勾配 ∇E を計算して this.grad に格納する（wantE が true なら全エネルギー E も返す）
   * 【関係式】力 F = -∇E なので、grad には「力と逆向きのベクトル（引き戻そうとする方向）」が入る
   */
  gradient(wantE) {
    const P = this.P;
    const k = P.k;             // 剛性係数 k
    const kr = k * P.rho;      // 形状復元バネ定数 kr = ρ * k
    const kv = k * P.beta;     // 体積保持定数 kv = β * k
    const { n, x, rest, grad, sa, sb, sL, sW, tris } = this;
    grad.fill(0);
    let E = 0;

    // -----------------------------------------------------------------
    // 1. 構造・せん断・曲げ・内部支持バネ（フックの法則）
    // 【エネルギー】 E_spring = ½ * ks * (L - L0)²
    // 【勾配（力）】 ∇E = ks * (L - L0) * (x_i - x_j) / L
    // -----------------------------------------------------------------
    const ns = sa.length;
    for (let s = 0; s < ns; s++) {
      const i = sa[s] * 3, j = sb[s] * 3;
      const dx = x[i] - x[j], dy = x[i + 1] - x[j + 1], dz = x[i + 2] - x[j + 2];
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-12; // 現在のバネの長さ L（0割り防止の微小値付き）
      const dl = len - sL[s];                                     // 自然長からの伸び ΔL = L - L0
      const ks = k * sW[s];                                       // バネ種別ごとの実効剛性 ks
      const f = (ks * dl) / len;                                  // スカラー力 / 距離 L
      const gx = f * dx, gy = f * dy, gz = f * dz;
      // 作用・反作用の法則：頂点 i と頂点 j に逆向きの勾配を加算
      grad[i] += gx; grad[i + 1] += gy; grad[i + 2] += gz;
      grad[j] -= gx; grad[j + 1] -= gy; grad[j + 2] -= gz;
      if (wantE) E += 0.5 * ks * dl * dl;
    }

    // -----------------------------------------------------------------
    // 2. 形状復元バネ（元の 3D 形状 x0 へ戻す見えないバネ）
    // 【エネルギー】 E_restore = ½ * kr * |x - x0|²
    // 【勾配（力）】 ∇E = kr * (x - x0)
    // ※重力モード(this.free = true)では重心移動分(mx, my, mz)を差し引き、落下を妨げず形だけ戻す
    // -----------------------------------------------------------------
    let mx = 0, my = 0, mz = 0;
    if (this.free) {
      for (let i = 0; i < n; i++) {
        mx += x[3 * i] - rest[3 * i];
        my += x[3 * i + 1] - rest[3 * i + 1];
        mz += x[3 * i + 2] - rest[3 * i + 2];
      }
      mx /= n; my /= n; mz /= n; // 全体の平行移動量（重心の変位）
    }
    for (let i = 0; i < n; i++) {
      const o = 3 * i;
      const dx = x[o] - rest[o] - mx, dy = x[o + 1] - rest[o + 1] - my, dz = x[o + 2] - rest[o + 2] - mz;
      grad[o] += kr * dx; grad[o + 1] += kr * dy; grad[o + 2] += kr * dz;
      if (wantE) E += 0.5 * kr * (dx * dx + dy * dy + dz * dz);
    }

    // -----------------------------------------------------------------
    // 3. 体積保持（ガス圧モデル：潰れたら内圧で押し返す）
    // 【符号付き体積】 V = (1/6) * Σ (x_a · (x_b × x_c))  ※スカラー三重積（四面体体積の総和）
    // 【エネルギー】   U_vol(V) = (kv * V0 / aMean) * [ V/V0 - ln(V/V0) - 1 ]
    // 【勾配（圧力）】 ∂U_vol/∂x_a = (kv / aMean) * (1 - V0/V) * (x_b × x_c) / 6
    // -----------------------------------------------------------------
    const nt = tris.length / 3;
    let V = 0;
    for (let t = 0; t < nt; t++) {
      const a = tris[3 * t] * 3, b = tris[3 * t + 1] * 3, c = tris[3 * t + 2] * 3;
      // 外積 (b × c) と a の内積で原点と三角形が作る四面体の体積を足し合わせる
      V += x[a] * (x[b + 1] * x[c + 2] - x[b + 2] * x[c + 1])
        + x[a + 1] * (x[b + 2] * x[c] - x[b] * x[c + 2])
        + x[a + 2] * (x[b] * x[c + 1] - x[b + 1] * x[c]);
    }
    V /= 6;
    this.V = V;
    if (kv > 0 && this.V0 > 1e-9) { // 閉じたメッシュ（V0 > 0）でのみ体積保持を有効化
      const Vc = Math.max(V, 0.25 * this.V0); // 極端な圧縮による対数発散を防ぐ下限クランプ
      const coef = ((kv / this.aMean) * (1 - this.V0 / Vc)) / 6;
      for (let t = 0; t < nt; t++) {
        const a = tris[3 * t] * 3, b = tris[3 * t + 1] * 3, c = tris[3 * t + 2] * 3;
        // 各頂点の体積勾配は対辺の外積に比例する（∂V/∂a = (b×c)/6 など）
        const bcx = x[b + 1] * x[c + 2] - x[b + 2] * x[c + 1];
        const bcy = x[b + 2] * x[c] - x[b] * x[c + 2];
        const bcz = x[b] * x[c + 1] - x[b + 1] * x[c];
        const cax = x[c + 1] * x[a + 2] - x[c + 2] * x[a + 1];
        const cay = x[c + 2] * x[a] - x[c] * x[a + 2];
        const caz = x[c] * x[a + 1] - x[c + 1] * x[a];
        const abx = x[a + 1] * x[b + 2] - x[a + 2] * x[b + 1];
        const aby = x[a + 2] * x[b] - x[a] * x[b + 2];
        const abz = x[a] * x[b + 1] - x[a + 1] * x[b];
        grad[a] += coef * bcx; grad[a + 1] += coef * bcy; grad[a + 2] += coef * bcz;
        grad[b] += coef * cax; grad[b + 1] += coef * cay; grad[b + 2] += coef * caz;
        grad[c] += coef * abx; grad[c + 1] += coef * aby; grad[c + 2] += coef * abz;
      }
      if (wantE) E += ((kv * this.V0) / this.aMean) * (Vc / this.V0 - Math.log(Vc / this.V0) - 1);
    }
    return E;
  }

  /**
   * 1 サブステップ（dt = 1/360 秒）分の時間発展計算
   * 【数値積分法】半陰的オイラー法（Semi-Implicit Euler）
   *   Step 1: v(t + dt) = v(t) + a(t) * dt        （先に速度を更新）
   *   Step 2: x(t + dt) = x(t) + v(t + dt) * dt   （新しい速度で位置を更新）
   */
  step(dt, grab, wantTelemetry) {
    const P = this.P;
    const E = this.gradient(wantTelemetry); // 全エネルギー E と勾配 ∇E を取得
    const { n, x, v, grad } = this;
    const c = P.c;                          // 減衰係数 c（粘性抵抗）
    const g = this.free ? P.g : 0;          // 重力加速度 g（固定モード時は 0）

    // テレメトリ用：全頂点の中で最も大きい勾配ノルム max ||∇E|| を探す
    let tele = null;
    if (wantTelemetry) {
      let m2max = 0;
      for (let i = 0; i < n; i++) {
        const o = 3 * i;
        const m2 = grad[o] * grad[o] + grad[o + 1] * grad[o + 1] + grad[o + 2] * grad[o + 2];
        if (m2 > m2max) m2max = m2;
      }
      tele = { E, gradMax: Math.sqrt(m2max) };
    }

    // 1. 内力（-∇E）・減衰力（-c*v）・重力（-g）による速度の更新（質量 m = 1 のため F = a）
    const m = 3 * n;
    for (let i = 0; i < n; i++) {
      const o = 3 * i;
      v[o] += (-grad[o] - c * v[o]) * dt;
      v[o + 1] += (-grad[o + 1] - c * v[o + 1] - g) * dt;
      v[o + 2] += (-grad[o + 2] - c * v[o + 2]) * dt;
    }

    // 2. マウスで引っ張る外力 F_grab の加算
    // 【数式】 F_grab = (4 * k) * w * (目標座標 - 現在座標 x)
    if (grab) {
      const kg = P.k * 4;
      const { idx, w, start, delta, primary } = grab;
      for (let t = 0; t < idx.length; t++) {
        const o = 3 * idx[t], wt = w[t];
        const fx = kg * wt * (start[3 * t] + wt * delta[0] - x[o]);
        const fy = kg * wt * (start[3 * t + 1] + wt * delta[1] - x[o + 1]);
        const fz = kg * wt * (start[3 * t + 2] + wt * delta[2] - x[o + 2]);
        v[o] += fx * dt; v[o + 1] += fy * dt; v[o + 2] += fz * dt;
        if (idx[t] === primary) { this.grabForce[0] = fx; this.grabForce[1] = fy; this.grabForce[2] = fz; }
      }
    } else {
      this.grabForce[0] = this.grabForce[1] = this.grabForce[2] = 0;
    }

    // 3. 更新された新しい速度 v を使って座標 x を進める（半陰的オイラー法の肝）
    for (let i = 0; i < m; i++) x[i] += v[i] * dt;

    // 4. 床との衝突判定と反発応答（重力モード時）
    // 【数式】 めり込み補正: y = floorY ／ 跳ね返り: vy' = -e * vy ／ 摩擦減衰: vx *= 0.92, vz *= 0.92
    const fy = this.floorY;
    if (this.free && fy > -1e8) {
      for (let i = 0; i < n; i++) {
        const o = 3 * i + 1;
        if (x[o] < fy) {
          x[o] = fy;
          if (v[o] < 0) v[o] = v[o] < -0.8 ? -v[o] * this.restitution : 0;
          v[o - 1] *= 0.92;
          v[o + 1] *= 0.92;
        }
      }
    }
    return tele;
  }

  /**
   * マウス把持の重み分布を作成
   * 【距離減衰】 w(d) = (1 - d² / R²)²  （中心で1、半径Rで滑らかに0になる4次関数）
   * 【法線ゲート】 つかんだ面と反対側（裏側）の頂点が誤って引っ張られないよう法線の内積で制限
   */
  makeGrab(primary, R, normals) {
    const { n, x } = this;
    const p = 3 * primary;
    const cx = x[p], cy = x[p + 1], cz = x[p + 2];
    const gx = normals[p], gy = normals[p + 1], gz = normals[p + 2];
    const idx = [], w = [], start = [];
    const R2 = R * R;
    for (let i = 0; i < n; i++) {
      const o = 3 * i;
      const dx = x[o] - cx, dy = x[o + 1] - cy, dz = x[o + 2] - cz;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 >= R2) continue;
      const q = 1 - d2 / R2;
      let wt = q * q;
      const dn = normals[o] * gx + normals[o + 1] * gy + normals[o + 2] * gz; // 法線どうしの内積 cosθ
      const s = clamp((dn + 0.25) / 0.6, 0, 1);
      wt *= s * s * (3 - 2 * s); // エルミート補間（Smoothstep）で滑らかに減衰
      if (wt > 0.002 || i === primary) {
        idx.push(i); w.push(i === primary ? 1 : wt);
        start.push(x[o], x[o + 1], x[o + 2]);
      }
    }
    return {
      primary,
      idx: Int32Array.from(idx),
      w: Float64Array.from(w),
      start: Float64Array.from(start),
      p0: [cx, cy, cz],
      delta: [0, 0, 0],
    };
  }

  /* -----------------------------------------------------------------
   * ヘッセ行列 H（エネルギーの2階微分：剛性行列）の固有値推定
   *
   * 【定義】 H = ∂²E / ∂x² （システム全体の「硬さの分布」を表す巨大行列）
   * 【条件数】 κ(H) = λ_max / λ_min
   *   ・λ_max（最大固有値）：最も硬い変形モードの剛性
   *   ・λ_min（最小固有値）：最も柔らかい変形モードの剛性（概ね kr = ρ*k に近づく）
   *   ・メモリ節約のため巨大な行列 H を直接作らず、ベクトル積 H*u の関数（Matrix-Free法）で計算する
   * --------------------------------------------------------------- */

  initSpectrum() {
    const m = 3 * this.n;
    this.pv = new Float64Array(m); // 最大固有値推定用の固有ベクトル候補
    this.qv = new Float64Array(m); // 最小固有値推定用の固有ベクトル候補
    this.yv = new Float64Array(m);
    this.tmp = new Float64Array(m);
    this.cgR = new Float64Array(m); // 共役勾配法(CG)の残差ベクトル r
    this.cgP = new Float64Array(m); // 共役勾配法(CG)の探索方向 p
    this.cgA = new Float64Array(m);
    let pp = 0, qq = 0;
    for (let i = 0; i < m; i++) {
      this.pv[i] = Math.random() - 0.5; this.qv[i] = Math.random() - 0.5;
      pp += this.pv[i] * this.pv[i]; qq += this.qv[i] * this.qv[i];
    }
    pp = 1 / Math.sqrt(pp); qq = 1 / Math.sqrt(qq);
    for (let i = 0; i < m; i++) { this.pv[i] *= pp; this.qv[i] *= qq; }
    this.lamMax = this.P.k * 8;
    this.lamMin = this.P.k * this.P.rho;
  }

  /**
   * 行列ベクトル積 out = H * u を計算する（Matrix-Free）
   * バネの接線剛性行列（引張時は半正定値になるよう Gauss-Newton 近似）と復元バネ kr の和
   */
  hessVec(u, out) {
    const P = this.P;
    const k = P.k, kr = k * P.rho;
    const { n, x, sa, sb, sL, sW } = this;
    out.fill(0);
    const ns = sa.length;
    for (let s = 0; s < ns; s++) {
      const i = sa[s] * 3, j = sb[s] * 3;
      const dx = x[i] - x[j], dy = x[i + 1] - x[j + 1], dz = x[i + 2] - x[j + 2];
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-12;
      const ux = dx / len, uy = dy / len, uz = dz / len; // バネ方向の単位ベクトル u_hat
      const wx = u[i] - u[j], wy = u[i + 1] - u[j + 1], wz = u[i + 2] - u[j + 2];
      const dd = ux * wx + uy * wy + uz * wz;
      const t = Math.max(0, 1 - sL[s] / len);            // 圧縮時の負固有値を防ぐクランプ
      const ks = k * sW[s];
      const hx = ks * (t * wx + (1 - t) * dd * ux);
      const hy = ks * (t * wy + (1 - t) * dd * uy);
      const hz = ks * (t * wz + (1 - t) * dd * uz);
      out[i] += hx; out[i + 1] += hy; out[i + 2] += hz;
      out[j] -= hx; out[j + 1] -= hy; out[j + 2] -= hz;
    }
    const m = 3 * n;
    for (let i = 0; i < m; i++) out[i] += kr * u[i];
  }

  /**
   * 最大固有値 λ_max と最小固有値 λ_min を数値計算で推定し、条件数 κ(H) = λ_max / λ_min を返す
   * ・λ_max：べき乗法 (Power Iteration) — H を繰り返し掛けて最大成分を取り出す
   * ・λ_min：逆反復法 (Inverse Iteration) — 共役勾配法 (CG法) で連立方程式 H*y = q を解いて最小成分を取り出す
   */
  updateSpectrum() {
    const { pv, qv, yv, tmp, cgR, cgP, cgA } = this;
    const m = pv.length;

    // 1. べき乗法による λ_max の推定（レイリー商 λ = (p^T * H * p) / (p^T * p)）
    let lam = this.lamMax;
    for (let it = 0; it < 2; it++) {
      this.hessVec(pv, tmp);
      let pp = 0, pt = 0, tt = 0;
      for (let i = 0; i < m; i++) { pp += pv[i] * pv[i]; pt += pv[i] * tmp[i]; tt += tmp[i] * tmp[i]; }
      lam = pt / (pp + 1e-30);
      const inv = 1 / Math.sqrt(tt + 1e-30);
      for (let i = 0; i < m; i++) pv[i] = tmp[i] * inv;
    }
    this.lamMax = lam;

    // 2. 共役勾配法 (Conjugate Gradient) を用いた逆反復法による λ_min の推定
    const guess = 1 / Math.max(this.lamMin, 1e-9);
    for (let i = 0; i < m; i++) yv[i] = qv[i] * guess;
    this.hessVec(yv, tmp);
    let rr = 0;
    for (let i = 0; i < m; i++) { cgR[i] = qv[i] - tmp[i]; cgP[i] = cgR[i]; rr += cgR[i] * cgR[i]; }
    for (let iter = 0; iter < 6 && rr > 1e-22; iter++) {
      this.hessVec(cgP, cgA);
      let pAp = 0;
      for (let i = 0; i < m; i++) pAp += cgP[i] * cgA[i];
      if (pAp <= 0) break;
      const a = rr / pAp;
      let rrNew = 0;
      for (let i = 0; i < m; i++) { yv[i] += a * cgP[i]; cgR[i] -= a * cgA[i]; rrNew += cgR[i] * cgR[i]; }
      const b = rrNew / rr;
      rr = rrNew;
      for (let i = 0; i < m; i++) cgP[i] = cgR[i] + b * cgP[i];
    }
    let qy = 0, yy = 0;
    for (let i = 0; i < m; i++) { qy += qv[i] * yv[i]; yy += yv[i] * yv[i]; }
    if (qy > 0 && yy > 0) {
      this.lamMin = 1 / qy;
      const inv = 1 / Math.sqrt(yy);
      for (let i = 0; i < m; i++) qv[i] = yv[i] * inv;
    }
    return { lamMax: this.lamMax, lamMin: this.lamMin, kappa: this.lamMax / Math.max(this.lamMin, 1e-30) };
  }
}

/*
 * Part 2: 起動
 *
 *Three.js は静的 import ではなく動的読み込み（CDN -> CDN -> ローカル）。
 *失敗してもモジュールは止まらず、UI（言語・テーマ・スライダー・画像スロット等）は動作し続ける。
 *UI の各初期化は safe() で個別に保護し、一箇所の例外が全体を止めないようにしている。
 */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let THREE = null; // 動的読み込み後に代入される

const DEFAULT_PARAMS = {
  res: 40, bgth: 0.10, relief: 0.35, depth: 0.45, contrast: 1.3,
  k: 500, c: 1.2, g: 9.8, R: 0.5, beta: 1.0, rho: 0.25,
};
const params = { ...DEFAULT_PARAMS };

const SUB_DT = 1 / 360;      // 物理計算の1ステップの時間幅 Δt = 1/360秒（約2.78ミリ秒ごとに位置と速度を積分する）
const MAX_SUBSTEPS = 12;     // 1フレームあたりの最大計算回数（PCが重くなっても物理時間が飛びすぎて爆発しないための上限）
const TELE_EVERY = 6;        // テレメトリ（エネルギーやヘッセ条件数）を画面に更新する間隔（6フレームに1回 ≒ 10Hz）

const state = {
  lang: 'ja',
  theme: 'dark',
  mode: 'solid',
  phys: 'fixed',             // fixed（固定モード）| gravity（重力モード）
  paused: false,             // S キーで時間停止             // solid | wire | plexus
  collapsed: false,
  source: 'model',           // model | image（左パネルのタブ。起動時は 3Dモデル）
  pendingModel: null,        // 外部ファイル待ちのモデル
  model: null,               // { name, parsed, info, mtls, textures, mtlFiles } 読み込み済みの .obj と色情報
  slots: {},                 // key -> { rgba, w, h, label, analysis }
  views: {},
  mesh: null,
  body: null,
  isSample: true,
  grab: null,
  pointer: { x: 0, y: 0 },
  frame: 0,
  fps: 60,
  kappaSmooth: null,
  status: { key: 'st.init', kind: '', args: {} },
  teleState: 'idle',
};
let engineReady = false;

/*
 * i18n（日本語 / English）
 */

const I18N = {
  ja: {
    'app.title': 'うごけスライム',
    'viewport.aria': '3Dビューポート',
    'hdr.resetView': '視点リセット',
    'hdr.displayMode': '表示モード',
    'hdr.theme': 'テーマ',
    'hdr.lang': '言語',
    'hdr.shot': 'スクリーンショット',
    'hdr.pause': '時間停止',
    'hdr.physics': '物理モード',
    'phys.fixed': '固定モード',
    'phys.gravity': '重力モード',
    'mode.solid': 'ソリッド',
    'mode.wire': 'ワイヤーフレーム',
    'mode.plexus': 'ネットワーク',
    'mode.relief': 'レリーフ',
    'mode.hull': '多視点交差',
    'theme.light.title': 'ライトモード',
    'theme.dark.title': 'ダークモード',
    'panel.aria': '設定パネル',
    'panel.title': '設定',
    'panel.collapse': 'メニューを閉じる',
    'panel.expand': 'メニューを開く',
    'tab.image': '画像から生成',
    'tab.model': '3Dモデルから生成',
    'sec.images': '入力画像',
    'sec.mesh': '3Dメッシュ生成',
    'sec.phys': '物理パラメータ',
    'slot.front.name': '正面',
    'slot.right.name': '右側面',
    'slot.back.name': '背面',
    'slot.left.name': '左側面',
    'slot.top.name': '上面',
    'slot.clear': '削除',
    'slot.main.cta': 'クリックまたは画像をドロップ',
    'acc.label': 'その他の面',
    'model.drop.title': '3Dモデル',
    'model.drop.cta': '3Dモデルをドロップまたは選択',
    'model.drop.replace': '別の3Dモデルをドロップまたは選択',
    'model.files': '追加ファイル：{list}',
    'model.color.vertex': '色：頂点カラー',
    'model.color.mtl': '色：マテリアル',
    'model.color.tex': '色：テクスチャ',
    'model.color.mix': '色：頂点カラー・マテリアル・テクスチャ',
    'model.color.none': '色：標準色',
    'model.warn.nomtl': 'マテリアルファイル {name} が見つかりません。モデルと一緒に選択してください',
    'model.warn.notex': 'テクスチャ {name} が見つかりません。画像も一緒に選択してください',
    'model.stats': '{v} 頂点 / {t} 三角形 / {s} バネ / 支持 {i}',
    'model.warn.heavy': '頂点数が多く、動作が重くなる可能性があります',
    'model.warn.decimated': '頂点数が {before} と非常に多いため、約 {after} 頂点に自動で間引きました',
    'model.warn.open': 'メッシュが閉じていないため、体積保持は無効になります',
    'p.res': 'ボクセル解像度',
    'p.bgth': '背景しきい値',
    'p.relief': 'レリーフ強度',
    'p.depth': '押し出し厚み',
    'p.contrast': '深度コントラスト',
    'p.k': '剛性係数',
    'p.c': '減衰係数',
    'p.g': '重力加速度',
    'p.R': '把持影響半径',
    'p.beta': '体積保持係数',
    'p.rho': '形状復元比',
    'tele.aria': 'シミュレーション状態',
    'tele.title': 'シミュレーション状態',
    'tele.E': '総弾性エネルギー',
    'tele.grad': '最大勾配ノルム',
    'tele.kappa': 'ヘッセ条件数',
    'tele.force': '把持力',
    'tele.state.idle': '待機中',
    'tele.state.running': '実行中',
    'tele.state.grab': '把持中',
    'tele.state.paused': '時間停止',
    'pause.chip': '時間停止中 / Sキーで再開',
    'unit.fps': 'fps',
    'hint.grab.k': '左ドラッグ',
    'hint.grab.v': '引っ張る',
    'hint.rot.k': 'Shift + 左ドラッグ',
    'hint.rot.v': '視点回転',
    'hint.zoom.k': 'ホイール',
    'hint.zoom.v': 'ズーム',
    'hint.pan.k': '右ドラッグ',
    'hint.pan.v': '平行移動',
    'hint.pause.k': 'Sキー',
    'hint.pause.v': '時間停止／再開',
    'reset.btn': 'すべて初期化',
    'reset.title': '初期化の確認',
    'reset.msg': 'すべての画像とパラメータを初期状態に戻しますか？',
    'modal.ok': 'OK',
    'modal.cancel': 'キャンセル',
    'err.title': '3Dエンジンを開始できません',
    'err.load': 'Three.js を読み込めませんでした。ネットワーク接続を確認するか、vendor/three.module.js を配置してください。画像や3Dモデルの読み込み、各種設定は引き続き使えます。',
    'err.webgl': 'WebGL を初期化できませんでした。グラフィックスドライバーの更新、またはハードウェアアクセラレーションの設定を確認してください。',
    'err.retry': '再試行',
    'st.init': '初期化中',
    'st.loading3d': '3Dエンジンを読み込み中 {i}/{n}',
    'st.building': 'メッシュ生成中',
    'st.sample': '見本モデル / {ms} ms',
    'st.done': '{mode}：{views}枚 / {ms} ms',
    'st.doneWarn': '{mode}：{views}枚 / {ms} ms / 背景を検出できない画像あり',
    'st.model': '3Dモデル：{name} / {v} 頂点 / {ms} ms',
    'st.modelWarn': '3Dモデル：{name} / {v} 頂点 / {ms} ms / 注意：{warn}',
    'st.nomesh': 'シルエットから立体を生成できませんでした。背景しきい値を調整してください',
    'st.imgfail': '画像を読み込めませんでした：{name}',
    'st.notimage': '画像ファイルではありません：{name}',
    'st.notmodel': '対応していない3Dモデルです：{name}',
    'st.needmodel': '先に3Dモデルを選択してください',
    'st.texfail': 'テクスチャ画像を読み込めませんでした：{name}',
    'st.needfile': '必要なファイル {name} が見つかりません。モデルと一緒に選択してください',
    'st.modelfail': '3Dモデルを読み取れませんでした：{name}',
    'st.diverged': '数値発散を検出したため初期形状へ復帰しました',
    'st.noexport': '書き出せるメッシュがありません',
    'st.shotDone': 'スクリーンショットを保存しました',
    'st.glLost': 'WebGL コンテキストが失われました',
    'st.reset': '初期状態に戻しました',
    'st.raw': '{msg}',
  },
  en: {
    'app.title': 'Ugoke Slime',
    'viewport.aria': '3D viewport',
    'hdr.resetView': 'Reset View',
    'hdr.displayMode': 'Display mode',
    'hdr.theme': 'Theme',
    'hdr.lang': 'Language',
    'hdr.shot': 'Screenshot',
    'hdr.pause': 'Stop time',
    'hdr.physics': 'Physics mode',
    'phys.fixed': 'Fixed Mode',
    'phys.gravity': 'Gravity Mode',
    'mode.solid': 'Solid',
    'mode.wire': 'Wireframe',
    'mode.plexus': 'Plexus',
    'mode.relief': 'Relief',
    'mode.hull': 'Multi-view hull',
    'theme.light.title': 'Light mode',
    'theme.dark.title': 'Dark mode',
    'panel.aria': 'Settings panel',
    'panel.title': 'Settings',
    'panel.collapse': 'Collapse panel',
    'panel.expand': 'Open panel',
    'tab.image': 'Generate from Images',
    'tab.model': 'Generate from 3D Model',
    'sec.images': 'Input Images',
    'sec.mesh': '3D Mesh Generation',
    'sec.phys': 'Physics Parameters',
    'slot.front.name': 'Front',
    'slot.right.name': 'Right',
    'slot.back.name': 'Back',
    'slot.left.name': 'Left',
    'slot.top.name': 'Top',
    'slot.clear': 'Remove',
    'slot.main.cta': 'Click or drop an image',
    'acc.label': 'Other views',
    'model.drop.title': '3D model',
    'model.drop.cta': 'Drop or select a 3D model',
    'model.drop.replace': 'Drop or select another 3D model',
    'model.files': 'Extra files: {list}',
    'model.color.vertex': 'Color: vertex colors',
    'model.color.mtl': 'Color: materials',
    'model.color.tex': 'Color: textures',
    'model.color.mix': 'Color: vertex colors, materials, textures',
    'model.color.none': 'Color: default',
    'model.warn.nomtl': 'Material file {name} was not found. Select it together with the model',
    'model.warn.notex': 'Texture {name} was not found. Select the image together with the model',
    'model.stats': '{v} vertices / {t} triangles / {s} springs / {i} supports',
    'model.warn.heavy': 'High vertex count: the simulation may run slowly',
    'model.warn.decimated': 'The model has {before} vertices, so it was automatically reduced to about {after}',
    'model.warn.open': 'The mesh is not closed, so volume preservation is disabled',
    'p.res': 'Voxel resolution',
    'p.bgth': 'Background threshold',
    'p.relief': 'Relief strength',
    'p.depth': 'Extrusion depth',
    'p.contrast': 'Depth contrast',
    'p.k': 'Stiffness',
    'p.c': 'Damping',
    'p.g': 'Gravity',
    'p.R': 'Grasp radius',
    'p.beta': 'Volume preservation',
    'p.rho': 'Shape restoring ratio',
    'tele.aria': 'Simulation status',
    'tele.title': 'Simulation Status',
    'tele.E': 'Total Elastic Energy',
    'tele.grad': 'Max Gradient Norm',
    'tele.kappa': 'Hessian condition number',
    'tele.force': 'Grasp Force',
    'tele.state.idle': 'Idle',
    'tele.state.running': 'Running',
    'tele.state.grab': 'Grasping',
    'tele.state.paused': 'Time stopped',
    'pause.chip': 'Time stopped / Press S to resume',
    'unit.fps': 'fps',
    'hint.grab.k': 'Left drag',
    'hint.grab.v': 'Pull',
    'hint.rot.k': 'Shift + Left drag',
    'hint.rot.v': 'Orbit camera',
    'hint.zoom.k': 'Wheel',
    'hint.zoom.v': 'Zoom',
    'hint.pan.k': 'Right drag',
    'hint.pan.v': 'Pan',
    'hint.pause.k': 'S key',
    'hint.pause.v': 'Stop / resume time',
    'reset.btn': 'Reset All',
    'reset.title': 'Confirm reset',
    'reset.msg': 'Reset all images and parameters to their defaults?',
    'modal.ok': 'OK',
    'modal.cancel': 'Cancel',
    'err.title': 'Cannot start the 3D engine',
    'err.load': 'Three.js could not be loaded. Check your network connection or place vendor/three.module.js next to index.html. Loading images and 3D models and changing settings still work.',
    'err.webgl': 'WebGL could not be initialized. Update your graphics driver or check the hardware acceleration setting.',
    'err.retry': 'Retry',
    'st.init': 'Initializing',
    'st.loading3d': 'Loading 3D engine {i}/{n}',
    'st.building': 'Generating mesh',
    'st.sample': 'Sample model / {ms} ms',
    'st.done': '{mode}: {views} images / {ms} ms',
    'st.doneWarn': '{mode}: {views} images / {ms} ms / background not detected in some image',
    'st.model': '3D model: {name} / {v} vertices / {ms} ms',
    'st.modelWarn': '3D model: {name} / {v} vertices / {ms} ms / Note: {warn}',
    'st.nomesh': 'Could not build a solid from the silhouettes. Adjust the background threshold',
    'st.imgfail': 'Could not load image: {name}',
    'st.notimage': 'Not an image file: {name}',
    'st.notmodel': 'Unsupported 3D model: {name}',
    'st.needmodel': 'Select a 3D model first',
    'st.texfail': 'Could not load texture image: {name}',
    'st.needfile': 'Required file {name} was not found. Select it together with the model',
    'st.modelfail': 'Could not read the 3D model: {name}',
    'st.diverged': 'Numerical divergence detected; restored the rest shape.',
    'st.noexport': 'There is no mesh to export.',
    'st.shotDone': 'Screenshot saved',
    'st.glLost': 'WebGL context lost',
    'st.reset': 'Reset to defaults',
    'st.raw': '{msg}',
  },
};

function t(key, vars) {
  const dict = I18N[state.lang] || I18N.ja;
  let s = dict[key];
  if (s === undefined) s = I18N.ja[key];
  if (s === undefined) return key;
  if (vars) for (const k of Object.keys(vars)) s = s.split('{' + k + '}').join(String(vars[k]));
  return s;
}

function loadSetting(key, fallback) {
  try { const v = localStorage.getItem('sbs.' + key); return v == null ? fallback : v; } catch (e) { return fallback; }
}
function saveSetting(key, value) {
  try { localStorage.setItem('sbs.' + key, String(value)); } catch (e) { /* 保存不可でも続行 */ }
}

function safe(name, fn) {
  try { return fn(); } catch (err) {
    console.error('[init:' + name + ']', err);
    setStatus('st.raw', 'error', { msg: name + ': ' + (err && err.message ? err.message : err) });
    return undefined;
  }
}

// ステータス / テレメトリ状態表示

function setStatus(key, kind = '', args = {}) {
  state.status = { key, kind, args };
  renderStatus();
}
function renderStatus() {
  const el = $('#status');
  if (!el) return;
  const args = {};
  for (const k of Object.keys(state.status.args)) {
    const v = state.status.args[k];
    args[k] = typeof v === 'string' && v.charAt(0) === '@' ? t(v.slice(1), state.status.args) : v;
  }
  el.textContent = t(state.status.key, args);
  el.className = 'status' + (state.status.kind ? ' is-' + state.status.kind : '');
}

function setTeleState(s) {
  if (state.teleState === s) return;
  state.teleState = s;
  renderTeleState();
}
function renderTeleState() {
  const el = $('#tele-state');
  if (!el) return;
  el.dataset.state = state.teleState;
  $('#tele-state-text').textContent = t('tele.state.' + state.teleState);
}

function applyI18n() {
  $$('[data-i18n]').forEach((el) => { el.textContent = t(el.dataset.i18n); });
  $$('[data-i18n-title]').forEach((el) => { el.title = t(el.dataset.i18nTitle); });
  $$('[data-i18n-aria]').forEach((el) => { el.setAttribute('aria-label', t(el.dataset.i18nAria)); });
  document.documentElement.lang = state.lang;
  VIEW_KEYS.forEach(updateSlotUI);
  renderModelPane();
  renderStatus();
  renderTeleState();
}

function setLang(lang) {
  state.lang = lang === 'en' ? 'en' : 'ja';
  saveSetting('lang', state.lang);
  $$('[data-lang-btn]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.langBtn === state.lang)));
  applyI18n();
}

// テーマ（AM = ライト / PM = ダーク）と表示モード

const THEME_GL = {
  light: {
    bg: 0x008080, gridMajor: 0x00FF00, gridMinor: 0x00A040,   // Windows 95 のデスクトップ色 + ネオングリーンのグリッド
    wire: 0x00FF00, wireOpacity: 1.0, ghost: 0.80,
    dot: [0.0, 1.0, 0.0], hot: [1.0, 0.1, 0.1], additive: false, glow: 0.6, lineAlpha: 0.8,
  },
  dark: {
    bg: 0x000000, gridMajor: 0x00FFFF, gridMinor: 0x007F7F,   // 真っ黒 + 白い星 + シアンのグリッド
    wire: 0x00FF00, wireOpacity: 1.0, ghost: 0.84,
    dot: [0.0, 1.0, 1.0], hot: [1.0, 0.0, 1.0], additive: true, glow: 1.0, lineAlpha: 0.7,
  },
};

function setTheme(theme) {
  state.theme = theme === 'dark' ? 'dark' : 'light';
  document.documentElement.dataset.theme = state.theme;
  $$('[data-theme-btn]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.themeBtn === state.theme)));
  saveSetting('theme', state.theme);
  if (engineReady) applyThemeGL();
}

function setMode(mode) {
  state.mode = ['solid', 'wire', 'plexus'].includes(mode) ? mode : 'solid';
  $$('[data-mode]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.mode === state.mode)));
  saveSetting('mode', state.mode);
  if (engineReady) applyDisplayMode();
}

function setCollapsed(collapsed) {
  state.collapsed = !!collapsed;
  $('#layout').classList.toggle('is-collapsed', state.collapsed);
  $('#btn-collapse').setAttribute('aria-expanded', String(!state.collapsed));
  saveSetting('collapsed', state.collapsed ? '1' : '0');
}

// Three.js の動的読み込み（CDN -> CDN -> ローカル）

const THREE_SOURCES = [
  { url: 'three', timeout: 8000 }, // index.html の importmap -> unpkg CDN
  { url: 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js', timeout: 8000 },
  { url: './node_modules/three/build/three.module.js', timeout: 4000 },
  { url: './vendor/three.module.js', timeout: 4000 },
];

function importWithTimeout(url, ms) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('timeout (' + ms + ' ms)')), ms);
    import(/* webpackIgnore: true */ url).then(
      (mod) => { clearTimeout(timer); resolve(mod); },
      (err) => { clearTimeout(timer); reject(err); },
    );
  });
}

async function loadThree(onProgress) {
  const details = [];
  for (let i = 0; i < THREE_SOURCES.length; i++) {
    const src = THREE_SOURCES[i];
    if (onProgress) onProgress(i + 1, THREE_SOURCES.length);
    try {
      const mod = await importWithTimeout(src.url, src.timeout);
      if (mod && mod.WebGLRenderer) return { mod, url: src.url };
      details.push(src.url + ' -> invalid module');
    } catch (err) {
      details.push(src.url + ' -> ' + (err && err.message ? err.message : err));
    }
  }
  const error = new Error('Three.js load failed');
  error.code = 'load';
  error.details = details;
  throw error;
}

function showEngineError(err) {
  const box = $('#engine-error');
  if (!box) return;
  $('#engine-error-msg').textContent = err && err.code === 'webgl' ? t('err.webgl') : t('err.load');
  const detail = err && err.details ? err.details.join('\n') : String(err && err.message ? err.message : err);
  $('#engine-error-detail').textContent = detail;
  box.hidden = false;
  setStatus('st.raw', 'error', { msg: t('err.title') });
}

let renderer = null, scene = null, camera = null, floor = null;
let pullArrow = null, pullMarker = null, raycaster = null, dragPlane = null;
let glObjs = null;             // { geo, obj, wire, dots, links, material, textures, strainAttr, ... }
let floorLevel = -1.3;
let loopStarted = false, eventsBound = false;
let canvasEl = null;
const tmpV = { v: null, target: null };

const CAM_DEFAULT = { theta: 0.45, phi: 1.3, radius: 5.4 };
const cam = { ...CAM_DEFAULT, target: null };

function applyCamera() {
  if (!camera) return;
  const sp = Math.sin(cam.phi);
  camera.position.set(
    cam.target.x + cam.radius * sp * Math.sin(cam.theta),
    cam.target.y + cam.radius * Math.cos(cam.phi),
    cam.target.z + cam.radius * sp * Math.cos(cam.theta),
  );
  camera.lookAt(cam.target);
  camera.updateMatrixWorld(true);
}

function resetView() {
  Object.assign(cam, CAM_DEFAULT);
  if (cam.target) cam.target.set(0, 0, 0);
  applyCamera();
}

/* シェーダー */

const SOLID_VERT = /* glsl */ `
  attribute vec3 aRest;
  attribute vec3 aRestNormal;
  attribute vec3 aColor;
  varying vec3 vCol;
  varying vec3 vPos;
  varying vec3 vNrm;
  varying vec3 vRest;
  varying vec3 vRestN;
  void main() {
    vPos = position;
    vNrm = normal;
    vRest = aRest;
    vRestN = aRestNormal;
    vCol = aColor;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const SOLID_FRAG = /* glsl */ `
  uniform sampler2D uTex0;
  uniform sampler2D uTex1;
  uniform sampler2D uTex2;
  uniform sampler2D uTex3;
  uniform sampler2D uTex4;
  uniform vec4 uMap[5];
  uniform float uHas[5];
  uniform float uGhost;
  uniform float uFace;
  uniform float uVC;
  varying vec3 vCol;
  uniform vec3 uBg;
  varying vec3 vPos;
  varying vec3 vNrm;
  varying vec3 vRest;
  varying vec3 vRestN;

  vec2 viewUV(int i, vec3 p) {
    float a; float b;
    if (i == 0) { a = p.x; b = -p.y; }
    else if (i == 1) { a = -p.z; b = -p.y; }
    else if (i == 2) { a = -p.x; b = -p.y; }
    else if (i == 3) { a = p.z; b = -p.y; }
    else { a = p.x; b = p.z; }
    vec4 m = uMap[i];
    return clamp(vec2(m.x + a * m.z, m.y + b * m.w), vec2(0.0005), vec2(0.9995));
  }

  // 画像が無いときの見本モデル（ぴんくスライム + 顔）
  vec3 sampleAlbedo(vec3 rn, vec3 p) {
    vec3 base = vec3(1.0, 0.28, 0.65);
    float shade = clamp(0.5 + 0.5 * rn.y, 0.0, 1.0);
    vec3 col = mix(base * 0.85, vec3(1.0, 0.65, 0.85), shade * 0.55);
    if (uFace > 0.5 && rn.z > 0.35) {
      float face = smoothstep(0.35, 0.8, rn.z);
      vec2 q = p.xy;
      vec2 e1 = (q - vec2(-0.27, 0.10)) / vec2(0.085, 0.12);
      vec2 e2 = (q - vec2( 0.27, 0.10)) / vec2(0.085, 0.12);
      float eye = max(1.0 - smoothstep(0.85, 1.0, length(e1)), 1.0 - smoothstep(0.85, 1.0, length(e2)));
      vec2 p1 = (q - vec2(-0.265, 0.085)) / vec2(0.040, 0.062);
      vec2 p2 = (q - vec2( 0.275, 0.085)) / vec2(0.040, 0.062);
      float pupil = max(1.0 - smoothstep(0.8, 1.0, length(p1)), 1.0 - smoothstep(0.8, 1.0, length(p2)));
      vec2 m = q - vec2(0.0, -0.02);
      float smile = (1.0 - smoothstep(0.010, 0.018, abs(length(m) - 0.13))) * (1.0 - smoothstep(-0.05, -0.03, m.y));
      vec3 ink = vec3(0.06, 0.09, 0.16);
      col = mix(col, vec3(1.0), eye * face);
      col = mix(col, ink, pupil * eye * face);
      col = mix(col, ink, smile * face);
    }
    return col;
  }

  void main() {
    vec3 rn = normalize(vRestN);
    // 法線と各カメラ方向（正面 +Z / 右 +X / 背面 -Z / 左 -X / 上 +Y）の内積で重みを決める
    float w0 = uHas[0] * pow(max( rn.z, 0.0), 4.0);
    float w1 = uHas[1] * pow(max( rn.x, 0.0), 4.0);
    float w2 = uHas[2] * pow(max(-rn.z, 0.0), 4.0);
    float w3 = uHas[3] * pow(max(-rn.x, 0.0), 4.0);
    float w4 = uHas[4] * pow(max( rn.y, 0.0), 4.0);
    if (w0 + w1 + w2 + w3 + w4 < 1e-4) {            // どのカメラにも正対しない面は、最も近い視点で補う
      w0 = uHas[0] * (max( rn.z, 0.0) + 1e-3);
      w1 = uHas[1] * (max( rn.x, 0.0) + 1e-3);
      w2 = uHas[2] * (max(-rn.z, 0.0) + 1e-3);
      w3 = uHas[3] * (max(-rn.x, 0.0) + 1e-3);
      w4 = uHas[4] * (max( rn.y, 0.0) + 1e-3);
    }

    vec3 col = vec3(0.0);
    float ws = 0.0;
    if (w0 > 0.0) { col += w0 * texture2D(uTex0, viewUV(0, vRest)).rgb; ws += w0; }
    if (w1 > 0.0) { col += w1 * texture2D(uTex1, viewUV(1, vRest)).rgb; ws += w1; }
    if (w2 > 0.0) { col += w2 * texture2D(uTex2, viewUV(2, vRest)).rgb; ws += w2; }
    if (w3 > 0.0) { col += w3 * texture2D(uTex3, viewUV(3, vRest)).rgb; ws += w3; }
    if (w4 > 0.0) { col += w4 * texture2D(uTex4, viewUV(4, vRest)).rgb; ws += w4; }
    vec3 albedo = uVC > 0.5 ? vCol : (ws > 0.0 ? col / ws : sampleAlbedo(rn, vRest));

    vec3 N = normalize(vNrm);
    vec3 V = normalize(cameraPosition - vPos);
    vec3 L1 = normalize(vec3(0.5, 0.9, 0.7));
    vec3 L2 = normalize(vec3(-0.7, 0.2, 0.4));
    float d1 = max(dot(N, L1), 0.0);
    float d2 = max(dot(N, L2), 0.0);
    float hemi = 0.5 + 0.5 * N.y;
    // 90 年代の初期 3DCG 風：陰影は 4 段階にパキッと分け、ハイライトも硬く 1 点だけ
    float band = d1 > 0.72 ? 1.0 : (d1 > 0.42 ? 0.82 : (d1 > 0.14 ? 0.64 : 0.5));
    vec3 lit = albedo * (band + 0.10 * step(0.3, d2) + 0.06 * hemi);
    vec3 H = normalize(L1 + V);
    float spec = step(0.975, dot(N, H)) * 0.35;
    vec3 outCol = clamp(lit + vec3(spec), 0.0, 1.0);
    gl_FragColor = vec4(mix(outCol, uBg, uGhost), 1.0);
  }
`;

const DOT_VERT = /* glsl */ `
  attribute float aStrain;
  uniform float uSize;
  uniform float uPixelRatio;
  varying float vS;
  void main() {
    vS = aStrain;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = clamp(uSize * uPixelRatio * (6.0 / max(-mv.z, 0.1)) * (1.0 + 1.6 * aStrain), 1.5, 26.0);
  }
`;

const DOT_FRAG = /* glsl */ `
  uniform vec3 uBase;
  uniform vec3 uHot;
  uniform float uGlow;
  varying float vS;
  void main() {
    float d = length(gl_PointCoord - vec2(0.5)) * 2.0;
    if (d > 1.0) discard;
    float core = 1.0 - smoothstep(0.0, 0.55, d);
    float halo = pow(1.0 - d, 2.0);
    vec3 col = mix(uBase, uHot, clamp(vS, 0.0, 1.0));
    float a = clamp(core + halo * uGlow, 0.0, 1.0);
    gl_FragColor = vec4(col * (0.7 + 0.6 * core), a);
  }
`;

const LINK_VERT = /* glsl */ `
  attribute float aStrain;
  varying float vS;
  void main() {
    vS = aStrain;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const LINK_FRAG = /* glsl */ `
  uniform vec3 uBase;
  uniform vec3 uHot;
  uniform float uAlpha;
  varying float vS;
  void main() {
    float s = clamp(vS, 0.0, 1.0);
    gl_FragColor = vec4(mix(uBase, uHot, s), uAlpha * (0.6 + 0.4 * clamp(s * 2.0, 0.0, 1.0)));
  }
`;

/* マテリアル */

function makeSolidMaterial(views) {
  const placeholder = new THREE.DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1, THREE.RGBAFormat);
  placeholder.needsUpdate = true;
  const textures = [placeholder];
  const uniforms = {
    uTex0: { value: placeholder }, uTex1: { value: placeholder }, uTex2: { value: placeholder },
    uTex3: { value: placeholder }, uTex4: { value: placeholder },
    uMap: { value: [0, 1, 2, 3, 4].map(() => new THREE.Vector4(0.5, 0.5, 1, 1)) },
    uHas: { value: [0, 0, 0, 0, 0] },
    uGhost: { value: 0 },
    uFace: { value: 1 },
    uVC: { value: 0 },
    uBg: { value: new THREE.Vector3(1, 1, 1) },
  };
  VIEW_KEYS.forEach((key, i) => {
    const v = views[key];
    if (!v) return;
    const data = new Uint8Array(v.tex.buffer, v.tex.byteOffset, v.tex.byteLength);
    const tex = new THREE.DataTexture(data, v.w, v.h, THREE.RGBAFormat);
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.generateMipmaps = false;
    tex.needsUpdate = true;
    textures.push(tex);
    uniforms['uTex' + i].value = tex;
    uniforms.uMap.value[i].set(v.bbox.cx / v.w, v.bbox.cy / v.h, v.ppu / v.w, v.ppu / v.h);
    uniforms.uHas.value[i] = 1;
  });
  const material = new THREE.ShaderMaterial({
    uniforms,
    vertexShader: SOLID_VERT,
    fragmentShader: SOLID_FRAG,
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  });
  return { material, textures };
}

function styleWire(mat) {
  const th = THEME_GL[state.theme];
  mat.color.set(th.wire);
  mat.opacity = th.wireOpacity;
}

function styleDots(mat) {
  const th = THEME_GL[state.theme];
  mat.uniforms.uBase.value.set(th.dot[0], th.dot[1], th.dot[2]);
  mat.uniforms.uHot.value.set(th.hot[0], th.hot[1], th.hot[2]);
  mat.uniforms.uGlow.value = th.glow;
  mat.blending = th.additive ? THREE.AdditiveBlending : THREE.NormalBlending;
  mat.needsUpdate = true;
}

function styleLinks(mat) {
  const th = THEME_GL[state.theme];
  mat.uniforms.uBase.value.set(th.dot[0], th.dot[1], th.dot[2]);
  mat.uniforms.uHot.value.set(th.hot[0], th.hot[1], th.hot[2]);
  mat.uniforms.uAlpha.value = th.lineAlpha;
  mat.blending = th.additive ? THREE.AdditiveBlending : THREE.NormalBlending;
  mat.needsUpdate = true;
}

function makeDotMaterial() {
  const mat = new THREE.ShaderMaterial({
    uniforms: {
      uBase: { value: new THREE.Vector3() }, uHot: { value: new THREE.Vector3() },
      uGlow: { value: 1 }, uSize: { value: 3.4 }, uPixelRatio: { value: renderer.getPixelRatio() },
    },
    vertexShader: DOT_VERT,
    fragmentShader: DOT_FRAG,
    transparent: true,
    depthWrite: false,
  });
  styleDots(mat);
  return mat;
}

function makeLinkMaterial() {
  const mat = new THREE.ShaderMaterial({
    uniforms: { uBase: { value: new THREE.Vector3() }, uHot: { value: new THREE.Vector3() }, uAlpha: { value: 0.4 } },
    vertexShader: LINK_VERT,
    fragmentShader: LINK_FRAG,
    transparent: true,
    depthWrite: false,
  });
  styleLinks(mat);
  return mat;
}

/* 床グリッドと座標軸 */

let stars = null;

function buildFloor() {
  if (floor) {
    scene.remove(floor);
    floor.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) o.material.dispose();
    });
  }
  const th = THEME_GL[state.theme];
  floor = new THREE.Group();
  floor.add(new THREE.GridHelper(12, 24, th.gridMajor, th.gridMinor));
  // 原点の座標軸：X = 赤 / Y = 緑 / Z = 青
  const axes = new THREE.AxesHelper(3);
  axes.position.y = 0.003;
  axes.material.fog = false;
  { const ca = axes.geometry.attributes.color; for (let k = 0; k < 3; k++) for (let e = 0; e < 2; e++) ca.setXYZ(2 * k + e, k === 0 ? 1 : 0, k === 1 ? 1 : 0, k === 2 ? 1 : 0); ca.needsUpdate = true; } // 軸の色は両端とも純色
  floor.add(axes);
  floor.position.y = floorLevel;
  scene.add(floor);
  // 暗い画面：白いドットの星空
  if (!stars) {
    const n = 260, p = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const u = Math.random() * 2 - 1, a = Math.random() * Math.PI * 2, r = 24, q = Math.sqrt(1 - u * u);
      p[3 * i] = r * q * Math.cos(a); p[3 * i + 1] = Math.abs(r * u) * 0.9 + 1; p[3 * i + 2] = r * q * Math.sin(a);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(p, 3));
    stars = new THREE.Group();
    stars.add(new THREE.Points(g, new THREE.PointsMaterial({ color: 0xFFFFFF, size: 3, sizeAttenuation: false, fog: false })));
    // Universe Q 風：遠くに浮かぶ平らな色の惑星
    const planet = (r, color, x, y, z, ring) => {
      const m = new THREE.Mesh(new THREE.SphereGeometry(r, 16, 12), new THREE.MeshBasicMaterial({ color, fog: false }));
      m.position.set(x, y, z);
      if (ring) {
        const rg = new THREE.Mesh(new THREE.RingGeometry(r * 1.4, r * 2.0, 32), new THREE.MeshBasicMaterial({ color: 0xFFC040, side: THREE.DoubleSide, fog: false }));
        rg.rotation.x = Math.PI / 2.4;
        m.add(rg);
      }
      stars.add(m);
    };
    planet(1.4, 0x2F6BFF, -11, 5.5, -16);
    planet(0.9, 0xFF8030, 12, 7, -18, true);
    planet(1.8, 0x30C060, 2, 11, -22);
    planet(0.6, 0xFF40C0, -4, 3.5, -15);
    scene.add(stars);
  }
  stars.visible = state.theme === 'dark';
}

function hexToRgb01(hex) {
  return [((hex >> 16) & 255) / 255, ((hex >> 8) & 255) / 255, (hex & 255) / 255];
}

function applyThemeGL() {
  if (!renderer) return;
  renderer.setClearColor(THEME_GL[state.theme].bg, 1);
  scene.fog.color.set(THEME_GL[state.theme].bg);
  buildFloor();
  if (glObjs) {
    applyDisplayMode();
    styleWire(glObjs.wire.material);
    styleDots(glObjs.dots.material);
    styleLinks(glObjs.links.material);
  }
}

function applyDisplayMode() {
  if (!glObjs) return;
  const m = state.mode;
  // ワイヤーフレーム = 背景色へ溶かしたソリッド（隠線処理）+ 構造バネの線。
  // プレクサスではソリッドを非表示にするが、非表示でも Raycaster の対象になるためつかめる。
  const th = THEME_GL[state.theme];
  glObjs.obj.visible = m !== 'plexus';
  glObjs.material.uniforms.uGhost.value = m === 'wire' ? th.ghost : 0;
  glObjs.material.uniforms.uBg.value.set(...hexToRgb01(th.bg));
  glObjs.wire.visible = m === 'wire';
  glObjs.dots.visible = m === 'plexus';
  glObjs.links.visible = m === 'plexus';
}

/* メッシュの設置・破棄 */

function disposeMesh() {
  if (!glObjs) return;
  const { geo, wireGeo, dotGeo, linkGeo, obj, wire, dots, links, material, textures } = glObjs;
  [obj, wire, dots, links].forEach((o) => scene.remove(o));
  [geo, wireGeo, dotGeo, linkGeo].forEach((g) => g.dispose());
  wire.material.dispose();
  dots.material.dispose();
  links.material.dispose();
  material.dispose();
  textures.forEach((tx) => tx.dispose());
  glObjs = null;
  state.body = null;
  state.mesh = null;
  state.grab = null;
  if (pullArrow) pullArrow.visible = false;
  if (pullMarker) pullMarker.visible = false;
}

function installMesh(views, mesh) {
  disposeMesh();
  const n = mesh.n;

  const geo = new THREE.BufferGeometry();
  const posAttr = new THREE.BufferAttribute(Float32Array.from(mesh.pos), 3);
  posAttr.setUsage(THREE.DynamicDrawUsage);
  const nrmAttr = new THREE.BufferAttribute(Float32Array.from(mesh.normals), 3);
  nrmAttr.setUsage(THREE.DynamicDrawUsage);
  const strainAttr = new THREE.BufferAttribute(new Float32Array(n), 1);
  strainAttr.setUsage(THREE.DynamicDrawUsage);
  geo.setAttribute('position', posAttr);
  geo.setAttribute('normal', nrmAttr);
  geo.setAttribute('aRest', new THREE.BufferAttribute(Float32Array.from(mesh.pos), 3));
  geo.setAttribute('aColor', new THREE.BufferAttribute(mesh.colors ? Float32Array.from(mesh.colors) : new Float32Array(3 * n), 3));
  geo.setAttribute('aRestNormal', new THREE.BufferAttribute(Float32Array.from(mesh.normals), 3));
  geo.setIndex(new THREE.BufferAttribute(Uint32Array.from(mesh.tris), 1));

  const { material, textures } = makeSolidMaterial(views);
  const obj = new THREE.Mesh(geo, material);
  obj.frustumCulled = false;
  scene.add(obj);

  // ワイヤーフレーム（構造バネ = 四角形グリッド）
  const wireGeo = new THREE.BufferGeometry();
  wireGeo.setAttribute('position', posAttr);
  wireGeo.setIndex(new THREE.BufferAttribute(mesh.structEdges, 1));
  const wireMat = new THREE.LineBasicMaterial({ transparent: true });
  styleWire(wireMat);
  const wire = new THREE.LineSegments(wireGeo, wireMat);
  wire.frustumCulled = false;
  scene.add(wire);

  // プレクサス（発光ドット + 細いリンク。ひずみで色が変わる）
  const dotGeo = new THREE.BufferGeometry();
  dotGeo.setAttribute('position', posAttr);
  dotGeo.setAttribute('aStrain', strainAttr);
  const dots = new THREE.Points(dotGeo, makeDotMaterial());
  dots.frustumCulled = false;
  dots.renderOrder = 2;
  scene.add(dots);

  const linkGeo = new THREE.BufferGeometry();
  linkGeo.setAttribute('position', posAttr);
  linkGeo.setAttribute('aStrain', strainAttr);
  linkGeo.setIndex(new THREE.BufferAttribute(mesh.triEdges, 1));
  const links = new THREE.LineSegments(linkGeo, makeLinkMaterial());
  links.frustumCulled = false;
  links.renderOrder = 1;
  scene.add(links);

  glObjs = { geo, wireGeo, dotGeo, linkGeo, obj, wire, dots, links, material, textures, strainAttr };
  state.mesh = mesh;
  state.views = views;
  state.isSample = !!(mesh.isSample || mesh.isModel); // 画像テクスチャを持たない（青いスライム色で出力）
  material.uniforms.uFace.value = mesh.isSample ? 1 : 0;
  material.uniforms.uVC.value = mesh.colors ? 1 : 0;
  state.body = new SoftBody(mesh, params);
  state.body.free = state.phys === 'gravity';
  state.kappaSmooth = null;

  // 床を物体の真下へ
  let minY = Infinity;
  for (let i = 0; i < n; i++) minY = Math.min(minY, mesh.pos[3 * i + 1]);
  floorLevel = minY - 0.14;
  if (floor) floor.position.y = floorLevel;
  state.body.floorY = floorLevel + 0.01;

  geo.computeBoundingSphere();
  applyDisplayMode();
  $('#tele-verts').textContent = n.toLocaleString('en-US');
  $('#tele-springs').textContent = mesh.spring.count.toLocaleString('en-US');
  syncGeometry();
}

function syncGeometry() {
  const { geo, strainAttr } = glObjs;
  const pos = geo.getAttribute('position');
  const x = state.body.x, arr = pos.array;
  for (let i = 0; i < arr.length; i++) arr[i] = x[i];
  pos.needsUpdate = true;
  geo.computeVertexNormals();
  if (state.mode === 'plexus') {
    state.body.computeStrain(strainAttr.array);
    strainAttr.needsUpdate = true;
  }
}

// 画像スロット

/**
 * サムネイルは <img> に表示し、CSS の object-fit: contain で縦横比を保ったまま全体を枠内に収める。
 * 元のファイル（Blob）から直接 URL を作るので、読み込みと同時に即表示される。
 */
function setThumb(key, blob) {
  const img = $(`.slot[data-view="${key}"] .slot-thumb`);
  if (!img) return;
  if (img._url) { URL.revokeObjectURL(img._url); img._url = null; }
  if (!blob) { img.removeAttribute('src'); img.hidden = true; return; }
  img._url = URL.createObjectURL(blob);
  img.src = img._url;
  img.hidden = false;
}

function initAccordion() {
  const btn = $('#btn-more'), wrap = $('#more-wrap');
  const set = (open) => { btn.setAttribute('aria-expanded', String(open)); wrap.classList.toggle('is-open', open); };
  btn.addEventListener('click', () => set(btn.getAttribute('aria-expanded') !== 'true'));
  set(false);
}

function updateSlotUI(key) {
  const el = $(`.slot[data-view="${key}"]`);
  if (!el) return;
  const s = state.slots[key];
  el.classList.toggle('is-filled', !!s);
  const st = $('.slot-state', el);
  if (st) st.textContent = s ? `${s.label} ${s.w}x${s.h}` : '';
  el.title = s ? s.name : '';
  $('.slot-clear', el).hidden = !s;
}

function setSlotFromSource(key, src, sw, sh, label, file) {
  const scale = Math.min(1, MAX_DIM / Math.max(sw, sh));
  const w = Math.max(2, Math.round(sw * scale)), h = Math.max(2, Math.round(sh * scale));
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  const ctx = cv.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(src, 0, 0, w, h);
  const img = ctx.getImageData(0, 0, w, h);
  state.slots[key] = {
    rgba: img.data, w, h,
    label: label.length > 14 ? label.slice(0, 11) + '...' : label,
    name: label,
    analysis: null,
    thumb: cv,
  };
  setThumb(key, file || null);
  updateSlotUI(key);
  scheduleRebuild(60);
}

async function decodeImageFile(file) {
  if (typeof createImageBitmap === 'function') {
    try {
      const bmp = await createImageBitmap(file);
      return { src: bmp, w: bmp.width, h: bmp.height, close: () => { if (bmp.close) bmp.close(); } };
    } catch (e) { /* <img> 経由へフォールバック */ }
  }
  const url = URL.createObjectURL(file);
  try {
    const img = await new Promise((resolve, reject) => {
      const im = new Image();
      im.onload = () => resolve(im);
      im.onerror = () => reject(new Error('decode failed'));
      im.src = url;
    });
    return { src: img, w: img.naturalWidth, h: img.naturalHeight, close: () => {} };
  } finally {
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }
}

async function loadFileToSlot(key, file) {
  if (!file) return;
  if (file.type && !file.type.startsWith('image/')) {
    setStatus('st.notimage', 'error', { name: file.name });
    return;
  }
  try {
    const dec = await decodeImageFile(file);
    setSlotFromSource(key, dec.src, dec.w, dec.h, file.name, file);
    dec.close();
  } catch (err) {
    console.error(err);
    setStatus('st.imgfail', 'error', { name: file.name });
  }
}

function clearSlot(key) {
  delete state.slots[key];
  setThumb(key, null);
  updateSlotUI(key);
  scheduleRebuild(60);
}

function initSlots() {
  VIEW_KEYS.forEach((key) => setThumb(key, null));
  VIEW_KEYS.forEach((key) => {
    const el = $(`.slot[data-view="${key}"]`);
    const input = $('input[type="file"]', el);
    input.addEventListener('change', () => {
      const f = input.files && input.files[0];
      if (f) loadFileToSlot(key, f);
      input.value = '';
    });
    const over = (e) => { e.preventDefault(); e.stopPropagation(); el.classList.add('is-drop'); };
    el.addEventListener('dragenter', over);
    el.addEventListener('dragover', over);
    el.addEventListener('dragleave', (e) => { e.preventDefault(); el.classList.remove('is-drop'); });
    el.addEventListener('drop', (e) => {
      e.preventDefault();
      e.stopPropagation();
      el.classList.remove('is-drop');
      const files = e.dataTransfer && e.dataTransfer.files;
      if (files && files[0]) loadFileToSlot(key, files[0]);
    });
    $('.slot-clear', el).addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      clearSlot(key);
    });
  });
  // ウィンドウ全体へのドロップでファイルが開かれてしまうのを防ぐ
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => e.preventDefault());
}

// メッシュ再生成（画像なし = 見本モデル）

let rebuildTimer = null, buildToken = 0;

function scheduleRebuild(delay = 200) {
  if (!engineReady) return;           // エンジン起動後に rebuild() が呼ばれる
  clearTimeout(rebuildTimer);
  rebuildTimer = setTimeout(rebuild, delay);
}

function analyzeAll() {
  const views = {};
  for (const key of VIEW_KEYS) {
    const s = state.slots[key];
    if (!s) continue;
    if (!s.analysis || s.analysis.bgUsed !== params.bgth) {
      s.analysis = analyzeImage(s.rgba, s.w, s.h, params.bgth);
      s.analysis.bgUsed = params.bgth;
      s.analysis.contrastUsed = null;
    }
    if (s.analysis.contrastUsed !== params.contrast) {
      computeHeight(s.analysis, params.contrast);
      s.analysis.contrastUsed = params.contrast;
    }
    views[key] = s.analysis;
  }
  return views;
}

async function rebuild() {
  if (!engineReady) return;
  const token = ++buildToken;
  const keys = VIEW_KEYS.filter((k) => state.slots[k]);
  setStatus('st.building', 'busy');
  await sleep(24);
  if (token !== buildToken) return;

  const t0 = performance.now();
  try {
    let mesh, views = {};
    if (state.source === 'model' && state.model) {
      mesh = buildObjMesh(state.model.parsed, state.model);
      if (!mesh) { const e = new Error('objfail'); e.i18n = 'st.modelfail'; e.args = { name: state.model.name }; throw e; }
    } else if (!keys.length || state.source === 'model') {
      mesh = buildSampleMesh(params.res);
    } else {
      views = analyzeAll();
      mesh = buildMesh(views, { resolution: params.res, relief: params.relief, reliefDepth: params.depth });
      if (!mesh) { const e = new Error('nomesh'); e.i18n = 'st.nomesh'; throw e; }
    }
    if (token !== buildToken) return;
    installMesh(views, mesh);
    const ms = Math.round(performance.now() - t0);
    if (mesh.isModel) {
      state.model.info = mesh.info;
      renderModelPane();
      const warns = modelWarnings(mesh.info);
      const args = { name: state.model.name, v: mesh.info.verts.toLocaleString('en-US'), ms };
      if (warns.length) { args.warn = '@' + warns[0].key; Object.assign(args, warns[0].args); setStatus('st.modelWarn', 'warn', args); }
      else setStatus('st.model', '', args);
    } else if (mesh.isSample) {
      setStatus('st.sample', '', { ms });
    } else {
      const warn = keys.some((k) => views[k].fallback);
      setStatus(warn ? 'st.doneWarn' : 'st.done', '', { mode: mesh.ext.freeAxis ? '@mode.relief' : '@mode.hull', views: keys.length, ms });
    }
    setTeleState('running');
  } catch (err) {
    console.error(err);
    if (err && err.i18n) {
      setStatus(err.i18n, 'error', err.args || {});
      if (err.i18n === 'st.modelfail') { state.model = null; renderModelPane(); }
    }
    else setStatus('st.raw', 'error', { msg: err && err.message ? err.message : String(err) });
  }
}

// スライダー（数値表示と赤い塗りバーをリアルタイム連動）
 
const SLIDERS = {
  res: { mesh: true, fmt: (v) => String(Math.round(v)) },
  bgth: { mesh: true, fmt: (v) => v.toFixed(2) },
  relief: { mesh: true, fmt: (v) => v.toFixed(2) },
  depth: { mesh: true, fmt: (v) => v.toFixed(2) },
  contrast: { mesh: true, fmt: (v) => v.toFixed(1) },
  k: { fmt: (v) => String(Math.round(v)) },
  c: { fmt: (v) => v.toFixed(1) },
  g: { fmt: (v) => v.toFixed(1) },
  R: { fmt: (v) => v.toFixed(2) },
  beta: { fmt: (v) => v.toFixed(1) },
  rho: { fmt: (v) => v.toFixed(2) },
};

function syncSlider(name) {
  const input = $('#p-' + name), out = $('#o-' + name);
  if (!input || !out) return;
  const v = parseFloat(input.value);
  params[name] = v;
  out.textContent = SLIDERS[name].fmt(v);
  const min = parseFloat(input.min), max = parseFloat(input.max);
  input.style.setProperty('--pf', String(max > min ? (v - min) / (max - min) : 0));
}

function initSliders() {
  Object.keys(SLIDERS).forEach((name) => {
    const input = $('#p-' + name);
    if (!input) return;
    input.addEventListener('input', () => {
      syncSlider(name);
      if (SLIDERS[name].mesh && state.source === 'image') scheduleRebuild(220);
    });
    syncSlider(name);
  });
}

function resetParams() {
  Object.assign(params, DEFAULT_PARAMS);
  Object.keys(SLIDERS).forEach((name) => {
    const input = $('#p-' + name);
    if (input) input.value = String(DEFAULT_PARAMS[name]);
    syncSlider(name);
  });
}

// 3Dモデル（.obj）読込タブ


function modelWarnings(info) {
  const out = [];
  if (info.vertsIn > MODEL_WARN_VERTS) out.push({ key: 'model.warn.heavy', args: {} });
  if (info.decimated) out.push({ key: 'model.warn.decimated', args: { before: info.vertsIn.toLocaleString('en-US'), after: info.verts.toLocaleString('en-US') } });
  if (!info.closed) out.push({ key: 'model.warn.open', args: {} });
  return out;
}

/** 色の取得元（頂点カラー / .mtl の Kd / テクスチャ）と、足りないファイルの警告を調べる */
function modelColorSummary(m) {
  const p = m.parsed;
  const used = Array.from(new Set(p.faceMat.filter(Boolean)));
  let kd = false, tex = false;
  const missingTex = new Set();
  for (const name of used) {
    const mt = m.mtls[name];
    if (!mt) continue;
    if (mt.kd) kd = true;
    if (mt.map) { if (m.textures[mt.map.toLowerCase()]) tex = true; else missingTex.add(mt.map); }
  }
  const vc = !!p.vcol;
  const kinds = [vc, kd, tex].filter(Boolean).length;
  const colored = !m.info || m.info.colored;
  let key = 'model.color.none';
  if (colored && kinds) key = kinds > 1 ? 'model.color.mix' : vc ? 'model.color.vertex' : kd ? 'model.color.mtl' : 'model.color.tex';
  const warns = [];
  const loaded = new Set(m.mtlFiles.map((n) => n.toLowerCase()));
  const missingMtl = p.mtllibs.filter((n) => !loaded.has(n.toLowerCase()));
  if (used.length && missingMtl.length && !used.every((u) => m.mtls[u])) warns.push({ key: 'model.warn.nomtl', args: { name: missingMtl[0] } });
  for (const name of Array.from(missingTex).slice(0, 2)) warns.push({ key: 'model.warn.notex', args: { name } });
  return { key, warns };
}

function renderModelPane() {
  const info = $('#model-info');
  if (!info) return;
  const m = state.model;
  const zone = $('#slot-model');
  zone.classList.toggle('is-filled', !!m);
  $('#btn-model-clear').hidden = !m;
  $('#model-title').textContent = m ? m.name : t('model.drop.title');
  $('#model-cta').textContent = m ? t('model.drop.replace') : t('model.drop.cta');
  info.hidden = !(m && m.info);
  if (!(m && m.info)) return;
  const i = m.info;
  $('#model-name').textContent = m.name;
  $('#model-stats').textContent = t('model.stats', { v: i.verts.toLocaleString('en-US'), t: i.tris.toLocaleString('en-US'), s: i.springs.toLocaleString('en-US'), i: i.supports.toLocaleString('en-US') });
  const extra = m.mtlFiles.concat(m.texFiles);
  const fl = $('#model-files');
  fl.hidden = !extra.length;
  fl.textContent = extra.length ? t('model.files', { list: extra.join(', ') }) : '';
  const cs = modelColorSummary(m);
  $('#model-color').textContent = t(cs.key);
  $('#model-color').dataset.color = cs.key === 'model.color.none' ? 'none' : 'on';
  const ul = $('#model-warns');
  ul.textContent = '';
  for (const w of modelWarnings(i).concat(cs.warns)) {
    const li = document.createElement('li');
    li.textContent = t(w.key, w.args);
    ul.appendChild(li);
  }
}

function setSource(src) {
  state.source = src === 'model' ? 'model' : 'image';
  const isModel = state.source === 'model';
  $$('.tab').forEach((b) => {
    const on = b.dataset.src === state.source;
    b.classList.toggle('is-active', on);
    b.setAttribute('aria-selected', String(on));
  });
  $('#pane-image').hidden = isModel;
  $('#pane-model').hidden = !isModel;
  $('#card-images').dataset.source = state.source;
  $('#panel').dataset.source = state.source;
  cancelGrab();
  scheduleRebuild(30);
}

const IMG_EXT = /\.(png|jpe?g|webp|bmp|gif)$/i;

/** テクスチャ画像を RGBA 配列にする（大きい画像は最大 1024 px に縮小） */
async function decodeTexture(file) {
  const bmp = await createImageBitmap(file);
  const sc = Math.min(1, 1024 / Math.max(bmp.width, bmp.height));
  const w = Math.max(1, Math.round(bmp.width * sc)), h = Math.max(1, Math.round(bmp.height * sc));
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const g = c.getContext('2d', { willReadFrequently: true });
  g.drawImage(bmp, 0, 0, w, h);
  if (bmp.close) bmp.close();
  return { w, h, data: g.getImageData(0, 0, w, h).data };
}

/**
 * 3Dモデルを読み込む。形式は拡張子から自動判別する。
 * 一緒に選ばれた マテリアル・テクスチャ画像・外部データ も使い、あとから足されたファイルは合流させる。
 */
async function loadModelFiles(list) {
  const files = Array.from(list || []);
  if (!files.length) return;
  const modelFile = files.find((f) => MODEL_EXTS.indexOf(fileExt(f.name)) >= 0);
  const mtls = files.filter((f) => fileExt(f.name) === 'mtl');
  const imgs = files.filter((f) => IMG_EXT.test(f.name) || (f.type || '').indexOf('image/') === 0);
  const others = files.filter((f) => f !== modelFile && mtls.indexOf(f) < 0 && imgs.indexOf(f) < 0);
  const wantsAux = !!state.pendingModel || (state.model && state.model.kind === 'gltf');   // 外部データを足せるのは glTF だけ
  if (!modelFile && !mtls.length && !imgs.length && !wantsAux) { setStatus('st.notmodel', 'error', { name: files[0].name }); return; }
  if (!modelFile && !state.model && !state.pendingModel) { setStatus(mtls.length || imgs.length || others.length ? 'st.needmodel' : 'st.notmodel', 'error', { name: files[0].name }); return; }
  let model = state.pendingModel || state.model;
  try {
    setStatus('st.building', 'busy');
    if (modelFile) {
      model = { name: modelFile.name, kind: fileExt(modelFile.name), file: modelFile, aux: new Map(), parsed: null, info: null, mtls: {}, textures: {}, mtlFiles: [], texFiles: [] };
    }
    for (const f of imgs.concat(others)) model.aux.set(f.name.toLowerCase(), f);
    for (const f of mtls) {
      Object.assign(model.mtls, parseMTL(await f.text()));
      if (model.mtlFiles.indexOf(f.name) < 0) model.mtlFiles.push(f.name);
    }
    if (model.kind === 'obj') {
      if (modelFile) model.parsed = parseOBJ(await modelFile.text());
      for (const f of imgs) {
        try {
          model.textures[f.name.toLowerCase()] = await decodeTexture(f);
          if (model.texFiles.indexOf(f.name) < 0) model.texFiles.push(f.name);
        } catch (err) {
          console.error(err);
          setStatus('st.texfail', 'error', { name: f.name });
        }
      }
    } else if (modelFile || imgs.length || others.length) {
      // obj 以外：外部ファイルが足りないときは、そのファイル名を知らせる。あとから足されたら読み直す
      const r = await parseModelFile(model.file, model.aux);
      model.parsed = r.parsed; model.mtls = r.mtls; model.textures = r.textures;
    }
    if (!model.parsed || model.parsed.pos.length < 9 || !model.parsed.faces.length) { setStatus('st.modelfail', 'error', { name: model.name }); return; }
    state.model = model;
    state.pendingModel = null;
    renderModelPane();
    setSource('model');
    if (engineReady) { clearTimeout(rebuildTimer); await rebuild(); }
  } catch (err) {
    if (!(err && err.missing)) console.error(err);
    if (err && err.missing) state.pendingModel = model;   // 外部ファイルが足りない：あとから足されたら続きを読む
    if (err && err.missing) setStatus('st.needfile', 'error', { name: err.missing });
    else setStatus('st.modelfail', 'error', { name: (modelFile || files[0]).name });
  }
}

function clearModel() {
  state.model = null;
  state.pendingModel = null;
  renderModelPane();
  scheduleRebuild(30);
}

function initModel() {
  $$('.tab').forEach((b) => b.addEventListener('click', () => setSource(b.dataset.src)));
  const el = $('#slot-model'), input = $('#file-model');
  input.addEventListener('change', () => {
    const fl = input.files ? Array.from(input.files) : [];
    input.value = '';
    if (fl.length) loadModelFiles(fl);
  });
  const over = (e) => { e.preventDefault(); e.stopPropagation(); el.classList.add('is-drop'); };
  el.addEventListener('dragenter', over);
  el.addEventListener('dragover', over);
  el.addEventListener('dragleave', (e) => { e.preventDefault(); el.classList.remove('is-drop'); });
  el.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    el.classList.remove('is-drop');
    const fl = e.dataTransfer && e.dataTransfer.files ? Array.from(e.dataTransfer.files) : [];
    if (fl.length) loadModelFiles(fl);
  });
  // ビューポートへ直接ドロップしても読み込める（3Dモデルのタブが開いているとき、または .obj / .mtl のとき）
  const vp = $('#viewport');
  vp.addEventListener('dragover', (e) => e.preventDefault());
  vp.addEventListener('drop', (e) => {
    e.preventDefault();
    const fl = e.dataTransfer && e.dataTransfer.files ? Array.from(e.dataTransfer.files) : [];
    if (fl.some((f) => MODEL_EXTS.indexOf(fileExt(f.name)) >= 0 || fileExt(f.name) === 'mtl') || (state.source === 'model' && fl.length)) loadModelFiles(fl);
  });
  $('#btn-model-clear').addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); clearModel(); });
  renderModelPane();
}

// 確認モーダルと「初期化」

function confirmModal(title, message) {
  return new Promise((resolve) => {
    const modal = $('#modal'), ok = $('#modal-ok'), cancel = $('#modal-cancel');
    $('#modal-title').textContent = title;
    $('#modal-msg').textContent = message;
    ok.textContent = t('modal.ok');
    cancel.textContent = t('modal.cancel');
    modal.hidden = false;

    const finish = (value) => {
      modal.hidden = true;
      ok.removeEventListener('click', onOk);
      cancel.removeEventListener('click', onCancel);
      modal.removeEventListener('mousedown', onBackdrop);
      document.removeEventListener('keydown', onKey, true);
      resolve(value);
    };
    const onOk = () => finish(true);
    const onCancel = () => finish(false);
    const onBackdrop = (e) => { if (e.target === modal) finish(false); };
    const onKey = (e) => { if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); finish(false); } };
    ok.addEventListener('click', onOk);
    cancel.addEventListener('click', onCancel);
    modal.addEventListener('mousedown', onBackdrop);
    document.addEventListener('keydown', onKey, true);
    ok.focus();
  });
}

function cancelGrab() {
  state.grab = null;
  dragMode = 'none';
  if (canvasEl) canvasEl.classList.remove('is-grabbing');
  if (pullArrow) pullArrow.visible = false;
  if (pullMarker) pullMarker.visible = false;
}

async function resetAll() {
  const ok = await confirmModal(t('reset.title'), t('reset.msg'));
  if (!ok) return;
  VIEW_KEYS.forEach((k) => { delete state.slots[k]; setThumb(k, null); updateSlotUI(k); });
  state.model = null;
  state.pendingModel = null;
  renderModelPane();
  setSource('model');
  setPhys('fixed');
  setPaused(false);
  resetParams();
  cancelGrab();
  resetView();
  if (engineReady) { clearTimeout(rebuildTimer); await rebuild(); }
  if (state.status.kind !== 'error') setStatus('st.reset');
}

// ヘッダー操作, エクスポート


function downloadBlob(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 3000);
}

/** スクリーンショット保存（3D ビューを PNG で書き出す。時間停止中でもそのまま撮れる） */
function saveScreenshot() {
  if (!renderer) { setStatus('st.noexport', 'error'); return false; }
  renderer.render(scene, camera);
  canvasEl.toBlob((blob) => {
    if (!blob) return;
    const d = new Date(), z = (v) => String(v).padStart(2, '0');
    const stamp = `${d.getFullYear()}${z(d.getMonth() + 1)}${z(d.getDate())}-${z(d.getHours())}${z(d.getMinutes())}${z(d.getSeconds())}`;
    downloadBlob(blob, `slime-${stamp}.png`);
    setStatus('st.shotDone');
  }, 'image/png');
  return true;
}

// 時間停止（Sキー）と 固定 / 重力モード

function setPaused(on) {
  state.paused = !!on;
  if (state.paused) cancelGrab();
  const chip = $('#pause-chip');
  if (chip) chip.hidden = !state.paused;
  const b = $('#btn-pause');
  if (b) b.setAttribute('aria-pressed', String(state.paused));
  document.body.classList.toggle('is-paused', state.paused);
  setTeleState(state.paused ? 'paused' : (state.grab ? 'grab' : 'running'));
}

function setPhys(mode) {
  const next = mode === 'gravity' ? 'gravity' : 'fixed';
  const changed = next !== state.phys;
  state.phys = next;
  saveSetting('phys', next);
  $$('[data-phys]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.phys === next)));
  const g = $('#p-g');
  if (g) { g.disabled = next !== 'gravity'; g.closest('.field').classList.toggle('is-off', next !== 'gravity'); }
  document.body.dataset.phys = next;
  const body = state.body;
  if (body) {
    body.free = next === 'gravity';
    if (changed && body.free) {            // 重力モードへ：少し持ち上げて落下させる
      const x = body.x, v = body.v;
      for (let i = 0; i < body.n; i++) { x[3 * i + 1] += 0.9; }
      v.fill(0);
    }
    if (changed && !body.free) body.v.fill(0);
  }
}

function initHeader() {
  $('#btn-reset-view').addEventListener('click', resetView);

  $$('[data-mode]').forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)));
  $$('[data-phys]').forEach((b) => b.addEventListener('click', () => setPhys(b.dataset.phys)));
  $$('[data-theme-btn]').forEach((b) => b.addEventListener('click', () => setTheme(b.dataset.themeBtn)));
  $$('[data-lang-btn]').forEach((b) => b.addEventListener('click', () => setLang(b.dataset.langBtn)));

  $('#btn-shot').addEventListener('click', () => saveScreenshot());
  $('#btn-pause').addEventListener('click', () => setPaused(!state.paused));
  // S キー：時間停止 / 再開（停止中もカメラは回せるので、好きな角度で撮影できる）
  document.addEventListener('keydown', (e) => {
    if (e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.code !== 'KeyS' && e.key !== 's' && e.key !== 'S') return;
    const m = $('#modal');
    if (m && !m.hidden) return;
    e.preventDefault();
    setPaused(!state.paused);
  });

  $('#btn-collapse').addEventListener('click', () => setCollapsed(true));
  $('#btn-expand').addEventListener('click', () => setCollapsed(false));
  $('#btn-reset-all').addEventListener('click', resetAll);
  $('#btn-engine-retry').addEventListener('click', () => startEngine());
}

/*
 * カメラ・把持の入力（完全分離）
 *   Shift + 左ドラッグ : カメラ回転
 *   左ドラッグ         : メッシュ表面を把持して引っ張る（静止しても毎フレーム牽引）
 *   右 / 中ドラッグ    : 平行移動,  ホイール : ズーム
 */

let dragMode = 'none';
let lastX = 0, lastY = 0;

/** 仮想ポインタ（vp）の位置から NDC を求める。ポインターロック中は画面外へも延長できる。 */
const vp = { x: 0, y: 0 };
function setPointerNdc() {
  const r = canvasEl.getBoundingClientRect();
  state.pointer.x = ((vp.x - r.left) / r.width) * 2 - 1;
  state.pointer.y = -(((vp.y - r.top) / r.height) * 2 - 1);
}

function beginGrab(e) {
  if (!state.body || !glObjs || state.paused) return false;
  vp.x = e.clientX; vp.y = e.clientY;
  setPointerNdc();
  glObjs.geo.computeBoundingSphere();
  raycaster.setFromCamera(state.pointer, camera);
  const hits = raycaster.intersectObject(glObjs.obj, false);
  if (!hits.length || !hits[0].face) return false;
  const hit = hits[0], f = hit.face, x = state.body.x;
  let best = f.a, bd = Infinity;
  for (const idx of [f.a, f.b, f.c]) {
    const dx = x[3 * idx] - hit.point.x, dy = x[3 * idx + 1] - hit.point.y, dz = x[3 * idx + 2] - hit.point.z;
    const d = dx * dx + dy * dy + dz * dz;
    if (d < bd) { bd = d; best = idx; }
  }
  const normals = glObjs.geo.getAttribute('normal').array;
  const grab = state.body.makeGrab(best, params.R, normals);
  camera.getWorldDirection(tmpV.v);
  dragPlane.setFromNormalAndCoplanarPoint(tmpV.v, new THREE.Vector3(grab.p0[0], grab.p0[1], grab.p0[2]));
  grab.target = grab.p0.slice();
  state.grab = grab;
  return true;
}

/** 毎フレーム：現在のポインタ位置から Raycaster でレイを引き、視線直交平面上の目標点を更新する */
function updateGrabTarget() {
  const grab = state.grab;
  if (!grab) return;
  raycaster.setFromCamera(state.pointer, camera);
  if (raycaster.ray.intersectPlane(dragPlane, tmpV.target)) {
    let dx = tmpV.target.x - grab.p0[0], dy = tmpV.target.y - grab.p0[1], dz = tmpV.target.z - grab.p0[2];
    const len = Math.hypot(dx, dy, dz), cap = 3;
    if (len > cap) { const s = cap / len; dx *= s; dy *= s; dz *= s; }
    grab.delta[0] = dx; grab.delta[1] = dy; grab.delta[2] = dz;
    grab.target[0] = grab.p0[0] + dx; grab.target[1] = grab.p0[1] + dy; grab.target[2] = grab.p0[2] + dz;
  }
}

function endDrag(e) {
  if (dragMode === 'grab') cancelGrab();
  dragMode = 'none';
  if (document.pointerLockElement === canvasEl && document.exitPointerLock) document.exitPointerLock();
  if (e && canvasEl.hasPointerCapture && canvasEl.hasPointerCapture(e.pointerId)) {
    try { canvasEl.releasePointerCapture(e.pointerId); } catch (err) { /* 既に解放済み */ }
  }
}

function bindCanvasEvents() {
  if (eventsBound) return;
  eventsBound = true;

  canvasEl.addEventListener('pointerdown', (e) => {
    lastX = e.clientX; lastY = e.clientY;
    if (e.button === 0 && e.shiftKey) {
      dragMode = 'rotate';
    } else if (e.button === 0) {
      dragMode = beginGrab(e) ? 'grab' : 'none';
      if (dragMode === 'grab') {
        canvasEl.classList.add('is-grabbing');
        // カーソルを固定して、画面端で止まらずに無限に引っ張れるようにする
        try {
          const req = canvasEl.requestPointerLock && canvasEl.requestPointerLock();
          if (req && req.catch) req.catch(() => { /* 取得できなくても通常のドラッグで続行 */ });
        } catch (err) { /* 続行 */ }
      }
    } else if (e.button === 1 || e.button === 2) {
      dragMode = 'pan';
    } else {
      return;
    }
    try { canvasEl.setPointerCapture(e.pointerId); } catch (err) { /* 無視 */ }
    e.preventDefault();
  });

  canvasEl.addEventListener('pointermove', (e) => {
    const locked = document.pointerLockElement === canvasEl;
    if (locked) { vp.x += e.movementX; vp.y += e.movementY; }
    else { vp.x = e.clientX; vp.y = e.clientY; }
    setPointerNdc();
    const dx = locked ? e.movementX : e.clientX - lastX, dy = locked ? e.movementY : e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    if (dragMode === 'rotate') {
      cam.theta -= dx * 0.008;
      cam.phi = clamp(cam.phi - dy * 0.008, 0.05, Math.PI - 0.05);
      applyCamera();
    } else if (dragMode === 'pan') {
      const unit = (2 * cam.radius * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) / Math.max(1, canvasEl.clientHeight);
      const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
      const up = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1);
      cam.target.addScaledVector(right, -dx * unit).addScaledVector(up, dy * unit);
      applyCamera();
    }
    // 'grab' は座標を保存するのみ。牽引は requestAnimationFrame 内で毎フレーム継続する。
  });

  let lockWas = false;
  document.addEventListener('pointerlockchange', () => {
    if (document.pointerLockElement === canvasEl) lockWas = true;
    else if (lockWas) { lockWas = false; if (dragMode === 'grab') endDrag(null); } // Esc などでロックが外れたら把持を終える
  });
  window.addEventListener('pointerup', (e) => { if (dragMode === 'grab') endDrag(e); });
  canvasEl.addEventListener('pointerup', endDrag);
  canvasEl.addEventListener('pointercancel', endDrag);
  canvasEl.addEventListener('lostpointercapture', () => { if (dragMode !== 'none') endDrag(null); });
  canvasEl.addEventListener('contextmenu', (e) => e.preventDefault());
  canvasEl.addEventListener('wheel', (e) => {
    e.preventDefault();
    cam.radius = clamp(cam.radius * Math.exp(e.deltaY * 0.0012), 1.8, 14);
    applyCamera();
  }, { passive: false });

  canvasEl.addEventListener('webglcontextlost', (e) => { e.preventDefault(); setStatus('st.glLost', 'error'); });
  canvasEl.addEventListener('webglcontextrestored', () => { rebuild(); });

  const syncShiftCursor = (e) => canvasEl.classList.toggle('is-rotate', !!e.shiftKey);
  window.addEventListener('keydown', syncShiftCursor);
  window.addEventListener('keyup', syncShiftCursor);
  window.addEventListener('blur', () => canvasEl.classList.remove('is-rotate'));
}

// テレメトリ表示
function fmtNum(v, digits = 4) {
  if (!isFinite(v)) return 'NaN';
  const a = Math.abs(v);
  if (a >= 1e5) return v.toExponential(3);
  if (a >= 1000) return v.toFixed(1);
  return v.toFixed(digits);
}

function updateTelemetry(tele, spec) {
  $('#tele-E').textContent = fmtNum(tele.E, 4);
  $('#tele-grad').textContent = fmtNum(tele.gradMax, 3);
  if (spec) {
    state.kappaSmooth = state.kappaSmooth == null ? spec.kappa : state.kappaSmooth * 0.7 + spec.kappa * 0.3;
    $('#tele-kappa').textContent = fmtNum(state.kappaSmooth, 2);
    $('#tele-lmax').textContent = fmtNum(spec.lamMax, 1);
    $('#tele-lmin').textContent = fmtNum(spec.lamMin, 2);
  }
}

// メインループ

let lastTime = performance.now();
let frameErrors = 0;

function stepFrame(now) {
  const dtFrame = Math.min((now - lastTime) / 1000, 1 / 20);
  lastTime = now;
  if (dtFrame > 0) state.fps = state.fps * 0.92 + (1 / dtFrame) * 0.08;
  state.frame++;

  const body = state.body;
  if (body && glObjs) {
    // 把持中は静止していても毎フレーム Raycaster を更新し、牽引し続ける
    if (!state.paused) updateGrabTarget();
    const steps = state.paused ? 0 : clamp(Math.round(dtFrame / SUB_DT), 1, MAX_SUBSTEPS);
    const wantTele = state.frame % TELE_EVERY === 0;
    let tele = null;
    for (let s = 0; s < steps; s++) {
      const tt = body.step(SUB_DT, state.grab, wantTele && s === steps - 1);
      if (tt) tele = tt;
    }
    if (tele) {
      if (!isFinite(tele.E)) {
        body.reset();
        setStatus('st.diverged', 'error');
      } else {
        updateTelemetry(tele, body.updateSpectrum());
      }
    }
    syncGeometry();

    const gf = body.grabForce;
    $('#tele-force').textContent = state.grab ? Math.hypot(gf[0], gf[1], gf[2]).toFixed(3) : '0.000';
    setTeleState(state.paused ? 'paused' : state.grab ? 'grab' : 'running');

    if (state.grab) {
      const g = state.grab, o = 3 * g.primary;
      const ox = body.x[o], oy = body.x[o + 1], oz = body.x[o + 2];
      const dx = g.target[0] - ox, dy = g.target[1] - oy, dz = g.target[2] - oz;
      const len = Math.hypot(dx, dy, dz);
      pullMarker.position.set(g.target[0], g.target[1], g.target[2]);
      pullMarker.visible = true;
      if (len > 0.02) {
        pullArrow.position.set(ox, oy, oz);
        pullArrow.setDirection(tmpV.v.set(dx / len, dy / len, dz / len));
        pullArrow.setLength(len, Math.min(0.14, len * 0.5), 0.08);
        pullArrow.visible = true;
      } else {
        pullArrow.visible = false;
      }
    }
  }

  if (state.frame % 15 === 0) $('#tele-fps').textContent = String(Math.round(state.fps));
  renderer.render(scene, camera);
}

function frame(now) {
  requestAnimationFrame(frame);
  try {
    stepFrame(now);
    frameErrors = 0;
  } catch (err) {
    frameErrors++;
    if (frameErrors === 1) {
      console.error(err);
      setStatus('st.raw', 'error', { msg: 'render: ' + (err && err.message ? err.message : err) });
    }
    if (frameErrors > 20 && glObjs) disposeMesh(); // 連続エラー時はメッシュを破棄してループを守る
  }
}

// 3D エンジンの初期化と起動

function resizeRenderer() {
  if (!renderer) return;
  const vp = $('#viewport');
  const w = Math.max(1, vp.clientWidth), h = Math.max(1, vp.clientHeight);
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

async function initEngine() {
  const loaded = await loadThree((i, n) => setStatus('st.loading3d', 'busy', { i, n }));
  THREE = loaded.mod;

  canvasEl = $('#gl');
  try {
    renderer = new THREE.WebGLRenderer({ canvas: canvasEl, antialias: true });
  } catch (e) {
    const err = new Error('WebGL init failed: ' + (e && e.message ? e.message : e));
    err.code = 'webgl';
    throw err;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(THEME_GL[state.theme].bg, 1);

  scene = new THREE.Scene();
  stars = null;
  scene.fog = new THREE.Fog(THEME_GL[state.theme].bg, 7, 15); // 床グリッドを遠方でフェード（ShaderMaterial には影響しない）
  camera = new THREE.PerspectiveCamera(40, 1, 0.05, 100);
  cam.target = new THREE.Vector3(0, 0, 0);
  raycaster = new THREE.Raycaster();
  dragPlane = new THREE.Plane();
  tmpV.v = new THREE.Vector3();
  tmpV.target = new THREE.Vector3();

  buildFloor();

  // 牽引ベクトル表示
  const BLUE = 0x005AFF;
  pullArrow = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(), 1, BLUE, 0.14, 0.08);
  pullArrow.line.material.depthTest = false;
  pullArrow.cone.material.depthTest = false;
  pullArrow.line.renderOrder = pullArrow.cone.renderOrder = 10;
  pullArrow.visible = false;
  scene.add(pullArrow);
  pullMarker = new THREE.Mesh(
    new THREE.SphereGeometry(0.035, 16, 12),
    new THREE.MeshBasicMaterial({ color: BLUE, depthTest: false }),
  );
  pullMarker.renderOrder = 11;
  pullMarker.visible = false;
  scene.add(pullMarker);

  resizeRenderer();
  new ResizeObserver(resizeRenderer).observe($('#viewport'));
  resetView();
  bindCanvasEvents();
}

async function startEngine() {
  const box = $('#engine-error');
  if (box) box.hidden = true;
  try {
    if (!renderer) await initEngine();
    engineReady = true;
    window.__studioReady = true;
    await rebuild();
    if (!loopStarted) {
      loopStarted = true;
      lastTime = performance.now();
      requestAnimationFrame(frame);
    }
  } catch (err) {
    console.error(err);
    renderer = null;
    engineReady = false;
    showEngineError(err);
  }
}

// 起動

function boot() {
  const jaLike = (navigator.language || 'ja').toLowerCase().startsWith('ja');
  state.lang = loadSetting('lang', jaLike ? 'ja' : 'en') === 'en' ? 'en' : 'ja';
  state.theme = loadSetting('theme', 'light') === 'dark' ? 'dark' : 'light';
  state.phys = 'fixed';
  state.mode = loadSetting('mode', 'solid');
  state.collapsed = loadSetting('collapsed', '0') === '1';

  safe('theme', () => setTheme(state.theme));
  safe('mode', () => setMode(state.mode));
  safe('lang', () => setLang(state.lang));
  safe('sidebar', () => setCollapsed(state.collapsed));
  safe('sliders', initSliders);
  safe('slots', initSlots);
  safe('accordion', initAccordion);
  safe('model', initModel);
  safe('source', () => setSource(state.source));
  safe('phys', () => setPhys(state.phys));
  safe('header', initHeader);

  // デバッグ, 自動テスト用のフック
  window.__studio = {
    state, params, cam, applyCamera,
    get engineReady() { return engineReady; },
    get gl() { return glObjs; },
    get scene() { return scene; },
    get stars() { return stars; },
    get floor() { return floor; },
    renderNow() { if (renderer) renderer.render(scene, camera); },
  };

  setStatus('st.loading3d', 'busy', { i: 1, n: THREE_SOURCES.length });
  return startEngine();
}

window.addEventListener('error', (e) => {
  if (e && e.message && !/ResizeObserver/i.test(e.message)) setStatus('st.raw', 'error', { msg: e.message });
});
window.addEventListener('unhandledrejection', (e) => {
  const r = e && e.reason;
  setStatus('st.raw', 'error', { msg: r && r.message ? r.message : String(r) });
});

boot().catch((err) => {
  console.error(err);
  setStatus('st.raw', 'error', { msg: err && err.message ? err.message : String(err) });
});