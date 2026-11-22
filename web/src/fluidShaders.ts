// web/src/fluidShaders.ts
// Aster Hobby JNR D51 — 七大零件级三维多相热流体与共轭传热实时可视化着色器系统
// (1) 23-1 直列 1×3 三头立式酒精燃烧器 + セラミックヤーン多孔陶瓷灯芯三维毛细汽化湍流反应火焰 (受三维内腔连通性物理约束!)
// (2) 锅炉内胆三维晃动沸腾自由液面 + 4 层水温梯度体 + 180 颗核态沸腾上升汽泡 + 11-7 驾驶室水面计实况水柱
// (3) RTX 5080 64×24 纵剖面实时三维热场云图切片 (DataTexture 硬件双线性插值)
// (4) 左右主气缸前后腔动态伸缩蒸汽膨胀做功云、滑阀进排汽口脉冲、缸底冷凝水膜与 4-2 置换注油器油水分层内芯
// (5) 真实 CAD 零件网格沿程相变着色器: 17-5 四程过热管湿蒸汽→过热橙红渐变、6-6/6-8 给水脉冲、火室顶板 6 分区局部干烧红热

import * as THREE from "three";
import { CylinderCalibration, SpatialFluidState, StepOutput } from "./thermo1d";

export interface CadShaderUniforms {
  uTime: { value: number };
  uXray: { value: number };
  uSteamFlow: { value: number };
  uDomeDryness: { value: number };
  uSuperheatNorm: { value: number };
  uExhaustPulse: { value: number };
  uFeedPhase: { value: number };
  uFeedStrength: { value: number };
  uCrownTemps: { value: number[] };
  uFlueTemps: { value: number[] };
  uFlueBowMm: { value: number[] };
  uCrownBulgeMm: { value: number };
}

export interface PipeCalloutSnapshot {
  id: string;
  partNo: string;
  title: string;
  zone: "tender" | "umbilical" | "cab" | "boiler" | "underframe" | "cylinder";
  worldPos: THREE.Vector3;
  colorCss: string;
  tempC: number;
  flowText: string;
  dirSymbol: string;
  active: boolean;
}

const THERMAL_STOPS: { t: number; r: number; g: number; b: number }[] = [
  { t: 20.0, r: 0.114, g: 0.306, b: 0.847 },  // #1d4ed8 常温冷水/冷醇 (深钴蓝)
  { t: 45.0, r: 0.024, g: 0.714, b: 0.831 },  // #06b6d4 预热回水/温醇 (青蓝)
  { t: 80.0, r: 0.063, g: 0.725, b: 0.506 },  // #10b981 暖机汽缸油/热给水 (翠绿)
  { t: 105.0, r: 0.918, g: 0.702, b: 0.031 }, // #eab308 常压沸水/膨胀乏汽 (明黄)
  { t: 148.0, r: 0.961, g: 0.620, b: 0.043 }, // #f59e0b 3.8kgf饱和锅水与湿蒸汽 (金琥珀)
  { t: 210.0, r: 0.976, g: 0.451, b: 0.086 }, // #f97316 四程过热活汽 (炽橙)
  { t: 420.0, r: 0.937, g: 0.267, b: 0.267 }, // #ef4444 火管高温燃气/赤热顶板 (绯红)
  { t: 750.0, r: 0.996, g: 0.941, b: 0.541 }, // #fef08a 酒精外焰核心 (白炽黄)
];

/** 全车液体、气体、蒸汽、油、酒精与管道统一温度正相关色标 (严格单调随温度 °C 升高由深蓝→青→绿→黄→琥珀→橙→红→白炽过渡) */
export function temperatureToThermalColor(tempC: number, out?: THREE.Color): THREE.Color {
  const c = out ?? new THREE.Color();
  const t = Math.max(THERMAL_STOPS[0].t, Math.min(THERMAL_STOPS[THERMAL_STOPS.length - 1].t, tempC));
  for (let i = 0; i < THERMAL_STOPS.length - 1; i++) {
    const a = THERMAL_STOPS[i];
    const b = THERMAL_STOPS[i + 1];
    if (t <= b.t) {
      const u = (t - a.t) / Math.max(1e-6, b.t - a.t);
      return c.setRGB(a.r + (b.r - a.r) * u, a.g + (b.g - a.g) * u, a.b + (b.b - a.b) * u);
    }
  }
  const last = THERMAL_STOPS[THERMAL_STOPS.length - 1];
  return c.setRGB(last.r, last.g, last.b);
}

export function temperatureToThermalCss(tempC: number): string {
  const c = temperatureToThermalColor(tempC);
  const r = Math.round(Math.max(0, Math.min(1, c.r)) * 255);
  const g = Math.round(Math.max(0, Math.min(1, c.g)) * 255);
  const b = Math.round(Math.max(0, Math.min(1, c.b)) * 255);
  return `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;
}

/** 精确 G1 圆弧导角三维折线曲线 (彻底杜绝 CatmullRom 在短 U 弯处产生尖刺自交过冲，100% 贴合 CAD make_swept_tube!) */
class ExactFilletedPipeCurve3 {
  readonly pts: THREE.Vector3[];
  readonly cumLen: Float64Array;
  readonly totalLength: number;

  constructor(ptsM: THREE.Vector3[]) {
    this.pts = ptsM.length >= 2 ? ptsM : [new THREE.Vector3(), new THREE.Vector3(0.001, 0, 0)];
    this.cumLen = new Float64Array(this.pts.length);
    let acc = 0.0;
    for (let i = 1; i < this.pts.length; i++) {
      acc += this.pts[i].distanceTo(this.pts[i - 1]);
      this.cumLen[i] = acc;
    }
    this.totalLength = Math.max(1e-6, acc);
  }

  getPointAt(u: number, target = new THREE.Vector3()): THREE.Vector3 {
    const s = Math.max(0.0, Math.min(1.0, u)) * this.totalLength;
    let lo = 0;
    let hi = this.pts.length - 1;
    while (lo + 1 < hi) {
      const mid = (lo + hi) >> 1;
      if (this.cumLen[mid] <= s) lo = mid;
      else hi = mid;
    }
    const segLen = Math.max(1e-9, this.cumLen[hi] - this.cumLen[lo]);
    const f = Math.max(0.0, Math.min(1.0, (s - this.cumLen[lo]) / segLen));
    return target.lerpVectors(this.pts[lo], this.pts[hi], f);
  }

  getTangentAt(u: number, target = new THREE.Vector3()): THREE.Vector3 {
    const s = Math.max(0.0, Math.min(1.0, u)) * this.totalLength;
    let lo = 0;
    let hi = this.pts.length - 1;
    while (lo + 1 < hi) {
      const mid = (lo + hi) >> 1;
      if (this.cumLen[mid] <= s) lo = mid;
      else hi = mid;
    }
    target.subVectors(this.pts[hi], this.pts[lo]);
    if (target.lengthSq() < 1e-16 && hi + 1 < this.pts.length) {
      target.subVectors(this.pts[hi + 1], this.pts[lo]);
    }
    return target.lengthSq() > 1e-16 ? target.normalize() : target.set(1, 0, 0);
  }
}

function buildFilletedPointsM(ptsMm: [number, number, number][], bendRadiusMm: number): THREE.Vector3[] {
  const raw = ptsMm.map(([x, y, z]) => new THREE.Vector3(x, y, z));
  if (raw.length <= 2) {
    return subdivideStraightRunsM(raw, 4.5);
  }
  const outMm: THREE.Vector3[] = [raw[0].clone()];
  const vIn = new THREE.Vector3();
  const vOut = new THREE.Vector3();
  const vNext = new THREE.Vector3();
  let i = 1;
  while (i < raw.length - 1) {
    // 专门识别 180° 半圆回弯 (如 17-5 四程过热器 U 型弯与 17-3 烟室 J 型弯): 两段反向平行主管夹一段短横桥
    if (i + 2 < raw.length) {
      vIn.subVectors(raw[i], raw[i - 1]);
      vOut.subVectors(raw[i + 1], raw[i]);
      vNext.subVectors(raw[i + 2], raw[i + 1]);
      const lIn = vIn.length();
      const lBridge = vOut.length();
      const lNext = vNext.length();
      if (lIn > 3.5 && lNext > 3.5 && lBridge > 1.2 && lBridge <= 7.2) {
        const uIn = vIn.clone().divideScalar(lIn);
        const uBridge = vOut.clone().divideScalar(lBridge);
        const uNext = vNext.clone().divideScalar(lNext);
        if (uIn.dot(uNext) < -0.94 && Math.abs(uIn.dot(uBridge)) < 0.18) {
          const rU = 0.5 * lBridge;
          const trim = Math.min(rU, 0.45 * lIn, 0.45 * lNext);
          const pEnter = raw[i].clone().addScaledVector(uIn, -trim);
          const center = raw[i]
            .clone()
            .add(raw[i + 1])
            .multiplyScalar(0.5)
            .addScaledVector(uIn, -trim);
          const nArc = 18;
          for (let k = 0; k <= nArc; k++) {
            const ang = (Math.PI * k) / nArc;
            // ang=0 -> pEnter (-rU * uBridge), ang=pi/2 -> apex (+rU * uIn), ang=pi -> pExit (+rU * uBridge)
            const pt = center
              .clone()
              .addScaledVector(uBridge, -rU * Math.cos(ang))
              .addScaledVector(uIn, trim * Math.sin(ang));
            if (pt.distanceToSquared(outMm[outMm.length - 1]) > 1e-6) {
              outMm.push(pt);
            }
          }
          i += 2;
          continue;
        }
      }
    }

    vIn.subVectors(raw[i], raw[i - 1]);
    vOut.subVectors(raw[i + 1], raw[i]);
    const l1 = vIn.length();
    const l2 = vOut.length();
    if (l1 < 1e-4 || l2 < 1e-4) {
      i++;
      continue;
    }
    const u1 = vIn.clone().divideScalar(l1);
    const u2 = vOut.clone().divideScalar(l2);
    const cosTurn = Math.max(-1.0, Math.min(1.0, u1.dot(u2)));
    const turnAng = Math.acos(cosTurn);
    if (turnAng < 0.04 || turnAng > Math.PI - 0.02) {
      outMm.push(raw[i].clone());
      i++;
      continue;
    }
    const halfTan = Math.tan(0.5 * turnAng);
    const maxTrim = Math.min(0.45 * l1, 0.45 * l2);
    const dt = Math.min(bendRadiusMm * halfTan, maxTrim);
    const rEff = dt / Math.max(1e-5, halfTan);
    const pA = raw[i].clone().addScaledVector(u1, -dt);
    const pB = raw[i].clone().addScaledVector(u2, +dt);
    // 圆心位于角平分内法向: nIn = normalize(u2 - cosTurn * u1)
    const nIn = u2.clone().addScaledVector(u1, -cosTurn).normalize();
    const arcCenter = pA.clone().addScaledVector(nIn, rEff);
    const eRadA = pA.clone().sub(arcCenter).divideScalar(Math.max(1e-5, rEff));
    const nSteps = Math.max(8, Math.round((turnAng / (0.5 * Math.PI)) * 10));
    for (let k = 0; k <= nSteps; k++) {
      const phi = (turnAng * k) / nSteps;
      const pt = arcCenter
        .clone()
        .addScaledVector(eRadA, rEff * Math.cos(phi))
        .addScaledVector(u1, rEff * Math.sin(phi));
      if (k === nSteps) pt.copy(pB);
      if (pt.distanceToSquared(outMm[outMm.length - 1]) > 1e-6) {
        outMm.push(pt);
      }
    }
    i++;
  }
  outMm.push(raw[raw.length - 1].clone());
  return subdivideStraightRunsM(outMm, 4.5);
}

function subdivideStraightRunsM(ptsMm: THREE.Vector3[], maxSegMm: number): THREE.Vector3[] {
  const resM: THREE.Vector3[] = [ptsMm[0].clone().multiplyScalar(0.001)];
  for (let i = 1; i < ptsMm.length; i++) {
    const a = ptsMm[i - 1];
    const b = ptsMm[i];
    const dist = a.distanceTo(b);
    const nSub = Math.max(1, Math.ceil(dist / maxSegMm));
    for (let s = 1; s <= nSub; s++) {
      resM.push(new THREE.Vector3().lerpVectors(a, b, s / nSub).multiplyScalar(0.001));
    }
  }
  return resM;
}

/** 基于无扭转 Bishop 平行输运标架精确扫掠圆管网格 (零尖角折扁、零竖直管扭麻花、精确弧长 vUv.x!) */
function buildBishopPipeGeometry(curve: ExactFilletedPipeCurve3, radiusM: number, radialSegs = 12): THREE.BufferGeometry {
  const pts = curve.pts;
  const nRings = pts.length;
  const verts = new Float32Array(nRings * (radialSegs + 1) * 3);
  const norms = new Float32Array(nRings * (radialSegs + 1) * 3);
  const uvs = new Float32Array(nRings * (radialSegs + 1) * 2);
  const indices: number[] = [];

  const tangents: THREE.Vector3[] = [];
  for (let i = 0; i < nRings; i++) {
    const t = new THREE.Vector3();
    if (i === 0) t.subVectors(pts[1], pts[0]);
    else if (i === nRings - 1) t.subVectors(pts[nRings - 1], pts[nRings - 2]);
    else t.subVectors(pts[i + 1], pts[i - 1]);
    tangents.push(t.lengthSq() > 1e-16 ? t.normalize() : new THREE.Vector3(1, 0, 0));
  }

  const normal = new THREE.Vector3();
  const binormal = new THREE.Vector3();
  const refUp = Math.abs(tangents[0].z) < 0.88 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(1, 0, 0);
  normal.crossVectors(tangents[0], refUp).normalize();
  const rotQ = new THREE.Quaternion();

  for (let i = 0; i < nRings; i++) {
    if (i > 0) {
      rotQ.setFromUnitVectors(tangents[i - 1], tangents[i]);
      normal.applyQuaternion(rotQ);
      normal.addScaledVector(tangents[i], -normal.dot(tangents[i])).normalize();
    }
    binormal.crossVectors(tangents[i], normal).normalize();
    const uCoord = curve.cumLen[i] / curve.totalLength;

    for (let j = 0; j <= radialSegs; j++) {
      const phi = (2.0 * Math.PI * j) / radialSegs;
      const cs = Math.cos(phi);
      const sn = Math.sin(phi);
      const nx = cs * normal.x + sn * binormal.x;
      const ny = cs * normal.y + sn * binormal.y;
      const nz = cs * normal.z + sn * binormal.z;
      const idx = i * (radialSegs + 1) + j;
      verts[3 * idx] = pts[i].x + radiusM * nx;
      verts[3 * idx + 1] = pts[i].y + radiusM * ny;
      verts[3 * idx + 2] = pts[i].z + radiusM * nz;
      norms[3 * idx] = nx;
      norms[3 * idx + 1] = ny;
      norms[3 * idx + 2] = nz;
      uvs[2 * idx] = uCoord;
      uvs[2 * idx + 1] = j / radialSegs;
    }
  }

  for (let i = 0; i < nRings - 1; i++) {
    for (let j = 0; j < radialSegs; j++) {
      const a = i * (radialSegs + 1) + j;
      const b = (i + 1) * (radialSegs + 1) + j;
      const c = (i + 1) * (radialSegs + 1) + (j + 1);
      const d = i * (radialSegs + 1) + (j + 1);
      indices.push(a, b, d, b, c, d);
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(verts, 3));
  geom.setAttribute("normal", new THREE.BufferAttribute(norms, 3));
  geom.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
  geom.setIndex(indices);
  return geom;
}

/** ① 管内受压蒸汽流向指示器: 锐利细长分叉燕尾飞镖 (长径比 3.2:1, 尖端亮白聚焦, 专用于受管壁约束的高速蒸汽!) */
function createStreamlinedFlowDartGeometry(): THREE.BufferGeometry {
  const profile = [
    new THREE.Vector2(0.0, -0.46),
    new THREE.Vector2(0.84, -1.16),
    new THREE.Vector2(0.62, -0.52),
    new THREE.Vector2(0.28, +0.58),
    new THREE.Vector2(0.0, +1.74),
  ];
  const geom = new THREE.LatheGeometry(profile, 12);
  const pos = geom.attributes.position;
  const vCols = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    const y = pos.getY(i);
    const tipBoost = Math.max(0.0, Math.min(1.0, (y + 0.5) / 2.2));
    const lum = 0.82 + 0.54 * tipBoost * tipBoost;
    vCols[3 * i] = lum;
    vCols[3 * i + 1] = lum;
    vCols[3 * i + 2] = lum;
  }
  geom.setAttribute("color", new THREE.BufferAttribute(vCols, 3));
  geom.computeVertexNormals();
  return geom;
}

/** ② 管内/管外液体流向指示器: 饱满圆润表面张力三维水滴珠 (半球圆尾 + 柔润液滴前鼻 + 湿润高光, 与尖锐气体飞镖一眼区分!) */
function createLiquidDropletDartGeometry(): THREE.BufferGeometry {
  const profile = [
    new THREE.Vector2(0.0, -0.82),
    new THREE.Vector2(0.56, -0.72),
    new THREE.Vector2(0.86, -0.44),
    new THREE.Vector2(0.92, -0.08),
    new THREE.Vector2(0.76, +0.34),
    new THREE.Vector2(0.46, +0.82),
    new THREE.Vector2(0.18, +1.18),
    new THREE.Vector2(0.0, +1.36),
  ];
  const geom = new THREE.LatheGeometry(profile, 14);
  const pos = geom.attributes.position;
  const vCols = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    const y = pos.getY(i);
    const x = pos.getX(i);
    const nose = Math.max(0.0, Math.min(1.0, (y + 0.65) / 2.0));
    const specBand = Math.exp(-Math.pow((x - 0.52) / 0.24, 2.0)) * Math.max(0.0, 1.0 - Math.abs(y) * 0.7);
    const lum = 0.80 + 0.36 * nose * nose + 0.28 * specBand;
    vCols[3 * i] = lum;
    vCols[3 * i + 1] = lum;
    vCols[3 * i + 2] = lum;
  }
  geom.setAttribute("color", new THREE.BufferAttribute(vCols, 3));
  geom.computeVertexNormals();
  return geom;
}

/** ③ 管外开放腔室/火管/车外大气自由气流指示器: 镂空双 V 风羽 (» 正交双翼薄带, 中间留白通透, 绝不造成实体圆管或液滴假象!) */
function createUnconfinedGasChevronGeometry(): THREE.BufferGeometry {
  const verts: number[] = [];
  const cols: number[] = [];
  const pushQuad = (a: [number, number, number], b: [number, number, number], c: [number, number, number], d: [number, number, number], lumHi: number, lumLo: number) => {
    verts.push(...a, ...b, ...c, ...a, ...c, ...d);
    cols.push(
      lumHi, lumHi, lumHi,
      lumHi, lumHi, lumHi,
      lumLo, lumLo, lumLo,
      lumHi, lumHi, lumHi,
      lumLo, lumLo, lumLo,
      lumLo, lumLo, lumLo
    );
  };
  const addChevronPairPlane = (inZY: boolean) => {
    const mapPt = (uSpan: number, yAx: number): [number, number, number] =>
      inZY ? [0.0, yAx, uSpan] : [uSpan, yAx, 0.0];
    // 前后串联双 V 开放风羽 (yOffsets: 前羽 +0.58, 后羽 -0.34, 中间完全镂空!)
    const chevrons = [
      { yTip: +1.42, yBack: +1.08, yWingOut: +0.36, yWingIn: +0.04, span: 1.04, lumTip: 1.32, lumWing: 0.86 },
      { yTip: +0.32, yBack: -0.02, yWingOut: -0.72, yWingIn: -1.04, span: 0.90, lumTip: 1.12, lumWing: 0.74 },
    ];
    for (const cv of chevrons) {
      for (const sgn of [-1.0, +1.0]) {
        pushQuad(
          mapPt(0.0, cv.yTip),
          mapPt(0.0, cv.yBack),
          mapPt(sgn * cv.span, cv.yWingIn),
          mapPt(sgn * cv.span, cv.yWingOut),
          cv.lumTip,
          cv.lumWing
        );
      }
    }
  };
  addChevronPairPlane(false);
  addChevronPairPlane(true);
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
  geom.setAttribute("color", new THREE.Float32BufferAttribute(cols, 3));
  geom.computeVertexNormals();
  return geom;
}

type PipeMediumKind = 0 | 1 | 2; // 0: 管内致密液体, 1: 管内受压蒸汽, 2: 锅筒火管热燃气通道(X=-48..+92mm)

interface InternalPipeRuntime {
  id: string;
  partNo: string;
  title: string;
  zone: "tender" | "umbilical" | "cab" | "boiler" | "underframe" | "cylinder";
  frame: "loco" | "tender" | "umbilical";
  mediumKind: PipeMediumKind;
  curve: ExactFilletedPipeCurve3;
  branchCurves: ExactFilletedPipeCurve3[];
  branchMeshes: THREE.Mesh[];
  branchArrowCounts: number[];
  umbEndsMm?: { tdr: [number, number, number]; loco: [number, number, number] };
  radiusM: number;
  mesh: THREE.Mesh;
  mat: THREE.ShaderMaterial;
  arrowCount: number;
  phase: number;
  labelT: number;
  tempProfile: [number, number, number, number];
  arrowColTmp: THREE.Color;
  colorCss: string;
  evalTempProfileC: (out: StepOutput, sp: SpatialFluidState) => [number, number, number, number];
  evalFlow: (out: StepOutput, sp: SpatialFluidState) => {
    signedRate: number; // >0 along curvepts[0]->[-1], <0 reversed!
    fill: number;       // 0..1 lumen presence
    unit: string;
    colOverride?: [number, number, number];
  };
}

interface OpenFlowStreamRuntime {
  id: string;
  partNo: string;
  title: string;
  zone: InternalPipeRuntime["zone"];
  frame: "loco" | "tender";
  medium: "open_gas" | "open_liquid";
  curve: ExactFilletedPipeCurve3;
  branchCurves: ExactFilletedPipeCurve3[];
  branchLines: THREE.Line[];
  branchArrowCounts: number[];
  mat: THREE.ShaderMaterial;
  markerScaleM: number;
  phase: number;
  labelT: number;
  tempProfile: [number, number, number, number];
  arrowColTmp: THREE.Color;
  evalTempProfileC: InternalPipeRuntime["evalTempProfileC"];
  evalFlow: InternalPipeRuntime["evalFlow"];
}

const ARROW_BASE_AXIS = new THREE.Vector3(0, 1, 0);

function createPipeLumenMaterial(mediumKind: PipeMediumKind = 0): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    uniforms: {
      uPhase: { value: 0.0 },
      uFill: { value: 0.8 },
      uActive: { value: 0.0 },
      uMediumKind: { value: mediumKind },
      uTemps: { value: new THREE.Vector4(25.0, 25.0, 25.0, 25.0) },
    },
    vertexShader: /* glsl */ `
      varying vec2 vUv;
      varying float vEdge;
      void main() {
        vUv = uv;
        vec3 mvPos = (modelViewMatrix * vec4(position, 1.0)).xyz;
        vec3 mvNorm = normalize(normalMatrix * normal);
        vEdge = 1.0 - abs(dot(mvNorm, normalize(-mvPos)));
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      uniform float uPhase;
      uniform float uFill;
      uniform float uActive;
      uniform float uMediumKind;
      uniform vec4 uTemps;
      varying vec2 vUv;
      varying float vEdge;

      vec3 thermalColorGLSL(float tc) {
        float t = clamp(tc, 20.0, 750.0);
        if (t <= 45.0)  return mix(vec3(0.114, 0.306, 0.847), vec3(0.024, 0.714, 0.831), (t - 20.0) / 25.0);
        if (t <= 80.0)  return mix(vec3(0.024, 0.714, 0.831), vec3(0.063, 0.725, 0.506), (t - 45.0) / 35.0);
        if (t <= 105.0) return mix(vec3(0.063, 0.725, 0.506), vec3(0.918, 0.702, 0.031), (t - 80.0) / 25.0);
        if (t <= 148.0) return mix(vec3(0.918, 0.702, 0.031), vec3(0.961, 0.620, 0.043), (t - 105.0) / 43.0);
        if (t <= 210.0) return mix(vec3(0.961, 0.620, 0.043), vec3(0.976, 0.451, 0.086), (t - 148.0) / 62.0);
        if (t <= 420.0) return mix(vec3(0.976, 0.451, 0.086), vec3(0.937, 0.267, 0.267), (t - 210.0) / 210.0);
        return mix(vec3(0.937, 0.267, 0.267), vec3(0.996, 0.941, 0.541), (t - 420.0) / 330.0);
      }

      float sampleAxialTemp(float x) {
        float s = clamp(x, 0.0, 1.0) * 3.0;
        if (s <= 1.0) return mix(uTemps.x, uTemps.y, s);
        if (s <= 2.0) return mix(uTemps.y, uTemps.z, s - 1.0);
        return mix(uTemps.z, uTemps.w, s - 2.0);
      }

      void main() {
        if (uFill < 0.02) discard;
        float act = clamp(uActive, 0.0, 1.0);
        float tBase = sampleAxialTemp(vUv.x);
        float edge = clamp(vEdge, 0.0, 1.0);

        if (uMediumKind < 0.5) {
          // ① 管内致密液体 (水/醇/汽缸油): 中心厚实高折射液柱 + 湿润玻璃管壁镜面高光带
          float wave = 0.5 + 0.5 * sin(vUv.x * 36.0 - uPhase);
          float bolus = mix(0.55, pow(wave, 1.45), act);
          vec3 col = thermalColorGLSL(tBase + act * 5.0 * (bolus - 0.5));
          float wetSpec = exp(-pow((edge - 0.72) / 0.16, 2.0));
          col += vec3(0.16, 0.19, 0.22) * (0.35 + 0.65 * act) * wetSpec;
          float coreDense = 1.0 - 0.26 * edge * edge;
          float alpha = uFill * coreDense * mix(0.46, 0.68 + 0.22 * bolus, act);
          gl_FragColor = vec4(clamp(col, 0.0, 1.0), clamp(alpha, 0.0, 0.92));
          return;
        }

        if (uMediumKind < 1.5) {
          // ② 管内受压蒸汽: 中空透亮菲涅尔管壁轮廓 + 高速轴向可压缩蒸汽丝纹 (与实心液柱截然不同!)
          float streak = 0.5 + 0.5 * sin(vUv.x * 88.0 - uPhase * 1.75 + sin(vUv.y * 18.85) * 1.4);
          float rim = pow(edge, 1.65);
          vec3 col = thermalColorGLSL(tBase + act * 7.5 * (streak - 0.45));
          col = mix(col, vec3(1.0, 0.99, 0.96), 0.22 * rim + 0.16 * act * pow(streak, 3.0));
          float alpha = uFill * (0.12 + 0.44 * rim + 0.22 * act * pow(streak, 1.8));
          gl_FragColor = vec4(clamp(col, 0.0, 1.0), clamp(alpha, 0.0, 0.78));
          return;
        }

        // ③ 锅筒 5 烟管高温燃气通道 (严格止于 X=-48..+92mm 前后管板): 柔和径向热气光晕, 绝不遮蔽内部 17-5 过热蒸汽管!
        float endFade = smoothstep(0.0, 0.06, vUv.x) * smoothstep(1.0, 0.94, vUv.x);
        float wisp = 0.5 + 0.5 * sin(vUv.x * 48.0 - uPhase * 1.35);
        float radialGlow = pow(max(0.0, 1.0 - edge), 1.25);
        vec3 col = thermalColorGLSL(tBase + act * 8.0 * (wisp - 0.5));
        float alpha = uFill * endFade * radialGlow * (0.14 + 0.16 * act * wisp);
        gl_FragColor = vec4(clamp(col, 0.0, 1.0), clamp(alpha, 0.0, 0.24));
      }
    `,
  });
}

/** 构造无管壁开放空间流迹几何 (THREE.Line 1D 曲线 + 归一化弧长属性 aArcU, 绝不生成实体圆管网格!) */
function buildStreamlineLineGeometry(curve: ExactFilletedPipeCurve3, samples = 56): THREE.BufferGeometry {
  const pos = new Float32Array((samples + 1) * 3);
  const arcU = new Float32Array(samples + 1);
  const pt = new THREE.Vector3();
  for (let i = 0; i <= samples; i++) {
    const u = i / samples;
    curve.getPointAt(u, pt);
    pos[3 * i] = pt.x;
    pos[3 * i + 1] = pt.y;
    pos[3 * i + 2] = pt.z;
    arcU[i] = u;
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geom.setAttribute("aArcU", new THREE.BufferAttribute(arcU, 1));
  return geom;
}

/** 管外开放空间自由气体/落水行进虚线着色器 (无管壁边界! 气体为长风丝虚线, 自由液流为落水珠链虚线) */
function createDashedGasStreamlineMaterial(isLiquidJet = false): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    depthTest: false,
    uniforms: {
      uPhase: { value: 0.0 },
      uFill: { value: 0.0 },
      uActive: { value: 0.0 },
      uIsLiquid: { value: isLiquidJet ? 1.0 : 0.0 },
      uTemps: { value: new THREE.Vector4(25.0, 25.0, 25.0, 25.0) },
    },
    vertexShader: /* glsl */ `
      attribute float aArcU;
      varying float vArcU;
      void main() {
        vArcU = aArcU;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      uniform float uPhase;
      uniform float uFill;
      uniform float uActive;
      uniform float uIsLiquid;
      uniform vec4 uTemps;
      varying float vArcU;

      vec3 thermalColorGLSL(float tc) {
        float t = clamp(tc, 20.0, 750.0);
        if (t <= 45.0)  return mix(vec3(0.114, 0.306, 0.847), vec3(0.024, 0.714, 0.831), (t - 20.0) / 25.0);
        if (t <= 80.0)  return mix(vec3(0.024, 0.714, 0.831), vec3(0.063, 0.725, 0.506), (t - 45.0) / 35.0);
        if (t <= 105.0) return mix(vec3(0.063, 0.725, 0.506), vec3(0.918, 0.702, 0.031), (t - 80.0) / 25.0);
        if (t <= 148.0) return mix(vec3(0.918, 0.702, 0.031), vec3(0.961, 0.620, 0.043), (t - 105.0) / 43.0);
        if (t <= 210.0) return mix(vec3(0.961, 0.620, 0.043), vec3(0.976, 0.451, 0.086), (t - 148.0) / 62.0);
        if (t <= 420.0) return mix(vec3(0.976, 0.451, 0.086), vec3(0.937, 0.267, 0.267), (t - 210.0) / 210.0);
        return mix(vec3(0.937, 0.267, 0.267), vec3(0.996, 0.941, 0.541), (t - 420.0) / 330.0);
      }

      float sampleAxialTemp(float x) {
        float s = clamp(x, 0.0, 1.0) * 3.0;
        if (s <= 1.0) return mix(uTemps.x, uTemps.y, s);
        if (s <= 2.0) return mix(uTemps.y, uTemps.z, s - 1.0);
        return mix(uTemps.z, uTemps.w, s - 2.0);
      }

      void main() {
        if (uFill < 0.02 || uActive < 0.02) discard;
        float freq = uIsLiquid > 0.5 ? 78.0 : 50.0;
        float dashWave = sin(vArcU * freq - uPhase * 2.1);
        float dashMask = smoothstep(-0.10, 0.38, dashWave);
        if (dashMask < 0.04) discard;
        float endFade = smoothstep(0.0, 0.09, vArcU) * smoothstep(1.0, 0.88, vArcU);
        vec3 col = thermalColorGLSL(sampleAxialTemp(vArcU));
        float baseAlpha = uIsLiquid > 0.5 ? 0.58 : 0.42;
        gl_FragColor = vec4(col, clamp(endFade * dashMask * baseAlpha * clamp(uActive, 0.25, 1.0), 0.0, 0.62));
      }
    `,
  });
}

export class FluidPartVisualizer {
  readonly root = new THREE.Group();
  readonly tenderRoot = new THREE.Group();
  readonly worldRoot = new THREE.Group();
  readonly uniforms: CadShaderUniforms;

  private wickFlames: THREE.Mesh[] = [];
  private flameMat: THREE.ShaderMaterial;

  private waterSurfaceMat: THREE.ShaderMaterial;
  private waterVolumeMat: THREE.ShaderMaterial;
  private steamHeadspaceMat: THREE.ShaderMaterial;
  private smokeboxBlastMat: THREE.ShaderMaterial;
  private waterSurfaceMesh: THREE.Mesh;
  private waterVolumeMesh: THREE.Mesh;
  private steamHeadspaceMesh: THREE.Mesh;
  private smokeboxBlastMeshes: THREE.Mesh[] = [];
  private blowerJetMat: THREE.ShaderMaterial;
  private blowerJetMeshes: THREE.Mesh[] = [];

  private bubblePoints: THREE.Points;
  private bubblePositions: Float32Array;
  private bubbleSeeds: Float32Array;
  private readonly bubbleCount = 180;
  private bubbleMat: THREE.ShaderMaterial;

  private sightGlassWaterMesh: THREE.Mesh;
  private sightGlassMeniscusMesh: THREE.Mesh;

  private sliceTexData: Uint8Array<ArrayBuffer>;
  private sliceTex: THREE.DataTexture;
  private sliceMat: THREE.ShaderMaterial;
  private sliceMesh: THREE.Mesh;

  private cylHeadMeshes: [THREE.Mesh, THREE.Mesh];
  private cylCrankMeshes: [THREE.Mesh, THREE.Mesh];
  private cylCondMeshes: [THREE.Mesh, THREE.Mesh];
  private portGlowMeshes: [THREE.Mesh, THREE.Mesh, THREE.Mesh, THREE.Mesh];

  private lubeCoreMat: THREE.ShaderMaterial;
  private lubeCoreMesh: THREE.Mesh;
  private lubeStandpipeMesh: THREE.Mesh;

  private tenderWaterMesh: THREE.Mesh;
  private tenderOpenBayWaterMesh: THREE.Mesh;
  private tenderOpenBayMat: THREE.ShaderMaterial;
  private tenderWaveTexData: Uint8Array<ArrayBuffer>;
  private tenderWaveTex: THREE.DataTexture;
  private tenderAlcoholMesh: THREE.Mesh;
  private returnStreamMesh: THREE.Mesh;
  private waterFillStreamMesh: THREE.Mesh;
  private tenderDrainStreamMesh: THREE.Mesh;
  private mariotteBubblePoints: THREE.Points;
  private mariotteBubblePos: Float32Array;

  private pipeRuntimes: InternalPipeRuntime[] = [];
  private openStreamRuntimes: OpenFlowStreamRuntime[] = [];
  private locoArrowInst!: THREE.InstancedMesh;       // 管内受压蒸汽锐利燕尾镖 (➤)
  private locoLiquidArrowInst!: THREE.InstancedMesh; // 机车管内+管外液体圆润水滴珠 (💧)
  private tenderArrowInst!: THREE.InstancedMesh;     // 煤水车管内+管外液体圆润水滴珠 (💧)
  private worldArrowInst!: THREE.InstancedMesh;      // 跨车软管液体圆润水滴珠 (💧)
  private openGasArrowInst!: THREE.InstancedMesh;    // 管外开放腔室/火管/大气自由气流双V风羽 (»)
  private calloutSnapshots: PipeCalloutSnapshot[] = [];
  private tmpVecA = new THREE.Vector3();
  private tmpVecB = new THREE.Vector3();
  private tmpTan = new THREE.Vector3();
  private tmpQuat = new THREE.Quaternion();
  private tmpScale = new THREE.Vector3();
  private tmpMat = new THREE.Matrix4();

  private elapsed = 0.0;
  private feedPhase = 0.0;

  constructor() {
    this.root.matrixAutoUpdate = false;
    this.tenderRoot.matrixAutoUpdate = false;
    this.worldRoot.matrixAutoUpdate = true;

    this.uniforms = {
      uTime: { value: 0.0 },
      uXray: { value: 0.0 },
      uSteamFlow: { value: 0.0 },
      uDomeDryness: { value: 0.95 },
      uSuperheatNorm: { value: 0.2 },
      uExhaustPulse: { value: 0.0 },
      uFeedPhase: { value: 0.0 },
      uFeedStrength: { value: 0.0 },
      uCrownTemps: { value: [156, 156, 156, 156, 156, 156] },
      uFlueTemps: { value: [420, 340, 280, 240, 210, 190, 178, 168] },
      uFlueBowMm: { value: [0, 0, 0, 0, 0] },
      uCrownBulgeMm: { value: 0.0 },
    };

    // -----------------------------------------------------------------------
    // 1. 23-1 纵向直列 1×3 三头立式酒精燃烧器三维横流流线弯折火焰 (X=[-93, -77, -61]mm, Z_rim=+18.8mm -> 5烟管喉口 X=-48.2mm)
    // -----------------------------------------------------------------------
    this.flameMat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      uniforms: {
        uTime: { value: 0.0 },
        uLit: { value: 1.0 },
        uPhi: { value: 1.0 },
        uTiltRad: { value: 0.85 },
        uWickTilts: { value: new THREE.Vector3(0.75, 0.98, 1.20) },
        uChuffPulse: { value: 0.0 },
        uDraftNorm: { value: 0.65 },
        uIntensity: { value: 1.0 },
      },
      vertexShader: /* glsl */ `
        uniform float uTime;
        uniform vec3 uWickTilts;
        uniform float uChuffPulse;
        uniform float uDraftNorm;
        varying vec2 vUv;
        varying float vSeed;
        varying float vDraft;
        attribute float aSeed;
        attribute float aWickIdx;
        attribute vec3 aAnchor;
        attribute vec2 aCrossDir;
        void main() {
          vUv = uv;
          vSeed = aSeed;
          vDraft = uDraftNorm;
          float s = clamp(uv.y, 0.0, 1.0);
          float uCross = position.x;

          float baseTilt = (aWickIdx < 0.5) ? uWickTilts.x : ((aWickIdx < 1.5) ? uWickTilts.y : uWickTilts.z);
          float effTilt = clamp(baseTilt + 0.11 * uChuffPulse, 0.04, 1.36);

          // 浮力射流在烟室强负压横流中的三维流线轨迹 C(s): 根部自 ø10mm 陶瓷芯微升，随即顺风弯向 X=-48.2mm 的 5 根烟管喉口!
          float arcLen = 0.0235 + 0.0125 * sin(effTilt) + 0.0034 * uChuffPulse;
          float dx = arcLen * sin(effTilt) * pow(s, 1.32);
          float dz = arcLen * (cos(effTilt) * pow(s, 0.86) + 0.24 * sin(effTilt) * s * (1.0 - 0.62 * s));

          float cx = aAnchor.x + dx;
          float cz = aAnchor.z + dz;

          // 严格遵循 Type-"C" 12-6 紫铜冠板下壁 (Z=+0.0448m) 与后管板 5 烟管入口喉部边界 (绝不刺穿顶板进入锅水!)
          float throatPull = smoothstep(-0.056, -0.044, cx);
          cz = mix(min(cz, 0.0438 - 0.0010 * (1.0 - s)), clamp(cz, 0.0328, 0.0445), throatPull);
          cx = min(cx, -0.0375);

          // 局部流线切向角 alpha(s) 与无瘪塌正交截面标架 N_xz = (cos alpha, 0, -sin alpha), N_y = (0, 1, 0)
          float alpha = effTilt * (0.14 + 0.86 * pow(s, 0.80));
          vec3 center = vec3(cx, aAnchor.y, cz);
          vec3 nXZ = vec3(cos(alpha), 0.0, -sin(alpha));
          vec3 nY  = vec3(0.0, 1.0, 0.0);

          float freq = 18.0 + 14.0 * uDraftNorm;
          float flutterN = 0.0014 * s * sin(uTime * freq + s * 9.5 + aSeed * 6.28);
          float flutterY = 0.0011 * s * cos(uTime * (freq * 0.83) + s * 7.8 + aSeed * 4.13);

          vec3 p = center + (uCross * aCrossDir.x + flutterN) * nXZ + (uCross * aCrossDir.y + flutterY) * nY;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        uniform float uTime;
        uniform float uLit;
        uniform float uPhi;
        uniform float uIntensity;
        varying vec2 vUv;
        varying float vSeed;
        varying float vDraft;

        float hash21(vec2 p) {
          p = fract(p * vec2(123.34, 456.21));
          p += dot(p, p + 45.32);
          return fract(p.x * p.y);
        }
        float noise2(vec2 p) {
          vec2 i = floor(p);
          vec2 f = fract(p);
          f = f * f * (3.0 - 2.0 * f);
          float a = hash21(i);
          float b = hash21(i + vec2(1.0, 0.0));
          float c = hash21(i + vec2(0.0, 1.0));
          float d = hash21(i + vec2(1.0, 1.0));
          return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
        }
        void main() {
          if (uLit < 0.02) discard;
          float h = vUv.y;
          float r = abs(vUv.x - 0.5) * 2.0;
          float advRate = 8.5 + 11.5 * clamp(vDraft, 0.0, 1.3);
          float n1 = noise2(vec2(vUv.x * 5.0 + vSeed * 3.1, h * 6.5 - uTime * advRate));
          float n2 = noise2(vec2(vUv.x * 11.0 - vSeed * 1.7, h * 13.0 - uTime * (advRate * 1.55)));
          float turb = 0.65 * n1 + 0.35 * n2;

          // 火焰外轮廓: 底部贴合 ø10mm 陶瓷芯，强引风下收束成流线型高速火舌吸入烟管
          float tipWidth = mix(0.20, 0.10, clamp(vDraft, 0.0, 1.0));
          float halfWidth = mix(0.78, tipWidth, pow(h, 0.82)) + 0.22 * (turb - 0.5) * h;
          float body = smoothstep(halfWidth, halfWidth * 0.20, r) * smoothstep(0.0, 0.09, h) * smoothstep(1.0, 0.32, h);
          if (body < 0.015) discard;

          // aster-manual.pdf p.36 乙醇火焰真实光谱: 强引风化学计量燃烧时主体呈清澈醇蓝紫、舌尖微带高温金橙；缺风/漫油富燃时呈明亮黄橙色扩散焰!
          float coreMask = smoothstep(0.62, 0.06, h) * smoothstep(0.62, 0.08, r);
          float richFrac = clamp((uPhi - 0.86) / 0.56, 0.0, 1.0);
          vec3 leanOuter = vec3(0.20, 0.58, 1.00);
          vec3 richOuter = vec3(1.00, 0.46, 0.08);
          vec3 tipGlow   = vec3(1.00, 0.78, 0.28);
          vec3 innerBlue = vec3(0.16, 0.48, 1.00);

          vec3 outerCol = mix(leanOuter, richOuter, richFrac);
          outerCol = mix(outerCol, tipGlow, smoothstep(0.38, 0.92, h) * (0.28 + 0.62 * richFrac));
          vec3 col = mix(outerCol, innerBlue, coreMask * (1.0 - 0.32 * richFrac));

          float alpha = body * (0.52 + 0.46 * uIntensity);
          gl_FragColor = vec4(col * (1.22 + 0.55 * turb), clamp(alpha, 0.0, 0.90));
        }
      `,
    });

    // 23-1 纵向单列 1×3 三头立式 ø10mm 燃烧器火焰 (X = [-93, -77, -61] mm, Y = 0, Z_rim = +18.8 mm)
    const wickCoords: [number, number, number][] = [
      [-0.093, 0.0, 0.0188],
      [-0.077, 0.0, 0.0188],
      [-0.061, 0.0, 0.0188],
    ];
    wickCoords.forEach(([wx, wy, wz], idx) => {
      [0.0, Math.PI / 3.0, (2.0 * Math.PI) / 3.0].forEach((ang, pIdx) => {
        const g = new THREE.PlaneGeometry(0.0130, 1.0, 6, 20);
        g.translate(0.0, 0.5, 0.0);
        const vCount = g.attributes.position.count;
        const seeds = new Float32Array(vCount).fill(idx * 0.29 + pIdx * 0.17);
        const wIdxs = new Float32Array(vCount).fill(idx);
        const anchors = new Float32Array(vCount * 3);
        const crossDirs = new Float32Array(vCount * 2);
        const ca = Math.cos(ang);
        const sa = Math.sin(ang);
        for (let k = 0; k < vCount; k++) {
          anchors[3 * k] = wx;
          anchors[3 * k + 1] = wy;
          anchors[3 * k + 2] = wz;
          crossDirs[2 * k] = ca;
          crossDirs[2 * k + 1] = sa;
        }
        g.setAttribute("aSeed", new THREE.BufferAttribute(seeds, 1));
        g.setAttribute("aWickIdx", new THREE.BufferAttribute(wIdxs, 1));
        g.setAttribute("aAnchor", new THREE.BufferAttribute(anchors, 3));
        g.setAttribute("aCrossDir", new THREE.BufferAttribute(crossDirs, 2));
        const m = new THREE.Mesh(g, this.flameMat);
        m.frustumCulled = false;
        m.renderOrder = 4;
        this.wickFlames.push(m);
        this.root.add(m);
      });
    });

    // -----------------------------------------------------------------------
    // 2. 锅炉三维晃动自由水面 + 四层水温梯度水体 + 核态沸腾上升汽泡 + 11-7 驾驶室水面计实况水柱
    // -----------------------------------------------------------------------
    const waterUniforms = {
      uTime: { value: 0.0 },
      uWaterZ: { value: 0.0555 },
      uSloshTan: { value: 0.0 },
      uBoil: { value: 0.25 },
      uTempBottom: { value: 145.0 },
      uTempTop: { value: 150.0 },
    };

    this.waterSurfaceMat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      uniforms: waterUniforms,
      vertexShader: /* glsl */ `
        uniform float uTime;
        uniform float uWaterZ;
        uniform float uSloshTan;
        uniform float uBoil;
        varying vec3 vLocalPos;
        varying float vWave;
        void main() {
          vec3 p = position;
          float wave = (0.00025 + 0.00125 * uBoil) * (
            sin(p.x * 95.0 - uTime * 11.0 + p.y * 40.0) * 0.6 +
            cos(p.x * 155.0 + uTime * 16.5 - p.y * 85.0) * 0.4
          );
          p.z = uWaterZ + (p.x + 0.010) * uSloshTan + wave;
          vLocalPos = p;
          vWave = wave;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        uniform float uBoil;
        uniform float uTempTop;
        varying vec3 vLocalPos;
        varying float vWave;
        void main() {
          // 精确裁剪于 ø51.2mm 紫铜锅筒圆截面内壁 (z_axis = 0.0440m, r_in = 0.0256m)
          float dz = vLocalPos.z - 0.0440;
          float rad2 = vLocalPos.y * vLocalPos.y + dz * dz;
          float rMax = 0.0256;
          if (rad2 > rMax * rMax) discard;
          // Type-"C" 12-6 退台紫铜锅筒真值: X < -47.0mm 后段仅在 Z >= 46.8mm 冠板上方有水，绝无两侧湿式水腿!
          if (vLocalPos.x < -0.0470 && vLocalPos.z < 0.0468) discard;

          float edge = smoothstep(rMax * rMax, rMax * rMax * 0.72, rad2);
          float hotFrac = clamp((uTempTop - 30.0) / 130.0, 0.0, 1.0);
          vec3 coldWater = vec3(0.10, 0.48, 0.88);
          vec3 boilWater = vec3(0.14, 0.82, 0.92);
          vec3 foamCol   = vec3(0.86, 0.98, 1.00);

          float foam = clamp((1.0 - edge) * 0.55 + uBoil * (0.32 + 260.0 * abs(vWave)), 0.0, 1.0);
          vec3 col = mix(mix(coldWater, boilWater, hotFrac), foamCol, foam * 0.72);
          gl_FragColor = vec4(col, 0.56 + 0.22 * foam);
        }
      `,
    });

    const surfGeom = new THREE.PlaneGeometry(0.199, 0.052, 52, 18);
    surfGeom.translate(-0.010, 0.0, 0.0);
    this.waterSurfaceMesh = new THREE.Mesh(surfGeom, this.waterSurfaceMat);
    this.waterSurfaceMesh.renderOrder = 3;
    this.waterSurfaceMesh.visible = false;
    this.root.add(this.waterSurfaceMesh);

    this.waterVolumeMat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      side: THREE.BackSide,
      uniforms: waterUniforms,
      vertexShader: /* glsl */ `
        varying vec3 vLocalPos;
        void main() {
          vLocalPos = position;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        uniform float uTime;
        uniform float uWaterZ;
        uniform float uSloshTan;
        uniform float uBoil;
        uniform float uTempBottom;
        uniform float uTempTop;
        varying vec3 vLocalPos;
        void main() {
          float wave = (0.0002 + 0.0010 * uBoil) * sin(vLocalPos.x * 95.0 - uTime * 11.0);
          float zSurf = uWaterZ + (vLocalPos.x + 0.010) * uSloshTan + wave;
          if (vLocalPos.z > zSurf) discard;
          if (vLocalPos.x < -0.0470 && vLocalPos.z < 0.0468) discard;

          float depthFrac = clamp((vLocalPos.z - 0.019) / max(0.008, zSurf - 0.019), 0.0, 1.0);
          float tLocal = mix(uTempBottom, uTempTop, depthFrac);
          float hot = clamp((tLocal - 25.0) / 135.0, 0.0, 1.0);
          vec3 deepCol = mix(vec3(0.05, 0.28, 0.62), vec3(0.06, 0.52, 0.72), hot);
          vec3 topCol  = mix(vec3(0.12, 0.52, 0.86), vec3(0.18, 0.84, 0.94), hot);
          vec3 col = mix(deepCol, topCol, depthFrac);
          gl_FragColor = vec4(col, 0.28);
        }
      `,
    });

    const volGeom = new THREE.CylinderGeometry(0.0254, 0.0254, 0.199, 32, 16, false);
    volGeom.rotateZ(Math.PI * 0.5);
    volGeom.translate(-0.010, 0.0, 0.0440);
    this.waterVolumeMesh = new THREE.Mesh(volGeom, this.waterVolumeMat);
    this.waterVolumeMesh.renderOrder = 2;
    this.waterVolumeMesh.visible = false;
    this.root.add(this.waterVolumeMesh);

    // 2A. 锅筒汽空间 (z > zSurf) 三维加压饱和蒸汽体积云着色器 (随锅炉表压、沸腾强度与主蒸汽抽吸流动翻涌)
    this.steamHeadspaceMat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      side: THREE.BackSide,
      blending: THREE.NormalBlending,
      uniforms: {
        uTime: { value: 0.0 },
        uWaterZ: { value: 0.0555 },
        uSloshTan: { value: 0.0 },
        uPressureNorm: { value: 0.0 },
        uBoil: { value: 0.0 },
        uSteamFlow: { value: 0.0 },
      },
      vertexShader: /* glsl */ `
        varying vec3 vLocalPos;
        void main() {
          vLocalPos = position;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        uniform float uTime;
        uniform float uWaterZ;
        uniform float uSloshTan;
        uniform float uPressureNorm;
        uniform float uBoil;
        uniform float uSteamFlow;
        varying vec3 vLocalPos;

        float hash21(vec2 p) {
          p = fract(p * vec2(127.1, 311.7));
          p += dot(p, p + 34.5);
          return fract(p.x * p.y);
        }
        float noise2(vec2 p) {
          vec2 i = floor(p);
          vec2 f = fract(p);
          f = f * f * (3.0 - 2.0 * f);
          return mix(
            mix(hash21(i), hash21(i + vec2(1.0, 0.0)), f.x),
            mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), f.x),
            f.y
          );
        }
        void main() {
          float dens = clamp(0.65 * uPressureNorm + 0.55 * uBoil, 0.0, 1.0);
          if (dens < 0.03) discard;
          float zSurf = uWaterZ + (vLocalPos.x + 0.010) * uSloshTan;
          if (vLocalPos.z <= zSurf + 0.0005) discard;
          if (vLocalPos.x < -0.0470 && vLocalPos.z < 0.0468) discard;

          float drift = uTime * (0.9 + 3.2 * uSteamFlow);
          float f1 = noise2(vec2(vLocalPos.x * 46.0 + drift, vLocalPos.z * 78.0 - uTime * 1.4));
          float f2 = noise2(vec2(vLocalPos.x * 96.0 - drift * 0.7, (vLocalPos.y + vLocalPos.z) * 110.0 + uTime * 2.1));
          float wisp = 0.62 * f1 + 0.38 * f2;

          float crownGlow = smoothstep(0.044, 0.069, vLocalPos.z);
          vec3 satMist = mix(vec3(0.76, 0.90, 0.98), vec3(0.96, 0.98, 1.00), wisp);
          float alpha = dens * (0.12 + 0.24 * wisp) * (0.65 + 0.35 * crownGlow);
          gl_FragColor = vec4(satMist, clamp(alpha, 0.0, 0.42));
        }
      `,
    });
    const hsGeom = new THREE.CylinderGeometry(0.0252, 0.0252, 0.198, 32, 12, false);
    hsGeom.rotateZ(Math.PI * 0.5);
    hsGeom.translate(-0.010, 0.0, 0.0440);
    this.steamHeadspaceMesh = new THREE.Mesh(hsGeom, this.steamHeadspaceMat);
    this.steamHeadspaceMesh.renderOrder = 3;
    this.steamHeadspaceMesh.visible = false;
    this.root.add(this.steamHeadspaceMesh);

    // 2B. 烟室 20-7 乏汽喷嘴 / 17-3 通风喷嘴 -> 18-28 ペチコート -> 18-25 烟囱 三维文丘里高速射流锥
    this.smokeboxBlastMat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      uniforms: {
        uTime: { value: 0.0 },
        uJet: { value: 0.0 },
        uDryness: { value: 0.92 },
      },
      vertexShader: /* glsl */ `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        uniform float uTime;
        uniform float uJet;
        uniform float uDryness;
        varying vec2 vUv;

        float hash21(vec2 p) {
          p = fract(p * vec2(233.34, 511.21));
          p += dot(p, p + 29.17);
          return fract(p.x * p.y);
        }
        float noise2(vec2 p) {
          vec2 i = floor(p);
          vec2 f = fract(p);
          f = f * f * (3.0 - 2.0 * f);
          return mix(
            mix(hash21(i), hash21(i + vec2(1.0, 0.0)), f.x),
            mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), f.x),
            f.y
          );
        }
        void main() {
          if (uJet < 0.02) discard;
          float h = clamp(vUv.y, 0.0, 1.0);
          float r = abs(vUv.x - 0.5) * 2.0;
          // 射流自 ø4.2mm 喷嘴 (h=0) 扩散射入 18-28 喇叭口喉部 (h=0.45) 并直冲烟囱顶 (h=1.0)
          float jetHalfW = mix(0.18, 0.62, pow(h, 0.72));
          float n = 0.65 * noise2(vec2(vUv.x * 7.0, h * 9.0 - uTime * (16.0 + 18.0 * uJet)))
                  + 0.35 * noise2(vec2(vUv.x * 15.0 + 2.3, h * 18.0 - uTime * 26.0));
          float cone = smoothstep(jetHalfW + 0.14 * n, jetHalfW * 0.12, r) * smoothstep(0.0, 0.06, h) * smoothstep(1.0, 0.82, h);
          if (cone < 0.015) discard;
          vec3 coreCol = mix(vec3(0.55, 0.88, 1.00), vec3(0.94, 0.98, 1.00), clamp(1.0 - h * 0.5 + 0.3 * (1.0 - uDryness), 0.0, 1.0));
          float alpha = cone * clamp(0.18 + 0.58 * uJet, 0.0, 0.75) * (0.68 + 0.42 * n);
          gl_FragColor = vec4(coreCol, alpha);
        }
      `,
    });
    for (const ang of [0.0, Math.PI * 0.5]) {
      const bjGeom = new THREE.PlaneGeometry(0.017, 0.057, 4, 12);
      bjGeom.rotateX(Math.PI * 0.5);
      bjGeom.rotateZ(ang);
      bjGeom.translate(0.130, 0.0, 0.0630);
      const bjMesh = new THREE.Mesh(bjGeom, this.smokeboxBlastMat);
      bjMesh.renderOrder = 5;
      bjMesh.visible = false;
      this.smokeboxBlastMeshes.push(bjMesh);
      this.root.add(bjMesh);
    }

    // 17-3 通風ノズル (+130.0, +4.2, +35.2mm) 独立活汽引射细射流锥 (停车开通风阀时自 J 字管喷口直射 18-28 喇叭口制造烟室负压!)
    this.blowerJetMat = this.smokeboxBlastMat.clone();
    this.blowerJetMat.uniforms = {
      uTime: { value: 0.0 },
      uJet: { value: 0.0 },
      uDryness: { value: 0.98 },
    };
    for (const ang of [0.0, Math.PI * 0.5]) {
      const blwGeom = new THREE.PlaneGeometry(0.0085, 0.048, 4, 10);
      blwGeom.rotateX(Math.PI * 0.5);
      blwGeom.rotateZ(ang);
      blwGeom.translate(0.130, +0.0042, 0.0592);
      const blwMesh = new THREE.Mesh(blwGeom, this.blowerJetMat);
      blwMesh.renderOrder = 5;
      blwMesh.visible = false;
      this.blowerJetMeshes.push(blwMesh);
      this.root.add(blwMesh);
    }

    // 180 颗核态沸腾上升蒸汽泡 (从紫铜火室顶板及 5 根烟管外壁核化升起至自由液面破裂)
    this.bubblePositions = new Float32Array(this.bubbleCount * 3);
    this.bubbleSeeds = new Float32Array(this.bubbleCount);
    for (let i = 0; i < this.bubbleCount; i++) {
      this.respawnBubble(i, Math.random());
      this.bubbleSeeds[i] = Math.random();
    }
    const bubbleGeom = new THREE.BufferGeometry();
    bubbleGeom.setAttribute("position", new THREE.BufferAttribute(this.bubblePositions, 3));
    bubbleGeom.setAttribute("aSeed", new THREE.BufferAttribute(this.bubbleSeeds, 1));

    this.bubbleMat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      uniforms: {
        uBoil: { value: 0.0 },
        uPixelRatio: { value: Math.min(window.devicePixelRatio || 1, 2) },
      },
      vertexShader: /* glsl */ `
        attribute float aSeed;
        uniform float uBoil;
        uniform float uPixelRatio;
        varying float vAlpha;
        void main() {
          vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
          float sz = (2.2 + 3.2 * aSeed) * (0.45 + 0.75 * uBoil) * uPixelRatio;
          gl_PointSize = clamp(sz * (0.45 / max(0.08, -mvPos.z)), 1.5, 16.0);
          vAlpha = clamp(uBoil * 1.35, 0.0, 0.90);
          gl_Position = projectionMatrix * mvPos;
        }
      `,
      fragmentShader: /* glsl */ `
        varying float vAlpha;
        void main() {
          if (vAlpha < 0.02) discard;
          vec2 c = gl_PointCoord * 2.0 - 1.0;
          float r2 = dot(c, c);
          if (r2 > 1.0) discard;
          float rim = smoothstep(0.15, 0.92, r2) * (1.0 - smoothstep(0.85, 1.0, r2));
          float core = (1.0 - r2) * 0.45;
          gl_FragColor = vec4(vec3(0.75, 0.97, 1.0), (rim + core) * vAlpha);
        }
      `,
    });
    this.bubblePoints = new THREE.Points(bubbleGeom, this.bubbleMat);
    this.bubblePoints.renderOrder = 4;
    this.bubblePoints.visible = false;
    this.root.add(this.bubblePoints);

    // 11-7 驾驶室右侧外缘 (-Y=-0.0313) パイレックス玻璃管水面计实况液柱 (X=-0.1185, Y=-0.0313, Z=0.0375..0.0545)
    const sgGeom = new THREE.CylinderGeometry(0.00108, 0.00108, 1.0, 14);
    sgGeom.rotateX(Math.PI * 0.5);
    sgGeom.translate(0, 0, 0.5);
    const sgMat = new THREE.MeshBasicMaterial({
      color: 0x2ee6ff,
      transparent: true,
      opacity: 0.88,
    });
    this.sightGlassWaterMesh = new THREE.Mesh(sgGeom, sgMat);
    this.sightGlassWaterMesh.position.set(-0.1185, -0.0313, 0.0375);
    this.sightGlassWaterMesh.scale.set(1, 1, 0.0085);
    this.sightGlassWaterMesh.renderOrder = 5;
    this.root.add(this.sightGlassWaterMesh);

    const menGeom = new THREE.CylinderGeometry(0.00126, 0.00126, 0.0007, 14);
    menGeom.rotateX(Math.PI * 0.5);
    this.sightGlassMeniscusMesh = new THREE.Mesh(
      menGeom,
      new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.96 })
    );
    this.sightGlassMeniscusMesh.position.set(-0.1185, -0.0313, 0.0460);
    this.sightGlassMeniscusMesh.renderOrder = 6;
    this.root.add(this.sightGlassMeniscusMesh);

    // -----------------------------------------------------------------------
    // 3. RTX 5080 64×24 纵剖面实时三维热场云图切片 (Y = 0 中心纵剖面)
    // -----------------------------------------------------------------------
    this.sliceTexData = new Uint8Array(new ArrayBuffer(64 * 24 * 4));
    this.sliceTex = new THREE.DataTexture(this.sliceTexData, 64, 24, THREE.RGBAFormat);
    this.sliceTex.magFilter = THREE.LinearFilter;
    this.sliceTex.minFilter = THREE.LinearFilter;
    this.sliceTex.needsUpdate = true;

    this.sliceMat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      uniforms: {
        uTex: { value: this.sliceTex },
      },
      vertexShader: /* glsl */ `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        uniform sampler2D uTex;
        varying vec2 vUv;

        vec3 ironbow(float t) {
          float x = clamp(t, 0.0, 1.0);
          if (x < 0.25) return mix(vec3(0.02, 0.08, 0.35), vec3(0.08, 0.55, 0.85), x / 0.25);
          if (x < 0.55) return mix(vec3(0.08, 0.55, 0.85), vec3(0.95, 0.42, 0.08), (x - 0.25) / 0.30);
          if (x < 0.82) return mix(vec3(0.95, 0.42, 0.08), vec3(1.00, 0.82, 0.18), (x - 0.55) / 0.27);
          return mix(vec3(1.00, 0.82, 0.18), vec3(1.00, 0.98, 0.92), (x - 0.82) / 0.18);
        }
        void main() {
          float tempNorm = texture2D(uTex, vUv).r;
          if (tempNorm < 0.05) discard;
          // 边缘柔化防止矩形硬边
          float border = smoothstep(0.0, 0.06, vUv.x) * smoothstep(1.0, 0.94, vUv.x)
                       * smoothstep(0.0, 0.08, vUv.y) * smoothstep(1.0, 0.92, vUv.y);
          vec3 col = ironbow(tempNorm);
          float alpha = clamp((tempNorm - 0.05) * 0.46, 0.0, 0.38) * border;
          gl_FragColor = vec4(col, alpha);
        }
      `,
    });

    const sliceGeom = new THREE.PlaneGeometry(0.276, 0.078);
    sliceGeom.rotateX(Math.PI * 0.5);
    sliceGeom.translate(0.024, 0.0, 0.049);
    this.sliceMesh = new THREE.Mesh(sliceGeom, this.sliceMat);
    this.sliceMesh.renderOrder = 3;
    this.sliceMesh.visible = false;
    this.root.add(this.sliceMesh);

    // -----------------------------------------------------------------------
    // 4. 左右主气缸前后腔蒸汽膨胀云 + 滑阀进排汽口脉冲 + 底部冷凝水池 + 4-2 置换注油器油水分层芯
    // -----------------------------------------------------------------------
    const makeCylChamberMesh = (yCenter: number): THREE.Mesh => {
      const g = new THREE.CylinderGeometry(0.0070, 0.0070, 1.0, 22);
      g.rotateZ(Math.PI * 0.5);
      const m = new THREE.ShaderMaterial({
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        uniforms: {
          uTime: { value: 0.0 },
          uPressNorm: { value: 0.0 },
          uSuperheatNorm: { value: 0.3 },
          uOpacity: { value: 0.0 },
        },
        vertexShader: /* glsl */ `
          varying vec3 vLocal;
          void main() {
            vLocal = position;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
        `,
        fragmentShader: /* glsl */ `
          uniform float uTime;
          uniform float uPressNorm;
          uniform float uSuperheatNorm;
          uniform float uOpacity;
          varying vec3 vLocal;

          float hash21(vec2 p) {
            p = fract(p * vec2(129.1, 317.7));
            p += dot(p, p + 41.2);
            return fract(p.x * p.y);
          }
          float noise2(vec2 p) {
            vec2 i = floor(p);
            vec2 f = fract(p);
            f = f * f * (3.0 - 2.0 * f);
            return mix(
              mix(hash21(i), hash21(i + vec2(1.0, 0.0)), f.x),
              mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), f.x),
              f.y
            );
          }
          void main() {
            if (uOpacity < 0.015) discard;
            float n = 0.65 * noise2(vec2(vLocal.x * 6.5 - uTime * 11.0, (vLocal.y + vLocal.z) * 180.0))
                    + 0.35 * noise2(vec2(vLocal.x * 14.0 + uTime * 17.0, (vLocal.y - vLocal.z) * 320.0));
            vec3 exhMist = vec3(0.56, 0.88, 1.00);
            vec3 liveHot = mix(vec3(0.95, 0.96, 1.00), vec3(1.00, 0.66, 0.24), uSuperheatNorm);
            vec3 col = mix(exhMist, liveHot, clamp(uPressNorm * 1.25, 0.0, 1.0));
            gl_FragColor = vec4(col, clamp(uOpacity * (0.68 + 0.52 * n), 0.0, 0.85));
          }
        `,
      });
      const mesh = new THREE.Mesh(g, m);
      mesh.position.set(0.1332, yCenter, 0.0);
      mesh.renderOrder = 4;
      mesh.visible = false;
      this.root.add(mesh);
      return mesh;
    };

    // 左右 ø15mm 活塞工作腔精确同轴对齐主气缸中心 Y = ±32.5mm (杜绝 ±36.0mm 外偏造成单侧视觉遮挡!)
    this.cylHeadMeshes = [makeCylChamberMesh(+0.0325), makeCylChamberMesh(-0.0325)];
    this.cylCrankMeshes = [makeCylChamberMesh(+0.0325), makeCylChamberMesh(-0.0325)];

    const makeCondensateMesh = (yCenter: number): THREE.Mesh => {
      const g = new THREE.BoxGeometry(0.025, 0.0085, 0.0026);
      const m = new THREE.MeshBasicMaterial({
        color: 0x1fd2ff,
        transparent: true,
        opacity: 0.0,
        depthWrite: false,
      });
      const mesh = new THREE.Mesh(g, m);
      mesh.position.set(0.1332, yCenter, -0.0056);
      mesh.renderOrder = 4;
      mesh.visible = false;
      this.root.add(mesh);
      return mesh;
    };
    this.cylCondMeshes = [makeCondensateMesh(+0.0325), makeCondensateMesh(-0.0325)];

    const makePortGlow = (xPos: number, yPos: number): THREE.Mesh => {
      const g = new THREE.BoxGeometry(0.0022, 0.0056, 0.0036);
      const m = new THREE.MeshBasicMaterial({
        color: 0xffaa44,
        transparent: true,
        opacity: 0.0,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });
      const mesh = new THREE.Mesh(g, m);
      mesh.position.set(xPos, yPos, 0.0108);
      mesh.renderOrder = 4;
      mesh.visible = false;
      this.root.add(mesh);
      return mesh;
    };
    this.portGlowMeshes = [
      makePortGlow(0.1292, +0.0360), // L Head port (1-9 Cross-Port Rear mirror slot -> Head chamber)
      makePortGlow(0.1372, +0.0360), // L Crank port (1-9 Cross-Port Front mirror slot -> Crank chamber)
      makePortGlow(0.1292, -0.0360), // R Head port
      makePortGlow(0.1372, -0.0360), // R Crank port
    ];

    // 4-2 前车架卧式置换注油器油水分层内芯 + 内部银焊竖直溢流立管 (Roscoe Standpipe: X=+0.1462, Y=0, Z∈[-0.0006,+0.0109])
    this.lubeCoreMat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      uniforms: {
        uWaterFrac: { value: 0.20 },
        uTime: { value: 0.0 },
        uWaterCol: { value: new THREE.Color(0.06, 0.58, 0.76) },
        uOilCol: { value: new THREE.Color(0.18, 0.76, 0.42) },
      },
      vertexShader: /* glsl */ `
        varying vec3 vLocal;
        void main() {
          vLocal = position;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        uniform float uWaterFrac;
        uniform float uTime;
        uniform vec3 uWaterCol;
        uniform vec3 uOilCol;
        varying vec3 vLocal;
        void main() {
          float r = 0.0050;
          float zRel = (vLocal.z - 0.0062) / (2.0 * r) + 0.5;
          float split = clamp(uWaterFrac + 0.012 * sin(vLocal.x * 260.0 + uTime * 4.0), 0.0, 0.90);
          float topHead = smoothstep(0.90, 0.96, zRel);
          float interfaceBand = smoothstep(split - 0.035, split + 0.035, zRel);
          vec3 liqCol = mix(uWaterCol, uOilCol, interfaceBand);
          // Z > +10.9mm 顶空高温蒸汽凝结区 + 油层中按 Stokes 定律沉降的微细凝结水珠闪烁
          float dropWave = step(0.965, fract(sin(dot(floor(vec2(vLocal.x * 1600.0, (vLocal.z + uTime * 0.0025) * 1800.0)), vec2(12.9898, 78.233))) * 43758.5453));
          liqCol = mix(liqCol, uWaterCol + vec3(0.25), dropWave * interfaceBand * (1.0 - topHead) * 0.65);
          vec3 steamCapCol = vec3(0.98, 0.68, 0.16);
          vec3 col = mix(liqCol, steamCapCol, topHead);
          float men = 1.0 - abs(interfaceBand * 2.0 - 1.0);
          col += vec3(0.32, 0.36, 0.38) * men * 0.55;
          gl_FragColor = vec4(col, mix(0.84, 0.52, topHead));
        }
      `,
    });
    const lubeGeom = new THREE.CylinderGeometry(0.0050, 0.0050, 0.0202, 22);
    lubeGeom.rotateZ(Math.PI * 0.5);
    lubeGeom.translate(0.1540, 0.0, 0.0062);
    this.lubeCoreMesh = new THREE.Mesh(lubeGeom, this.lubeCoreMat);
    this.lubeCoreMesh.renderOrder = 3;
    this.lubeCoreMesh.visible = false;
    this.root.add(this.lubeCoreMesh);

    // 4-2 内部银焊竖直溢流立管 (自 -Z 底孔 Z=-0.6mm 直通油箱顶部 Z=+10.9mm: 物理隔离罐底沉水、收集顶层浮油!)
    const spGeom = new THREE.CylinderGeometry(0.00105, 0.00105, 0.0115, 16, 1, true);
    spGeom.rotateX(Math.PI * 0.5);
    spGeom.translate(0.1462, 0.0, 0.00515);
    this.lubeStandpipeMesh = new THREE.Mesh(
      spGeom,
      new THREE.MeshBasicMaterial({
        color: 0xfbbf24,
        transparent: true,
        opacity: 0.92,
        side: THREE.DoubleSide,
        depthWrite: false,
      })
    );
    this.lubeStandpipeMesh.renderOrder = 5;
    this.lubeStandpipeMesh.visible = false;
    this.root.add(this.lubeStandpipeMesh);

    // -----------------------------------------------------------------------
    // 5. P3 煤水车 (Tender) 31-10 后置独立水箱 + 35-1 敞口湾 Domain 6 浅水波纹面 + 36-4 前置上法兰吊挂酒精箱
    // -----------------------------------------------------------------------
    const twGeom = new THREE.BoxGeometry(0.082, 0.074, 1.0);
    twGeom.translate(-0.04675, 0.0, 0.5);
    this.tenderWaterMesh = new THREE.Mesh(
      twGeom,
      new THREE.MeshBasicMaterial({ color: 0x1e88e5, transparent: true, opacity: 0.34, depthWrite: false })
    );
    this.tenderWaterMesh.position.set(0.0, 0.0, 0.0150);
    this.tenderWaterMesh.scale.set(1.0, 1.0, 0.036);
    this.tenderWaterMesh.renderOrder = 3;
    this.tenderWaterMesh.visible = false;
    this.tenderRoot.add(this.tenderWaterMesh);

    // 35-1 后置敞口注水槽自由液面 (X∈[-0.0840, -0.0185], Y∈[-0.0168, +0.0168], Domain 6 24×10 GPU 浅水波高纹理)
    this.tenderWaveTexData = new Uint8Array(new ArrayBuffer(24 * 10 * 4));
    for (let i = 0; i < 24 * 10; i++) {
      this.tenderWaveTexData[4 * i] = 128;
      this.tenderWaveTexData[4 * i + 1] = 128;
      this.tenderWaveTexData[4 * i + 2] = 128;
      this.tenderWaveTexData[4 * i + 3] = 255;
    }
    this.tenderWaveTex = new THREE.DataTexture(this.tenderWaveTexData, 24, 10, THREE.RGBAFormat);
    this.tenderWaveTex.magFilter = THREE.LinearFilter;
    this.tenderWaveTex.minFilter = THREE.LinearFilter;
    this.tenderWaveTex.needsUpdate = true;

    this.tenderOpenBayMat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      uniforms: {
        uWaveTex: { value: this.tenderWaveTex },
        uTime: { value: 0.0 },
        uWaterZ: { value: 0.0525 },
        uTempC: { value: 25.0 },
        uJetAmp: { value: 0.0 },
      },
      vertexShader: /* glsl */ `
        uniform sampler2D uWaveTex;
        uniform float uTime;
        uniform float uWaterZ;
        uniform float uJetAmp;
        varying vec2 vUv;
        varying float vEtaMm;
        void main() {
          vUv = uv;
          vec3 p = position;
          float texEta = (texture2D(uWaveTex, uv).r - 0.50196) * 5.0; // ±2.5 mm
          float micro = uJetAmp * 0.35 * sin(uv.x * 28.0 - uTime * 15.0) * cos(uv.y * 22.0 + uTime * 11.0);
          vEtaMm = texEta + micro;
          p.z = uWaterZ + vEtaMm * 0.001;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        uniform float uTempC;
        varying vec2 vUv;
        varying float vEtaMm;
        void main() {
          float warm = clamp((uTempC - 24.0) / 36.0, 0.0, 1.0);
          vec3 coldCol = vec3(0.08, 0.52, 0.90);
          vec3 warmCol = vec3(0.16, 0.82, 0.94);
          vec3 crest   = vec3(0.86, 0.98, 1.00);
          float ripple = clamp(abs(vEtaMm) * 0.75, 0.0, 1.0);
          float men = 1.0 - smoothstep(0.0, 0.07, min(min(vUv.x, 1.0 - vUv.x), min(vUv.y, 1.0 - vUv.y)));
          vec3 col = mix(mix(coldCol, warmCol, warm), crest, clamp(ripple * 0.65 + men * 0.35, 0.0, 0.85));
          gl_FragColor = vec4(col, 0.76 + 0.16 * ripple);
        }
      `,
    });
    const obGeom = new THREE.PlaneGeometry(0.0630, 0.0316, 23, 9);
    obGeom.translate(-0.0450, 0.0, 0.0);
    this.tenderOpenBayWaterMesh = new THREE.Mesh(obGeom, this.tenderOpenBayMat);
    this.tenderOpenBayWaterMesh.renderOrder = 4;
    this.tenderOpenBayWaterMesh.visible = false;
    this.tenderRoot.add(this.tenderOpenBayWaterMesh);

    const taGeom = new THREE.BoxGeometry(0.0576, 0.0576, 1.0);
    taGeom.translate(+0.0302, 0.0, 0.5);
    this.tenderAlcoholMesh = new THREE.Mesh(
      taGeom,
      new THREE.MeshBasicMaterial({ color: 0xf7b731, transparent: true, opacity: 0.38, depthWrite: false })
    );
    this.tenderAlcoholMesh.position.set(0.0, 0.0, 0.0258);
    this.tenderAlcoholMesh.scale.set(1.0, 1.0, 0.0350);
    this.tenderAlcoholMesh.renderOrder = 3;
    this.tenderAlcoholMesh.visible = false;
    this.tenderRoot.add(this.tenderAlcoholMesh);

    // 31-14 左前防虹吸鹅颈回水喷流 (回水): 自 (-0.0260, +0.0110, +0.0537) 落入 35-1 敞口槽水面
    const rsGeom = new THREE.CylinderGeometry(0.0009, 0.0014, 1.0, 12);
    rsGeom.rotateX(Math.PI * 0.5);
    rsGeom.translate(0, 0, -0.5);
    this.returnStreamMesh = new THREE.Mesh(
      rsGeom,
      new THREE.MeshBasicMaterial({
        color: 0x38e8ff,
        transparent: true,
        opacity: 0.85,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      })
    );
    this.returnStreamMesh.position.set(-0.0260, 0.0115, 0.0536);
    this.returnStreamMesh.scale.set(1.0, 1.0, 0.012);
    this.returnStreamMesh.renderOrder = 5;
    this.returnStreamMesh.visible = false;
    this.tenderRoot.add(this.returnStreamMesh);

    // 35-1 敞口注水槽 加水 水柱: 自上方 (-0.0450, 0.0, +0.0820) 垂直注入自由液面
    const wfGeom = new THREE.CylinderGeometry(0.0018, 0.0013, 1.0, 14);
    wfGeom.rotateX(Math.PI * 0.5);
    wfGeom.translate(0, 0, -0.5);
    this.waterFillStreamMesh = new THREE.Mesh(
      wfGeom,
      new THREE.MeshBasicMaterial({
        color: 0x52b8ff,
        transparent: true,
        opacity: 0.82,
        depthWrite: false,
      })
    );
    this.waterFillStreamMesh.position.set(-0.0450, 0.0, 0.0820);
    this.waterFillStreamMesh.scale.set(1.0, 1.0, 0.032);
    this.waterFillStreamMesh.renderOrder = 5;
    this.waterFillStreamMesh.visible = false;
    this.tenderRoot.add(this.waterFillStreamMesh);

    // 31-11 -> 31-9 底部排水口 放水 水流: 自煤水车底板 (-0.0410, -0.0103, +0.014) 排向轨面
    const tdGeom = new THREE.CylinderGeometry(0.0014, 0.0020, 0.034, 12);
    tdGeom.rotateX(Math.PI * 0.5);
    this.tenderDrainStreamMesh = new THREE.Mesh(
      tdGeom,
      new THREE.MeshBasicMaterial({
        color: 0x38c8ff,
        transparent: true,
        opacity: 0.84,
        depthWrite: false,
      })
    );
    this.tenderDrainStreamMesh.position.set(-0.0410, -0.0103, -0.0030);
    this.tenderDrainStreamMesh.renderOrder = 5;
    this.tenderDrainStreamMesh.visible = false;
    this.tenderRoot.add(this.tenderDrainStreamMesh);

    this.mariotteBubblePos = new Float32Array(28 * 3);
    for (let i = 0; i < 28; i++) {
      // 36-6 单根真实马氏定液位立管位置 (+21.7, -6.0 mm, イラスト 36)
      this.mariotteBubblePos[3 * i] = 0.0217 + (Math.random() - 0.5) * 0.0036;
      this.mariotteBubblePos[3 * i + 1] = -0.0060 + (Math.random() - 0.5) * 0.0036;
      this.mariotteBubblePos[3 * i + 2] = 0.0265 + Math.random() * 0.032;
    }
    const mbGeom = new THREE.BufferGeometry();
    mbGeom.setAttribute("position", new THREE.BufferAttribute(this.mariotteBubblePos, 3));
    this.mariotteBubblePoints = new THREE.Points(
      mbGeom,
      new THREE.PointsMaterial({ color: 0xfff3c4, size: 0.0028, transparent: true, opacity: 0.88, depthWrite: false })
    );
    this.mariotteBubblePoints.renderOrder = 5;
    this.mariotteBubblePoints.visible = false;
    this.tenderRoot.add(this.mariotteBubblePoints);

    // -----------------------------------------------------------------------
    // 6. 全车 35 路真实物理内腔流道 (煤水车内管 + 三平行跨车软管 + 底盘泵油醇 + 锅炉/驾驶室 + 双缸可逆气道) & 三维流向箭头场
    // -----------------------------------------------------------------------
    this.initAllPipeLumensAndArrows();
  }

  private initAllPipeLumensAndArrows(): void {
    type Spec = {
      id: string;
      partNo: string;
      title: string;
      zone: InternalPipeRuntime["zone"];
      frame: InternalPipeRuntime["frame"];
      ptsMm: [number, number, number][];
      extraBranchesMm?: [number, number, number][][];
      bendRadiusMm?: number;
      umbEndsMm?: { tdr: [number, number, number]; loco: [number, number, number] };
      radiusMm: number;
      arrows: number;
      labelT: number;
      colHexA: number;
      colHexB: number;
      css: string;
      evalFlow: InternalPipeRuntime["evalFlow"];
    };

    const specs: Spec[] = [
      // === A. 煤水车内部全套管路 (7 路) ===
      {
        id: "tdr_36_6_mariotte",
        partNo: "36-6",
        title: "マリオット定液位立管 (醇下降/補気泡上昇)",
        zone: "tender",
        frame: "tender",
        ptsMm: [[21.7, -6.0, 58.0], [21.7, -6.0, 32.0], [21.7, -6.0, 9.6]],
        radiusMm: 1.45,
        arrows: 3,
        labelT: 0.35,
        colHexA: 0xd98a12,
        colHexB: 0xffe066,
        css: "#fbbf24",
        evalFlow: (out, sp) => ({
          signedRate: out.alcoholMl > 0.3 ? (out.wickCapillaryMlS ?? sp.wickCapillaryMlS ?? 0.0) + (out.mariotteFloodAlarm ? 0.35 : 0.0) : 0.0,
          fill: out.alcoholMl > 0.3 ? 0.90 : 0.0,
          unit: "ml/s",
        }),
      },
      {
        id: "tdr_34_2_fuel",
        partNo: "34-2",
        title: "恒液位サンプ→前端供醇銅管",
        zone: "tender",
        frame: "tender",
        ptsMm: [[28.4, 0.0, 6.2], [42.2, 0.0, 6.2], [62.0, 0.0, 7.0], [107.2, 0.0, 7.0]],
        radiusMm: 1.30,
        arrows: 4,
        labelT: 0.55,
        colHexA: 0xd98a12,
        colHexB: 0xffe066,
        css: "#fbbf24",
        evalFlow: (out, sp) => ({
          signedRate: out.alcoholMl > 0.2 ? (out.wickCapillaryMlS ?? sp.wickCapillaryMlS ?? 0.0) : 0.0,
          fill: out.alcoholMl > 0.2 ? 0.88 : 0.0,
          unit: "ml/s",
        }),
      },
      {
        id: "tdr_31_5_31_13_hp",
        partNo: "31-6→31-5→31-13",
        title: "炭水車給水吸込・手動ポンプ共用塔 (SB-ø4×2)→吐出管",
        zone: "tender",
        frame: "tender",
        ptsMm: [[-56.0, 4.0, 17.5], [-69.5, 4.0, 17.5], [-69.5, 4.0, 35.7], [-69.5, -21.3, 35.7], [-16.4, -21.3, 35.7], [-16.4, -21.3, 18.2]],
        radiusMm: 0.95,
        arrows: 5,
        labelT: 0.46,
        colHexA: 0x1d4ed8,
        colHexB: 0x06b6d4,
        css: "#06b6d4",
        evalFlow: (out) => ({
          signedRate: out.tenderWaterMl > 0.3 ? out.feedwaterGS + out.returnWaterMlS : 0.0,
          fill: out.tenderWaterMl > 0.3 ? 0.90 : 0.0,
          unit: "ml/s",
        }),
      },
      {
        id: "tdr_34_6_feed_R",
        partNo: "34-6(右/-Y)",
        title: "炭水車床下給水本管 (吸込/圧送共用)",
        zone: "tender",
        frame: "tender",
        ptsMm: [[-16.4, -21.3, 18.2], [-16.4, -21.3, 16.2], [45.0, -21.3, 16.2], [102.2, -21.3, 16.2], [103.8, -21.3, -5.1], [109.5, -21.3, -5.1]],
        radiusMm: 0.95,
        arrows: 5,
        labelT: 0.52,
        colHexA: 0x1d4ed8,
        colHexB: 0x06b6d4,
        css: "#06b6d4",
        evalFlow: (out) => ({
          signedRate: out.tenderWaterMl > 0.3 ? out.feedwaterGS + out.returnWaterMlS : 0.0,
          fill: out.tenderWaterMl > 0.3 ? 0.90 : 0.0,
          unit: "ml/s",
        }),
      },
      {
        id: "tdr_34_6_ret_L",
        partNo: "34-6(左/+Y)",
        title: "炭水車床下バイパス無圧戻水管",
        zone: "tender",
        frame: "tender",
        ptsMm: [[109.5, 21.3, -5.1], [103.8, 21.3, -5.1], [102.2, 21.3, 16.2], [42.0, 21.3, 16.2], [-16.4, 21.3, 16.2], [-16.4, 21.3, 18.2]],
        radiusMm: 0.95,
        arrows: 5,
        labelT: 0.50,
        colHexA: 0x0891b2,
        colHexB: 0x67e8f9,
        css: "#22d3ee",
        evalFlow: (out) => ({
          signedRate: out.returnWaterMlS,
          fill: out.returnWaterMlS > 0.005 ? 0.88 : 0.25,
          unit: "ml/s",
        }),
      },
      {
        id: "tdr_31_14_gooseneck",
        partNo: "31-14",
        title: "高位グースネック防虹吸戻水立管",
        zone: "tender",
        frame: "tender",
        ptsMm: [[-16.4, 21.3, 18.2], [-16.4, 21.3, 54.2], [-22.0, 15.0, 55.8], [-26.0, 11.5, 53.6]],
        radiusMm: 0.95,
        arrows: 3,
        labelT: 0.72,
        colHexA: 0x0891b2,
        colHexB: 0x7ef9ff,
        css: "#22d3ee",
        evalFlow: (out) => ({
          signedRate: out.returnWaterMlS,
          fill: out.returnWaterMlS > 0.005 ? 0.90 : 0.18,
          unit: "ml/s",
        }),
      },
      {
        id: "tdr_31_9_drain",
        partNo: "31-11/31-9",
        title: "水槽底スタンドパイプ放水口",
        zone: "tender",
        frame: "tender",
        ptsMm: [[-41.0, -10.3, 24.0], [-41.0, -10.3, 14.8], [-41.0, -10.3, -4.0]],
        radiusMm: 1.40,
        arrows: 2,
        labelT: 0.65,
        colHexA: 0x1d4ed8,
        colHexB: 0x60a5fa,
        css: "#60a5fa",
        evalFlow: (out) => ({
          signedRate: out.tenderDrainMlS ?? 0.0,
          fill: (out.tenderDrainMlS ?? 0.0) > 0.01 ? 0.92 : 0.15,
          unit: "ml/s",
        }),
      },

      // === B. 跨车三平行弹性软管 (3 路, 动态桥接煤水车与机车!) ===
      {
        id: "umb_st7_fuel",
        partNo: "ST-ø7",
        title: "中央平行シリコン供醇ホース",
        zone: "umbilical",
        frame: "umbilical",
        ptsMm: [[-0.20, 0, 0.048], [-0.175, 0, 0.045], [-0.15, 0, 0.045]],
        umbEndsMm: { tdr: [106.0, 0.0, 7.0], loco: [-150.0, 0.0, 3.6] },
        radiusMm: 1.55,
        arrows: 3,
        labelT: 0.50,
        colHexA: 0xd98a12,
        colHexB: 0xffe066,
        css: "#fbbf24",
        evalFlow: (out, sp) => ({
          signedRate: out.alcoholMl > 0.2 ? (out.wickCapillaryMlS ?? sp.wickCapillaryMlS ?? 0.0) : 0.0,
          fill: out.alcoholMl > 0.2 ? 0.88 : 0.0,
          unit: "ml/s",
        }),
      },
      {
        id: "umb_bt5_feed",
        partNo: "BT-ø5(-Y)",
        title: "右側給水ホース＋34-8活接",
        zone: "umbilical",
        frame: "umbilical",
        ptsMm: [[-0.20, -0.02, 0.034], [-0.175, -0.02, 0.032], [-0.15, -0.0192, 0.033]],
        umbEndsMm: { tdr: [108.5, -21.3, -5.1], loco: [-150.0, -19.2, -8.5] },
        radiusMm: 1.15,
        arrows: 3,
        labelT: 0.50,
        colHexA: 0x0969da,
        colHexB: 0x4cc9f0,
        css: "#38bdf8",
        evalFlow: (out) => ({
          signedRate: out.tenderWaterMl > 0.3 ? out.feedwaterGS + out.returnWaterMlS : 0.0,
          fill: out.tenderWaterMl > 0.3 ? 0.90 : 0.0,
          unit: "ml/s",
        }),
      },
      {
        id: "umb_bt5_return",
        partNo: "BT-ø5(+Y)",
        title: "左側バイパス無圧戻水ホース",
        zone: "umbilical",
        frame: "umbilical",
        ptsMm: [[-0.15, 0.0192, 0.033], [-0.175, 0.02, 0.032], [-0.20, 0.02, 0.034]],
        umbEndsMm: { tdr: [108.5, 21.3, -5.1], loco: [-150.0, 19.2, -8.5] }, // reversed below!
        radiusMm: 1.15,
        arrows: 3,
        labelT: 0.50,
        colHexA: 0x0891b2,
        colHexB: 0x67e8f9,
        css: "#22d3ee",
        evalFlow: (out) => ({
          signedRate: out.returnWaterMlS,
          fill: out.returnWaterMlS > 0.005 ? 0.88 : 0.22,
          unit: "ml/s",
        }),
      },

      // === C. 机车底盘给水泵、燃烧器、注油器与全部 5 烟管流道 ===
      {
        id: "loco_23_1_burner",
        partNo: "23-1",
        title: "バーナー供醇本管→3連芯カップ毛細吸上",
        zone: "underframe",
        frame: "loco",
        ptsMm: [[-150.5, 0.0, 3.6], [-132.0, 0.0, 10.2], [-93.0, 0.0, 10.2], [-77.0, 0.0, 10.2], [-61.0, 0.0, 10.2], [-61.0, 0.0, 17.5]],
        extraBranchesMm: [
          [[-93.0, 0.0, 10.2], [-93.0, 0.0, 17.5]],
          [[-77.0, 0.0, 10.2], [-77.0, 0.0, 17.5]],
        ],
        bendRadiusMm: 2.2,
        radiusMm: 1.18,
        arrows: 8,
        labelT: 0.38,
        colHexA: 0xd98a12,
        colHexB: 0xffe066,
        css: "#fbbf24",
        evalFlow: (out, sp) => ({
          signedRate: out.burnerLumenConnected === false ? 0.0 : (out.wickCapillaryMlS ?? sp.wickCapillaryMlS ?? 0.0),
          fill: out.alcoholMl > 0.2 && out.burnerLumenConnected !== false ? 0.90 : 0.0,
          unit: "ml/s",
        }),
      },
      {
        id: "loco_6_10_suction",
        partNo: "6-11→6-10",
        title: "軸動ポンプ車架内側吸込銅管",
        zone: "underframe",
        frame: "loco",
        ptsMm: [[-150.8, -19.2, -8.5], [-142.5, -19.2, -8.5], [-136.0, -8.2, 8.5], [-85.0, -8.2, 8.5], [-38.8, -8.2, 8.5], [-38.8, -8.2, -6.2], [-38.8, 0.0, -6.2]],
        bendRadiusMm: 2.4,
        radiusMm: 0.78,
        arrows: 9,
        labelT: 0.56,
        colHexA: 0x0969da,
        colHexB: 0x4cc9f0,
        css: "#38bdf8",
        evalFlow: (out) => ({
          signedRate: out.tenderWaterMl > 0.3 ? out.feedwaterGS + out.returnWaterMlS : 0.0,
          fill: out.tenderWaterMl > 0.3 ? 0.90 : 0.0,
          unit: "ml/s",
        }),
      },
      {
        id: "loco_6_5_6_6_deliv",
        partNo: "6-5→6-6→4-17",
        title: "軸動ポンプ双球室→車架内側吐出管(Y=-8.2)→バイパス弁",
        zone: "underframe",
        frame: "loco",
        ptsMm: [[-38.8, 0.0, -6.2], [-38.8, 0.0, 9.6], [-38.8, -8.2, 13.8], [-85.0, -8.2, 14.3], [-128.0, -8.2, 14.8], [-135.5, -5.5, 15.0]],
        bendRadiusMm: 2.4,
        radiusMm: 0.78,
        arrows: 9,
        labelT: 0.42,
        colHexA: 0x0284c7,
        colHexB: 0x38f8ff,
        css: "#38bdf8",
        evalFlow: (out) => ({
          signedRate: out.feedwaterGS + out.returnWaterMlS,
          fill: out.tenderWaterMl > 0.3 ? 0.92 : 0.15,
          unit: "ml/s",
        }),
      },
      {
        id: "loco_6_8_13_2_riser",
        partNo: "4-17→6-8→13-2",
        title: "バイパス弁4-17→給水立管6-8→中位逆止弁(SB-ø3)",
        zone: "cab",
        frame: "loco",
        ptsMm: [[-135.5, -11.4, 15.0], [-131.5, -11.4, 15.0], [-115.8, -11.8, 16.2], [-115.8, -11.8, 47.5], [-109.2, -11.8, 47.5]],
        bendRadiusMm: 2.4,
        radiusMm: 0.78,
        arrows: 6,
        labelT: 0.62,
        colHexA: 0x0284c7,
        colHexB: 0x7ef9ff,
        css: "#38bdf8",
        evalFlow: (out) => ({
          signedRate: out.feedwaterGS,
          fill: out.feedwaterGS > 0.005 ? 0.94 : 0.30,
          unit: "g/s",
        }),
      },
      {
        id: "loco_4_16_byp_ret",
        partNo: "4-17→4-16",
        title: "バイパス弁4-17下口バンジョー→4-16戻し管(左+Y)",
        zone: "cab",
        frame: "loco",
        ptsMm: [[-135.5, 8.5, 15.0], [-135.5, 8.5, 11.2], [-138.2, 13.7, 11.2], [-141.5, 19.2, 6.2], [-144.5, 19.2, -8.5], [-150.8, 19.2, -8.5]],
        bendRadiusMm: 2.2,
        radiusMm: 0.78,
        arrows: 5,
        labelT: 0.50,
        colHexA: 0x0891b2,
        colHexB: 0x67e8f9,
        css: "#22d3ee",
        evalFlow: (out) => ({
          signedRate: out.returnWaterMlS,
          fill: out.returnWaterMlS > 0.005 ? 0.90 : 0.22,
          unit: "ml/s",
        }),
      },
      {
        id: "loco_4_2_lub_oil",
        partNo: "4-2→7-9",
        title: "ロスコ置換注油器 内立管(Z=+10.9)→-Z底Uループ→7-9給気管",
        zone: "underframe",
        frame: "loco",
        ptsMm: [[146.2, 0.0, 10.9], [146.2, 0.0, -0.6], [146.2, 0.0, -3.4], [142.0, 4.2, -3.4], [136.5, 7.5, 3.2], [123.5, 7.5, 5.2], [123.5, 7.5, 18.5]],
        bendRadiusMm: 2.0,
        radiusMm: 0.62,
        arrows: 6,
        labelT: 0.54,
        colHexA: 0x10b981,
        colHexB: 0xf59e0b,
        css: "#10b981",
        evalFlow: (out) => {
          const oilRate = out.lubricatorOilPct > 1.0 && (out.steamOutGS ?? out.steamGenGS) > 0.01 ? 0.08 + 0.35 * (out.steamOutGS ?? out.steamGenGS) : 0.0;
          return { signedRate: oilRate, fill: out.lubricatorOilPct > 1.0 ? 0.92 : 0.1, unit: "mg/m" };
        },
      },
      {
        id: "boil_flues_upper",
        partNo: "12-6(上2煙管)",
        title: "左右上段ø11.5過熱大煙管×2(Y=±11.5, X=-48..+92止) 高温燃焼ガス流",
        zone: "boiler",
        frame: "loco",
        ptsMm: [[-48.0, 11.5, 46.5], [22.0, 11.5, 46.5], [92.0, 11.5, 46.5]],
        extraBranchesMm: [
          [[-48.0, -11.5, 46.5], [22.0, -11.5, 46.5], [92.0, -11.5, 46.5]],
        ],
        radiusMm: 3.40,
        arrows: 10,
        labelT: 0.45,
        colHexA: 0xea580c,
        colHexB: 0xfef08a,
        css: "#fb923c",
        evalFlow: (out, sp) => ({
          signedRate: sp.flueGasProfileC[0] > 60.0 || out.smokeboxVacuumPa > 4.0 ? Math.max(0.05, out.smokeboxVacuumPa * 0.018) : 0.0,
          fill: sp.flueGasProfileC[0] > 60.0 ? 0.78 : 0.0,
          unit: "g/s",
        }),
      },
      {
        id: "boil_flues_lower",
        partNo: "12-6(下3火管)",
        title: "下段ø8.0火管×3(Y=-13/0/+13, X=-48..+92止) 燃焼ガス熱伝達流",
        zone: "boiler",
        frame: "loco",
        ptsMm: [[-48.0, 0.0, 33.2], [22.0, 0.0, 33.2], [92.0, 0.0, 33.2]],
        extraBranchesMm: [
          [[-48.0, +13.0, 35.0], [22.0, +13.0, 35.0], [92.0, +13.0, 35.0]],
          [[-48.0, -13.0, 35.0], [22.0, -13.0, 35.0], [92.0, -13.0, 35.0]],
        ],
        radiusMm: 1.95,
        arrows: 10,
        labelT: 0.55,
        colHexA: 0xdc2626,
        colHexB: 0xfdba74,
        css: "#f97316",
        evalFlow: (out, sp) => ({
          signedRate: sp.flueGasProfileC[0] > 60.0 || out.smokeboxVacuumPa > 4.0 ? Math.max(0.04, out.smokeboxVacuumPa * 0.015) : 0.0,
          fill: sp.flueGasProfileC[0] > 60.0 ? 0.76 : 0.0,
          unit: "g/s",
        }),
      },

      // === D. 锅炉—驾驶室跨界连续管网与左右双U四程过热器 ===
      {
        id: "boil_13_4_11_15_turret",
        partNo: "13-4→11-15",
        title: "缶頂主蒸気管→縦長蒸気分配器(18mm)",
        zone: "cab",
        frame: "loco",
        ptsMm: [[-98.5, 0.0, 70.5], [-98.5, 0.0, 72.4], [-118.5, 0.0, 72.4], [-118.5, 8.2, 72.4], [-118.5, 8.2, 66.5], [-130.0, 8.2, 66.5]],
        bendRadiusMm: 2.2,
        radiusMm: 0.90,
        arrows: 5,
        labelT: 0.48,
        colHexA: 0x93c5fd,
        colHexB: 0xffffff,
        css: "#bae6fd",
        evalFlow: (out) => ({
          signedRate: out.boilerPressureKgf > 0.03 ? Math.max(0.02, out.steamOutGS ?? out.steamGenGS) : 0.0,
          fill: out.boilerPressureKgf > 0.03 ? 0.92 : 0.15,
          unit: "g/s",
        }),
      },
      {
        id: "boil_13_9_drypipe",
        partNo: "11-10→13-9",
        title: "加減管→後妻板同軸貫通→缶内ドライパイプ",
        zone: "boiler",
        frame: "loco",
        ptsMm: [[-124.5, 8.2, 66.5], [-124.5, -3.6, 66.5], [-124.5, -3.6, 48.0], [-112.8, -3.6, 48.0], [-105.0, -3.6, 54.5], [-96.0, 0.0, 58.8], [0.0, 0.0, 58.8], [96.5, 0.0, 58.8], [108.5, 0.0, 58.8]],
        bendRadiusMm: 3.2,
        radiusMm: 1.02,
        arrows: 14,
        labelT: 0.22,
        colHexA: 0x7dd3fc,
        colHexB: 0xfef3c7,
        css: "#7dd3fc",
        evalFlow: (out) => {
          const regFlow = Math.max(0.0, (out.steamOutGS ?? out.steamGenGS) - (out.blowerGS ?? 0) - out.whistleSteamGS - (out.safetyGS ?? 0));
          return { signedRate: out.steamChestKgf > 0.02 ? Math.max(0.03, regFlow) : 0.0, fill: out.steamChestKgf > 0.02 ? 0.90 : 0.18, unit: "g/s" };
        },
      },
      {
        id: "boil_17_5_sh_4pass",
        partNo: "17-5(L)→17-4",
        title: "左上煙管(+Y=+11.5) 半円U過熱ループ→過熱蒸気降下管",
        zone: "boiler",
        frame: "loco",
        ptsMm: [[108.5, 0.0, 58.8], [108.5, 0.0, 48.35], [108.5, 11.5, 48.35], [-24.0, 11.5, 48.35], [-24.0, 11.5, 44.65], [108.5, 11.5, 44.65], [108.5, 0.0, 44.65], [108.5, 0.0, 26.5], [119.5, 0.0, 21.8], [123.5, 0.0, 18.5]],
        bendRadiusMm: 1.85,
        radiusMm: 0.88,
        arrows: 16,
        labelT: 0.82,
        colHexA: 0xfb923c,
        colHexB: 0xfef08a,
        css: "#fb923c",
        evalFlow: (out) => {
          const regFlow = 0.5 * Math.max(0.0, (out.steamOutGS ?? out.steamGenGS) - (out.blowerGS ?? 0) - out.whistleSteamGS - (out.safetyGS ?? 0));
          return { signedRate: out.steamChestKgf > 0.02 ? Math.max(0.025, regFlow) : 0.0, fill: out.steamChestKgf > 0.02 ? 0.94 : 0.18, unit: "g/s" };
        },
      },
      {
        id: "boil_17_5_sh_loop_R",
        partNo: "17-5(R)→17-4",
        title: "右上煙管(-Y=-11.5) 半円U過熱ループ→下クロスヘッダ合流",
        zone: "boiler",
        frame: "loco",
        ptsMm: [[108.5, 0.0, 48.35], [108.5, -11.5, 48.35], [-24.0, -11.5, 48.35], [-24.0, -11.5, 44.65], [108.5, -11.5, 44.65], [108.5, 0.0, 44.65], [108.5, 0.0, 26.5], [119.5, 0.0, 21.8], [123.5, 0.0, 18.5]],
        bendRadiusMm: 1.85,
        radiusMm: 0.88,
        arrows: 15,
        labelT: 0.35,
        colHexA: 0xfb923c,
        colHexB: 0xfef08a,
        css: "#fb923c",
        evalFlow: (out) => {
          const regFlow = 0.5 * Math.max(0.0, (out.steamOutGS ?? out.steamGenGS) - (out.blowerGS ?? 0) - out.whistleSteamGS - (out.safetyGS ?? 0));
          return { signedRate: out.steamChestKgf > 0.02 ? Math.max(0.025, regFlow) : 0.0, fill: out.steamChestKgf > 0.02 ? 0.94 : 0.18, unit: "g/s" };
        },
      },
      {
        id: "boil_13_8_17_3_blower",
        partNo: "13-8→17-3",
        title: "通風弁→炉内ステー管→17-3煙室J字180°通風ノズル(停車強制ドラフト)",
        zone: "boiler",
        frame: "loco",
        ptsMm: [[-128.0, 8.2, 66.0], [-128.0, 8.2, 54.5], [-120.5, 2.6, 48.0], [-112.8, 2.6, 48.0], [-105.0, 2.6, 49.5], [-44.0, 2.6, 49.5], [86.0, 4.2, 40.5], [124.2, 4.2, 40.5], [124.2, 4.2, 23.8], [130.0, 4.2, 23.8], [130.0, 4.2, 35.2]],
        bendRadiusMm: 2.8,
        radiusMm: 0.58,
        arrows: 14,
        labelT: 0.86,
        colHexA: 0x67e8f9,
        colHexB: 0xffffff,
        css: "#67e8f9",
        evalFlow: (out) => ({
          signedRate: out.blowerGS ?? 0.0,
          fill: (out.blowerGS ?? 0.0) > 0.002 ? 0.94 : 0.22,
          unit: "g/s",
        }),
      },
      {
        id: "boil_13_3_11_8_whistle",
        partNo: "11-1→13-3→11-8",
        title: "汽笛弁→左直立管→車底真鍮共鳴汽笛",
        zone: "cab",
        frame: "loco",
        ptsMm: [[-117.6, 15.2, 65.5], [-117.6, 20.2, 65.5], [-117.6, 20.2, 25.5], [-114.5, 20.2, 25.5], [-72.0, 20.2, 25.5]],
        bendRadiusMm: 2.0,
        radiusMm: 0.62,
        arrows: 7,
        labelT: 0.45,
        colHexA: 0xa5f3fc,
        colHexB: 0xffffff,
        css: "#a5f3fc",
        evalFlow: (out) => ({
          signedRate: out.whistleSteamGS,
          fill: out.whistleSteamGS > 0.002 ? 0.95 : 0.14,
          unit: "g/s",
        }),
      },
      {
        id: "boil_13_1_siphon",
        partNo: "13-1→PG-ø20",
        title: "180°U字凝結水液封サイフォン圧力管",
        zone: "cab",
        frame: "loco",
        ptsMm: [[-124.5, 0.8, 66.5], [-124.5, 0.8, 62.0], [-123.5, -6.5, 43.0], [-122.8, -11.8, 39.0], [-122.0, -16.5, 43.2], [-122.0, -16.5, 57.5]],
        bendRadiusMm: 2.4,
        radiusMm: 0.56,
        arrows: 5,
        labelT: 0.62,
        colHexA: 0x38bdf8,
        colHexB: 0xe0f2fe,
        css: "#7dd3fc",
        evalFlow: (out) => ({
          signedRate: out.boilerPressureKgf > 0.05 ? 0.03 * Math.min(1.0, out.boilerPressureKgf / 2.0) : 0.0,
          fill: 0.88,
          unit: "kgf",
        }),
      },
      {
        id: "boil_13_6_sg_steam",
        partNo: "13-6→11-7",
        title: "水面計上蒸気連通管 (缶頂ドーム↔ガラス管)",
        zone: "cab",
        frame: "loco",
        ptsMm: [[-106.8, -15.2, 58.3], [-106.8, -17.5, 60.5], [-118.5, -17.5, 60.5], [-118.5, -31.3, 60.5], [-118.5, -31.3, 55.0]],
        bendRadiusMm: 1.8,
        radiusMm: 0.68,
        arrows: 4,
        labelT: 0.55,
        colHexA: 0x93c5fd,
        colHexB: 0xffffff,
        css: "#bae6fd",
        evalFlow: (out) => ({
          signedRate: out.sightPurgeGS > 0.002 ? out.sightPurgeGS : out.boilerPressureKgf > 0.05 ? 0.025 : 0.0,
          fill: out.boilerPressureKgf > 0.03 ? 0.88 : 0.25,
          unit: "g/s",
        }),
      },
      {
        id: "boil_13_10_sg_water",
        partNo: "13-10↔11-7",
        title: "水面計100%水平下水連通管 (二階連通器双方向)",
        zone: "cab",
        frame: "loco",
        ptsMm: [[-43.2, -20.8, 32.6], [-82.0, -20.8, 32.6], [-118.5, -20.8, 32.6], [-118.5, -31.3, 32.6], [-118.5, -31.3, 38.0]],
        bendRadiusMm: 2.0,
        radiusMm: 0.72,
        arrows: 8,
        labelT: 0.68,
        colHexA: 0x0ea5e9,
        colHexB: 0x67e8f9,
        css: "#38bdf8",
        evalFlow: (out, sp) => {
          const vMmS = out.sightGlassVelMmS ?? 0.0;
          const purge = out.sightPurgeGS > 0.002 ? 0.45 + out.sightPurgeGS : 0.0;
          const signed = purge > 0 ? purge : Math.abs(vMmS) > 0.04 ? vMmS * 0.05 : sp.waterFreeZMm > 33.0 ? 0.02 * (out.feedwaterGS - out.steamGenGS) : 0.0;
          return {
            signedRate: signed,
            fill: sp.waterFreeZMm > 32.8 ? 0.92 : 0.08,
            unit: "mm/s",
          };
        },
      },
      {
        id: "boil_13_11_blowdown",
        partNo: "13-11→11-16",
        title: "缶底排泥管→左床下マイナス溝ブロー弁",
        zone: "underframe",
        frame: "loco",
        ptsMm: [[-42.5, 0.0, 17.2], [-42.5, 0.0, 15.6], [-54.0, 7.2, 15.4], [-72.0, 16.2, 15.2], [-108.0, 16.2, 15.5], [-115.5, 16.2, 15.8], [-115.5, 19.2, 12.0]],
        bendRadiusMm: 2.2,
        radiusMm: 0.70,
        arrows: 7,
        labelT: 0.72,
        colHexA: 0x0284c7,
        colHexB: 0xe0f2fe,
        css: "#38bdf8",
        evalFlow: (out) => ({
          signedRate: out.blowdownGS,
          fill: out.blowdownGS > 0.002 ? 0.94 : 0.25,
          unit: "g/s",
        }),
      },

      // === E. 左右对称双气缸给汽室、1-9 真三维交叉四气道(可逆)、左右双排汽支管与左右全 4 口排液阀 ===
      {
        id: "cyl_7_9_inlet_L",
        partNo: "7-9→1-20(L)",
        title: "左シリンダ蒸気室(+Y)・含油過熱給気横管&蒸気室ヘッダ",
        zone: "cylinder",
        frame: "loco",
        ptsMm: [[123.5, 0.0, 18.5], [123.5, 29.0, 18.5], [127.2, 36.0, 18.5], [134.2, 36.0, 16.6]],
        bendRadiusMm: 1.8,
        radiusMm: 1.15,
        arrows: 5,
        labelT: 0.52,
        colHexA: 0xf97316,
        colHexB: 0xfef08a,
        css: "#fb923c",
        evalFlow: (out) => ({
          signedRate: out.steamChestKgf > 0.03 ? 0.5 * Math.max(0.04, out.steamOutGS ?? out.steamGenGS) : 0.0,
          fill: out.steamChestKgf > 0.03 ? 0.92 : 0.16,
          unit: "g/s",
        }),
      },
      {
        id: "cyl_7_9_inlet_R",
        partNo: "7-9→1-20(R)",
        title: "右シリンダ蒸気室(-Y)・含油過熱給気横管&蒸気室ヘッダ",
        zone: "cylinder",
        frame: "loco",
        ptsMm: [[123.5, 0.0, 18.5], [123.5, -29.0, 18.5], [127.2, -36.0, 18.5], [134.2, -36.0, 16.6]],
        bendRadiusMm: 1.8,
        radiusMm: 1.15,
        arrows: 5,
        labelT: 0.52,
        colHexA: 0xf97316,
        colHexB: 0xfef08a,
        css: "#fb923c",
        evalFlow: (out) => ({
          signedRate: out.steamChestKgf > 0.03 ? 0.5 * Math.max(0.04, out.steamOutGS ?? out.steamGenGS) : 0.0,
          fill: out.steamChestKgf > 0.03 ? 0.92 : 0.16,
          unit: "g/s",
        }),
      },
      {
        id: "cyl_port_L_head",
        partNo: "1-9(左交叉・後口→前室)",
        title: "左1-9交叉ポート(後弁口→前気室Head・給排気可逆)",
        zone: "cylinder",
        frame: "loco",
        ptsMm: [[129.2, 36.0, 10.6], [129.2, 38.1, 7.8], [129.2, 40.3, 5.4], [143.2, 40.3, 5.4], [143.2, 32.5, 5.2]],
        bendRadiusMm: 1.4,
        radiusMm: 0.66,
        arrows: 4,
        labelT: 0.55,
        colHexA: 0xf97316,
        colHexB: 0xfef08a,
        css: "#f97316",
        evalFlow: (out, sp) => {
          const pv = sp.valvePortOpenL[0] ?? 0;
          const sweepDrive = Math.min(0.85, Math.abs(out.wheelRpm) * 0.012) + (sp.cylLPKgf[0] ?? 0.0) * 0.65;
          const rate = pv >= 0.02
            ? (out.steamChestKgf > 0.015 ? pv * (0.35 + 0.45 * Math.min(1.0, out.steamChestKgf)) : 0.0)
            : pv <= -0.02
              ? pv * Math.max(0.18, sweepDrive)
              : 0.0;
          return {
            signedRate: rate,
            fill: Math.abs(rate) > 0.01 ? 0.95 : 0.22,
            unit: pv >= 0 ? "給気" : "排気",
          };
        },
      },
      {
        id: "cyl_port_L_crank",
        partNo: "1-9(左交叉・前口→後室)",
        title: "左1-9交叉ポート(前弁口→後気室Crank・給排気可逆)",
        zone: "cylinder",
        frame: "loco",
        ptsMm: [[137.2, 36.0, 10.6], [137.2, 40.9, 9.4], [125.2, 40.9, 9.4], [125.2, 38.7, 7.8], [125.0, 32.5, 5.2]],
        bendRadiusMm: 1.4,
        radiusMm: 0.66,
        arrows: 4,
        labelT: 0.55,
        colHexA: 0xf97316,
        colHexB: 0xfef08a,
        css: "#f97316",
        evalFlow: (out, sp) => {
          const pv = sp.valvePortOpenL[1] ?? 0;
          const sweepDrive = Math.min(0.85, Math.abs(out.wheelRpm) * 0.012) + (sp.cylLPKgf[1] ?? 0.0) * 0.65;
          const rate = pv >= 0.02
            ? (out.steamChestKgf > 0.015 ? pv * (0.35 + 0.45 * Math.min(1.0, out.steamChestKgf)) : 0.0)
            : pv <= -0.02
              ? pv * Math.max(0.18, sweepDrive)
              : 0.0;
          return {
            signedRate: rate,
            fill: Math.abs(rate) > 0.01 ? 0.95 : 0.22,
            unit: pv >= 0 ? "給気" : "排気",
          };
        },
      },
      {
        id: "cyl_port_R_head",
        partNo: "1-9(右交叉・後口→前室)",
        title: "右1-9交叉ポート(後弁口→前気室Head・給排気可逆)",
        zone: "cylinder",
        frame: "loco",
        ptsMm: [[129.2, -36.0, 10.6], [129.2, -38.1, 7.8], [129.2, -40.3, 5.4], [143.2, -40.3, 5.4], [143.2, -32.5, 5.2]],
        bendRadiusMm: 1.4,
        radiusMm: 0.66,
        arrows: 4,
        labelT: 0.55,
        colHexA: 0xf97316,
        colHexB: 0xfef08a,
        css: "#f97316",
        evalFlow: (out, sp) => {
          const pv = sp.valvePortOpenR[0] ?? 0;
          const sweepDrive = Math.min(0.85, Math.abs(out.wheelRpm) * 0.012) + (sp.cylRPKgf[0] ?? 0.0) * 0.65;
          const rate = pv >= 0.02
            ? (out.steamChestKgf > 0.015 ? pv * (0.35 + 0.45 * Math.min(1.0, out.steamChestKgf)) : 0.0)
            : pv <= -0.02
              ? pv * Math.max(0.18, sweepDrive)
              : 0.0;
          return {
            signedRate: rate,
            fill: Math.abs(rate) > 0.01 ? 0.95 : 0.22,
            unit: pv >= 0 ? "給気" : "排気",
          };
        },
      },
      {
        id: "cyl_port_R_crank",
        partNo: "1-9(右交叉・前口→後室)",
        title: "右1-9交叉ポート(前弁口→後気室Crank・給排気可逆)",
        zone: "cylinder",
        frame: "loco",
        ptsMm: [[137.2, -36.0, 10.6], [137.2, -40.9, 9.4], [125.2, -40.9, 9.4], [125.2, -38.7, 7.8], [125.0, -32.5, 5.2]],
        bendRadiusMm: 1.4,
        radiusMm: 0.66,
        arrows: 4,
        labelT: 0.55,
        colHexA: 0xf97316,
        colHexB: 0xfef08a,
        css: "#f97316",
        evalFlow: (out, sp) => {
          const pv = sp.valvePortOpenR[1] ?? 0;
          const sweepDrive = Math.min(0.85, Math.abs(out.wheelRpm) * 0.012) + (sp.cylRPKgf[1] ?? 0.0) * 0.65;
          const rate = pv >= 0.02
            ? (out.steamChestKgf > 0.015 ? pv * (0.35 + 0.45 * Math.min(1.0, out.steamChestKgf)) : 0.0)
            : pv <= -0.02
              ? pv * Math.max(0.18, sweepDrive)
              : 0.0;
          return {
            signedRate: rate,
            fill: Math.abs(rate) > 0.01 ? 0.95 : 0.22,
            unit: pv >= 0 ? "給気" : "排気",
          };
        },
      },
      {
        id: "cyl_7_8_20_7_blast",
        partNo: "1-9(L)→7-8→20-7",
        title: "左シリンダ中央排気腔(+Y)→7-8合流管→20-7ブラストノズル(ø4.2)",
        zone: "cylinder",
        frame: "loco",
        ptsMm: [[133.2, 36.0, 10.2], [133.2, 36.0, 9.15], [130.0, 28.0, 9.15], [130.0, 25.3, 12.5], [130.0, 0.0, 12.5], [130.0, 0.0, 35.0]],
        bendRadiusMm: 1.8,
        radiusMm: 1.08,
        arrows: 6,
        labelT: 0.76,
        colHexA: 0xeab308,
        colHexB: 0xfde047,
        css: "#eab308",
        evalFlow: (out) => ({
          signedRate: out.exhaustGS ?? out.exhaustPlume * 0.35,
          fill: (out.exhaustGS ?? 0) > 0.005 || out.exhaustPlume > 0.04 ? 0.92 : 0.18,
          unit: "g/s",
        }),
      },
      {
        id: "cyl_7_8_exhaust_R",
        partNo: "1-9(R)→7-8→20-7",
        title: "右シリンダ中央排気腔(-Y)→内壁ソケット→7-8排気合流管→20-7",
        zone: "cylinder",
        frame: "loco",
        ptsMm: [[133.2, -36.0, 10.2], [133.2, -36.0, 9.15], [130.0, -28.0, 9.15], [130.0, -25.3, 12.5], [130.0, 0.0, 12.5], [130.0, 0.0, 35.0]],
        bendRadiusMm: 1.8,
        radiusMm: 1.08,
        arrows: 6,
        labelT: 0.42,
        colHexA: 0xeab308,
        colHexB: 0xfde047,
        css: "#eab308",
        evalFlow: (out) => ({
          signedRate: out.exhaustGS ?? out.exhaustPlume * 0.35,
          fill: (out.exhaustGS ?? 0) > 0.005 || out.exhaustPlume > 0.04 ? 0.92 : 0.18,
          unit: "g/s",
        }),
      },
      {
        id: "cyl_1_16_drains",
        partNo: "1-16×2(R)",
        title: "右シリンダ前後底ばねポペット二相排液弁(-Y)",
        zone: "cylinder",
        frame: "loco",
        ptsMm: [[133.8, -30.8, -6.2], [133.8, -30.8, -16.8]],
        extraBranchesMm: [
          [[123.8, -30.8, -6.2], [123.8, -30.8, -16.8]],
        ],
        radiusMm: 0.62,
        arrows: 3,
        labelT: 0.60,
        colHexA: 0x06b6d4,
        colHexB: 0xe0f2fe,
        css: "#22d3ee",
        evalFlow: (out) => ({
          signedRate: out.drainBlowdownGS ?? (out.drainPlume * 0.18),
          fill: (out.drainBlowdownGS ?? 0) > 0.002 || out.drainPlume > 0.02 ? 0.95 : 0.14,
          unit: "g/s",
        }),
      },
      {
        id: "cyl_1_16_drains_L",
        partNo: "1-16×2(L)",
        title: "左シリンダ前後底ばねポペット二相排液弁(+Y)",
        zone: "cylinder",
        frame: "loco",
        ptsMm: [[133.8, 30.8, -6.2], [133.8, 30.8, -16.8]],
        extraBranchesMm: [
          [[123.8, 30.8, -6.2], [123.8, 30.8, -16.8]],
        ],
        radiusMm: 0.62,
        arrows: 3,
        labelT: 0.60,
        colHexA: 0x06b6d4,
        colHexB: 0xe0f2fe,
        css: "#22d3ee",
        evalFlow: (out) => ({
          signedRate: out.drainBlowdownGS ?? (out.drainPlume * 0.18),
          fill: (out.drainBlowdownGS ?? 0) > 0.002 || out.drainPlume > 0.02 ? 0.95 : 0.14,
          unit: "g/s",
        }),
      },
    ];

    let nLocoArrows = 0; // Gate 25.1 38-Pipe LVS 切片锚点 (严禁删除或移位!)
    let nLocoLiquidArrows = 0;
    let nTdrArrows = 0;
    let nWorldArrows = 0;
    let nOpenGasArrows = 0;

    type OpenStreamSpec = {
      id: string;
      partNo: string;
      title: string;
      zone: InternalPipeRuntime["zone"];
      frame: "loco" | "tender";
      medium: "open_gas" | "open_liquid";
      ptsMm: [number, number, number][];
      extraBranchesMm?: [number, number, number][][];
      bendRadiusMm?: number;
      markerScaleMm: number;
      labelT: number;
      evalFlow: InternalPipeRuntime["evalFlow"];
    };

    // 管外开放腔室 & 车外大气交互自由气/液流迹 (严禁套用 buildBishopPipeGeometry 圆管网格! 统一采用无管壁行进虚线 + 双V开放风羽» / 自由落水滴💧)
    const gasStreamSpecs: OpenStreamSpec[] = [
      {
        id: "gas_grate_to_firebox",
        partNo: "外気→23-1炉床→火室",
        title: "車外大気→炉床一次空気吸込→3芯横流湾曲火炎→後管板5煙管入口(無管壁自由気流)",
        zone: "boiler",
        frame: "loco",
        medium: "open_gas",
        ptsMm: [[-98.0, 0.0, 10.0], [-92.0, 0.0, 21.5], [-74.0, 0.0, 35.5], [-49.5, 0.0, 33.2]],
        extraBranchesMm: [
          [[-86.0, +9.5, 10.0], [-76.0, +10.5, 22.0], [-60.0, +11.5, 39.5], [-49.5, +11.5, 46.5]],
          [[-86.0, -9.5, 10.0], [-76.0, -10.5, 22.0], [-60.0, -11.5, 39.5], [-49.5, -11.5, 46.5]],
          [[-68.0, +11.0, 10.0], [-60.0, +12.5, 22.5], [-52.0, +13.0, 33.5], [-49.5, +13.0, 35.0]],
          [[-68.0, -11.0, 10.0], [-60.0, -12.5, 22.5], [-52.0, -13.0, 33.5], [-49.5, -13.0, 35.0]],
        ],
        bendRadiusMm: 5.0,
        markerScaleMm: 1.05,
        labelT: 0.42,
        evalFlow: (out, sp) => {
          const back = out.backdraftAlarm ?? sp.backdraftAlarm ?? false;
          const gr = out.grateAirGS ?? sp.grateAirGS ?? (out.smokeboxVacuumPa > 2.0 ? 0.12 + out.smokeboxVacuumPa * 0.011 : 0.08);
          const lit = sp.flueGasProfileC[0] > 70.0;
          const rate = back ? -0.32 : (lit || out.smokeboxVacuumPa > 2.0 ? gr : 0.0);
          return {
            signedRate: rate,
            fill: Math.abs(rate) > 0.01 ? 0.85 : 0.0,
            unit: back ? "逆火g/s" : "g/s",
          };
        },
      },
      {
        id: "gas_12_1_firehole",
        partNo: "12-1焚口二次風",
        title: "運転室外気↔12-1バタフライ焚口戸 二次空気吸込/逆火吹出(無管壁自由気流)",
        zone: "cab",
        frame: "loco",
        medium: "open_gas",
        ptsMm: [[-121.0, 0.0, 28.8], [-107.5, 0.0, 28.8], [-84.0, 0.0, 38.0], [-49.5, 0.0, 33.2]],
        bendRadiusMm: 4.5,
        markerScaleMm: 0.96,
        labelT: 0.24,
        evalFlow: (out, sp) => {
          const back = out.backdraftAlarm ?? sp.backdraftAlarm ?? false;
          const fh = out.fireholeAirGS ?? sp.fireholeAirGS ?? (out.smokeboxVacuumPa > 2.0 ? 0.035 : 0.0);
          const rate = back ? -0.38 : fh;
          return {
            signedRate: rate,
            fill: Math.abs(rate) > 0.008 ? 0.85 : 0.0,
            unit: back ? "逆火g/s" : "g/s",
          };
        },
      },
      {
        id: "gas_smokebox_to_chimney",
        partNo: "煙室空間→18-28→18-25煙突",
        title: "前管板(X=+92止)出口排ガス→煙室開放空間負圧吸引→18-28ペチコート単一中心合流→18-25煙突排出(無管壁自由気流)",
        zone: "boiler",
        frame: "loco",
        medium: "open_gas",
        // 主流线: 下中央火管出口前方自由空间(X=+95.5) -> 18-28 集烟罩喉口(130,0,49) -> 18-25 烟囱中心轴线(Z=+86.5 < 91.5mm 烟囱口!)
        ptsMm: [[95.5, 0.0, 33.2], [116.0, 0.0, 37.5], [129.0, 0.0, 48.5], [130.0, 0.0, 52.0], [130.0, 0.0, 86.5]],
        // 其余 4 股前管板出口自由烟气在烟室开放腔内汇聚至 18-28 喇叭口(130, 0, 49.5)，绝不向烟囱内伸入任何平行假管！
        extraBranchesMm: [
          [[95.5, +11.5, 46.5], [116.5, +9.4, 47.2], [128.0, +2.6, 48.8], [130.0, 0.0, 49.8]],
          [[95.5, -11.5, 46.5], [116.5, -9.4, 47.2], [128.0, -2.6, 48.8], [130.0, 0.0, 49.8]],
          [[95.5, +13.0, 35.0], [116.0, +9.8, 39.8], [128.0, +2.6, 47.8], [130.0, 0.0, 49.2]],
          [[95.5, -13.0, 35.0], [116.0, -9.8, 39.8], [128.0, -2.6, 47.8], [130.0, 0.0, 49.2]],
        ],
        bendRadiusMm: 5.5,
        markerScaleMm: 1.08,
        labelT: 0.72,
        evalFlow: (out, sp) => {
          const fg = out.flueGasGS ?? sp.flueGasGS ?? (out.smokeboxVacuumPa > 2.0 ? 0.14 + out.smokeboxVacuumPa * 0.012 : 0.08);
          const tot = (sp.flueGasProfileC[0] > 60.0 || out.smokeboxVacuumPa > 3.0) ? fg + (out.exhaustGS ?? 0) + (out.blowerGS ?? 0) : 0.0;
          return {
            signedRate: tot,
            fill: tot > 0.02 ? 0.88 : 0.0,
            unit: "g/s",
          };
        },
      },
      {
        id: "gas_blast_to_petticoat",
        partNo: "20-7/17-3噴口→18-28吸引射流",
        title: "20-7ブラスト&17-3通風ノズル噴出→煙室自由空間ベンチュリ高速蒸気射流",
        zone: "boiler",
        frame: "loco",
        medium: "open_gas",
        ptsMm: [[130.0, 0.0, 35.5], [130.0, 0.0, 49.0]],
        extraBranchesMm: [
          [[130.0, +4.2, 35.5], [130.0, +2.2, 43.0], [130.0, +0.6, 49.0]],
        ],
        bendRadiusMm: 3.0,
        markerScaleMm: 0.92,
        labelT: 0.50,
        evalFlow: (out) => {
          const jet = (out.exhaustGS ?? out.exhaustPlume * 0.35) + (out.blowerGS ?? 0.0);
          return {
            signedRate: jet,
            fill: jet > 0.004 ? 0.90 : 0.0,
            unit: "g/s",
          };
        },
      },
      {
        id: "gas_ext_28_6_safety",
        partNo: "28-6×2安全弁→車外放気",
        title: "28-6複式安全弁開弁→車外大気への高温蒸気吹出流",
        zone: "boiler",
        frame: "loco",
        medium: "open_gas",
        ptsMm: [[-58.0, 0.0, 73.5], [-59.0, 0.0, 86.0], [-61.0, 0.0, 98.0]],
        extraBranchesMm: [
          [[-70.0, 0.0, 73.5], [-71.0, 0.0, 86.0], [-73.0, 0.0, 98.0]],
        ],
        bendRadiusMm: 3.5,
        markerScaleMm: 1.05,
        labelT: 0.55,
        evalFlow: (out) => ({
          signedRate: out.safetyGS ?? 0.0,
          fill: (out.safetyGS ?? 0.0) > 0.003 ? 0.92 : 0.0,
          unit: "g/s",
        }),
      },
      {
        id: "gas_ext_11_8_whistle",
        partNo: "11-8汽笛窓→車底放気",
        title: "11-8車底真鍮共鳴汽笛(-Z下向き窓)→車外大気への音響蒸気噴流",
        zone: "underframe",
        frame: "loco",
        medium: "open_gas",
        ptsMm: [[-72.0, +20.2, 24.2], [-73.5, +22.5, 14.5], [-75.5, +25.0, 5.5]],
        bendRadiusMm: 3.0,
        markerScaleMm: 0.95,
        labelT: 0.55,
        evalFlow: (out) => ({
          signedRate: out.whistleSteamGS,
          fill: out.whistleSteamGS > 0.002 ? 0.92 : 0.0,
          unit: "g/s",
        }),
      },
      // === 管外自由液体流向 (落水/加水/放水/排液: 无管壁珠链虚线 + 圆润水滴珠💧) ===
      {
        id: "liq_ext_31_14_return",
        partNo: "31-14吐出口→水槽面自由落水",
        title: "31-14グースネック先端→炭水車気相空間を自由落下する戻水液流(管外液体)",
        zone: "tender",
        frame: "tender",
        medium: "open_liquid",
        ptsMm: [[-26.0, 11.5, 53.2], [-28.2, 9.8, 41.5], [-29.5, 8.8, 28.0]],
        bendRadiusMm: 3.0,
        markerScaleMm: 0.92,
        labelT: 0.50,
        evalFlow: (out) => ({
          signedRate: out.returnWaterMlS,
          fill: out.returnWaterMlS > 0.005 ? 0.92 : 0.0,
          unit: "ml/s",
        }),
      },
      {
        id: "liq_ext_35_1_fill",
        partNo: "35-1注水口→後部開放槽落水",
        title: "外部補給水→35-1後部開放給水槽への自由落下水流(管外液体)",
        zone: "tender",
        frame: "tender",
        medium: "open_liquid",
        ptsMm: [[-84.0, 0.0, 81.5], [-84.0, 0.0, 58.0], [-84.0, 0.0, 34.0]],
        bendRadiusMm: 2.0,
        markerScaleMm: 1.05,
        labelT: 0.45,
        evalFlow: (out) => ({
          signedRate: out.waterFillMlS ?? 0.0,
          fill: (out.waterFillMlS ?? 0.0) > 0.02 ? 0.95 : 0.0,
          unit: "ml/s",
        }),
      },
      {
        id: "liq_ext_31_9_drain",
        partNo: "31-9放水口→車外排水",
        title: "31-11スタンドパイプ開栓→31-9底穴から軌道下への自由放水流(管外液体)",
        zone: "tender",
        frame: "tender",
        medium: "open_liquid",
        ptsMm: [[-41.0, -10.3, -4.2], [-41.0, -10.3, -14.0], [-41.0, -10.3, -24.0]],
        bendRadiusMm: 2.0,
        markerScaleMm: 0.98,
        labelT: 0.55,
        evalFlow: (out) => ({
          signedRate: out.tenderDrainMlS ?? 0.0,
          fill: (out.tenderDrainMlS ?? 0.0) > 0.01 ? 0.95 : 0.0,
          unit: "ml/s",
        }),
      },
      {
        id: "liq_ext_11_16_blowdown",
        partNo: "11-16排泥口→車外噴出",
        title: "11-16ブロー弁開弁→車外軌道下への高温缶水・フラッシュ放流(管外液体)",
        zone: "underframe",
        frame: "loco",
        medium: "open_liquid",
        ptsMm: [[-115.5, 19.2, 11.6], [-116.5, 21.0, 1.5], [-117.8, 23.0, -9.0]],
        bendRadiusMm: 2.5,
        markerScaleMm: 0.90,
        labelT: 0.55,
        evalFlow: (out) => ({
          signedRate: out.blowdownGS,
          fill: out.blowdownGS > 0.002 ? 0.95 : 0.0,
          unit: "g/s",
        }),
      },
      {
        id: "liq_ext_1_16_drains",
        partNo: "1-16×4排液口→車外排出",
        title: "左右シリンダ底1-16×4ドレンコック→車外軌道面への凝縮水噴出(管外液体)",
        zone: "cylinder",
        frame: "loco",
        medium: "open_liquid",
        ptsMm: [[133.8, -30.8, -17.0], [133.8, -31.6, -28.5]],
        extraBranchesMm: [
          [[123.8, -30.8, -17.0], [123.8, -31.6, -28.5]],
          [[133.8, +30.8, -17.0], [133.8, +31.6, -28.5]],
          [[123.8, +30.8, -17.0], [123.8, +31.6, -28.5]],
        ],
        bendRadiusMm: 2.0,
        markerScaleMm: 0.78,
        labelT: 0.55,
        evalFlow: (out) => ({
          signedRate: out.drainBlowdownGS ?? (out.drainPlume * 0.18),
          fill: (out.drainBlowdownGS ?? 0) > 0.002 || out.drainPlume > 0.02 ? 0.94 : 0.0,
          unit: "g/s",
        }),
      },
    ];

    const classifyPipeMediumKind = (id: string): PipeMediumKind => {
      if (id === "boil_flues_upper" || id === "boil_flues_lower") return 2;
      if (
        id.startsWith("tdr_") ||
        id.startsWith("umb_") ||
        id === "loco_23_1_burner" ||
        id === "loco_6_10_suction" ||
        id === "loco_6_5_6_6_deliv" ||
        id === "loco_6_8_13_2_riser" ||
        id === "loco_4_16_byp_ret" ||
        id === "loco_4_2_lub_oil" ||
        id === "boil_13_1_siphon" ||
        id === "boil_13_10_sg_water" ||
        id === "boil_13_11_blowdown" ||
        id === "cyl_1_16_drains" ||
        id === "cyl_1_16_drains_L"
      ) {
        return 0;
      }
      return 1;
    };

    // 全车耦合节点—沿程连续温度场评估器: 共享物理接头的上下游管道在端点处严格保持 C0 温度连续，消除跳变断层！
    const makePipeTempProfileEvaluator = (id: string): InternalPipeRuntime["evalTempProfileC"] => {
      return (out, sp) => {
        const tBoil = out.boilerTempC;
        const tTdr = out.tenderWaterTempC;
        const tAlc = out.alcoholTempC ?? 28.0;
        const tWick = Math.max(tAlc + 6.0, Math.min(94.0, out.wickTipTempC ?? 78.0));
        const tPump = tTdr + 0.05 * Math.max(0.0, tBoil - tTdr);
        const tByp = tTdr + 0.12 * Math.max(0.0, tBoil - tTdr);
        const tRetUmb = tTdr + 0.08 * Math.max(0.0, tBoil - tTdr);
        const tRetEnd = tTdr + 0.04 * Math.max(0.0, tBoil - tTdr);
        const tClack = out.feedwaterGS > 0.01 ? 0.18 * tByp + 0.82 * tBoil : 0.08 * tByp + 0.92 * tBoil;
        const tTurret = Math.max(25.0, tBoil - 1.2);
        const tSh = Math.max(tBoil, out.superheatTempC);
        const tChest = Math.max(25.0, tSh - 1.5);
        const tOil = out.lubeOilTempC ?? Math.max(30.0, 0.62 * out.cylWallTempC);
        const tExh = out.boilerPressureKgf > 0.05 ? Math.max(98.0, Math.min(118.0, 0.78 * out.cylWallTempC + 10.0)) : out.cylWallTempC;
        const tSiphEnd = 25.0 + 0.14 * Math.max(0.0, tBoil - 25.0);
        const tSgBot = 25.0 + 0.72 * Math.max(0.0, tBoil - 25.0);

        switch (id) {
          case "tdr_36_6_mariotte":
            return [tAlc, tAlc, tAlc + 0.5, tAlc + 1.0];
          case "tdr_34_2_fuel":
            return [tAlc + 1.0, tAlc + 1.2, tAlc + 1.6, tAlc + 2.0];
          case "umb_st7_fuel":
            return [tAlc + 2.0, tAlc + 2.5, tAlc + 3.2, tAlc + 4.0];
          case "loco_23_1_burner":
            return [tAlc + 4.0, tAlc + 0.25 * (tWick - tAlc), tAlc + 0.62 * (tWick - tAlc), tWick];
          case "tdr_31_5_31_13_hp":
          case "tdr_31_9_drain":
            return [tTdr, tTdr, tTdr + 0.3, tTdr + 0.6];
          case "tdr_34_6_feed_R":
            return [tTdr + 0.6, tTdr + 0.8, tTdr + 1.0, tTdr + 1.2];
          case "umb_bt5_feed":
            return [tTdr + 1.2, tTdr + 1.5, tTdr + 1.8, tTdr + 2.2];
          case "loco_6_10_suction":
            return [tTdr + 2.2, 0.70 * tTdr + 0.30 * tPump, 0.35 * tTdr + 0.65 * tPump, tPump];
          case "loco_6_5_6_6_deliv":
            return [tPump, 0.72 * tPump + 0.28 * tByp, 0.35 * tPump + 0.65 * tByp, tByp];
          case "loco_6_8_13_2_riser":
            return [tByp, 0.72 * tByp + 0.28 * tClack, 0.34 * tByp + 0.66 * tClack, tClack];
          case "loco_4_16_byp_ret":
            return [tByp, 0.68 * tByp + 0.32 * tRetUmb, 0.34 * tByp + 0.66 * tRetUmb, tRetUmb];
          case "umb_bt5_return":
            return [tRetUmb, 0.66 * tRetUmb + 0.34 * tRetEnd, 0.33 * tRetUmb + 0.67 * tRetEnd, tRetEnd];
          case "tdr_34_6_ret_L":
            return [tRetEnd, 0.75 * tRetEnd + 0.25 * tTdr, 0.50 * tRetEnd + 0.50 * tTdr, 0.35 * tRetEnd + 0.65 * tTdr];
          case "tdr_31_14_gooseneck":
            return [0.35 * tRetEnd + 0.65 * tTdr, 0.25 * tRetEnd + 0.75 * tTdr, 0.15 * tRetEnd + 0.85 * tTdr, tTdr + 1.0];
          case "loco_4_2_lub_oil":
            return [tOil, tOil + 0.15 * (tSh - tOil), tOil + 0.52 * (tSh - tOil), tSh];
          case "boil_flues_upper":
          case "boil_flues_lower": {
            const g0 = Math.max(tBoil, sp.flueGasProfileC[0] ?? tBoil);
            const g1 = Math.max(tBoil, sp.flueGasProfileC[2] ?? tBoil);
            const g2 = Math.max(tBoil, sp.flueGasProfileC[5] ?? tBoil);
            const g3 = Math.max(tBoil, sp.flueGasProfileC[7] ?? tBoil);
            return [g0, g1, g2, g3];
          }
          case "gas_grate_to_firebox": {
            const g0 = Math.max(tBoil, sp.flueGasProfileC[0] ?? tBoil);
            const lit = sp.flueGasProfileC[0] > 70.0;
            return lit ? [26.0, 265.0, 0.58 * g0 + 185.0, g0] : [25.0, 26.0, 0.5 * (26.0 + tBoil), tBoil];
          }
          case "gas_12_1_firehole": {
            const g0 = Math.max(tBoil, sp.flueGasProfileC[0] ?? tBoil);
            const lit = sp.flueGasProfileC[0] > 70.0;
            const back = out.backdraftAlarm ?? sp.backdraftAlarm ?? false;
            return back ? [460.0, 560.0, 640.0, g0] : lit ? [27.0, 145.0, 0.52 * g0 + 140.0, g0] : [25.0, 26.0, 32.0, tBoil];
          }
          case "gas_smokebox_to_chimney": {
            const g3 = Math.max(tBoil, sp.flueGasProfileC[7] ?? out.smokeboxGasTempC ?? tBoil);
            const tStack = out.stackExitTempC ?? sp.stackExitTempC ?? Math.max(85.0, 0.72 * g3 + 0.28 * 105.0);
            return [g3, 0.65 * g3 + 0.35 * tStack, tStack, Math.max(65.0, tStack - 16.0)];
          }
          case "gas_blast_to_petticoat":
            return [tExh + 6.0, tExh + 2.0, tExh - 3.0, Math.max(92.0, tExh - 8.0)];
          case "gas_ext_28_6_safety":
            return [tBoil, tBoil - 14.0, Math.max(88.0, tBoil - 34.0), Math.max(68.0, tBoil - 56.0)];
          case "gas_ext_11_8_whistle":
            return [Math.max(96.0, tTurret - 14.0), 86.0, 68.0, 52.0];
          case "liq_ext_31_14_return":
            return [tTdr + 1.5, tTdr + 1.0, tTdr + 0.5, tTdr];
          case "liq_ext_35_1_fill":
            return [22.0, 22.0, 22.5, 23.0];
          case "liq_ext_31_9_drain":
            return [tTdr, tTdr, Math.max(20.0, tTdr - 0.5), Math.max(20.0, tTdr - 1.0)];
          case "liq_ext_11_16_blowdown":
            return [Math.min(tBoil, 102.0), 92.0, 78.0, 64.0];
          case "liq_ext_1_16_drains": {
            const tDrn = Math.max(28.0, Math.min(100.0, 0.78 * out.cylWallTempC));
            return [tDrn, Math.max(25.0, tDrn - 8.0), Math.max(24.0, tDrn - 16.0), Math.max(22.0, tDrn - 22.0)];
          }
          case "boil_13_4_11_15_turret":
            return [tBoil, tBoil - 0.4, tBoil - 0.8, tTurret];
          case "boil_13_9_drypipe":
            return [tTurret, tBoil - 0.6, tBoil, tBoil];
          case "boil_17_5_sh_4pass":
          case "boil_17_5_sh_loop_R":
            return [tBoil, tBoil + 0.36 * (tSh - tBoil), tBoil + 0.74 * (tSh - tBoil), tSh];
          case "boil_13_8_17_3_blower":
            return [tTurret, tBoil, tBoil + 0.25 * (tSh - tBoil), Math.max(98.0, tBoil - 8.0)];
          case "boil_13_3_11_8_whistle":
            return [tTurret, tTurret - 3.5, tTurret - 8.0, Math.max(96.0, tTurret - 14.0)];
          case "boil_13_1_siphon":
            return [tTurret, 0.62 * tTurret + 0.38 * tSiphEnd, 0.22 * tTurret + 0.78 * tSiphEnd, tSiphEnd];
          case "boil_13_6_sg_steam":
            return [tBoil, tBoil - 1.5, tBoil - 3.5, tBoil - 5.5];
          case "boil_13_10_sg_water":
            return [tBoil - 1.5, 0.68 * tBoil + 0.32 * tSgBot, 0.32 * tBoil + 0.68 * tSgBot, tSgBot];
          case "boil_13_11_blowdown":
            return [tBoil - 1.5, tBoil - 4.0, 0.65 * tBoil + 32.0, Math.min(tBoil, 100.0)];
          case "cyl_7_9_inlet_L":
          case "cyl_7_9_inlet_R":
            return [tSh, tSh - 0.5, tChest, tChest];
          case "cyl_port_L_head": {
            const adm = (sp.valvePortOpenL[0] ?? 0) >= 0;
            const tIn = adm ? tChest : tExh;
            const tOut = adm ? Math.max(tExh, 0.72 * tChest + 0.28 * out.cylWallTempC) : Math.max(92.0, 0.85 * tExh + 0.15 * out.cylWallTempC);
            return [tIn, 0.68 * tIn + 0.32 * tOut, 0.34 * tIn + 0.66 * tOut, tOut];
          }
          case "cyl_port_L_crank": {
            const adm = (sp.valvePortOpenL[1] ?? 0) >= 0;
            const tIn = adm ? tChest : tExh;
            const tOut = adm ? Math.max(tExh, 0.72 * tChest + 0.28 * out.cylWallTempC) : Math.max(92.0, 0.85 * tExh + 0.15 * out.cylWallTempC);
            return [tIn, 0.68 * tIn + 0.32 * tOut, 0.34 * tIn + 0.66 * tOut, tOut];
          }
          case "cyl_port_R_head": {
            const adm = (sp.valvePortOpenR[0] ?? 0) >= 0;
            const tIn = adm ? tChest : tExh;
            const tOut = adm ? Math.max(tExh, 0.72 * tChest + 0.28 * out.cylWallTempC) : Math.max(92.0, 0.85 * tExh + 0.15 * out.cylWallTempC);
            return [tIn, 0.68 * tIn + 0.32 * tOut, 0.34 * tIn + 0.66 * tOut, tOut];
          }
          case "cyl_port_R_crank": {
            const adm = (sp.valvePortOpenR[1] ?? 0) >= 0;
            const tIn = adm ? tChest : tExh;
            const tOut = adm ? Math.max(tExh, 0.72 * tChest + 0.28 * out.cylWallTempC) : Math.max(92.0, 0.85 * tExh + 0.15 * out.cylWallTempC);
            return [tIn, 0.68 * tIn + 0.32 * tOut, 0.34 * tIn + 0.66 * tOut, tOut];
          }
          case "cyl_7_8_20_7_blast":
          case "cyl_7_8_exhaust_R":
            return [tExh, tExh - 1.2, tExh - 2.4, Math.max(94.0, tExh - 4.0)];
          case "cyl_1_16_drains":
          case "cyl_1_16_drains_L": {
            const tDrn = Math.max(28.0, Math.min(104.0, 0.82 * out.cylWallTempC));
            return [tDrn + 4.0, tDrn + 1.5, tDrn, Math.max(26.0, tDrn - 4.0)];
          }
          default:
            return [tBoil, tBoil, tBoil, tBoil];
        }
      };
    };

    const countArrowsForCurve = (c: ExactFilletedPipeCurve3, isUmb: boolean): number => {
      if (isUmb) return 6;
      const lenMm = c.totalLength * 1000.0;
      return Math.max(3, Math.min(22, Math.round(lenMm / 8.5)));
    };

    // 严格仅对 38 条具备物理管壁的硬件管道 (specs) 生成 Bishop 圆管网格！绝不对开放腔室/车外气流生成假圆管！
    for (const s of specs) {
      const mediumKind = classifyPipeMediumKind(s.id);
      const mat = createPipeLumenMaterial(mediumKind);
      const bendR = s.bendRadiusMm ?? Math.max(1.6, s.radiusMm * 2.0);
      const mainCurve =
        s.frame === "umbilical"
          ? new ExactFilletedPipeCurve3(s.ptsMm.map(([x, y, z]) => new THREE.Vector3(x, y, z)))
          : new ExactFilletedPipeCurve3(buildFilletedPointsM(s.ptsMm, bendR));
      const radiusM = s.radiusMm * 0.001;
      const geom = buildBishopPipeGeometry(mainCurve, radiusM, 12);
      const mesh = new THREE.Mesh(geom, mat);
      mesh.renderOrder = mediumKind === 2 ? 5 : 6;
      mesh.visible = false;

      const branchCurves: ExactFilletedPipeCurve3[] = [mainCurve];
      const branchMeshes: THREE.Mesh[] = [mesh];
      const branchArrowCounts: number[] = [countArrowsForCurve(mainCurve, s.frame === "umbilical")];

      if (s.extraBranchesMm && s.frame !== "umbilical") {
        for (const ebPts of s.extraBranchesMm) {
          const bc = new ExactFilletedPipeCurve3(buildFilletedPointsM(ebPts, bendR));
          const bm = new THREE.Mesh(buildBishopPipeGeometry(bc, radiusM, 12), mat);
          bm.renderOrder = mediumKind === 2 ? 5 : 6;
          bm.visible = false;
          branchCurves.push(bc);
          branchMeshes.push(bm);
          branchArrowCounts.push(countArrowsForCurve(bc, false));
        }
      }

      const totPipeArrows = branchArrowCounts.reduce((a, b) => a + b, 0);
      const targetGroup = s.frame === "loco" ? this.root : s.frame === "tender" ? this.tenderRoot : this.worldRoot;
      for (const bm of branchMeshes) {
        targetGroup.add(bm);
      }
      if (s.frame === "tender") nTdrArrows += totPipeArrows;
      else if (s.frame === "umbilical") nWorldArrows += totPipeArrows;
      else if (mediumKind === 0) nLocoLiquidArrows += totPipeArrows;
      else if (mediumKind === 2) nOpenGasArrows += totPipeArrows;
      else nLocoArrows += totPipeArrows;

      this.pipeRuntimes.push({
        id: s.id,
        partNo: s.partNo,
        title: s.title,
        zone: s.zone,
        frame: s.frame,
        mediumKind,
        curve: mainCurve,
        branchCurves,
        branchMeshes,
        branchArrowCounts,
        umbEndsMm: s.umbEndsMm,
        radiusM,
        mesh,
        mat,
        arrowCount: totPipeArrows,
        phase: Math.random() * 6.28,
        labelT: s.labelT,
        tempProfile: [25, 25, 25, 25],
        arrowColTmp: new THREE.Color(),
        colorCss: s.css,
        evalTempProfileC: makePipeTempProfileEvaluator(s.id),
        evalFlow: s.evalFlow,
      });
    }

    // 管外开放空间自由气体/落水流迹: 采用 1D 行进虚线 (THREE.Line) + 镂空双V风羽(») / 自由落水滴(💧)，零实体管壁！
    const countMarkersForOpenCurve = (c: ExactFilletedPipeCurve3): number => {
      const lenMm = c.totalLength * 1000.0;
      return Math.max(2, Math.min(10, Math.round(lenMm / 10.5)));
    };
    for (const gs of gasStreamSpecs) {
      const isLiq = gs.medium === "open_liquid";
      const mat = createDashedGasStreamlineMaterial(isLiq);
      const bendR = gs.bendRadiusMm ?? 4.0;
      const mainCurve = new ExactFilletedPipeCurve3(buildFilletedPointsM(gs.ptsMm, bendR));
      const mainLine = new THREE.Line(buildStreamlineLineGeometry(mainCurve), mat);
      mainLine.renderOrder = 7;
      mainLine.visible = false;

      const branchCurves: ExactFilletedPipeCurve3[] = [mainCurve];
      const branchLines: THREE.Line[] = [mainLine];
      const branchArrowCounts: number[] = [countMarkersForOpenCurve(mainCurve)];

      if (gs.extraBranchesMm) {
        for (const ebPts of gs.extraBranchesMm) {
          const bc = new ExactFilletedPipeCurve3(buildFilletedPointsM(ebPts, bendR));
          const bl = new THREE.Line(buildStreamlineLineGeometry(bc), mat);
          bl.renderOrder = 7;
          bl.visible = false;
          branchCurves.push(bc);
          branchLines.push(bl);
          branchArrowCounts.push(countMarkersForOpenCurve(bc));
        }
      }

      const totMarkers = branchArrowCounts.reduce((a, b) => a + b, 0);
      const targetGroup = gs.frame === "loco" ? this.root : this.tenderRoot;
      for (const bl of branchLines) {
        targetGroup.add(bl);
      }
      if (!isLiq) {
        nOpenGasArrows += totMarkers;
      } else if (gs.frame === "loco") {
        nLocoLiquidArrows += totMarkers;
      } else {
        nTdrArrows += totMarkers;
      }

      this.openStreamRuntimes.push({
        id: gs.id,
        partNo: gs.partNo,
        title: gs.title,
        zone: gs.zone,
        frame: gs.frame,
        medium: gs.medium,
        curve: mainCurve,
        branchCurves,
        branchLines,
        branchArrowCounts,
        mat,
        markerScaleM: gs.markerScaleMm * 0.001,
        phase: Math.random() * 6.28,
        labelT: gs.labelT,
        tempProfile: [25, 25, 25, 25],
        arrowColTmp: new THREE.Color(),
        evalTempProfileC: makePipeTempProfileEvaluator(gs.id),
        evalFlow: gs.evalFlow,
      });
    }

    // 三套正交形态流向指示器网格: ① 管内蒸汽锐利燕尾镖(➤) ② 液体圆润水滴珠(💧) ③ 管外开放气流镂空双V风羽(»)
    const steamDartGeom = createStreamlinedFlowDartGeometry();
    const liquidDropGeom = createLiquidDropletDartGeometry();
    const openGasChevronGeom = createUnconfinedGasChevronGeometry();
    const solidMarkerMat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      vertexColors: true,
      transparent: true,
      opacity: 0.94,
      depthTest: false,
      depthWrite: false,
    });
    const openGasMarkerMat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      vertexColors: true,
      transparent: true,
      opacity: 0.80,
      side: THREE.DoubleSide,
      depthTest: false,
      depthWrite: false,
    });
    const makeArrowInst = (
      geom: THREE.BufferGeometry,
      mat: THREE.Material,
      count: number,
      parent: THREE.Group
    ): THREE.InstancedMesh => {
      const inst = new THREE.InstancedMesh(geom, mat, Math.max(1, count));
      inst.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      inst.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, count) * 3), 3);
      inst.frustumCulled = false;
      inst.renderOrder = 9;
      inst.visible = false;
      parent.add(inst);
      return inst;
    };
    this.locoArrowInst = makeArrowInst(steamDartGeom, solidMarkerMat, nLocoArrows, this.root);
    this.locoLiquidArrowInst = makeArrowInst(liquidDropGeom, solidMarkerMat, nLocoLiquidArrows, this.root);
    this.tenderArrowInst = makeArrowInst(liquidDropGeom, solidMarkerMat, nTdrArrows, this.tenderRoot);
    this.worldArrowInst = makeArrowInst(liquidDropGeom, solidMarkerMat, nWorldArrows, this.worldRoot);
    this.openGasArrowInst = makeArrowInst(openGasChevronGeom, openGasMarkerMat, nOpenGasArrows, this.root);
  }

  getPipeCallouts(): PipeCalloutSnapshot[] {
    return this.calloutSnapshots;
  }

  private respawnBubble(i: number, zFrac = 0.0, waterZ = 0.0555): void {
    // 严格仅在当前真实液面 (waterZ) 以下浸没受热面上核化汽泡！干烧低水位时顶板/上烟管裸露处绝不悬空冒汽泡！
    let x: number, y: number, zBase: number;
    if (waterZ > 0.0492 && Math.random() < 0.45) {
      x = -0.105 + Math.random() * 0.056;
      y = (Math.random() - 0.5) * 0.036;
      zBase = 0.0478;
    } else if (waterZ > 0.0375) {
      x = -0.046 + Math.random() * 0.134;
      const validFlues = waterZ > 0.0505 ? [0, 1, 2, 3, 4] : [2, 3, 4];
      const fluePick = validFlues[Math.floor(Math.random() * validFlues.length)];
      const fy = [-0.0115, +0.0115, -0.013, 0.0, +0.013][fluePick];
      const fz = [0.0465, 0.0465, 0.035, 0.0332, 0.035][fluePick];
      y = fy + (Math.random() - 0.5) * 0.006;
      zBase = Math.min(waterZ - 0.002, fz + 0.0025);
    } else {
      // 极低水位 (<37.5mm): Type-"C" 锅筒仅在前段圆筒腹底 (X >= -46mm) 存有残水沸腾核化，后段无湿水腿!
      x = -0.046 + Math.random() * 0.132;
      y = (Math.random() - 0.5) * 0.016;
      zBase = Math.max(0.0186, Math.min(waterZ - 0.0014, 0.0195 + Math.random() * 0.006));
    }
    const maxRise = Math.max(0.0005, waterZ - 0.001 - zBase);
    this.bubblePositions[3 * i] = x;
    this.bubblePositions[3 * i + 1] = y;
    this.bubblePositions[3 * i + 2] = zBase + zFrac * Math.min(0.012, maxRise);
  }

  private updateMidplaneTexture(sp: SpatialFluidState, boilerTempC: number): void {
    const NX = 64, NZ = 24;
    if (sp.midplaneTempB64 && sp.midplaneTempB64.length >= 1600) {
      const bin = atob(sp.midplaneTempB64);
      const n = Math.min(NX * NZ, bin.length);
      for (let idx = 0; idx < n; idx++) {
        const v = bin.charCodeAt(idx);
        const o = idx * 4;
        this.sliceTexData[o] = v;
        this.sliceTexData[o + 1] = v;
        this.sliceTexData[o + 2] = v;
        this.sliceTexData[o + 3] = 255;
      }
      this.sliceTex.needsUpdate = true;
      return;
    }

    // 浏览器独立 WASM 模式: 由 8 站烟温、4 层水温与过热管温度分布实时合成 64×24 纵剖面热场
    const xMin = -0.114, xMax = 0.162, zMin = 0.010, zMax = 0.088;
    const zWater = sp.waterFreeZMm * 0.001;
    const tFlame = Math.max(...sp.wickTempsC, boilerTempC);
    for (let iz = 0; iz < NZ; iz++) {
      const z = zMin + ((iz + 0.5) / NZ) * (zMax - zMin);
      for (let ix = 0; ix < NX; ix++) {
        const x = xMin + ((ix + 0.5) / NX) * (xMax - xMin);
        let tempC = 25.0;
        if (x >= -0.106 && x <= -0.047 && z >= 0.018 && z <= 0.0475) {
          // 火室火焰腔
          const h = (z - 0.018) / 0.0295;
          tempC = boilerTempC + (tFlame - boilerTempC) * (1.0 - 0.35 * h);
        } else if (x > -0.047 && x <= 0.091 && ((z >= 0.030 && z <= 0.038) || (z >= 0.042 && z <= 0.051))) {
          // 5 根烟管燃气通道
          const s = Math.max(0, Math.min(7, ((x + 0.047) / 0.138) * 7.0));
          const i0 = Math.floor(s), i1 = Math.min(7, i0 + 1);
          tempC = sp.flueGasProfileC[i0] * (1 - (s - i0)) + sp.flueGasProfileC[i1] * (s - i0);
        } else if (x >= -0.110 && x <= 0.090 && z >= 0.019 && z <= 0.069) {
          if (z <= zWater) {
            tempC = sp.waterLayerTempsC[Math.min(3, Math.floor(((z - 0.019) / 0.042) * 4))];
          } else {
            tempC = boilerTempC;
          }
        } else if (x > 0.091 && x <= 0.158 && z >= 0.018 && z <= 0.072) {
          // 烟室高温乏汽/烟气混合流
          const gasExit = sp.flueGasProfileC[7] || boilerTempC;
          const rChim = Math.abs(x - 0.130);
          tempC = rChim < 0.014 ? gasExit * 0.92 : 25.0 + (gasExit - 25.0) * 0.42;
        }
        const u8 = Math.max(0, Math.min(255, Math.round(((tempC - 25.0) / 825.0) * 255.0)));
        const o = (iz * NX + ix) * 4;
        this.sliceTexData[o] = u8;
        this.sliceTexData[o + 1] = u8;
        this.sliceTexData[o + 2] = u8;
        this.sliceTexData[o + 3] = 255;
      }
    }
    this.sliceTex.needsUpdate = true;
  }

  private styleSteamChamber(mesh: THREE.Mesh, xStart: number, xEnd: number, pKgf: number, shTempC: number): void {
    const len = Math.max(0.0012, xEnd - xStart);
    mesh.position.x = 0.5 * (xStart + xEnd);
    mesh.scale.set(len, 1.0, 1.0);
    const mat = mesh.material as THREE.ShaderMaterial;
    const pNorm = Math.max(0.0, Math.min(1.0, pKgf / 3.4));
    const shNorm = Math.max(0.0, Math.min(1.0, (shTempC - 135.0) / 85.0));
    mat.uniforms.uTime.value = this.elapsed;
    mat.uniforms.uPressNorm.value = pNorm;
    mat.uniforms.uSuperheatNorm.value = shNorm;
    mat.uniforms.uOpacity.value = pKgf > 0.03 ? Math.min(0.78, 0.18 + 0.58 * pNorm) : 0.0;
  }

  update(
    dt: number,
    locoMatrix: THREE.Matrix4,
    out: StepOutput,
    qPL: number,
    qPR: number,
    calib: CylinderCalibration,
    xray: boolean,
    burnerLit: boolean,
    tenderMatrix?: THREE.Matrix4 | null,
    showFlowArrows = false,
    showPipeLabels = false
  ): void {
    this.elapsed += dt;
    const sp = out.spatial;
    this.feedPhase = (this.feedPhase + dt * (1.2 + 9.5 * sp.feedwaterVelMS)) % 1000.0;

    this.root.matrix.copy(locoMatrix);
    this.root.matrixWorldNeedsUpdate = true;
    if (tenderMatrix) {
      this.tenderRoot.visible = true;
      this.tenderRoot.matrix.copy(tenderMatrix);
      this.tenderRoot.matrixWorldNeedsUpdate = true;
    } else {
      this.tenderRoot.visible = false;
    }

    // 同步 CAD 材质全局流体着色器 Uniforms
    this.uniforms.uTime.value = this.elapsed;
    this.uniforms.uXray.value = xray ? 1.0 : 0.0;
    this.uniforms.uSteamFlow.value = Math.min(1.0, out.steamGenGS / 0.32);
    this.uniforms.uDomeDryness.value = sp.steamDrynessDome;
    this.uniforms.uSuperheatNorm.value = Math.max(0.0, Math.min(1.0, (out.superheatTempC - 120.0) / 105.0));
    this.uniforms.uExhaustPulse.value = out.exhaustPlume;
    this.uniforms.uFeedPhase.value = this.feedPhase;
    this.uniforms.uFeedStrength.value = Math.min(1.0, sp.feedwaterVelMS / 0.35 + (out.feedwaterGS > 0.02 ? 0.35 : 0.0));
    for (let i = 0; i < 6; i++) {
      this.uniforms.uCrownTemps.value[i] = sp.crownHotspotsC[i] ?? out.crownTempC;
    }
    for (let i = 0; i < 8; i++) {
      this.uniforms.uFlueTemps.value[i] = sp.flueGasProfileC[i] ?? out.boilerTempC;
    }
    const fbows = out.flueDeflectionMm ?? sp.flueDeflectionMm ?? [0, 0, 0, 0, 0];
    for (let i = 0; i < 5; i++) {
      this.uniforms.uFlueBowMm.value[i] = fbows[i] ?? 0.0;
    }
    this.uniforms.uCrownBulgeMm.value = out.crownBulgeMm ?? sp.crownBulgeMm ?? 0.0;

    // 1. 更新 23-1 直列 1×3 三头酒精火焰 (逐芯横流弯折角 uWickTilts + 四拍乏汽脉动 uChuffPulse)
    const lumenOk = sp.burnerLumenConnected !== false;
    const wickFlowOk = sp.wickCapillaryMlS === undefined || sp.wickCapillaryMlS > 0.002;
    const litVal = burnerLit && lumenOk && wickFlowOk && sp.airFuelPhi > 0.05 ? 1.0 : 0.0;
    const meanTiltDeg = out.flameTiltDeg ?? sp.flameTiltDeg ?? 12.0;
    const wt3 = out.wickTiltDeg3 ?? sp.wickTiltDeg3 ?? [
      Math.max(5.0, meanTiltDeg * 0.76),
      meanTiltDeg,
      Math.min(76.5, meanTiltDeg * 1.22),
    ];
    this.flameMat.uniforms.uTime.value = this.elapsed;
    this.flameMat.uniforms.uLit.value = litVal;
    this.flameMat.uniforms.uPhi.value = sp.airFuelPhi || 1.0;
    this.flameMat.uniforms.uTiltRad.value = (meanTiltDeg * Math.PI) / 180.0;
    this.flameMat.uniforms.uWickTilts.value.set(
      ((wt3[0] ?? meanTiltDeg) * Math.PI) / 180.0,
      ((wt3[1] ?? meanTiltDeg) * Math.PI) / 180.0,
      ((wt3[2] ?? meanTiltDeg) * Math.PI) / 180.0
    );
    this.flameMat.uniforms.uChuffPulse.value = Math.min(1.0, out.exhaustPlume * 0.85);
    this.flameMat.uniforms.uDraftNorm.value = Math.min(1.3, out.smokeboxVacuumPa / 80.0);
    this.flameMat.uniforms.uIntensity.value = Math.min(1.25, 0.55 + out.smokeboxVacuumPa / 120.0);

    // 2. 更新 11-7 驾驶室右侧外缘 (-Y=-31.3mm) 水面计实况液柱 (二阶连通器真实弯月面高度!)
    const sgLevel = out.sightGlassLevelPct ?? out.waterLevelPct;
    const sgVisible = sgLevel > 0.35;
    this.sightGlassWaterMesh.visible = sgVisible;
    this.sightGlassMeniscusMesh.visible = sgVisible;
    if (sgVisible) {
      const sgFrac = Math.max(0.005, Math.min(0.98, sgLevel / 100.0));
      const sgHeight = 0.0170 * sgFrac;
      this.sightGlassWaterMesh.scale.set(1.0, 1.0, sgHeight);
      this.sightGlassMeniscusMesh.position.z = 0.0375 + sgHeight;
    }

    // 2b. 更新煤水车 35-1 后置敞口注水槽自由液面波纹 + 加水 / 31-14 回水 / 31-11 放水
    const hasTdr = !!tenderMatrix;
    const wFrac = Math.max(0.01, Math.min(1.0, out.tenderWaterMl / 340.0));
    const zBayWater = 0.0152 + 0.0445 * wFrac;
    this.tenderOpenBayWaterMesh.visible = hasTdr && out.tenderWaterMl > 0.5;
    this.returnStreamMesh.visible = hasTdr && out.returnWaterMlS > 0.02;
    this.waterFillStreamMesh.visible = hasTdr && (out.waterFillMlS ?? 0) > 0.05;
    this.tenderDrainStreamMesh.visible = hasTdr && (out.tenderDrainMlS ?? 0) > 0.05;

    if (hasTdr) {
      if (sp.tenderWaveB64 && sp.tenderWaveB64.length >= 200) {
        const bin = atob(sp.tenderWaveB64);
        const n = Math.min(24 * 10, bin.length);
        for (let idx = 0; idx < n; idx++) {
          const v = bin.charCodeAt(idx);
          const o = idx * 4;
          this.tenderWaveTexData[o] = v;
          this.tenderWaveTexData[o + 1] = v;
          this.tenderWaveTexData[o + 2] = v;
          this.tenderWaveTexData[o + 3] = 255;
        }
        this.tenderWaveTex.needsUpdate = true;
      }
      this.tenderOpenBayMat.uniforms.uTime.value = this.elapsed;
      this.tenderOpenBayMat.uniforms.uWaterZ.value = zBayWater;
      this.tenderOpenBayMat.uniforms.uTempC.value = sp.tenderBayTempC ?? out.tenderWaterTempC;
      this.tenderOpenBayMat.uniforms.uJetAmp.value = Math.min(
        1.5,
        out.returnWaterMlS * 0.45 + (out.waterFillMlS ?? 0) * 0.08 + (out.tenderDrainMlS ?? 0) * 0.06
      );

      if (this.returnStreamMesh.visible) {
        const dropLen = Math.max(0.003, 0.0530 - zBayWater);
        this.returnStreamMesh.scale.set(1.0, 1.0, dropLen);
        const rsMat = this.returnStreamMesh.material as THREE.MeshBasicMaterial;
        rsMat.opacity = Math.min(0.92, 0.38 + 0.48 * Math.min(1.0, out.returnWaterMlS / 0.8));
      }
      if (this.waterFillStreamMesh.visible) {
        const fillLen = Math.max(0.008, 0.0820 - zBayWater);
        const wScale = 0.65 + 0.55 * Math.min(1.0, (out.waterFillMlS ?? 0) / 18.0);
        this.waterFillStreamMesh.scale.set(wScale, wScale, fillLen);
      }
      if (this.tenderDrainStreamMesh.visible) {
        const dScale = 0.65 + 0.55 * Math.min(1.0, (out.tenderDrainMlS ?? 0) / 5.0);
        this.tenderDrainStreamMesh.scale.set(dScale, dScale, 1.0);
      }
    }

    // 3. X-Ray 内窥零件级三维流体场更新 (水位低于 19.0mm 即 <0.5ml 时锅水彻底蒸干消失!)
    const waterZ = sp.waterFreeZMm * 0.001;
    const hasBoilerWater = waterZ > 0.0190 && (out.boilerWaterMl === undefined || out.boilerWaterMl > 0.3);
    const exhaustJetOnly = Math.min(1.0, out.exhaustPlume * 1.15);
    const blowerJetOnly = (out.blowerGS ?? 0) > 0.002 ? Math.min(1.0, 0.22 + (out.blowerGS ?? 0) * 8.5) : 0.0;
    const blastJetStrength = Math.min(1.0, exhaustJetOnly + 0.35 * blowerJetOnly);
    this.waterSurfaceMesh.visible = xray && hasBoilerWater;
    this.waterVolumeMesh.visible = xray && hasBoilerWater;
    this.steamHeadspaceMesh.visible = xray && (out.boilerPressureKgf > 0.03 || sp.boilingIntensity > 0.03);
    for (const bm of this.smokeboxBlastMeshes) {
      bm.visible = xray && exhaustJetOnly > 0.02;
    }
    for (const blm of this.blowerJetMeshes) {
      blm.visible = (xray || showFlowArrows) && blowerJetOnly > 0.02;
    }
    this.blowerJetMat.uniforms.uTime.value = this.elapsed;
    this.blowerJetMat.uniforms.uJet.value = blowerJetOnly;
    this.blowerJetMat.uniforms.uDryness.value = 0.98;
    this.bubblePoints.visible = xray && hasBoilerWater && sp.boilingIntensity > 0.02;
    this.sliceMesh.visible = xray;
    this.lubeCoreMesh.visible = xray;
    this.lubeStandpipeMesh.visible = xray;
    this.tenderWaterMesh.visible = xray && hasTdr && out.tenderWaterMl > 1.0;
    this.tenderAlcoholMesh.visible = xray && hasTdr && out.alcoholMl > 0.5;
    this.mariotteBubblePoints.visible = xray && hasTdr && (out.mariotteBubbling || out.mariotteFloodAlarm);
    for (const m of [...this.cylHeadMeshes, ...this.cylCrankMeshes, ...this.cylCondMeshes, ...this.portGlowMeshes]) {
      m.visible = xray;
    }

    const pipesVisible = xray || showFlowArrows;
    for (const pr of this.pipeRuntimes) {
      const vis = pipesVisible && (pr.frame === "loco" || hasTdr);
      for (const bm of pr.branchMeshes) {
        bm.visible = vis;
      }
    }
    for (const os of this.openStreamRuntimes) {
      const vis = showFlowArrows && (os.frame === "loco" || hasTdr);
      for (const bl of os.branchLines) {
        bl.visible = vis;
      }
    }
    this.locoArrowInst.visible = showFlowArrows;
    this.locoLiquidArrowInst.visible = showFlowArrows;
    this.tenderArrowInst.visible = showFlowArrows && hasTdr;
    this.worldArrowInst.visible = showFlowArrows && hasTdr;
    this.openGasArrowInst.visible = showFlowArrows;

    if (pipesVisible || showPipeLabels) {
      this.updateAllPipeLumensAndArrows(dt, locoMatrix, tenderMatrix ?? null, out, sp, showFlowArrows, showPipeLabels);
    } else {
      this.calloutSnapshots.length = 0;
    }

    if (!xray) return;

    if (hasTdr) {
      this.tenderWaterMesh.scale.set(1.0, 1.0, 0.0445 * wFrac);
      const twMat = this.tenderWaterMesh.material as THREE.MeshBasicMaterial;
      temperatureToThermalColor(out.tenderWaterTempC, twMat.color);

      const aFrac = Math.max(0.02, Math.min(1.0, out.alcoholMl / 130.0));
      this.tenderAlcoholMesh.scale.set(1.0, 1.0, 0.0350 * aFrac);
      const taMat = this.tenderAlcoholMesh.material as THREE.MeshBasicMaterial;
      if (out.mariotteFloodAlarm) {
        taMat.color.setHex(0xff4d29);
      } else {
        temperatureToThermalColor(out.alcoholTempC ?? 28.0, taMat.color);
      }

      if (this.mariotteBubblePoints.visible) {
        const zTop = 0.0258 + 0.0350 * aFrac;
        for (let i = 0; i < 28; i++) {
          this.mariotteBubblePos[3 * i + 2] += (out.mariotteFloodAlarm ? 0.065 : 0.024) * dt;
          if (this.mariotteBubblePos[3 * i + 2] > zTop) {
            this.mariotteBubblePos[3 * i + 2] = 0.0265;
          }
        }
        this.mariotteBubblePoints.geometry.attributes.position.needsUpdate = true;
      }
    }

    const sloshTan = Math.tan((sp.waterSloshPitchDeg * Math.PI) / 180.0);
    this.waterSurfaceMat.uniforms.uTime.value = this.elapsed;
    this.waterSurfaceMat.uniforms.uWaterZ.value = waterZ;
    this.waterSurfaceMat.uniforms.uSloshTan.value = sloshTan;
    this.waterSurfaceMat.uniforms.uBoil.value = sp.boilingIntensity;
    this.waterSurfaceMat.uniforms.uTempBottom.value = sp.waterLayerTempsC[0] ?? out.boilerTempC;
    this.waterSurfaceMat.uniforms.uTempTop.value = sp.waterLayerTempsC[3] ?? out.boilerTempC;

    this.steamHeadspaceMat.uniforms.uTime.value = this.elapsed;
    this.steamHeadspaceMat.uniforms.uWaterZ.value = hasBoilerWater ? waterZ : 0.0182;
    this.steamHeadspaceMat.uniforms.uSloshTan.value = sloshTan;
    this.steamHeadspaceMat.uniforms.uPressureNorm.value = Math.min(1.0, out.boilerPressureKgf / 3.8);
    this.steamHeadspaceMat.uniforms.uBoil.value = sp.boilingIntensity;
    this.steamHeadspaceMat.uniforms.uSteamFlow.value = this.uniforms.uSteamFlow.value;

    this.smokeboxBlastMat.uniforms.uTime.value = this.elapsed;
    this.smokeboxBlastMat.uniforms.uJet.value = blastJetStrength;
    this.smokeboxBlastMat.uniforms.uDryness.value = sp.exhaustDryness;

    if (hasBoilerWater && sp.boilingIntensity > 0.02) {
      this.bubbleMat.uniforms.uBoil.value = sp.boilingIntensity;
      const riseVel = 0.016 + 0.088 * sp.boilingIntensity;
      for (let i = 0; i < this.bubbleCount; i++) {
        const bx = this.bubblePositions[3 * i];
        this.bubblePositions[3 * i] += 0.004 * Math.sin(this.elapsed * 9.0 + i) * dt;
        this.bubblePositions[3 * i + 2] += riseVel * (0.7 + 0.6 * this.bubbleSeeds[i]) * dt;
        const zCeil = waterZ + (bx + 0.010) * sloshTan - 0.0008;
        if (this.bubblePositions[3 * i + 2] >= zCeil) {
          this.respawnBubble(i, 0.0, waterZ);
        }
      }
      this.bubblePoints.geometry.attributes.position.needsUpdate = true;
    }

    this.updateMidplaneTexture(sp, out.boilerTempC);

    // 4. 左右主气缸前后工作腔实时活塞追随与蒸汽膨胀云
    const sL = Math.max(0.0, Math.min(calib.stroke, calib.qFdcL - qPL));
    const sR = Math.max(0.0, Math.min(calib.stroke, calib.qFdcR - qPR));
    const xHeadWall = 0.1450;
    const xCrankWall = 0.1214;
    const xPistL = 0.1332 + 0.5 * calib.stroke - sL;
    const xPistR = 0.1332 + 0.5 * calib.stroke - sR;

    this.styleSteamChamber(this.cylHeadMeshes[0], xPistL + 0.0018, xHeadWall, sp.cylLPKgf[0] ?? 0, out.superheatTempC);
    this.styleSteamChamber(this.cylCrankMeshes[0], xCrankWall, xPistL - 0.0018, sp.cylLPKgf[1] ?? 0, out.superheatTempC);
    this.styleSteamChamber(this.cylHeadMeshes[1], xPistR + 0.0018, xHeadWall, sp.cylRPKgf[0] ?? 0, out.superheatTempC);
    this.styleSteamChamber(this.cylCrankMeshes[1], xCrankWall, xPistR - 0.0018, sp.cylRPKgf[1] ?? 0, out.superheatTempC);

    const condRatio = Math.max(0.0, Math.min(1.0, out.condensateG / 0.55));
    for (let side = 0; side < 2; side++) {
      const cm = this.cylCondMeshes[side];
      cm.scale.set(1.0, 1.0, 0.25 + 1.45 * condRatio);
      const cMat = cm.material as THREE.MeshBasicMaterial;
      cMat.color.setHex((side === 0 ? out.hydrolockL : out.hydrolockR) ? 0xff3b30 : 0x1fd2ff);
      cMat.opacity = condRatio > 0.03 ? 0.22 + 0.65 * condRatio : 0.0;
    }

    const portVals = [
      sp.valvePortOpenL[0] ?? 0,
      sp.valvePortOpenL[1] ?? 0,
      sp.valvePortOpenR[0] ?? 0,
      sp.valvePortOpenR[1] ?? 0,
    ];
    portVals.forEach((pv, k) => {
      const pm = this.portGlowMeshes[k].material as THREE.MeshBasicMaterial;
      if (pv >= 0) {
        temperatureToThermalColor(out.superheatTempC, pm.color);
        pm.opacity = Math.min(0.82, pv * (0.25 + 0.55 * Math.min(1.0, out.steamChestKgf / 3.0)));
      } else {
        temperatureToThermalColor(Math.max(28.0, Math.min(112.0, 0.78 * out.cylWallTempC + 8.0)), pm.color);
        pm.opacity = Math.min(0.68, -pv * 0.58);
      }
    });

    // 5. 4-2 卧式置换注油器油水分层与统一温标色更新 (底水稍凉于顶层受蒸汽直接加热的浮油)
    const tOilNow = out.lubeOilTempC ?? Math.max(28.0, 0.62 * out.cylWallTempC);
    this.lubeCoreMat.uniforms.uWaterFrac.value = sp.lubeWaterFrac;
    this.lubeCoreMat.uniforms.uTime.value = this.elapsed;
    temperatureToThermalColor(Math.max(22.0, tOilNow - 18.0), this.lubeCoreMat.uniforms.uWaterCol.value);
    temperatureToThermalColor(tOilNow, this.lubeCoreMat.uniforms.uOilCol.value);
    const sgMat = this.sightGlassWaterMesh.material as THREE.MeshBasicMaterial;
    temperatureToThermalColor(25.0 + 0.72 * Math.max(0.0, out.boilerTempC - 25.0), sgMat.color);
  }

  private updateAllPipeLumensAndArrows(
    dt: number,
    locoMatrix: THREE.Matrix4,
    tenderMatrix: THREE.Matrix4 | null,
    out: StepOutput,
    sp: SpatialFluidState,
    showFlowArrows: boolean,
    showPipeLabels: boolean
  ): void {
    let iLocoSteam = 0;
    let iLocoLiq = 0;
    let iTdrLiq = 0;
    let iWorldLiq = 0;
    let iOpenGas = 0;
    const hasTdr = tenderMatrix !== null;
    this.calloutSnapshots.length = 0;
    const flueLvsG = 0.5 * ((sp.edgeConductances?.["boil_flues_upper"] ?? 1.0) + (sp.edgeConductances?.["boil_flues_lower"] ?? 1.0));

    // 1. 更新 38 条具备物理管壁的硬件管道 (管内液体💧 / 管内受压蒸汽➤ / 锅筒5烟管热燃气»)
    for (const pr of this.pipeRuntimes) {
      if ((pr.frame === "tender" || pr.frame === "umbilical") && !hasTdr) {
        continue;
      }

      if (pr.frame === "umbilical" && pr.umbEndsMm && tenderMatrix) {
        const isRet = pr.id === "umb_bt5_return";
        const pTdr = this.tmpVecA
          .set(pr.umbEndsMm.tdr[0] * 0.001, pr.umbEndsMm.tdr[1] * 0.001, pr.umbEndsMm.tdr[2] * 0.001)
          .applyMatrix4(tenderMatrix);
        const pLoco = this.tmpVecB
          .set(pr.umbEndsMm.loco[0] * 0.001, pr.umbEndsMm.loco[1] * 0.001, pr.umbEndsMm.loco[2] * 0.001)
          .applyMatrix4(locoMatrix);
        const pStart = isRet ? pLoco : pTdr;
        const pEnd = isRet ? pTdr : pLoco;
        const nUmb = 12;
        const umbPts: THREE.Vector3[] = [];
        for (let s = 0; s <= nUmb; s++) {
          const f = s / nUmb;
          const pt = new THREE.Vector3().lerpVectors(pStart, pEnd, f);
          pt.z -= 0.0018 * Math.sin(Math.PI * f);
          umbPts.push(pt);
        }
        const newUmbCurve = new ExactFilletedPipeCurve3(umbPts);
        pr.curve = newUmbCurve;
        pr.branchCurves[0] = newUmbCurve;
        pr.mesh.geometry.dispose();
        pr.mesh.geometry = buildBishopPipeGeometry(newUmbCurve, pr.radiusM, 10);
      }

      const st = pr.evalFlow(out, sp);
      const prof = pr.evalTempProfileC(out, sp);
      pr.tempProfile[0] = prof[0];
      pr.tempProfile[1] = prof[1];
      pr.tempProfile[2] = prof[2];
      pr.tempProfile[3] = prof[3];
      const pipeMidTempC = 0.25 * (prof[0] + prof[1] + prof[2] + prof[3]);
      const dynCss = temperatureToThermalCss(pipeMidTempC);
      const lvsG = sp.edgeConductances?.[pr.id] ?? 1.0;
      const lvsOpen = lvsG > 0.02;
      const effSigned = lvsOpen ? st.signedRate : 0.0;
      const absRate = Math.abs(effSigned);
      const signDir = effSigned >= 0 ? 1.0 : -1.0;
      const isFlowing = absRate > 0.0025;
      // 气体/蒸汽流速显著快于致密液体流速，直观体现可压缩流体高速特性！
      const speedMult = pr.mediumKind === 0 ? 10.5 : 15.5;
      const baseSpd = pr.mediumKind === 0 ? 2.0 : 3.2;
      const speed = isFlowing ? signDir * (baseSpd + Math.min(18.0, absRate * speedMult)) : 0.0;
      pr.phase = (pr.phase + dt * speed) % 628.318;

      pr.mat.uniforms.uPhase.value = pr.phase;
      pr.mat.uniforms.uFill.value = lvsOpen ? st.fill : 0.08;
      pr.mat.uniforms.uActive.value = isFlowing ? Math.min(1.0, 0.38 + absRate * 2.5) : 0.0;
      pr.mat.uniforms.uTemps.value.set(prof[0], prof[1], prof[2], prof[3]);

      if (showFlowArrows) {
        const inst =
          pr.frame === "tender"
            ? this.tenderArrowInst
            : pr.frame === "umbilical"
              ? this.worldArrowInst
              : pr.mediumKind === 0
                ? this.locoLiquidArrowInst
                : pr.mediumKind === 2
                  ? this.openGasArrowInst
                  : this.locoArrowInst;

        const baseRadM =
          pr.mediumKind === 2
            ? 0.00095
            : Math.max(0.00046, Math.min(0.00092, pr.radiusM * (pr.mediumKind === 0 ? 0.72 : 0.78)));
        const flowBoost = 0.90 + 0.24 * Math.min(1.0, absRate * 2.0);
        const rScale = baseRadM * flowBoost;
        const lenScale =
          pr.mediumKind === 0
            ? baseRadM * (1.02 + 0.16 * Math.min(1.0, absRate * 2.0))
            : pr.mediumKind === 2
              ? baseRadM * (1.05 + 0.20 * Math.min(1.0, absRate * 2.0))
              : baseRadM * (1.16 + 0.24 * Math.min(1.0, absRate * 2.0));

        for (let bIdx = 0; bIdx < pr.branchCurves.length; bIdx++) {
          const bCurve = pr.branchCurves[bIdx];
          const bCount = pr.branchArrowCounts[bIdx];
          for (let k = 0; k < bCount; k++) {
            const idx =
              pr.frame === "tender"
                ? iTdrLiq++
                : pr.frame === "umbilical"
                  ? iWorldLiq++
                  : pr.mediumKind === 0
                    ? iLocoLiq++
                    : pr.mediumKind === 2
                      ? iOpenGas++
                      : iLocoSteam++;
            if (!isFlowing || st.fill < 0.05) {
              this.tmpMat.makeScale(0, 0, 0);
              inst.setMatrixAt(idx, this.tmpMat);
              continue;
            }
            const rawS = ((k + 0.5) / bCount + (pr.phase / 18.84955)) % 1.0;
            const u = Math.max(0.04, Math.min(0.96, rawS < 0 ? rawS + 1.0 : rawS));
            bCurve.getPointAt(u, this.tmpVecA);
            bCurve.getTangentAt(u, this.tmpTan);
            if (signDir < 0) {
              this.tmpTan.negate();
            }
            this.tmpQuat.setFromUnitVectors(ARROW_BASE_AXIS, this.tmpTan.normalize());
            this.tmpScale.set(rScale, lenScale, rScale);
            this.tmpMat.compose(this.tmpVecA, this.tmpQuat, this.tmpScale);
            inst.setMatrixAt(idx, this.tmpMat);

            const s3 = u * 3.0;
            const localT =
              s3 <= 1.0
                ? prof[0] + (prof[1] - prof[0]) * s3
                : s3 <= 2.0
                  ? prof[1] + (prof[2] - prof[1]) * (s3 - 1.0)
                  : prof[2] + (prof[3] - prof[2]) * (s3 - 2.0);
            temperatureToThermalColor(localT + 6.0, pr.arrowColTmp);
            inst.setColorAt(idx, pr.arrowColTmp);
          }
        }
      }

      if (showPipeLabels) {
        const wp = new THREE.Vector3();
        pr.curve.getPointAt(pr.labelT, wp);
        if (pr.frame === "loco") {
          wp.applyMatrix4(locoMatrix);
        } else if (pr.frame === "tender" && tenderMatrix) {
          wp.applyMatrix4(tenderMatrix);
        }
        const tag = pr.mediumKind === 0 ? "管内液💧" : pr.mediumKind === 1 ? "管内蒸気➤" : "煙管気»";
        const dirSymbol = !lvsOpen ? "⛔" : !isFlowing ? "⏸" : signDir > 0 ? "→" : "←";
        const valStr = !lvsOpen
          ? `[${tag}] LVS断流`
          : st.unit === "給気" || st.unit === "排気"
            ? `[${tag}] ${pipeMidTempC.toFixed(0)}°C · ${st.unit} ${(absRate * 100).toFixed(0)}%`
            : `[${tag}] ${pipeMidTempC.toFixed(0)}°C · ${absRate.toFixed(2)} ${st.unit}`;
        this.calloutSnapshots.push({
          id: pr.id,
          partNo: pr.partNo,
          title: pr.title,
          zone: pr.zone,
          worldPos: wp,
          colorCss: dynCss,
          tempC: pipeMidTempC,
          flowText: valStr,
          dirSymbol,
          active: isFlowing,
        });
      }
    }

    // 2. 更新管外开放空间自由气体(»+行进虚线)与自由落水/喷流(💧+珠链虚线) — 零圆管网格！
    for (const os of this.openStreamRuntimes) {
      if (os.frame === "tender" && !hasTdr) {
        continue;
      }
      const st = os.evalFlow(out, sp);
      const prof = os.evalTempProfileC(out, sp);
      os.tempProfile[0] = prof[0];
      os.tempProfile[1] = prof[1];
      os.tempProfile[2] = prof[2];
      os.tempProfile[3] = prof[3];
      const midTempC = 0.25 * (prof[0] + prof[1] + prof[2] + prof[3]);
      const dynCss = temperatureToThermalCss(midTempC);
      const lvsOpen = os.medium === "open_gas" ? flueLvsG > 0.02 : true;
      const effSigned = lvsOpen ? st.signedRate : 0.0;
      const absRate = Math.abs(effSigned);
      const signDir = effSigned >= 0 ? 1.0 : -1.0;
      const isFlowing = absRate > 0.0025 && st.fill > 0.05;
      const speed = isFlowing ? signDir * (3.0 + Math.min(18.0, absRate * 13.5)) : 0.0;
      os.phase = (os.phase + dt * speed) % 628.318;

      os.mat.uniforms.uPhase.value = os.phase;
      os.mat.uniforms.uFill.value = isFlowing ? st.fill : 0.0;
      os.mat.uniforms.uActive.value = isFlowing ? Math.min(1.0, 0.42 + absRate * 2.4) : 0.0;
      os.mat.uniforms.uTemps.value.set(prof[0], prof[1], prof[2], prof[3]);
      for (const bl of os.branchLines) {
        bl.visible = showFlowArrows && isFlowing;
      }

      if (showFlowArrows) {
        const isLiq = os.medium === "open_liquid";
        const inst = !isLiq
          ? this.openGasArrowInst
          : os.frame === "loco"
            ? this.locoLiquidArrowInst
            : this.tenderArrowInst;
        const rScale = os.markerScaleM * (0.92 + 0.22 * Math.min(1.0, absRate * 2.0));
        const lenScale = os.markerScaleM * (isLiq ? 1.08 : 1.02);

        for (let bIdx = 0; bIdx < os.branchCurves.length; bIdx++) {
          const bCurve = os.branchCurves[bIdx];
          const bCount = os.branchArrowCounts[bIdx];
          for (let k = 0; k < bCount; k++) {
            const idx = !isLiq ? iOpenGas++ : os.frame === "loco" ? iLocoLiq++ : iTdrLiq++;
            if (!isFlowing) {
              this.tmpMat.makeScale(0, 0, 0);
              inst.setMatrixAt(idx, this.tmpMat);
              continue;
            }
            const rawS = ((k + 0.5) / bCount + (os.phase / 18.84955)) % 1.0;
            const u = Math.max(0.06, Math.min(0.94, rawS < 0 ? rawS + 1.0 : rawS));
            bCurve.getPointAt(u, this.tmpVecA);
            bCurve.getTangentAt(u, this.tmpTan);
            if (signDir < 0) {
              this.tmpTan.negate();
            }
            this.tmpQuat.setFromUnitVectors(ARROW_BASE_AXIS, this.tmpTan.normalize());
            this.tmpScale.set(rScale, lenScale, rScale);
            this.tmpMat.compose(this.tmpVecA, this.tmpQuat, this.tmpScale);
            inst.setMatrixAt(idx, this.tmpMat);

            const s3 = u * 3.0;
            const localT =
              s3 <= 1.0
                ? prof[0] + (prof[1] - prof[0]) * s3
                : s3 <= 2.0
                  ? prof[1] + (prof[2] - prof[1]) * (s3 - 1.0)
                  : prof[2] + (prof[3] - prof[2]) * (s3 - 2.0);
            temperatureToThermalColor(localT + 6.0, os.arrowColTmp);
            inst.setColorAt(idx, os.arrowColTmp);
          }
        }
      }

      if (showPipeLabels && isFlowing) {
        const wp = new THREE.Vector3();
        os.curve.getPointAt(os.labelT, wp);
        if (os.frame === "loco") {
          wp.applyMatrix4(locoMatrix);
        } else if (tenderMatrix) {
          wp.applyMatrix4(tenderMatrix);
        }
        const tag = os.medium === "open_liquid" ? "管外液💧" : "自由気»";
        const dirSymbol = signDir > 0 ? "→" : "←";
        this.calloutSnapshots.push({
          id: os.id,
          partNo: os.partNo,
          title: os.title,
          zone: os.zone,
          worldPos: wp,
          colorCss: dynCss,
          tempC: midTempC,
          flowText: `[${tag}] ${midTempC.toFixed(0)}°C · ${absRate.toFixed(2)} ${st.unit}`,
          dirSymbol,
          active: true,
        });
      }
    }

    if (showFlowArrows) {
      const zeroTail = (inst: THREE.InstancedMesh, fromIdx: number) => {
        this.tmpMat.makeScale(0, 0, 0);
        for (let k = fromIdx; k < inst.count; k++) {
          inst.setMatrixAt(k, this.tmpMat);
        }
        inst.instanceMatrix.needsUpdate = true;
        if (inst.instanceColor) inst.instanceColor.needsUpdate = true;
      };
      zeroTail(this.locoArrowInst, iLocoSteam);
      zeroTail(this.locoLiquidArrowInst, iLocoLiq);
      zeroTail(this.openGasArrowInst, iOpenGas);
      if (hasTdr) {
        zeroTail(this.tenderArrowInst, iTdrLiq);
        zeroTail(this.worldArrowInst, iWorldLiq);
      }
    }
  }
}

// ============================================================================
// CAD 零件网格 PBR 表面沿程热流相变增强着色器 (基于精确 CAD 坐标属性 aCadPos)
// ============================================================================

export function attachSuperheaterShader(mat: THREE.MeshStandardMaterial, u: CadShaderUniforms): void {
  mat.onBeforeCompile = (shader) => {
    shader.uniforms.uTime = u.uTime;
    shader.uniforms.uSteamFlow = u.uSteamFlow;
    shader.uniforms.uDomeDryness = u.uDomeDryness;
    shader.uniforms.uSuperheatNorm = u.uSuperheatNorm;
    shader.uniforms.uExhaustPulse = u.uExhaustPulse;

    shader.vertexShader =
      `attribute vec3 aCadPos;\nvarying vec3 vCadPos;\n` +
      shader.vertexShader.replace("#include <begin_vertex>", "#include <begin_vertex>\nvCadPos = aCadPos;");

    shader.fragmentShader =
      `uniform float uTime;\nuniform float uSteamFlow;\nuniform float uDomeDryness;\nuniform float uSuperheatNorm;\nuniform float uExhaustPulse;\nvarying vec3 vCadPos;\n` +
      shader.fragmentShader.replace(
        "#include <emissivemap_fragment>",
        /* glsl */ `
        #include <emissivemap_fragment>
        // 炉内干汽管 (Z>0.056) -> 17-5 双 U 四程过热器 -> 中央下垂管 & 7-9 横向内侧给汽歧管 -> 7-8/20-7 排汽喷嘴
        if (vCadPos.z > 0.056 && vCadPos.x < 0.111) {
          float wave = 0.5 + 0.5 * sin(vCadPos.x * 130.0 - uTime * (4.0 + 18.0 * uSteamFlow));
          vec3 wetCol = mix(vec3(0.18, 0.78, 1.0), vec3(0.82, 0.96, 1.0), uDomeDryness);
          totalEmissiveRadiance += wetCol * uSteamFlow * (0.18 + 0.36 * wave);
        } else if (vCadPos.x >= -0.026 && vCadPos.x <= 0.111 && vCadPos.z >= 0.041 && vCadPos.z <= 0.056 && abs(vCadPos.y) < 0.019) {
          float s = clamp((vCadPos.x + 0.025) / 0.132, 0.0, 1.0);
          float wave = 0.5 + 0.5 * sin(vCadPos.x * 165.0 - uTime * (6.0 + 22.0 * uSteamFlow));
          vec3 drySteam = vec3(0.85, 0.94, 1.0);
          vec3 superCol = vec3(1.0, 0.38, 0.06);
          vec3 shCol = mix(drySteam, superCol, clamp(s * (0.35 + 0.85 * uSuperheatNorm), 0.0, 1.0));
          float glow = (0.14 + 0.44 * uSuperheatNorm) * (0.45 + 0.55 * uSteamFlow * wave);
          totalEmissiveRadiance += shCol * glow;
        } else if (vCadPos.x >= 0.107 && vCadPos.x <= 0.128 && vCadPos.z >= 0.014 && vCadPos.z < 0.048) {
          // 17-5 中央下垂管 -> 17-4 -> 7-9 单根水平内壁给汽横歧管 (Y∈[-0.021,+0.021], Z=+0.0185)
          float pathCoord = vCadPos.x - vCadPos.z * 1.1 + abs(vCadPos.y) * 1.2;
          float wave = 0.5 + 0.5 * sin(pathCoord * 155.0 - uTime * (5.0 + 20.0 * uSteamFlow));
          vec3 delivCol = mix(vec3(1.0, 0.62, 0.18), vec3(1.0, 0.28, 0.04), uSuperheatNorm);
          totalEmissiveRadiance += delivCol * (0.14 + 0.50 * uSteamFlow * wave) * (0.35 + 0.65 * uSuperheatNorm);
        } else if (vCadPos.x >= 0.125 && vCadPos.x <= 0.136 && vCadPos.z >= 0.008 && vCadPos.z <= 0.036) {
          // 7-8 / 20-7 排汽横歧管与文丘里喷嘴: 随四拍乏汽脉冲喷射发亮
          float wave = 0.5 + 0.5 * sin((vCadPos.z - abs(vCadPos.y)) * 190.0 - uTime * 28.0);
          totalEmissiveRadiance += vec3(0.52, 0.86, 1.0) * uExhaustPulse * (0.28 + 0.48 * wave);
        }
        `
      );
  };
}

export function attachFeedwaterShader(mat: THREE.MeshStandardMaterial, u: CadShaderUniforms, isHose = false): void {
  mat.onBeforeCompile = (shader) => {
    shader.uniforms.uFeedPhase = u.uFeedPhase;
    shader.uniforms.uFeedStrength = u.uFeedStrength;

    shader.vertexShader =
      (isHose ? `varying vec3 vLocalHose;\n` : `attribute vec3 aCadPos;\nvarying vec3 vCadPos;\n`) +
      shader.vertexShader.replace(
        "#include <begin_vertex>",
        isHose ? "#include <begin_vertex>\nvLocalHose = position;" : "#include <begin_vertex>\nvCadPos = aCadPos;"
      );

    shader.fragmentShader =
      `uniform float uFeedPhase;\nuniform float uFeedStrength;\n` +
      (isHose ? `varying vec3 vLocalHose;\n` : `varying vec3 vCadPos;\n`) +
      shader.fragmentShader.replace(
        "#include <emissivemap_fragment>",
        isHose
          ? /* glsl */ `
            #include <emissivemap_fragment>
            if (uFeedStrength > 0.02) {
              float pulse = pow(0.5 + 0.5 * sin(vLocalHose.x * 165.0 - uFeedPhase * 3.2), 3.0);
              totalEmissiveRadiance += vec3(0.08, 0.85, 1.0) * pulse * uFeedStrength * 0.75;
            }
          `
          : /* glsl */ `
            #include <emissivemap_fragment>
            if (uFeedStrength > 0.02) {
              // 6-5 轴动给水泵出水管 6-6 向后 (-X) 经 6-8 立管 (+Z) 顶开 13-7 逆止阀入锅
              float sPath = -vCadPos.x * 165.0 + vCadPos.z * 165.0;
              float pulse = pow(0.5 + 0.5 * sin(sPath - uFeedPhase * 3.2), 3.2);
              totalEmissiveRadiance += vec3(0.06, 0.82, 1.0) * pulse * uFeedStrength * 0.78;
            }
          `
      );
  };
}

export function attachCopperBoilerShader(mat: THREE.MeshStandardMaterial, u: CadShaderUniforms): void {
  mat.onBeforeCompile = (shader) => {
    shader.uniforms.uCrownTemps = u.uCrownTemps;
    shader.uniforms.uFlueTemps = u.uFlueTemps;
    shader.uniforms.uFlueBowMm = u.uFlueBowMm;
    shader.uniforms.uCrownBulgeMm = u.uCrownBulgeMm;
    shader.uniforms.uXray = u.uXray;

    shader.vertexShader =
      `attribute vec3 aCadPos;\nuniform float uFlueBowMm[5];\nuniform float uCrownBulgeMm;\nuniform float uXray;\nvarying vec3 vCadPos;\n` +
      shader.vertexShader.replace(
        "#include <begin_vertex>",
        /* glsl */ `
        #include <begin_vertex>
        vCadPos = aCadPos;
        // 5 根紫铜烟管与火室顶板在高温干烧+蒸汽外压下的真实热弹塑性挠曲/鼓曲三维顶点形变
        float visAmp = mix(1.0, 2.5, uXray);
        if (aCadPos.x > -0.047 && aCadPos.x < 0.092 && aCadPos.z > 0.028 && aCadPos.z < 0.053 && abs(aCadPos.y) < 0.018) {
          float xi = clamp((aCadPos.x + 0.047) / 0.139, 0.0, 1.0);
          float mode = pow(sin(3.14159265 * xi), 1.25);
          float bowMm = uFlueBowMm[3];
          if (aCadPos.z > 0.041) {
            bowMm = aCadPos.y < 0.0 ? uFlueBowMm[0] : uFlueBowMm[1];
          } else if (aCadPos.y < -0.006) {
            bowMm = uFlueBowMm[2];
          } else if (aCadPos.y > 0.006) {
            bowMm = uFlueBowMm[4];
          }
          transformed.z -= 0.001 * bowMm * mode * visAmp;
        } else if (aCadPos.x >= -0.106 && aCadPos.x <= -0.048 && aCadPos.z >= 0.045 && aCadPos.z <= 0.050 && abs(aCadPos.y) <= 0.018) {
          float cx = 1.0 - pow((aCadPos.x + 0.077) / 0.029, 2.0);
          float cy = 1.0 - pow(aCadPos.y / 0.018, 2.0);
          transformed.z -= 0.001 * uCrownBulgeMm * max(0.0, cx * cy) * visAmp;
        }
        `
      );

    shader.fragmentShader =
      `uniform float uCrownTemps[6];\nuniform float uFlueTemps[8];\nuniform float uXray;\nvarying vec3 vCadPos;\n` +
      shader.fragmentShader.replace(
        "#include <emissivemap_fragment>",
        /* glsl */ `
        #include <emissivemap_fragment>
        // 1. 火室紫铜顶板 6 分区轴向局部干烧黑体红热发光 (X ∈ [-0.107, -0.046], Z ∈ [0.043, 0.051])
        if (vCadPos.x >= -0.108 && vCadPos.x <= -0.045 && vCadPos.z >= 0.042 && vCadPos.z <= 0.052 && abs(vCadPos.y) <= 0.022) {
          float zf = clamp((vCadPos.x + 0.1065) / 0.0595, 0.0, 1.0) * 5.0;
          int idx = int(floor(zf));
          float tC = uCrownTemps[0];
          if (idx == 1) tC = mix(uCrownTemps[1], uCrownTemps[2], fract(zf));
          else if (idx == 2) tC = mix(uCrownTemps[2], uCrownTemps[3], fract(zf));
          else if (idx == 3) tC = mix(uCrownTemps[3], uCrownTemps[4], fract(zf));
          else if (idx >= 4) tC = mix(uCrownTemps[4], uCrownTemps[5], fract(zf));
          else tC = mix(uCrownTemps[0], uCrownTemps[1], fract(zf));

          float glow = clamp((tC - 215.0) / 260.0, 0.0, 1.25);
          vec3 blackbody = mix(vec3(0.95, 0.14, 0.02), vec3(1.0, 0.62, 0.14), clamp(glow - 0.35, 0.0, 1.0));
          totalEmissiveRadiance += blackbody * glow * 1.15;
        }
        // 2. X-Ray 模式下显示 5 根烟管自火室向烟室递减的高温换热微光
        if (uXray > 0.5 && vCadPos.x > -0.047 && vCadPos.x <= 0.091 && vCadPos.z >= 0.029 && vCadPos.z <= 0.053 && abs(vCadPos.y) <= 0.018) {
          float sf = clamp((vCadPos.x + 0.047) / 0.138, 0.0, 1.0) * 7.0;
          int fi = int(floor(sf));
          float tf = uFlueTemps[0];
          if (fi == 1) tf = mix(uFlueTemps[1], uFlueTemps[2], fract(sf));
          else if (fi == 2) tf = mix(uFlueTemps[2], uFlueTemps[3], fract(sf));
          else if (fi == 3) tf = mix(uFlueTemps[3], uFlueTemps[4], fract(sf));
          else if (fi == 4) tf = mix(uFlueTemps[4], uFlueTemps[5], fract(sf));
          else if (fi == 5) tf = mix(uFlueTemps[5], uFlueTemps[6], fract(sf));
          else if (fi >= 6) tf = mix(uFlueTemps[6], uFlueTemps[7], fract(sf));
          else tf = mix(uFlueTemps[0], uFlueTemps[1], fract(sf));

          float fGlow = clamp((tf - 175.0) / 420.0, 0.0, 0.85);
          totalEmissiveRadiance += vec3(1.0, 0.36, 0.06) * fGlow * 0.55;
        }
        `
      );
  };
}
