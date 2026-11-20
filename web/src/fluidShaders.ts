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

interface InternalPipeRuntime {
  id: string;
  partNo: string;
  title: string;
  zone: "tender" | "umbilical" | "cab" | "boiler" | "underframe" | "cylinder";
  frame: "loco" | "tender" | "umbilical";
  curve: THREE.CatmullRomCurve3;
  umbEndsMm?: { tdr: [number, number, number]; loco: [number, number, number] };
  radiusM: number;
  mesh: THREE.Mesh;
  mat: THREE.ShaderMaterial;
  arrowCount: number;
  phase: number;
  labelT: number;
  colorA: THREE.Color;
  colorB: THREE.Color;
  colorCss: string;
  evalTempC: (out: StepOutput, sp: SpatialFluidState) => number;
  evalFlow: (out: StepOutput, sp: SpatialFluidState) => {
    signedRate: number; // >0 along curvepts[0]->[-1], <0 reversed!
    fill: number;       // 0..1 lumen presence
    unit: string;
    colOverride?: [number, number, number];
  };
}

const ARROW_BASE_AXIS = new THREE.Vector3(0, 1, 0);

function mmPtsToCurve(ptsMm: [number, number, number][]): THREE.CatmullRomCurve3 {
  const v = ptsMm.map(([x, y, z]) => new THREE.Vector3(x * 0.001, y * 0.001, z * 0.001));
  return new THREE.CatmullRomCurve3(v, false, "catmullrom", 0.18);
}

function createPipeLumenMaterial(colA: THREE.Color, colB: THREE.Color): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    uniforms: {
      uPhase: { value: 0.0 },
      uFill: { value: 0.8 },
      uActive: { value: 0.0 },
      uColA: { value: colA.clone() },
      uColB: { value: colB.clone() },
    },
    vertexShader: /* glsl */ `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      uniform float uPhase;
      uniform float uFill;
      uniform float uActive;
      uniform vec3 uColA;
      uniform vec3 uColB;
      varying vec2 vUv;
      void main() {
        if (uFill < 0.02) discard;
        // 流动液团/气团脉动条纹: 严格沿 vUv.x 方向以 uPhase 相位推进 (正负速度自动反转!)
        float wave = 0.5 + 0.5 * sin(vUv.x * 37.699 - uPhase);
        float bolus = mix(0.45, pow(wave, 1.8), clamp(uActive, 0.0, 1.0));
        vec3 col = mix(uColA, uColB, bolus);
        float alpha = uFill * mix(0.28, 0.52 + 0.40 * bolus, clamp(uActive, 0.0, 1.0));
        gl_FragColor = vec4(col, clamp(alpha, 0.0, 0.94));
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
  private locoArrowInst!: THREE.InstancedMesh;
  private tenderArrowInst!: THREE.InstancedMesh;
  private worldArrowInst!: THREE.InstancedMesh;
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
    // 1. 23-1 四头酒精燃烧器三维湍流反应火焰 (X=[-0.089, -0.065], Y=[-0.0058, +0.0058], Z=0.0188)
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
        uTiltRad: { value: 0.25 },
        uIntensity: { value: 1.0 },
      },
      vertexShader: /* glsl */ `
        uniform float uTime;
        uniform float uTiltRad;
        varying vec2 vUv;
        varying float vSeed;
        attribute float aSeed;
        void main() {
          vUv = uv;
          vSeed = aSeed;
          vec3 p = position;
          float h = clamp(uv.y, 0.0, 1.0);
          // 烟室文丘里负压将火焰向 +X 方向 (5根烟管入口 x=-0.047m) 牵引倾伏
          float forwardBend = pow(h, 1.32) * sin(uTiltRad) * 0.024;
          float flutterX = 0.0016 * h * sin(uTime * 23.0 + h * 9.0 + aSeed * 6.28);
          float flutterY = 0.0012 * h * cos(uTime * 19.0 + h * 7.5 + aSeed * 4.13);
          p.x += forwardBend + flutterX;
          p.y += flutterY;
          p.z *= (0.72 + 0.34 * cos(uTiltRad * 0.6));
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
          float n1 = noise2(vec2(vUv.x * 5.0 + vSeed * 3.1, h * 6.5 - uTime * 9.5));
          float n2 = noise2(vec2(vUv.x * 11.0 - vSeed * 1.7, h * 13.0 - uTime * 15.0));
          float turb = 0.65 * n1 + 0.35 * n2;

          // 火焰外轮廓: 底部贴合 ø10mm 陶瓷芯，向上收拢并随湍流脉动
          float halfWidth = mix(0.76, 0.12, pow(h, 0.85)) + 0.24 * (turb - 0.5) * h;
          float body = smoothstep(halfWidth, halfWidth * 0.22, r) * smoothstep(0.0, 0.10, h) * smoothstep(1.0, 0.38, h);
          if (body < 0.015) discard;

          // 乙醇火焰光谱: 根部预混反应区呈醇蓝紫，外焰随当量比 φ 由贫燃清澈蓝青过渡至富燃明亮橙黄
          float coreMask = smoothstep(0.52, 0.05, h) * smoothstep(0.55, 0.08, r);
          float richFrac = clamp((uPhi - 0.78) / 0.58, 0.0, 1.0);
          vec3 leanOuter = vec3(0.22, 0.62, 1.00);
          vec3 richOuter = vec3(1.00, 0.48, 0.09);
          vec3 tipGlow   = vec3(1.00, 0.82, 0.32);
          vec3 innerBlue = vec3(0.18, 0.46, 1.00);

          vec3 outerCol = mix(leanOuter, richOuter, richFrac);
          outerCol = mix(outerCol, tipGlow, smoothstep(0.35, 0.88, h) * richFrac * 0.65);
          vec3 col = mix(outerCol, innerBlue, coreMask * (1.0 - 0.35 * richFrac));

          float alpha = body * (0.55 + 0.45 * uIntensity);
          gl_FragColor = vec4(col * (1.25 + 0.55 * turb), clamp(alpha, 0.0, 0.92));
        }
      `,
    });

    // 23-1 纵向单列 1×3 三头立式 ø10mm 燃烧器火焰 (X = [-93, -77, -61] mm, Y = 0)
    const wickCoords: [number, number, number][] = [
      [-0.093, 0.0, 0.0185],
      [-0.077, 0.0, 0.0185],
      [-0.061, 0.0, 0.0185],
    ];
    wickCoords.forEach(([wx, wy, wz], idx) => {
      for (const ang of [0.0, Math.PI * 0.5]) {
        const g = new THREE.PlaneGeometry(0.0135, 0.0285, 6, 14);
        g.rotateX(Math.PI * 0.5);
        g.rotateZ(ang);
        g.translate(wx, wy, wz + 0.01425);
        const seeds = new Float32Array(g.attributes.position.count).fill(idx * 0.27 + (ang > 0.1 ? 0.13 : 0.0));
        g.setAttribute("aSeed", new THREE.BufferAttribute(seeds, 1));
        const m = new THREE.Mesh(g, this.flameMat);
        m.renderOrder = 4;
        this.wickFlames.push(m);
        this.root.add(m);
      }
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
          // 若水位极低低于火室顶板 (z < 0.0475 && x < -0.047)，扣除中央火室燃气腔
          if (vLocalPos.x < -0.0468 && vLocalPos.z < 0.0475 && abs(vLocalPos.y) < 0.0195) discard;

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
          if (vLocalPos.x < -0.0468 && vLocalPos.z < 0.0475 && abs(vLocalPos.y) < 0.0195) discard;

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
          if (vLocalPos.x < -0.0468 && vLocalPos.z < 0.0475 && abs(vLocalPos.y) < 0.0195) discard;

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
      mesh.position.set(0.126, yCenter, 0.0);
      mesh.renderOrder = 4;
      mesh.visible = false;
      this.root.add(mesh);
      return mesh;
    };

    this.cylHeadMeshes = [makeCylChamberMesh(+0.0360), makeCylChamberMesh(-0.0360)];
    this.cylCrankMeshes = [makeCylChamberMesh(+0.0360), makeCylChamberMesh(-0.0360)];

    const makeCondensateMesh = (yCenter: number): THREE.Mesh => {
      const g = new THREE.BoxGeometry(0.025, 0.0085, 0.0026);
      const m = new THREE.MeshBasicMaterial({
        color: 0x1fd2ff,
        transparent: true,
        opacity: 0.0,
        depthWrite: false,
      });
      const mesh = new THREE.Mesh(g, m);
      mesh.position.set(0.126, yCenter, -0.0056);
      mesh.renderOrder = 4;
      mesh.visible = false;
      this.root.add(mesh);
      return mesh;
    };
    this.cylCondMeshes = [makeCondensateMesh(+0.0360), makeCondensateMesh(-0.0360)];

    const makePortGlow = (xPos: number, yPos: number): THREE.Mesh => {
      const g = new THREE.BoxGeometry(0.0024, 0.0062, 0.0075);
      const m = new THREE.MeshBasicMaterial({
        color: 0xffaa44,
        transparent: true,
        opacity: 0.0,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });
      const mesh = new THREE.Mesh(g, m);
      mesh.position.set(xPos, yPos, 0.0122);
      mesh.renderOrder = 4;
      mesh.visible = false;
      this.root.add(mesh);
      return mesh;
    };
    this.portGlowMeshes = [
      makePortGlow(0.1335, +0.0360), // L Head port
      makePortGlow(0.1185, +0.0360), // L Crank port
      makePortGlow(0.1335, -0.0360), // R Head port
      makePortGlow(0.1185, -0.0360), // R Crank port
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

      // === C. 机车底盘给水泵、燃烧器、注油器与烟管流道 (8 路) ===
      {
        id: "loco_23_1_burner",
        partNo: "23-1",
        title: "バーナー供醇本管→3連芯カップ毛細吸上",
        zone: "underframe",
        frame: "loco",
        ptsMm: [[-150.5, 0.0, 3.6], [-132.0, 0.0, 10.2], [-93.0, 0.0, 10.2], [-77.0, 0.0, 10.2], [-61.0, 0.0, 10.2], [-61.0, 0.0, 17.5]],
        radiusMm: 1.25,
        arrows: 5,
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
        radiusMm: 0.78,
        arrows: 5,
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
        radiusMm: 0.78,
        arrows: 5,
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
        radiusMm: 0.78,
        arrows: 3,
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
        radiusMm: 0.78,
        arrows: 3,
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
        radiusMm: 0.62,
        arrows: 4,
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
        title: "上段ø11.5過熱大煙管×2 高温燃焼ガス流",
        zone: "boiler",
        frame: "loco",
        ptsMm: [[-48.0, 11.5, 46.5], [22.0, 11.5, 46.5], [92.0, 11.5, 46.5]],
        radiusMm: 2.10,
        arrows: 4,
        labelT: 0.45,
        colHexA: 0xea580c,
        colHexB: 0xfef08a,
        css: "#fb923c",
        evalFlow: (out, sp) => ({
          signedRate: sp.flueGasProfileC[0] > 60.0 || out.smokeboxVacuumPa > 4.0 ? Math.max(0.05, out.smokeboxVacuumPa * 0.018) : 0.0,
          fill: sp.flueGasProfileC[0] > 60.0 ? 0.55 : 0.0,
          unit: "g/s",
        }),
      },
      {
        id: "boil_flues_lower",
        partNo: "12-6(下3火管)",
        title: "下段ø8.0火管×3 燃焼ガス熱伝達流",
        zone: "boiler",
        frame: "loco",
        ptsMm: [[-48.0, 0.0, 33.2], [22.0, 0.0, 33.2], [92.0, 0.0, 33.2]],
        radiusMm: 1.75,
        arrows: 4,
        labelT: 0.55,
        colHexA: 0xdc2626,
        colHexB: 0xfdba74,
        css: "#f97316",
        evalFlow: (out, sp) => ({
          signedRate: sp.flueGasProfileC[0] > 60.0 || out.smokeboxVacuumPa > 4.0 ? Math.max(0.04, out.smokeboxVacuumPa * 0.015) : 0.0,
          fill: sp.flueGasProfileC[0] > 60.0 ? 0.52 : 0.0,
          unit: "g/s",
        }),
      },

      // === D. 锅炉—驾驶室跨界连续管网与四程过热器 (9 路) ===
      {
        id: "boil_13_4_11_15_turret",
        partNo: "13-4→11-15",
        title: "缶頂主蒸気管→縦長蒸気分配器(18mm)",
        zone: "cab",
        frame: "loco",
        ptsMm: [[-98.5, 0.0, 70.5], [-98.5, 0.0, 72.4], [-118.5, 0.0, 72.4], [-118.5, 8.2, 72.4], [-118.5, 8.2, 66.5], [-130.0, 8.2, 66.5]],
        radiusMm: 0.90,
        arrows: 4,
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
        ptsMm: [[-124.5, 8.2, 66.5], [-124.5, -3.6, 66.5], [-124.5, -3.6, 48.0], [-112.8, -3.6, 48.0], [-105.0, -3.6, 54.5], [-96.0, 0.0, 58.8], [0.0, 0.0, 58.8], [96.5, 0.0, 58.8], [109.5, 0.0, 58.8]],
        radiusMm: 1.05,
        arrows: 6,
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
        partNo: "17-5→17-4",
        title: "双U四程スーパーヒータ→過熱蒸気降下管",
        zone: "boiler",
        frame: "loco",
        ptsMm: [[109.5, 0.0, 58.8], [109.5, 11.5, 48.8], [-24.0, 11.5, 48.8], [-24.0, 11.5, 44.2], [109.5, 11.5, 44.2], [109.5, 0.0, 44.2], [109.5, 0.0, 26.5], [119.5, 0.0, 21.8], [123.5, 0.0, 18.5]],
        radiusMm: 0.92,
        arrows: 6,
        labelT: 0.80,
        colHexA: 0xfb923c,
        colHexB: 0xfef08a,
        css: "#fb923c",
        evalFlow: (out) => {
          const regFlow = Math.max(0.0, (out.steamOutGS ?? out.steamGenGS) - (out.blowerGS ?? 0) - out.whistleSteamGS - (out.safetyGS ?? 0));
          return { signedRate: out.steamChestKgf > 0.02 ? Math.max(0.03, regFlow) : 0.0, fill: out.steamChestKgf > 0.02 ? 0.94 : 0.18, unit: "g/s" };
        },
      },
      {
        id: "boil_13_8_17_3_blower",
        partNo: "13-8→17-3",
        title: "通風弁→後妻板同軸貫通ステー管→煙室ノズル",
        zone: "boiler",
        frame: "loco",
        ptsMm: [[-128.0, 8.2, 63.2], [-128.0, 8.2, 54.5], [-120.5, 2.6, 48.0], [-112.8, 2.6, 48.0], [-106.0, 2.6, 48.8], [-44.0, 2.6, 48.8], [86.0, 4.0, 40.5], [96.2, 4.0, 40.5], [96.2, 4.0, 29.5], [125.0, 3.6, 29.5], [130.0, 3.6, 24.8], [130.0, 3.6, 35.2]],
        radiusMm: 0.62,
        arrows: 6,
        labelT: 0.15,
        colHexA: 0x67e8f9,
        colHexB: 0xffffff,
        css: "#67e8f9",
        evalFlow: (out) => ({
          signedRate: out.blowerGS ?? 0.0,
          fill: (out.blowerGS ?? 0.0) > 0.002 ? 0.92 : 0.16,
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
        radiusMm: 0.65,
        arrows: 4,
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
        radiusMm: 0.58,
        arrows: 3,
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
        radiusMm: 0.72,
        arrows: 3,
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
        radiusMm: 0.76,
        arrows: 4,
        labelT: 0.68,
        colHexA: 0x0ea5e9,
        colHexB: 0x67e8f9,
        css: "#38bdf8",
        evalFlow: (out, sp) => {
          const vMmS = out.sightGlassVelMmS ?? 0.0;
          const purge = out.sightPurgeGS > 0.002 ? 0.45 + out.sightPurgeGS : 0.0;
          // 液面上升 (vMmS>0) 或吹扫冲洗时水从湿锅筒 (-43.2) 流向 11-7 (+); 液面蒸发下降 (vMmS<0) 时水反向回流锅筒 (-)!
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
        radiusMm: 0.72,
        arrows: 4,
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

      // === E. 左右主气缸给汽支管、四条可逆进排汽斜道、排汽总管与四口排液阀 (8 路) ===
      {
        id: "cyl_7_9_inlet_L",
        partNo: "7-9→1-20(L)",
        title: "左シリンダ蒸気室・含油過熱給気管",
        zone: "cylinder",
        frame: "loco",
        ptsMm: [[123.5, 0.0, 18.5], [123.5, 20.5, 18.5], [123.5, 34.5, 18.5]],
        radiusMm: 1.35,
        arrows: 3,
        labelT: 0.65,
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
        title: "右シリンダ蒸気室・含油過熱給気管",
        zone: "cylinder",
        frame: "loco",
        ptsMm: [[123.5, 0.0, 18.5], [123.5, -20.5, 18.5], [123.5, -34.5, 18.5]],
        radiusMm: 1.35,
        arrows: 3,
        labelT: 0.65,
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
        partNo: "1-9(左前気道)",
        title: "左前ポート (滑弁→ピストン給気 / ピストン→滑弁排気 可逆)",
        zone: "cylinder",
        frame: "loco",
        ptsMm: [[133.8, 34.7, 15.6], [137.2, 33.2, 8.8], [138.6, 32.5, 1.5]],
        radiusMm: 0.76,
        arrows: 3,
        labelT: 0.50,
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
        partNo: "1-9(左後気道)",
        title: "左後ポート (滑弁→ピストン給気 / ピストン→滑弁排気 可逆)",
        zone: "cylinder",
        frame: "loco",
        ptsMm: [[126.2, 34.7, 15.6], [119.2, 33.2, 8.8], [113.8, 32.5, 1.5]],
        radiusMm: 0.76,
        arrows: 3,
        labelT: 0.50,
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
        partNo: "1-9(右前気道)",
        title: "右前ポート (滑弁→ピストン給気 / ピストン→滑弁排気 可逆)",
        zone: "cylinder",
        frame: "loco",
        ptsMm: [[133.8, -34.7, 15.6], [137.2, -33.2, 8.8], [138.6, -32.5, 1.5]],
        radiusMm: 0.76,
        arrows: 3,
        labelT: 0.50,
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
        partNo: "1-9(右後気道)",
        title: "右後ポート (滑弁→ピストン給気 / ピストン→滑弁排気 可逆)",
        zone: "cylinder",
        frame: "loco",
        ptsMm: [[126.2, -34.7, 15.6], [119.2, -33.2, 8.8], [113.8, -32.5, 1.5]],
        radiusMm: 0.76,
        arrows: 3,
        labelT: 0.50,
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
        title: "左シリンダ排気内孔→7-8合流管→20-7ブラストノズル(ø4.2)",
        zone: "cylinder",
        frame: "loco",
        ptsMm: [[130.0, 34.5, 15.6], [130.0, 27.5, 13.0], [130.0, 21.1, 12.5], [130.0, 0.0, 12.5], [130.0, 0.0, 35.0]],
        radiusMm: 1.25,
        arrows: 4,
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
        partNo: "1-9(R)→7-8",
        title: "右シリンダ中央排気腔→内壁ソケット→7-8排気合流管",
        zone: "cylinder",
        frame: "loco",
        ptsMm: [[130.0, -34.5, 15.6], [130.0, -27.5, 13.0], [130.0, -21.1, 12.5], [130.0, 0.0, 12.5]],
        radiusMm: 1.22,
        arrows: 3,
        labelT: 0.45,
        colHexA: 0xeab308,
        colHexB: 0xfde047,
        css: "#eab308",
        evalFlow: (out) => ({
          signedRate: (out.exhaustGS ?? out.exhaustPlume * 0.35) * 0.5,
          fill: (out.exhaustGS ?? 0) > 0.005 || out.exhaustPlume > 0.04 ? 0.92 : 0.18,
          unit: "g/s",
        }),
      },
      {
        id: "cyl_1_16_drains",
        partNo: "1-16×4",
        title: "シリンダ底ばねポペット二相フラッシュドレン弁",
        zone: "cylinder",
        frame: "loco",
        ptsMm: [[130.8, -30.8, -6.2], [130.8, -30.8, -15.5]],
        radiusMm: 0.65,
        arrows: 2,
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

    let nLocoArrows = 0;
    let nTdrArrows = 0;
    let nWorldArrows = 0;

    const makePipeTempEvaluator = (id: string): InternalPipeRuntime["evalTempC"] => {
      return (out, sp) => {
        if (id === "tdr_36_6_mariotte" || id === "tdr_34_2_fuel" || id === "umb_st7_fuel") {
          return out.alcoholTempC ?? 28.0;
        }
        if (id === "loco_23_1_burner") {
          return Math.min(72.0, (out.alcoholTempC ?? 30.0) + 12.0);
        }
        if (id === "tdr_31_5_31_13_hp" || id === "tdr_34_6_feed_R" || id === "umb_bt5_feed" || id === "loco_6_10_suction" || id === "tdr_31_9_drain") {
          return out.tenderWaterTempC;
        }
        if (id === "loco_6_5_6_6_deliv") {
          return out.tenderWaterTempC + 0.06 * Math.max(0.0, out.boilerTempC - out.tenderWaterTempC);
        }
        if (id === "loco_4_16_byp_ret" || id === "umb_bt5_return" || id === "tdr_34_6_ret_L" || id === "tdr_31_14_gooseneck") {
          return out.tenderWaterTempC + 0.10 * Math.max(0.0, out.boilerTempC - out.tenderWaterTempC);
        }
        if (id === "loco_6_8_13_2_riser") {
          return 0.55 * out.tenderWaterTempC + 0.45 * out.boilerTempC;
        }
        if (id === "loco_4_2_lub_oil") {
          return out.lubeOilTempC ?? Math.max(32.0, 0.62 * out.cylWallTempC);
        }
        if (id === "boil_flues_upper") {
          return Math.max(out.boilerTempC, 0.55 * (sp.flueGasProfileC[1] ?? out.boilerTempC) + 0.45 * (sp.flueGasProfileC[5] ?? out.boilerTempC));
        }
        if (id === "boil_flues_lower") {
          return Math.max(out.boilerTempC, 0.48 * (sp.flueGasProfileC[2] ?? out.boilerTempC) + 0.52 * (sp.flueGasProfileC[6] ?? out.boilerTempC));
        }
        if (id === "boil_17_5_sh_4pass" || id === "cyl_7_9_inlet_L" || id === "cyl_7_9_inlet_R") {
          return out.superheatTempC;
        }
        if (id === "boil_13_1_siphon") {
          return 25.0 + 0.35 * Math.max(0.0, out.boilerTempC - 25.0);
        }
        if (id === "boil_13_10_sg_water") {
          return 25.0 + 0.72 * Math.max(0.0, out.boilerTempC - 25.0);
        }
        if (id === "cyl_port_L_head") {
          return (sp.valvePortOpenL[0] ?? 0) >= 0 ? out.superheatTempC : Math.max(28.0, Math.min(112.0, 0.78 * out.cylWallTempC + 8.0));
        }
        if (id === "cyl_port_L_crank") {
          return (sp.valvePortOpenL[1] ?? 0) >= 0 ? out.superheatTempC : Math.max(28.0, Math.min(112.0, 0.78 * out.cylWallTempC + 8.0));
        }
        if (id === "cyl_port_R_head") {
          return (sp.valvePortOpenR[0] ?? 0) >= 0 ? out.superheatTempC : Math.max(28.0, Math.min(112.0, 0.78 * out.cylWallTempC + 8.0));
        }
        if (id === "cyl_port_R_crank") {
          return (sp.valvePortOpenR[1] ?? 0) >= 0 ? out.superheatTempC : Math.max(28.0, Math.min(112.0, 0.78 * out.cylWallTempC + 8.0));
        }
        if (id === "cyl_7_8_20_7_blast" || id === "cyl_7_8_exhaust_R") {
          return out.boilerPressureKgf > 0.05 ? Math.max(98.0, Math.min(118.0, 0.76 * out.cylWallTempC + 10.0)) : out.cylWallTempC;
        }
        if (id === "cyl_1_16_drains") {
          return Math.max(28.0, Math.min(104.0, 0.75 * out.cylWallTempC));
        }
        return out.boilerTempC;
      };
    };

    for (const s of specs) {
      const colA = new THREE.Color(s.colHexA);
      const colB = new THREE.Color(s.colHexB);
      const mat = createPipeLumenMaterial(colA, colB);
      const curve = s.frame === "umbilical"
        ? new THREE.CatmullRomCurve3(s.ptsMm.map(([x, y, z]) => new THREE.Vector3(x, y, z)))
        : mmPtsToCurve(s.ptsMm);
      const segs = Math.max(16, s.ptsMm.length * 10);
      const geom = new THREE.TubeGeometry(curve, segs, s.radiusMm * 0.001, 10, false);
      const mesh = new THREE.Mesh(geom, mat);
      mesh.renderOrder = 6;
      mesh.visible = false;
      if (s.frame === "loco") {
        this.root.add(mesh);
        nLocoArrows += s.arrows;
      } else if (s.frame === "tender") {
        this.tenderRoot.add(mesh);
        nTdrArrows += s.arrows;
      } else {
        this.worldRoot.add(mesh);
        nWorldArrows += s.arrows;
      }
      this.pipeRuntimes.push({
        id: s.id,
        partNo: s.partNo,
        title: s.title,
        zone: s.zone,
        frame: s.frame,
        curve,
        umbEndsMm: s.umbEndsMm,
        radiusM: s.radiusMm * 0.001,
        mesh,
        mat,
        arrowCount: s.arrows,
        phase: Math.random() * 6.28,
        labelT: s.labelT,
        colorA: colA,
        colorB: colB,
        colorCss: s.css,
        evalTempC: makePipeTempEvaluator(s.id),
        evalFlow: s.evalFlow,
      });
    }

    // 额外补充右排气支臂 + 23-1 前两芯垂直立孔内芯 (纯几何同步跟随主通道)
    const arrowGeom = new THREE.ConeGeometry(0.00185, 0.0052, 12);
    const arrowMat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.95,
      depthTest: false,
      depthWrite: false,
    });
    const makeArrowInst = (count: number, parent: THREE.Group): THREE.InstancedMesh => {
      const inst = new THREE.InstancedMesh(arrowGeom, arrowMat, Math.max(1, count));
      inst.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      inst.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, count) * 3), 3);
      inst.frustumCulled = false;
      inst.renderOrder = 9;
      inst.visible = false;
      parent.add(inst);
      return inst;
    };
    this.locoArrowInst = makeArrowInst(nLocoArrows, this.root);
    this.tenderArrowInst = makeArrowInst(nTdrArrows, this.tenderRoot);
    this.worldArrowInst = makeArrowInst(nWorldArrows, this.worldRoot);
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
      // 极低水位 (<37.5mm): 仅在锅底残水与火室两侧浸没水腿底部沸腾核化
      x = -0.095 + Math.random() * 0.175;
      y = x < -0.046 ? (Math.random() < 0.5 ? -0.0205 : +0.0205) : (Math.random() - 0.5) * 0.016;
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

    // 1. 更新 23-1 直列 1×3 三头酒精火焰 (严格受三维管路连通性与陶瓷灯芯毛细吸醇速率物理门控!)
    const lumenOk = sp.burnerLumenConnected !== false;
    const wickFlowOk = sp.wickCapillaryMlS === undefined || sp.wickCapillaryMlS > 0.002;
    const litVal = burnerLit && lumenOk && wickFlowOk && sp.airFuelPhi > 0.05 ? 1.0 : 0.0;
    this.flameMat.uniforms.uTime.value = this.elapsed;
    this.flameMat.uniforms.uLit.value = litVal;
    this.flameMat.uniforms.uPhi.value = sp.airFuelPhi || 1.0;
    this.flameMat.uniforms.uTiltRad.value = (sp.flameTiltDeg * Math.PI) / 180.0;
    this.flameMat.uniforms.uIntensity.value = Math.min(1.2, 0.55 + out.smokeboxVacuumPa / 140.0);

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
    const blastJetStrength = Math.min(
      1.0,
      out.exhaustPlume * 1.15 + ((out.blowerGS ?? 0) > 0.002 ? Math.min(0.85, (out.blowerGS ?? 0) * 8.0) : 0.0)
    );
    this.waterSurfaceMesh.visible = xray && hasBoilerWater;
    this.waterVolumeMesh.visible = xray && hasBoilerWater;
    this.steamHeadspaceMesh.visible = xray && (out.boilerPressureKgf > 0.03 || sp.boilingIntensity > 0.03);
    for (const bm of this.smokeboxBlastMeshes) {
      bm.visible = xray && blastJetStrength > 0.02;
    }
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
      pr.mesh.visible = pipesVisible && (pr.frame === "loco" || hasTdr);
    }
    this.locoArrowInst.visible = showFlowArrows;
    this.tenderArrowInst.visible = showFlowArrows && hasTdr;
    this.worldArrowInst.visible = showFlowArrows && hasTdr;

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
    const xHeadWall = 0.1392;
    const xCrankWall = 0.1128;
    const xPistL = 0.1260 + 0.5 * calib.stroke - sL;
    const xPistR = 0.1260 + 0.5 * calib.stroke - sR;

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
    let iLoco = 0;
    let iTdr = 0;
    let iWorld = 0;
    const hasTdr = tenderMatrix !== null;
    this.calloutSnapshots.length = 0;

    for (const pr of this.pipeRuntimes) {
      if ((pr.frame === "tender" || pr.frame === "umbilical") && !hasTdr) {
        continue;
      }

      // 动态更新跨车三平行软管曲线端点 (严格跟随牵引杆与 R2m 过弯姿态!)
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
        pr.curve.points[0].copy(pStart);
        pr.curve.points[1].lerpVectors(pStart, pEnd, 0.5).setZ(0.5 * (pStart.z + pEnd.z) - 0.0018);
        pr.curve.points[2].copy(pEnd);
        pr.mesh.geometry.dispose();
        pr.mesh.geometry = new THREE.TubeGeometry(pr.curve, 18, pr.radiusM, 8, false);
      }

      const st = pr.evalFlow(out, sp);
      const pipeTempC = pr.evalTempC(out, sp);
      temperatureToThermalColor(pipeTempC - 12.0, pr.colorA);
      temperatureToThermalColor(pipeTempC + 10.0, pr.colorB);
      const dynCss = temperatureToThermalCss(pipeTempC);
      const lvsG = sp.edgeConductances?.[pr.id] ?? 1.0;
      const lvsOpen = lvsG > 0.02;
      const effSigned = lvsOpen ? st.signedRate : 0.0;
      const absRate = Math.abs(effSigned);
      const signDir = effSigned >= 0 ? 1.0 : -1.0;
      const isFlowing = absRate > 0.0025;
      const speed = isFlowing ? signDir * (2.6 + Math.min(18.0, absRate * 14.0)) : 0.0;
      pr.phase = (pr.phase + dt * speed) % 628.318;

      pr.mat.uniforms.uPhase.value = pr.phase;
      pr.mat.uniforms.uFill.value = lvsOpen ? st.fill : 0.08;
      pr.mat.uniforms.uActive.value = isFlowing ? Math.min(1.0, 0.38 + absRate * 2.5) : 0.0;
      pr.mat.uniforms.uColA.value.copy(pr.colorA);
      if (st.colOverride) {
        pr.mat.uniforms.uColB.value.setRGB(st.colOverride[0], st.colOverride[1], st.colOverride[2]);
      } else {
        pr.mat.uniforms.uColB.value.copy(pr.colorB);
      }

      if (showFlowArrows) {
        const inst =
          pr.frame === "loco" ? this.locoArrowInst : pr.frame === "tender" ? this.tenderArrowInst : this.worldArrowInst;
        for (let k = 0; k < pr.arrowCount; k++) {
          const idx = pr.frame === "loco" ? iLoco++ : pr.frame === "tender" ? iTdr++ : iWorld++;
          if (!isFlowing || st.fill < 0.05) {
            this.tmpMat.makeScale(0, 0, 0);
            inst.setMatrixAt(idx, this.tmpMat);
            continue;
          }
          const rawS = ((k + 0.5) / pr.arrowCount + (pr.phase / 18.84955)) % 1.0;
          const u = Math.max(0.04, Math.min(0.96, rawS < 0 ? rawS + 1.0 : rawS));
          pr.curve.getPointAt(u, this.tmpVecA);
          pr.curve.getTangentAt(u, this.tmpTan);
          if (signDir < 0) {
            this.tmpTan.negate();
          }
          this.tmpQuat.setFromUnitVectors(ARROW_BASE_AXIS, this.tmpTan.normalize());
          const sc = 0.85 + 0.35 * Math.min(1.0, absRate * 2.2);
          this.tmpScale.set(sc, sc * 1.15, sc);
          this.tmpMat.compose(this.tmpVecA, this.tmpQuat, this.tmpScale);
          inst.setMatrixAt(idx, this.tmpMat);
          inst.setColorAt(idx, st.colOverride ? pr.mat.uniforms.uColB.value : pr.colorB);
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
        const dirSymbol = !lvsOpen ? "⛔" : !isFlowing ? "⏸" : signDir > 0 ? "→" : "←";
        const valStr = !lvsOpen
          ? "LVS断流"
          : st.unit === "給気" || st.unit === "排気"
            ? `${pipeTempC.toFixed(0)}°C · ${st.unit} ${(absRate * 100).toFixed(0)}%`
            : `${pipeTempC.toFixed(0)}°C · ${absRate.toFixed(2)} ${st.unit}`;
        this.calloutSnapshots.push({
          id: pr.id,
          partNo: pr.partNo,
          title: pr.title,
          zone: pr.zone,
          worldPos: wp,
          colorCss: dynCss,
          tempC: pipeTempC,
          flowText: valStr,
          dirSymbol,
          active: isFlowing,
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
      zeroTail(this.locoArrowInst, iLoco);
      if (hasTdr) {
        zeroTail(this.tenderArrowInst, iTdr);
        zeroTail(this.worldArrowInst, iWorld);
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
