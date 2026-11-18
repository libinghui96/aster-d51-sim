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
}

export class FluidPartVisualizer {
  readonly root = new THREE.Group();
  readonly tenderRoot = new THREE.Group();
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

  private elapsed = 0.0;
  private feedPhase = 0.0;

  constructor() {
    this.root.matrixAutoUpdate = false;
    this.tenderRoot.matrixAutoUpdate = false;

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

    // 4-2 前车架卧式置换注油器油水分层内芯 (X ∈ [0.1438, 0.1642], Y = 0, Z = 0.0062, r = 0.0050)
    this.lubeCoreMat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      uniforms: {
        uWaterFrac: { value: 0.20 },
        uTime: { value: 0.0 },
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
        varying vec3 vLocal;
        void main() {
          float r = 0.0050;
          float zRel = (vLocal.z - 0.0062) / (2.0 * r) + 0.5;
          float split = clamp(uWaterFrac + 0.015 * sin(vLocal.x * 260.0 + uTime * 4.0), 0.0, 1.0);
          // 下层深青色凝结沉水，上层金琥珀色高粘度汽缸油
          vec3 waterCol = vec3(0.12, 0.72, 0.98);
          vec3 oilCol   = vec3(0.96, 0.66, 0.10);
          float interfaceBand = smoothstep(split - 0.04, split + 0.04, zRel);
          vec3 col = mix(waterCol, oilCol, interfaceBand);
          float men = 1.0 - abs(interfaceBand * 2.0 - 1.0);
          col += vec3(0.35, 0.40, 0.42) * men * 0.6;
          gl_FragColor = vec4(col, 0.84);
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
    tenderMatrix?: THREE.Matrix4 | null
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

    // 1. 更新 23-1 直列 1×3 三头酒精火焰 (严格受三维管路连通性与陶瓷灯芯毛细吸醇速率物理门控!)
    const lumenOk = sp.burnerLumenConnected !== false;
    const wickFlowOk = sp.wickCapillaryMlS === undefined || sp.wickCapillaryMlS > 0.002;
    const litVal = burnerLit && lumenOk && wickFlowOk && sp.airFuelPhi > 0.05 ? 1.0 : 0.0;
    this.flameMat.uniforms.uTime.value = this.elapsed;
    this.flameMat.uniforms.uLit.value = litVal;
    this.flameMat.uniforms.uPhi.value = sp.airFuelPhi || 1.0;
    this.flameMat.uniforms.uTiltRad.value = (sp.flameTiltDeg * Math.PI) / 180.0;
    this.flameMat.uniforms.uIntensity.value = Math.min(1.2, 0.55 + out.smokeboxVacuumPa / 140.0);

    // 2. 更新 11-7 驾驶室右侧外缘 (-Y=-31.3mm) 水面计实况液柱 (0% 干烧时液柱真实归零排空!)
    const sgLevel = out.sightGlassLevelPct ?? out.waterLevelPct;
    const sgVisible = sgLevel > 0.35 && sp.waterFreeZMm > 46.8;
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
    this.tenderWaterMesh.visible = xray && hasTdr && out.tenderWaterMl > 1.0;
    this.tenderAlcoholMesh.visible = xray && hasTdr && out.alcoholMl > 0.5;
    this.mariotteBubblePoints.visible = xray && hasTdr && (out.mariotteBubbling || out.mariotteFloodAlarm);
    for (const m of [...this.cylHeadMeshes, ...this.cylCrankMeshes, ...this.cylCondMeshes, ...this.portGlowMeshes]) {
      m.visible = xray;
    }

    if (!xray) return;

    if (hasTdr) {
      this.tenderWaterMesh.scale.set(1.0, 1.0, 0.0445 * wFrac);
      const twMat = this.tenderWaterMesh.material as THREE.MeshBasicMaterial;
      const warm = Math.max(0.0, Math.min(1.0, (out.tenderWaterTempC - 25.0) / 40.0));
      twMat.color.setRGB(0.10 + 0.12 * warm, 0.52 + 0.30 * warm, 0.90);

      const aFrac = Math.max(0.02, Math.min(1.0, out.alcoholMl / 130.0));
      this.tenderAlcoholMesh.scale.set(1.0, 1.0, 0.0350 * aFrac);
      const taMat = this.tenderAlcoholMesh.material as THREE.MeshBasicMaterial;
      taMat.color.setHex(out.mariotteFloodAlarm ? 0xff4d29 : 0xf7b731);

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
        // 正向开度: 高温过热进汽射流 (琥珀橙白)
        pm.color.setRGB(1.0, 0.62, 0.18);
        pm.opacity = Math.min(0.82, pv * (0.25 + 0.55 * Math.min(1.0, out.steamChestKgf / 3.0)));
      } else {
        // 负向开度: 乏汽经滑阀背腔排向 20-7 排汽喷嘴 (冰青白)
        pm.color.setRGB(0.45, 0.85, 1.0);
        pm.opacity = Math.min(0.65, -pv * 0.55);
      }
    });

    // 5. 4-2 卧式置换注油器油水分层更新
    this.lubeCoreMat.uniforms.uWaterFrac.value = sp.lubeWaterFrac;
    this.lubeCoreMat.uniforms.uTime.value = this.elapsed;
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
    shader.uniforms.uXray = u.uXray;

    shader.vertexShader =
      `attribute vec3 aCadPos;\nvarying vec3 vCadPos;\n` +
      shader.vertexShader.replace("#include <begin_vertex>", "#include <begin_vertex>\nvCadPos = aCadPos;");

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
