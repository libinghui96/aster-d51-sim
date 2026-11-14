// web/src/particles.ts
// Aster Hobby JNR D51 — 干度耦合三维 FBM 体积凝结白烟粒子系统
// 还原过热蒸汽「烟囱口近透明热气核 -> 升空绝热冷却凝结翻滚白雾」与四拍乏汽脉动、双安全阀及排液旋塞喷射

import * as THREE from "three";

interface Puff {
  pos: THREE.Vector3;
  vel: THREE.Vector3;
  age: number;
  life: number;
  dryness: number;
  baseSize: number;
  seed: number;
}

export class LiveSteamParticles {
  private readonly maxCount = 1200;
  private puffs: Puff[] = [];
  private cursor = 0;
  private geom: THREE.BufferGeometry;
  private positions: Float32Array;
  private aNormAge: Float32Array;
  private aDryness: Float32Array;
  private aSize: Float32Array;
  private aSeed: Float32Array;
  private mat: THREE.ShaderMaterial;
  private avgExhaust = 0.0;
  private elapsed = 0.0;
  readonly points: THREE.Points;

  constructor() {
    this.positions = new Float32Array(this.maxCount * 3);
    this.aNormAge = new Float32Array(this.maxCount);
    this.aDryness = new Float32Array(this.maxCount);
    this.aSize = new Float32Array(this.maxCount);
    this.aSeed = new Float32Array(this.maxCount);

    for (let i = 0; i < this.maxCount; i++) {
      const seed = Math.random();
      this.puffs.push({
        pos: new THREE.Vector3(0, 0, -20),
        vel: new THREE.Vector3(),
        age: 99,
        life: 1,
        dryness: 0.9,
        baseSize: 0.024,
        seed,
      });
      this.positions[3 * i + 2] = -20;
      this.aNormAge[i] = 1.0;
      this.aDryness[i] = 0.9;
      this.aSize[i] = 0.0;
      this.aSeed[i] = seed;
    }

    this.geom = new THREE.BufferGeometry();
    this.geom.setAttribute("position", new THREE.BufferAttribute(this.positions, 3));
    this.geom.setAttribute("aNormAge", new THREE.BufferAttribute(this.aNormAge, 1));
    this.geom.setAttribute("aDryness", new THREE.BufferAttribute(this.aDryness, 1));
    this.geom.setAttribute("aSize", new THREE.BufferAttribute(this.aSize, 1));
    this.geom.setAttribute("aSeed", new THREE.BufferAttribute(this.aSeed, 1));

    this.mat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.NormalBlending,
      uniforms: {
        uTime: { value: 0.0 },
        uPixelRatio: { value: Math.min(window.devicePixelRatio || 1, 2) },
      },
      vertexShader: /* glsl */ `
        attribute float aNormAge;
        attribute float aDryness;
        attribute float aSize;
        attribute float aSeed;
        uniform float uPixelRatio;
        varying float vNormAge;
        varying float vDryness;
        varying float vSeed;
        void main() {
          vNormAge = aNormAge;
          vDryness = aDryness;
          vSeed = aSeed;
          vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
          float grow = 0.55 + 2.35 * pow(clamp(aNormAge, 0.0, 1.0), 0.72);
          float px = aSize * grow * uPixelRatio * (520.0 / max(0.12, -mvPos.z));
          gl_PointSize = aNormAge >= 1.0 ? 0.0 : clamp(px, 1.0, 145.0);
          gl_Position = projectionMatrix * mvPos;
        }
      `,
      fragmentShader: /* glsl */ `
        uniform float uTime;
        varying float vNormAge;
        varying float vDryness;
        varying float vSeed;

        float hash21(vec2 p) {
          p = fract(p * vec2(234.34, 435.345));
          p += dot(p, p + 34.23);
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
          if (vNormAge >= 1.0) discard;
          vec2 uv = gl_PointCoord * 2.0 - 1.0;
          float r = length(uv);
          if (r > 1.0) discard;

          float ang = vSeed * 6.283 + vNormAge * (vSeed > 0.5 ? 1.8 : -1.8);
          mat2 rot = mat2(cos(ang), -sin(ang), sin(ang), cos(ang));
          vec2 q = rot * uv * 2.2 + vec2(vSeed * 11.3, uTime * 0.65);
          float fbm = 0.62 * noise2(q) + 0.38 * noise2(q * 2.3 + 4.1);

          float mask = smoothstep(1.0, 0.18, r + 0.26 * (fbm - 0.5));
          // 过热高干度乏汽在喷口处呈半透明热气，随膨胀冷却迅速凝结成浓白雾团；湿蒸汽出口即浓白
          float condDelay = clamp((vDryness - 0.85) * 1.15, 0.0, 0.18);
          float condenseIn = smoothstep(0.0, 0.05 + condDelay, vNormAge);
          float fadeOut = pow(1.0 - clamp(vNormAge, 0.0, 1.0), 1.35);
          float wetBoost = mix(1.25, 0.72, clamp((vDryness - 0.82) / 0.16, 0.0, 1.0));

          float alpha = mask * condenseIn * fadeOut * 0.44 * wetBoost;
          if (alpha < 0.008) discard;

          vec3 lightTop = vec3(0.96, 0.98, 1.00);
          vec3 shadowBot = vec3(0.74, 0.79, 0.86);
          vec3 col = mix(shadowBot, lightTop, clamp(0.5 - 0.45 * uv.y + 0.35 * fbm, 0.0, 1.0));
          gl_FragColor = vec4(col, clamp(alpha, 0.0, 0.78));
        }
      `,
    });

    this.points = new THREE.Points(this.geom, this.mat);
    this.points.frustumCulled = false;
    this.points.renderOrder = 6;
  }

  private spawn(
    worldOrigin: THREE.Vector3,
    worldVel: THREE.Vector3,
    life: number,
    dryness: number,
    baseSize: number
  ): void {
    const p = this.puffs[this.cursor];
    this.cursor = (this.cursor + 1) % this.maxCount;
    p.pos.copy(worldOrigin);
    p.vel.copy(worldVel);
    p.age = 0.0;
    p.life = life;
    p.dryness = dryness;
    p.baseSize = baseSize;
  }

  update(
    dt: number,
    locoMatrix: THREE.Matrix4,
    exhaustIntensity: number,
    safetyIntensity: number,
    drainIntensity: number,
    exhaustDryness = 0.91
  ): void {
    this.elapsed += dt;
    this.mat.uniforms.uTime.value = this.elapsed;

    const tmpPos = new THREE.Vector3();
    const tmpVel = new THREE.Vector3();
    const rotOnly = new THREE.Matrix4().extractRotation(locoMatrix);

    // 四拍乏汽脉动包络检波 (每转 4 次交替排气脉冲团簇 + 连续引风底烟)
    const chuffPulse = Math.max(0.0, exhaustIntensity - this.avgExhaust * 0.86);
    this.avgExhaust += Math.min(1.0, dt * 8.0) * (exhaustIntensity - this.avgExhaust);

    // 1. 18-25 烟囱口主排汽白烟羽流 (X=0.130, Y=0.0, Z=0.095)
    if (exhaustIntensity > 0.025) {
      const burstCount = chuffPulse > 0.07 ? 4 : Math.random() < exhaustIntensity * 0.92 ? 2 : 0;
      for (let b = 0; b < burstCount; b++) {
        tmpPos
          .set(0.130 + (Math.random() - 0.5) * 0.005, (Math.random() - 0.5) * 0.005, 0.095)
          .applyMatrix4(locoMatrix);
        const wJet = 0.52 + 0.78 * exhaustIntensity + 0.62 * chuffPulse;
        tmpVel
          .set(-0.09 + (Math.random() - 0.5) * 0.14, (Math.random() - 0.5) * 0.14, wJet)
          .applyMatrix4(rotOnly);
        this.spawn(tmpPos, tmpVel, 1.05 + Math.random() * 0.60, exhaustDryness, 0.022 + 0.010 * chuffPulse);
      }
    }

    // 2. 28-6 锅筒顶部双安全阀超压泄放白雾柱 (X=-0.036, Y=±0.0038, Z=0.082)
    if (safetyIntensity > 0.1) {
      for (const sgn of [-1, 1]) {
        tmpPos.set(-0.036, sgn * 0.0038, 0.082).applyMatrix4(locoMatrix);
        tmpVel
          .set((Math.random() - 0.5) * 0.07, sgn * 0.045, 0.92 + Math.random() * 0.28)
          .applyMatrix4(rotOnly);
        this.spawn(tmpPos, tmpVel, 0.52, 0.90, 0.017);
      }
    }

    // 3. 1-16 左右气缸前后排液旋塞浓白湿汽喷流 (X=0.114/0.138, Y=±0.0325, Z=-0.012)
    if (drainIntensity > 0.035 && Math.random() < Math.min(1.0, drainIntensity * 0.98)) {
      const xNoz = Math.random() < 0.5 ? 0.114 : 0.138;
      for (const sgn of [-1, 1]) {
        tmpPos.set(xNoz, sgn * 0.033, -0.012).applyMatrix4(locoMatrix);
        tmpVel
          .set(0.12 + (Math.random() - 0.5) * 0.08, sgn * (0.38 + Math.random() * 0.16), -0.03 + Math.random() * 0.07)
          .applyMatrix4(rotOnly);
        this.spawn(tmpPos, tmpVel, 0.46, 0.82, 0.016);
      }
    }

    for (let i = 0; i < this.maxCount; i++) {
      const p = this.puffs[i];
      if (p.age < p.life) {
        p.age += dt;
        p.pos.addScaledVector(p.vel, dt);
        p.vel.multiplyScalar(0.958);
        p.vel.z += 0.15 * dt;
        this.positions[3 * i] = p.pos.x;
        this.positions[3 * i + 1] = p.pos.y;
        this.positions[3 * i + 2] = p.pos.z;
        this.aNormAge[i] = p.age / p.life;
        this.aDryness[i] = p.dryness;
        this.aSize[i] = p.baseSize;
      } else {
        this.positions[3 * i + 2] = -20.0;
        this.aNormAge[i] = 1.0;
        this.aSize[i] = 0.0;
      }
    }

    this.geom.attributes.position.needsUpdate = true;
    this.geom.attributes.aNormAge.needsUpdate = true;
    this.geom.attributes.aDryness.needsUpdate = true;
    this.geom.attributes.aSize.needsUpdate = true;
  }
}
