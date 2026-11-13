// web/src/main.ts
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import loadMujoco from "mujoco-js";
import { AsterD51Thermo1D, CabControls, CylinderCalibration } from "./thermo1d";
import { LiveSteamParticles } from "./particles";

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

const overlay = document.getElementById("loading-overlay")!;
const loadingText = document.getElementById("loading-text")!;

async function main() {
  const mujoco = await loadMujoco();
  mujoco.FS.mkdir("/working");
  mujoco.FS.mkdir("/working/meshes");

  const modelBase = `${import.meta.env.BASE_URL}models`;
  const manifestResp = await fetch(`${modelBase}/manifest.json`);
  const manifest: { xmls: string[]; meshes: string[] } = await manifestResp.json();
  const stlLoader = new STLLoader();
  const stlGeomMap = new Map<string, THREE.BufferGeometry>();

  for (const xmlName of manifest.xmls) {
    const txt = await (await fetch(`${modelBase}/${xmlName}`)).text();
    mujoco.FS.writeFile(`/working/${xmlName}`, txt);
  }
  for (const stlName of manifest.meshes) {
    loadingText.textContent = `正在挂载 CAD 零件网格: ${stlName} ...`;
    const buf = await (await fetch(`${modelBase}/meshes/${stlName}`)).arrayBuffer();
    mujoco.FS.writeFile(`/working/meshes/${stlName}`, new Uint8Array(buf));
    const geom = stlLoader.parse(buf);
    geom.scale(0.001, 0.001, 0.001);
    geom.computeVertexNormals();
    stlGeomMap.set(stlName.replace(".stl", ""), geom);
  }

  const container = document.getElementById("canvas-container")!;
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x11151c);
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.85;
  pmrem.dispose();

  const camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.005, 30.0);
  camera.position.set(0.28, -0.32, 0.17);

  const orbit = new OrbitControls(camera, renderer.domElement);
  orbit.enableDamping = true;
  orbit.target.set(0.02, 0.0, 0.065);

  scene.add(new THREE.HemisphereLight(0xddeeff, 0x2a3038, 1.15));
  const dirLight = new THREE.DirectionalLight(0xfff6e8, 2.4);
  dirLight.position.set(0.8, -1.2, 1.8);
  scene.add(dirLight);
  const rimLight = new THREE.DirectionalLight(0xc8dcff, 1.2);
  rimLight.position.set(-0.9, 1.1, 1.2);
  scene.add(rimLight);
  scene.add(new THREE.GridHelper(6, 60, 0x30363d, 0x1c2128).rotateX(Math.PI / 2));

  const steamParticles = new LiveSteamParticles();
  scene.add(steamParticles.points);

  let model: any = null;
  let data: any = null;
  let thermo: AsterD51Thermo1D;
  let geomMeshes: THREE.Mesh[] = [];
  let cylHousingMeshes: THREE.Mesh[] = [];
  let simGroup = new THREE.Group();
  scene.add(simGroup);

  const ctrl: CabControls = {
    fuelValve: 1.0, burnerLit: true, chimneyFan: false,
    blowerValve: 0.25, regulator: 0.0, reverser: 0.78,
    drainCock: true, bypassValve: 0.40, thermalWarp: 1.0,
  };

  let idxMap: Record<string, number> = {};
  let isRollerMode = true;
  let tenderDryMass = 1.15;

  function objId(objType: any, name: string): number {
    const t = typeof objType === "number" ? objType : objType.value;
    return mujoco.mj_name2id(model, t, name);
  }

  function injectRollerFlywheelInertia() {
    // [已同步修复] 根据 body_iquat 自动找出对应世界 Y 轴 (车轴旋转轴) 的主惯量分量，注入試運転用コロ滚子飞轮惯量
    const q = new THREE.Quaternion();
    const axisY = new THREE.Vector3(0, 1, 0);
    const localAxis = new THREE.Vector3();
    for (let idx = 1; idx <= 4; idx++) {
      const bid = objId(mujoco.mjtObj.mjOBJ_BODY, `wheelset_${idx}`);
      if (bid < 0) continue;
      q.set(model.body_iquat[4 * bid + 1], model.body_iquat[4 * bid + 2], model.body_iquat[4 * bid + 3], model.body_iquat[4 * bid]);
      localAxis.copy(axisY).applyQuaternion(q.invert());
      const comps = [Math.abs(localAxis.x), Math.abs(localAxis.y), Math.abs(localAxis.z)];
      const maxAx = comps.indexOf(Math.max(...comps));
      for (let k = 0; k < 3; k++) {
        model.body_inertia[3 * bid + k] += k === maxAx ? 0.00045 : 0.00020;
      }
    }
  }

  function loadSimulationScene(xmlFile: string) {
    if (data) data.delete();
    if (model) model.delete();
    scene.remove(simGroup);
    simGroup = new THREE.Group();
    scene.add(simGroup);
    geomMeshes = [];
    cylHousingMeshes = [];

    model = mujoco.MjModel.loadFromXML(`/working/${xmlFile}`);
    isRollerMode = xmlFile === "d51_full.xml";
    if (isRollerMode) {
      injectRollerFlywheelInertia();
    }

    const jRev = objId(mujoco.mjtObj.mjOBJ_JOINT, "j_reverser");
    const actRev = objId(mujoco.mjtObj.mjOBJ_ACTUATOR, "reverser_ctrl");
    model.jnt_range[2 * jRev] = -1.25;
    model.jnt_range[2 * jRev + 1] = 1.25;
    model.actuator_ctrlrange[2 * actRev] = -1.25;
    model.actuator_ctrlrange[2 * actRev + 1] = 1.25;
    model.actuator_gainprm[10 * actRev] = 80.0;
    model.actuator_biasprm[10 * actRev + 1] = -80.0;

    const jPL = objId(mujoco.mjtObj.mjOBJ_JOINT, "j_piston_L");
    const jPR = objId(mujoco.mjtObj.mjOBJ_JOINT, "j_piston_R");
    const jW3 = objId(mujoco.mjtObj.mjOBJ_JOINT, "j_wheel_3");
    const jPump = objId(mujoco.mjtObj.mjOBJ_JOINT, "j_pump_ram");

    const sAdr = (sName: string) => model.sensor_adr[objId(mujoco.mjtObj.mjOBJ_SENSOR, sName)];
    idxMap = {
      sPL: sAdr("piston_pos_L"), sVL: sAdr("piston_vel_L"), sValL: sAdr("valve_pos_L"),
      sPR: sAdr("piston_pos_R"), sVR: sAdr("piston_vel_R"), sValR: sAdr("valve_pos_R"),
      sW3: sAdr("wheel3_omega"), sPump: sAdr("pump_ram_vel"),
      actFL: objId(mujoco.mjtObj.mjOBJ_ACTUATOR, "steam_force_L"),
      actFR: objId(mujoco.mjtObj.mjOBJ_ACTUATOR, "steam_force_R"),
      actRev,
      qvelW3: model.jnt_dofadr[jW3],
      qvelPump: model.jnt_dofadr[jPump],
      locoBody: objId(mujoco.mjtObj.mjOBJ_BODY, "loco_frame"),
      tenderBody: objId(mujoco.mjtObj.mjOBJ_BODY, "tender_body"),
    };

    if (idxMap.tenderBody >= 0) {
      tenderDryMass = Math.max(0.6, model.body_mass[idxMap.tenderBody] - 0.422);
    }

    // [已同步修复] 对齐 sim/run_cosim.py: 临时屏蔽接触与重力做 5000 步架空慢转标定 + steamLap = 0.45 * leadAmp
    const savedDisable = model.opt.disableflags;
    const bitVal = (b: any) => (typeof b === "number" ? b : b.value);
    model.opt.disableflags =
      savedDisable |
      bitVal(mujoco.mjtDisableBit.mjDSBL_CONTACT) |
      bitVal(mujoco.mjtDisableBit.mjDSBL_GRAVITY);
    const calData = new mujoco.MjData(model);
    calData.ctrl[idxMap.actRev] = 0.0;
    let vMinL = 1e9, vMaxL = -1e9, vMinR = 1e9, vMaxR = -1e9;
    for (let i = 0; i < 5000; i++) {
      if (!isRollerMode) for (let k = 0; k < 6; k++) calData.qvel[k] = 0;
      calData.qfrc_applied[idxMap.qvelW3] = 0.04 * (2.0 * Math.PI - calData.qvel[idxMap.qvelW3]);
      mujoco.mj_step(model, calData);
      if (i >= 1500) {
        const vl = calData.sensordata[idxMap.sValL];
        const vr = calData.sensordata[idxMap.sValR];
        vMinL = Math.min(vMinL, vl); vMaxL = Math.max(vMaxL, vl);
        vMinR = Math.min(vMinR, vr); vMaxR = Math.max(vMaxR, vr);
      }
    }
    calData.delete();
    model.opt.disableflags = savedDisable;

    const leadAmp = 0.5 * (vMaxL - vMinL);
    const calib: CylinderCalibration = {
      qFdcL: model.jnt_range[2 * jPL + 1] - 0.002,
      qFdcR: model.jnt_range[2 * jPR + 1] - 0.002,
      stroke: model.jnt_range[2 * jPL + 1] - model.jnt_range[2 * jPL] - 0.004,
      qValveCenterL: 0.5 * (vMaxL + vMinL),
      qValveCenterR: 0.5 * (vMaxR + vMinR),
      steamLap: 0.45 * leadAmp,
      exhaustLap: 0.02 * leadAmp,
      portWidth: 0.85 * leadAmp,
    };

    data = new mujoco.MjData(model);
    mujoco.mj_forward(model, data);
    thermo = new AsterD51Thermo1D(calib);

    const matIron = new THREE.MeshStandardMaterial({ color: 0x282c34, metalness: 0.58, roughness: 0.38 });
    const matSteel = new THREE.MeshStandardMaterial({ color: 0xd8dee9, metalness: 0.88, roughness: 0.22 });
    const matBrass = new THREE.MeshStandardMaterial({ color: 0xd8a03c, metalness: 0.82, roughness: 0.26 });

    const xmlDoc = new DOMParser().parseFromString(
      new TextDecoder().decode(mujoco.FS.readFile(`/working/${xmlFile}`)), "text/xml"
    );
    const meshTags = Array.from(xmlDoc.querySelectorAll("asset > mesh"));
    const meshIdToKey = meshTags.map((el) => (el.getAttribute("file") || "").replace(".stl", ""));
    const alignedMeshCache = new Map<number, THREE.BufferGeometry>();
    const meshObjType = typeof mujoco.mjtObj.mjOBJ_MESH === "number" ? mujoco.mjtObj.mjOBJ_MESH : mujoco.mjtObj.mjOBJ_MESH.value;

    for (let g = 0; g < model.ngeom; g++) {
      const grp = model.geom_group[g];
      const rgbaA = model.geom_rgba[4 * g + 3];
      const gType = model.geom_type[g];
      let threeGeom: THREE.BufferGeometry | null = null;
      let mat: THREE.Material = matIron;
      let meshKey = "";

      if (gType === 7) {
        const dataId = model.geom_dataid[g];
        const mjMeshName: string = mujoco.mj_id2name(model, meshObjType, dataId) || "";
        meshKey = mjMeshName ? mjMeshName.replace(/^m_/, "") : (meshIdToKey[dataId] || "");

        // [关键坐标对齐] MuJoCo 编译 STL 时会将顶点平移 -mesh_pos 并旋转 inv(mesh_quat) 至主惯量系，
        // 且把该变换烘焙进 data.geom_xpos / data.geom_xmat；Three.js 原始 STL 几何必须同步施加同一逆变换！
        if (alignedMeshCache.has(dataId)) {
          threeGeom = alignedMeshCache.get(dataId)!;
        } else {
          const rawGeom = stlGeomMap.get(meshKey);
          if (rawGeom) {
            const aligned = rawGeom.clone();
            aligned.translate(
              -model.mesh_pos[3 * dataId],
              -model.mesh_pos[3 * dataId + 1],
              -model.mesh_pos[3 * dataId + 2]
            );
            const qInv = new THREE.Quaternion(
              model.mesh_quat[4 * dataId + 1],
              model.mesh_quat[4 * dataId + 2],
              model.mesh_quat[4 * dataId + 3],
              model.mesh_quat[4 * dataId]
            ).invert();
            aligned.applyQuaternion(qInv);
            aligned.computeVertexNormals();
            alignedMeshCache.set(dataId, aligned);
            threeGeom = aligned;
          }
        }

        if (meshKey.includes("rod") || meshKey.includes("link") || meshKey.includes("reverser")) {
          mat = matSteel;
        } else if (meshKey.includes("valve") || meshKey.includes("piston") || meshKey.includes("die") || meshKey.includes("driving_axlebox")) {
          mat = matBrass;
        } else if (meshKey.includes("cylinder_block")) {
          mat = matIron.clone();
        }
      } else if (grp < 3 && rgbaA > 0.05 && gType !== 0) {
        const sx = model.geom_size[3 * g];
        const sy = model.geom_size[3 * g + 1];
        const sz = model.geom_size[3 * g + 2];
        if (gType === 6) {
          threeGeom = new THREE.BoxGeometry(2 * sx, 2 * sy, 2 * sz);
        } else if (gType === 5) {
          threeGeom = new THREE.CylinderGeometry(sx, sx, 2 * sy, 20);
          threeGeom.rotateX(Math.PI / 2);
        } else if (gType === 3) {
          threeGeom = new THREE.CapsuleGeometry(sx, 2 * sy, 8, 16);
          threeGeom.rotateX(Math.PI / 2);
        }
        const col = new THREE.Color(model.geom_rgba[4 * g], model.geom_rgba[4 * g + 1], model.geom_rgba[4 * g + 2]);
        mat = sx < 0.008 && model.geom_rgba[4 * g] > 0.72
          ? matBrass
          : new THREE.MeshStandardMaterial({ color: col, metalness: 0.55, roughness: 0.42 });
      }

      const mesh = new THREE.Mesh(threeGeom || new THREE.BufferGeometry(), mat);
      mesh.visible = threeGeom !== null && grp < 3 && rgbaA > 0.05;
      if (threeGeom && gType === 7 && meshKey.includes("cylinder_block")) {
        cylHousingMeshes.push(mesh);
      }
      simGroup.add(mesh);
      geomMeshes.push(mesh);
    }
  }

  loadSimulationScene("d51_full.xml");
  overlay.style.display = "none";

  const bindRange = (id: string, valId: string, fmt: (v: number) => string, cb: (v: number) => void) => {
    const el = document.getElementById(id) as HTMLInputElement;
    const txt = document.getElementById(valId)!;
    el.addEventListener("input", () => { const v = parseFloat(el.value); txt.textContent = fmt(v); cb(v); });
  };
  bindRange("rng-regulator", "val-regulator", (v) => `${Math.round(v * 100)}%`, (v) => (ctrl.regulator = v));
  bindRange("rng-reverser", "val-reverser", (v) => `${v >= 0 ? "+" : ""}${Math.round(v * 100)}%`, (v) => (ctrl.reverser = v));
  bindRange("rng-blower", "val-blower", (v) => `${Math.round(v * 100)}%`, (v) => (ctrl.blowerValve = v));
  bindRange("rng-bypass", "val-bypass", (v) => `${Math.round(v * 100)}%`, (v) => (ctrl.bypassValve = v));

  const toggleBtn = (id: string, cb: () => boolean) => {
    const b = document.getElementById(id)!;
    b.addEventListener("click", () => b.classList.toggle("active", cb()));
  };
  toggleBtn("btn-burner", () => (ctrl.burnerLit = !ctrl.burnerLit));
  toggleBtn("btn-fan", () => (ctrl.chimneyFan = !ctrl.chimneyFan));
  toggleBtn("btn-drain", () => (ctrl.drainCock = !ctrl.drainCock));
  toggleBtn("btn-warp", () => { ctrl.thermalWarp = ctrl.thermalWarp > 2 ? 1.0 : 15.0; return ctrl.thermalWarp > 2; });
  document.getElementById("btn-handpump")!.addEventListener("click", () => thermo.triggerHandPump(5));
  document.getElementById("btn-hot-preset")!.addEventListener("click", () => thermo.resetHotReady());
  document.getElementById("btn-cold-reset")!.addEventListener("click", () => {
    mujoco.mj_resetData(model, data);
    mujoco.mj_forward(model, data);
    thermo.resetColdPrepared();
  });

  const selScene = document.getElementById("select-scene") as HTMLSelectElement;
  const selCam = document.getElementById("select-cam") as HTMLSelectElement;
  const applyXrayMode = () => {
    const xray = selCam.value === "xray";
    for (const m of cylHousingMeshes) {
      const mat = m.material as THREE.MeshStandardMaterial;
      mat.transparent = xray;
      mat.opacity = xray ? 0.28 : 1.0;
      mat.depthWrite = !xray;
      mat.needsUpdate = true;
    }
  };
  const locoMat4 = new THREE.Matrix4();
  const tmpQuat = new THREE.Quaternion();
  const prevLocoPos = new THREE.Vector3();

  selScene.addEventListener("change", () => {
    loadSimulationScene(selScene.value);
    prevLocoPos.set(0, 0, 0);
    if (isRollerMode) {
      camera.position.set(0.28, -0.32, 0.17);
      orbit.target.set(0.02, 0.0, 0.065);
    } else {
      camera.position.set(0.26, -0.46, 0.22);
      orbit.target.set(0.0, 0.0, 0.07);
    }
    applyXrayMode();
  });
  selCam.addEventListener("change", applyXrayMode);

  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  function animate() {
    requestAnimationFrame(animate);
    const dt = model.opt.timestep;
    let out: ReturnType<AsterD51Thermo1D["step"]> | null = null;

    for (let s = 0; s < 33; s++) {
      const sd = data.sensordata;
      const w3 = sd[idxMap.sW3];
      out = thermo.step(
        dt,
        sd[idxMap.sPL], sd[idxMap.sVL], sd[idxMap.sValL],
        sd[idxMap.sPR], sd[idxMap.sVR], sd[idxMap.sValR],
        w3, sd[idxMap.sPump], ctrl
      );
      data.ctrl[idxMap.actFL] = out.forceL;
      data.ctrl[idxMap.actFR] = out.forceR;
      data.ctrl[idxMap.actRev] = ctrl.reverser * 1.25;
      data.qfrc_applied[idxMap.qvelPump] = out.forcePump;
      if (isRollerMode) {
        data.qfrc_applied[idxMap.qvelW3] = -0.0022 * w3 - 0.00012 * w3 * Math.abs(w3);
      } else {
        data.qfrc_applied[idxMap.qvelW3] = -0.0055 * w3 - 0.00048 * w3 * Math.abs(w3);
        if (idxMap.tenderBody >= 0) {
          model.body_mass[idxMap.tenderBody] = tenderDryMass + (out.tenderWaterMl + out.alcoholMl * 0.789) * 1e-3;
        }
      }
      mujoco.mj_step(model, data);
    }

    for (let g = 0; g < geomMeshes.length; g++) {
      const mesh = geomMeshes[g];
      if (!mesh.visible) continue;
      mesh.position.set(data.geom_xpos[3 * g], data.geom_xpos[3 * g + 1], data.geom_xpos[3 * g + 2]);
      const m9 = g * 9;
      locoMat4.set(
        data.geom_xmat[m9], data.geom_xmat[m9 + 1], data.geom_xmat[m9 + 2], 0,
        data.geom_xmat[m9 + 3], data.geom_xmat[m9 + 4], data.geom_xmat[m9 + 5], 0,
        data.geom_xmat[m9 + 6], data.geom_xmat[m9 + 7], data.geom_xmat[m9 + 8], 0,
        0, 0, 0, 1
      );
      mesh.quaternion.setFromRotationMatrix(locoMat4);
    }

    const lb = idxMap.locoBody;
    const lx = data.xpos[3 * lb], ly = data.xpos[3 * lb + 1], lz = data.xpos[3 * lb + 2];
    tmpQuat.set(data.xquat[4 * lb + 1], data.xquat[4 * lb + 2], data.xquat[4 * lb + 3], data.xquat[4 * lb]);
    locoMat4.compose(new THREE.Vector3(lx, ly, lz), tmpQuat, new THREE.Vector3(1, 1, 1));

    if (!isRollerMode) {
      const curPos = new THREE.Vector3(lx, ly, lz + 0.035);
      if (prevLocoPos.lengthSq() > 0) {
        camera.position.add(curPos.clone().sub(prevLocoPos));
      }
      orbit.target.copy(curPos);
      prevLocoPos.copy(curPos);
    } else {
      prevLocoPos.set(0, 0, 0);
    }

    if (out) {
      steamParticles.update(1 / 60, locoMat4, out.exhaustPlume, out.safetyPlume, out.drainPlume);

      document.getElementById("txt-pressure")!.textContent = `${out.boilerPressureKgf.toFixed(2)} / 4.00 kgf/cm²`;
      (document.getElementById("bar-pressure") as HTMLElement).style.width = `${Math.min(100, (out.boilerPressureKgf / 4.5) * 100)}%`;
      document.getElementById("txt-water")!.textContent = `${out.waterLevelPct.toFixed(1)} %`;
      (document.getElementById("bar-water") as HTMLElement).style.width = `${out.waterLevelPct}%`;
      document.getElementById("txt-steam")!.textContent = `${out.steamChestKgf.toFixed(2)} kgf | ${out.superheatTempC.toFixed(0)}°C`;
      document.getElementById("txt-speed")!.textContent = `${out.wheelRpm.toFixed(0)} RPM | ${out.scaleSpeedKmh.toFixed(1)} km/h`;
      document.getElementById("txt-tanks")!.textContent = `${out.alcoholMl.toFixed(0)} ml | ${out.tenderWaterMl.toFixed(0)} ml`;
      document.getElementById("txt-forces")!.textContent = `L: ${out.forceL.toFixed(1)}N | R: ${out.forceR.toFixed(1)}N`;
      document.getElementById("txt-alarm")!.textContent = out.safetyValveOpen
        ? "安全弁噴出中 (4.0 kgf/cm²)"
        : out.dryFireAlarm
        ? "警告: 低水位 (空焚き注意!)"
        : "";
    }

    orbit.update();
    renderer.render(scene, camera);
  }

  animate();
}

main().catch((err) => {
  loadingText.textContent = `初始化失败: ${err.message}`;
  console.error(err);
});