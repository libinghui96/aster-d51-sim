// web/src/main.ts
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import loadMujoco from "mujoco-js";
import {
  AsterD51Thermo1D,
  AutoFiremanAgent,
  CabControls,
  CylinderCalibration,
  SmokeboxLutJson,
  SpatialFluidState,
  StepOutput,
} from "./thermo1d";
import { LiveSteamParticles } from "./particles";
import {
  FluidPartVisualizer,
  attachCopperBoilerShader,
  attachFeedwaterShader,
  attachSuperheaterShader,
} from "./fluidShaders";

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

const overlay = document.getElementById("loading-overlay")!;
const loadingText = document.getElementById("loading-text")!;

async function main() {
  const mujoco = await loadMujoco();
  mujoco.FS.mkdir("/working");
  mujoco.FS.mkdir("/working/meshes");

  const modelBase = `${import.meta.env.BASE_URL}models`;
  const manifestResp = await fetch(`${modelBase}/manifest.json`);
  const manifest: { xmls: string[]; meshes: string[]; configs?: string[] } = await manifestResp.json();
  const stlLoader = new STLLoader();
  const stlGeomMap = new Map<string, THREE.BufferGeometry>();

  let smokeboxLut: SmokeboxLutJson | null = null;
  let firemanPolicyJson: any = null;
  const cfgList = manifest.configs || [];
  if (cfgList.includes("smokebox_lut.json")) {
    try {
      smokeboxLut = await (await fetch(`${modelBase}/smokebox_lut.json`)).json();
    } catch {
      smokeboxLut = null;
    }
  }
  if (cfgList.includes("auto_fireman_policy.json")) {
    try {
      firemanPolicyJson = await (await fetch(`${modelBase}/auto_fireman_policy.json`)).json();
    } catch {
      firemanPolicyJson = null;
    }
  }
  const firemanAgent = new AutoFiremanAgent(firemanPolicyJson);

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
  const fireboxGlowLight = new THREE.PointLight(0xff4214, 0.0, 0.35);
  scene.add(fireboxGlowLight);
  scene.add(new THREE.GridHelper(6, 60, 0x30363d, 0x1c2128).rotateX(Math.PI / 2));

  const steamParticles = new LiveSteamParticles();
  scene.add(steamParticles.points);

  const fluidVis = new FluidPartVisualizer();
  scene.add(fluidVis.root);

  let model: any = null;
  let data: any = null;
  let thermo: AsterD51Thermo1D;
  let currentCalib: CylinderCalibration;
  let geomMeshes: THREE.Mesh[] = [];
  let cylHousingMeshes: THREE.Mesh[] = [];
  let boilerOuterMeshes: THREE.Mesh[] = [];
  let copperCoreMeshes: THREE.Mesh[] = [];
  let lubeShellMeshes: THREE.Mesh[] = [];
  let tenderOuterMeshes: THREE.Mesh[] = [];
  let boilerShellMats: THREE.MeshStandardMaterial[] = [];
  let copperCoreMats: THREE.MeshStandardMaterial[] = [];
  let simGroup = new THREE.Group();
  scene.add(simGroup);

  const ctrl: CabControls = {
    fuelValve: 1.0,
    burnerLit: true,
    chimneyFan: false,
    blowerValve: 0.25,
    regulator: 0.0,
    reverser: 0.78,
    drainCock: true,
    bypassValve: 0.40,
    thermalWarp: 1.0,
    gradeDeg: 0.0,
  };
  let autoFiremanEnabled = false;

  let idxMap: Record<string, number> = {};
  let isRollerMode = true;
  let tenderDryMass = 1.15;
  let prevFwdVel = 0.0;

  // -------------------------------------------------------------------------
  // 本地 NVIDIA RTX 5080 (571万体素真 3D 求解器) WebSocket 自动发现与热桥接
  // -------------------------------------------------------------------------
  let wsBridge: WebSocket | null = null;
  let wsConnected = false;
  let latestWsFrame: any = null;
  const urlParams = new URLSearchParams(window.location.search);
  const explicitWs = urlParams.get("ws");
  const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const hostName = window.location.hostname || "127.0.0.1";
  const wsCandidates: string[] = explicitWs
    ? [explicitWs]
    : import.meta.env.DEV
      ? [`ws://${hostName}:8765`, "ws://127.0.0.1:8765"]
      : [`${wsProto}//${window.location.host}/ws`, `ws://${hostName}:8765`, "ws://127.0.0.1:8765"];
  let wsCandidateIdx = 0;

  const sendWsMessage = (payload: Record<string, any>) => {
    if (wsBridge && wsConnected && wsBridge.readyState === WebSocket.OPEN) {
      wsBridge.send(JSON.stringify(payload));
    }
  };

  const pushControlsToWs = () => {
    sendWsMessage({ type: "ctrl", autoFireman: autoFiremanEnabled, ctrl });
  };

  const connectLocalGpuServer = () => {
    if (window.location.protocol === "https:" && !explicitWs) {
      return; // GitHub Pages 纯静态 HTTPS 环境默认使用浏览器内置引擎，零报错运行
    }
    const targetUrl = wsCandidates[wsCandidateIdx % wsCandidates.length];
    let openedOnce = false;
    try {
      const sock = new WebSocket(targetUrl);
      wsBridge = sock;
      sock.onopen = () => {
        openedOnce = true;
        wsConnected = true;
        sendWsMessage({ type: "scene", scene: isRollerMode ? "roller" : "curve" });
        pushControlsToWs();
      };
      sock.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg && msg.type === "frame") {
            latestWsFrame = msg;
          }
        } catch {
          // ignore malformed frame
        }
      };
      sock.onclose = () => {
        wsConnected = false;
        latestWsFrame = null;
        if (!openedOnce) {
          wsCandidateIdx = (wsCandidateIdx + 1) % wsCandidates.length;
          window.setTimeout(connectLocalGpuServer, wsCandidateIdx === 0 ? 2500 : 120);
        } else {
          window.setTimeout(connectLocalGpuServer, 2200);
        }
      };
      sock.onerror = () => {
        sock.close();
      };
    } catch {
      wsConnected = false;
    }
  };

  function objId(objType: any, name: string): number {
    const t = typeof objType === "number" ? objType : objType.value;
    return mujoco.mj_name2id(model, t, name);
  }

  function injectRollerFlywheelInertia() {
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
    boilerOuterMeshes = [];
    copperCoreMeshes = [];
    lubeShellMeshes = [];
    tenderOuterMeshes = [];
    boilerShellMats = [];
    copperCoreMats = [];
    prevFwdVel = 0.0;

    model = mujoco.MjModel.loadFromXML(`/working/${xmlFile}`);
    isRollerMode = xmlFile === "d51_full.xml";
    if (isRollerMode) {
      injectRollerFlywheelInertia();
    }

    const jRev = objId(mujoco.mjtObj.mjOBJ_JOINT, "j_reverser");
    const actRev = objId(mujoco.mjtObj.mjOBJ_ACTUATOR, "reverser_ctrl");
    model.jnt_range[2 * jRev] = -0.48;
    model.jnt_range[2 * jRev + 1] = 0.48;
    model.actuator_ctrlrange[2 * actRev] = -0.46;
    model.actuator_ctrlrange[2 * actRev + 1] = 0.46;
    model.actuator_gainprm[10 * actRev] = 60.0;
    model.actuator_biasprm[10 * actRev + 1] = -60.0;

    const jPL = objId(mujoco.mjtObj.mjOBJ_JOINT, "j_piston_L");
    const jPR = objId(mujoco.mjtObj.mjOBJ_JOINT, "j_piston_R");
    const jValL = objId(mujoco.mjtObj.mjOBJ_JOINT, "j_valve_L");
    const jValR = objId(mujoco.mjtObj.mjOBJ_JOINT, "j_valve_R");
    const jW3 = objId(mujoco.mjtObj.mjOBJ_JOINT, "j_wheel_3");
    const jPump = objId(mujoco.mjtObj.mjOBJ_JOINT, "j_pump_ram");
    const jDrain = objId(mujoco.mjtObj.mjOBJ_JOINT, "j_drain_cock");

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
      qvelValL: model.jnt_dofadr[jValL],
      qvelValR: model.jnt_dofadr[jValR],
      qvelDrain: jDrain >= 0 ? model.jnt_dofadr[jDrain] : -1,
      locoBody: objId(mujoco.mjtObj.mjOBJ_BODY, "loco_frame"),
      tenderBody: objId(mujoco.mjtObj.mjOBJ_BODY, "tender_body"),
    };

    if (idxMap.tenderBody >= 0) {
      tenderDryMass = Math.max(0.6, model.body_mass[idxMap.tenderBody] - 0.250);
    }

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
    currentCalib = {
      qFdcL: model.jnt_range[2 * jPL + 1] - 0.002,
      qFdcR: model.jnt_range[2 * jPR + 1] - 0.002,
      stroke: model.jnt_range[2 * jPL + 1] - model.jnt_range[2 * jPL] - 0.004,
      qValveCenterL: 0.5 * (vMaxL + vMinL),
      qValveCenterR: 0.5 * (vMaxR + vMinR),
      steamLap: 0.99 * leadAmp,
      exhaustLap: 0.02 * leadAmp,
      portWidth: 0.95 * leadAmp,
    };

    data = new mujoco.MjData(model);
    mujoco.mj_forward(model, data);
    thermo = new AsterD51Thermo1D(currentCalib, smokeboxLut);

    const matIron = new THREE.MeshStandardMaterial({ color: 0x282c34, metalness: 0.58, roughness: 0.38 });
    const matSteel = new THREE.MeshStandardMaterial({ color: 0xd8dee9, metalness: 0.88, roughness: 0.22 });
    const matBrass = new THREE.MeshStandardMaterial({ color: 0xd8a03c, metalness: 0.82, roughness: 0.26 });
    const matCopper = new THREE.MeshStandardMaterial({ color: 0xc96f46, metalness: 0.86, roughness: 0.24 });

    // 挂载零件级沿程相变与给水脉冲 PBR 着色器实例
    const matSuperheater = matSteel.clone();
    attachSuperheaterShader(matSuperheater, fluidVis.uniforms);
    const matFeedwater = matBrass.clone();
    attachFeedwaterShader(matFeedwater, fluidVis.uniforms, false);
    const matWaterHose = matBrass.clone();
    attachFeedwaterShader(matWaterHose, fluidVis.uniforms, true);

    const pbrMatByName: Record<string, THREE.MeshStandardMaterial> = {
      mat_black_iron: matIron,
      mat_machined_steel: matSteel,
      mat_brass: matBrass,
      mat_copper: matCopper,
    };

    const xmlDoc = new DOMParser().parseFromString(
      new TextDecoder().decode(mujoco.FS.readFile(`/working/${xmlFile}`)), "text/xml"
    );
    const meshTags = Array.from(xmlDoc.querySelectorAll("asset > mesh"));
    const meshIdToKey = meshTags.map((el) => (el.getAttribute("file") || "").replace(".stl", ""));
    const matTags = Array.from(xmlDoc.querySelectorAll("asset > material"));
    const matIdToName = matTags.map((el) => el.getAttribute("name") || "");
    const geomTags = Array.from(xmlDoc.querySelectorAll("worldbody geom"));
    const geomNameToMat = new Map<string, string>();
    for (const gel of geomTags) {
      const gn = gel.getAttribute("name");
      const gm = gel.getAttribute("material");
      if (gn && gm) geomNameToMat.set(gn, gm);
    }

    const alignedMeshCache = new Map<number, THREE.BufferGeometry>();
    const meshObjType = typeof mujoco.mjtObj.mjOBJ_MESH === "number" ? mujoco.mjtObj.mjOBJ_MESH : mujoco.mjtObj.mjOBJ_MESH.value;
    const matObjType = typeof mujoco.mjtObj.mjOBJ_MATERIAL === "number" ? mujoco.mjtObj.mjOBJ_MATERIAL : mujoco.mjtObj.mjOBJ_MATERIAL.value;
    const geomObjType = typeof mujoco.mjtObj.mjOBJ_GEOM === "number" ? mujoco.mjtObj.mjOBJ_GEOM : mujoco.mjtObj.mjOBJ_GEOM.value;

    const resolveMjcfMatName = (gIdx: number, gName: string): string => {
      const matId = model.geom_matid ? model.geom_matid[gIdx] : -1;
      if (matId >= 0) {
        const byApi = mujoco.mj_id2name(model, matObjType, matId) || "";
        if (byApi) return byApi;
        if (matIdToName[matId]) return matIdToName[matId];
      }
      return geomNameToMat.get(gName) || "";
    };

    for (let g = 0; g < model.ngeom; g++) {
      const grp = model.geom_group[g];
      const rgbaA = model.geom_rgba[4 * g + 3];
      const gType = model.geom_type[g];
      const gName: string = mujoco.mj_id2name(model, geomObjType, g) || "";
      const mjMatName = resolveMjcfMatName(g, gName);
      let threeGeom: THREE.BufferGeometry | null = null;
      let mat: THREE.Material = pbrMatByName[mjMatName] || matIron;
      let meshKey = "";

      if (gType === 7) {
        const dataId = model.geom_dataid[g];
        const mjMeshName: string = mujoco.mj_id2name(model, meshObjType, dataId) || "";
        meshKey = mjMeshName ? mjMeshName.replace(/^m_/, "") : (meshIdToKey[dataId] || "");

        if (alignedMeshCache.has(dataId)) {
          threeGeom = alignedMeshCache.get(dataId)!;
        } else {
          const rawGeom = stlGeomMap.get(meshKey);
          if (rawGeom) {
            const aligned = rawGeom.clone();
            // 保留未经 MuJoCo 主惯性轴旋转的原厂 CAD 坐标属性 aCadPos (m)，确保 GLSL 沿程管路着色 0.000mm 对齐
            aligned.setAttribute("aCadPos", rawGeom.getAttribute("position").clone());
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

        // Gate 7 + 五大流体着色器绑定
        const baseMat = pbrMatByName[mjMatName] || matIron;
        if (meshKey === "boiler_firebox_flues") {
          const cuMat = baseMat.clone();
          attachCopperBoilerShader(cuMat, fluidVis.uniforms);
          copperCoreMats.push(cuMat);
          mat = cuMat;
        } else if (meshKey === "superheater_piping") {
          mat = matSuperheater;
        } else if (meshKey === "feedwater_piping") {
          mat = matFeedwater;
        } else if (meshKey === "displacement_lubricator") {
          mat = baseMat.clone();
        } else if (meshKey === "boiler_superstructure") {
          const bMat = baseMat.clone();
          boilerShellMats.push(bMat);
          mat = bMat;
        } else if (meshKey === "cylinder_block_L" || meshKey === "cylinder_block_R" || meshKey === "tender_body") {
          mat = baseMat.clone();
        } else {
          mat = baseMat;
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
        if (gName === "umb_hose_suction" || gName === "umb_hose_return" || gName === "umb_hose_hand") {
          mat = matWaterHose;
        } else if (mjMatName && pbrMatByName[mjMatName]) {
          mat = pbrMatByName[mjMatName];
        } else {
          const col = new THREE.Color(model.geom_rgba[4 * g], model.geom_rgba[4 * g + 1], model.geom_rgba[4 * g + 2]);
          mat = new THREE.MeshStandardMaterial({ color: col, metalness: 0.55, roughness: 0.42 });
        }
      }

      const mesh = new THREE.Mesh(threeGeom || new THREE.BufferGeometry(), mat);
      mesh.visible = threeGeom !== null && grp < 3 && rgbaA > 0.05;
      if (threeGeom && gType === 7) {
        if (meshKey === "cylinder_block_L" || meshKey === "cylinder_block_R") {
          cylHousingMeshes.push(mesh);
        } else if (meshKey === "boiler_superstructure") {
          boilerOuterMeshes.push(mesh);
        } else if (meshKey === "boiler_firebox_flues") {
          copperCoreMeshes.push(mesh);
        } else if (meshKey === "displacement_lubricator") {
          lubeShellMeshes.push(mesh);
        } else if (meshKey === "tender_body") {
          tenderOuterMeshes.push(mesh);
        }
      }
      simGroup.add(mesh);
      geomMeshes.push(mesh);
    }
  }

  loadSimulationScene("d51_full.xml");
  connectLocalGpuServer();
  overlay.style.display = "none";

  const elReg = document.getElementById("rng-regulator") as HTMLInputElement;
  const elRev = document.getElementById("rng-reverser") as HTMLInputElement;
  const elBlw = document.getElementById("rng-blower") as HTMLInputElement;
  const elByp = document.getElementById("rng-bypass") as HTMLInputElement;
  const btnBurner = document.getElementById("btn-burner")!;
  const btnFan = document.getElementById("btn-fan")!;
  const btnDrain = document.getElementById("btn-drain")!;
  const btnAuto = document.getElementById("btn-autofireman")!;

  const syncDomFromControls = () => {
    elReg.value = ctrl.regulator.toFixed(2);
    document.getElementById("val-regulator")!.textContent = `${Math.round(ctrl.regulator * 100)}%`;
    elRev.value = ctrl.reverser.toFixed(2);
    document.getElementById("val-reverser")!.textContent = `${ctrl.reverser >= 0 ? "+" : ""}${Math.round(ctrl.reverser * 100)}%`;
    elBlw.value = ctrl.blowerValve.toFixed(2);
    document.getElementById("val-blower")!.textContent = `${Math.round(ctrl.blowerValve * 100)}%`;
    elByp.value = ctrl.bypassValve.toFixed(2);
    document.getElementById("val-bypass")!.textContent = `${Math.round(ctrl.bypassValve * 100)}%`;
    btnBurner.classList.toggle("active", ctrl.burnerLit);
    btnFan.classList.toggle("active", ctrl.chimneyFan);
    btnDrain.classList.toggle("active", ctrl.drainCock);
  };

  const bindRange = (id: string, valId: string, fmt: (v: number) => string, cb: (v: number) => void) => {
    const el = document.getElementById(id) as HTMLInputElement;
    const txt = document.getElementById(valId)!;
    el.addEventListener("input", () => {
      const v = parseFloat(el.value);
      txt.textContent = fmt(v);
      cb(v);
      pushControlsToWs();
    });
  };
  bindRange("rng-regulator", "val-regulator", (v) => `${Math.round(v * 100)}%`, (v) => (ctrl.regulator = v));
  bindRange("rng-reverser", "val-reverser", (v) => `${v >= 0 ? "+" : ""}${Math.round(v * 100)}%`, (v) => (ctrl.reverser = v));
  bindRange("rng-blower", "val-blower", (v) => `${Math.round(v * 100)}%`, (v) => (ctrl.blowerValve = v));
  bindRange("rng-bypass", "val-bypass", (v) => `${Math.round(v * 100)}%`, (v) => (ctrl.bypassValve = v));
  bindRange("rng-grade", "val-grade", (v) => `${v >= 0 ? "+" : ""}${v.toFixed(1)}°`, (v) => (ctrl.gradeDeg = v));

  const toggleBtn = (id: string, cb: () => boolean) => {
    const b = document.getElementById(id)!;
    b.addEventListener("click", () => {
      b.classList.toggle("active", cb());
      pushControlsToWs();
    });
  };
  toggleBtn("btn-burner", () => (ctrl.burnerLit = !ctrl.burnerLit));
  toggleBtn("btn-fan", () => (ctrl.chimneyFan = !ctrl.chimneyFan));
  toggleBtn("btn-drain", () => (ctrl.drainCock = !ctrl.drainCock));
  toggleBtn("btn-warp", () => {
    ctrl.thermalWarp = ctrl.thermalWarp > 2 ? 1.0 : 15.0;
    return ctrl.thermalWarp > 2;
  });

  btnAuto.addEventListener("click", () => {
    autoFiremanEnabled = !autoFiremanEnabled;
    btnAuto.classList.toggle("ai-active", autoFiremanEnabled);
    if (autoFiremanEnabled) {
      firemanAgent.reset(ctrl.reverser >= 0 ? 1.0 : -1.0);
    }
    pushControlsToWs();
  });

  document.getElementById("btn-handpump")!.addEventListener("click", () => {
    thermo.triggerHandPump(5);
    sendWsMessage({ type: "action", name: "handpump", strokes: 5 });
  });
  document.getElementById("btn-hot-preset")!.addEventListener("click", () => {
    thermo.resetHotReady();
    sendWsMessage({ type: "action", name: "hotPreset" });
  });
  document.getElementById("btn-cold-reset")!.addEventListener("click", () => {
    mujoco.mj_resetData(model, data);
    mujoco.mj_forward(model, data);
    thermo.resetColdPrepared();
    sendWsMessage({ type: "action", name: "coldReset" });
  });

  const selScene = document.getElementById("select-scene") as HTMLSelectElement;
  const selCam = document.getElementById("select-cam") as HTMLSelectElement;

  const setXrayMatGroup = (meshes: THREE.Mesh[], xray: boolean, alpha: number, order: number) => {
    for (const m of meshes) {
      const mat = m.material as THREE.MeshStandardMaterial;
      mat.transparent = xray;
      mat.opacity = xray ? alpha : 1.0;
      mat.depthWrite = !xray;
      m.renderOrder = xray ? order : 0;
      mat.needsUpdate = true;
    }
  };

  const applyVisualMode = () => {
    const xray = selCam.value === "xray";
    setXrayMatGroup(cylHousingMeshes, xray, 0.22, 2);
    setXrayMatGroup(boilerOuterMeshes, xray, 0.14, 2);
    setXrayMatGroup(copperCoreMeshes, xray, 0.34, 1);
    setXrayMatGroup(lubeShellMeshes, xray, 0.30, 2);
    setXrayMatGroup(tenderOuterMeshes, xray, 0.22, 2);
  };
  applyVisualMode();

  const locoMat4 = new THREE.Matrix4();
  const tmpQuat = new THREE.Quaternion();
  const prevLocoPos = new THREE.Vector3();
  const chaseCamLocal = new THREE.Vector3(0.28, -0.34, 0.14);
  const chaseTargetLocal = new THREE.Vector3(-0.04, 0.0, 0.055);
  const fireboxLocal = new THREE.Vector3(-0.085, 0.0, 0.038);

  selScene.addEventListener("change", () => {
    loadSimulationScene(selScene.value);
    sendWsMessage({ type: "scene", scene: isRollerMode ? "roller" : "curve" });
    pushControlsToWs();
    prevLocoPos.set(0, 0, 0);
    if (isRollerMode) {
      camera.position.set(0.28, -0.32, 0.17);
      orbit.target.set(0.02, 0.0, 0.065);
    } else {
      camera.position.set(0.26, -0.46, 0.22);
      orbit.target.set(0.0, 0.0, 0.07);
    }
    applyVisualMode();
  });
  selCam.addEventListener("change", applyVisualMode);

  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  function unpackWsFrameToOutput(frame: any): StepOutput {
    const t = frame.tel;
    const sp: SpatialFluidState = frame.spatial;
    const exhL =
      Math.max(0, -(sp.valvePortOpenL[0] ?? 0)) * (sp.cylLPKgf[0] ?? 0) +
      Math.max(0, -(sp.valvePortOpenL[1] ?? 0)) * (sp.cylLPKgf[1] ?? 0);
    const exhR =
      Math.max(0, -(sp.valvePortOpenR[0] ?? 0)) * (sp.cylRPKgf[0] ?? 0) +
      Math.max(0, -(sp.valvePortOpenR[1] ?? 0)) * (sp.cylRPKgf[1] ?? 0);
    const chuffExh = Math.min(1.0, 0.32 * (exhL + exhR) + 0.45 * Math.min(1.0, t.steamGenGS / 0.35));

    return {
      forceL: t.forceL,
      forceR: t.forceR,
      forcePump: t.fPump ?? 0.0,
      valveFrictionL: t.valveFrictionL,
      valveFrictionR: t.valveFrictionR,
      boilerPressureKgf: t.boilerPressureKgf,
      steamChestKgf: t.steamChestKgf,
      boilerTempC: t.boilerTempC,
      superheatTempC: t.superheatTempC,
      crownTempC: t.crownTempC,
      crownSubmergedMm: t.crownSubmergedMm,
      copperAnnealDamage: t.copperAnnealDamage,
      sealHealth: t.sealHealth,
      waterLevelPct: t.waterLevelPct,
      trueWaterLevelPct: t.trueWaterLevelPct,
      cylWallTempC: t.cylWallTempC,
      condensateG: t.condensateG,
      smokeboxVacuumPa: t.smokeboxVacuumPa,
      lutActive: t.lutActive,
      steamGenGS: t.steamGenGS,
      feedwaterGS: t.feedwaterGS,
      alcoholMl: t.alcoholMl,
      tenderWaterMl: t.tenderWaterMl,
      lubricatorOilPct: t.lubricatorOilPct,
      wheelRpm: t.wheelRpm,
      scaleSpeedKmh: t.scaleSpeedKmh,
      safetyValveOpen: t.safetyValveOpen,
      dryFireAlarm: t.dryFireAlarm,
      flashSteamSpike: t.flashSteamSpike,
      hydrolockL: t.hydrolockL,
      hydrolockR: t.hydrolockR,
      exhaustPlume: chuffExh,
      safetyPlume: t.safetyValveOpen ? 1.0 : 0.0,
      drainPlume: ctrl.drainCock && t.steamChestKgf > 0.08 ? Math.min(1.0, 0.28 + t.condensateG * 1.4) : 0.0,
      spatial: sp,
    };
  }

  function animate() {
    requestAnimationFrame(animate);
    const dt = model.opt.timestep;
    let out: StepOutput | null = null;
    const lb = idxMap.locoBody;
    const expectedScene = isRollerMode ? "roller" : "curve";

    const useLiveGpu =
      wsConnected &&
      latestWsFrame &&
      latestWsFrame.scene === expectedScene &&
      Array.isArray(latestWsFrame.qpos) &&
      latestWsFrame.qpos.length === model.nq;

    if (useLiveGpu) {
      const qArr: number[] = latestWsFrame.qpos;
      for (let i = 0; i < qArr.length; i++) {
        data.qpos[i] = qArr[i];
      }
      mujoco.mj_forward(model, data);
      if (latestWsFrame.autoFireman && latestWsFrame.ctrl) {
        Object.assign(ctrl, latestWsFrame.ctrl);
        syncDomFromControls();
      }
      out = unpackWsFrameToOutput(latestWsFrame);
    } else {
      for (let s = 0; s < 33; s++) {
        const sd = data.sensordata;
        const w3 = sd[idxMap.sW3];
        const vVL = data.qvel[idxMap.qvelValL];
        const vVR = data.qvel[idxMap.qvelValR];

        let pitchRad = 0.0;
        let accelX = 0.0;
        if (!isRollerMode && lb >= 0) {
          const r20 = Math.max(-0.35, Math.min(0.35, data.xmat[9 * lb + 6]));
          pitchRad = Math.asin(r20);
          const fwdVel = w3 * 0.0219;
          accelX = (fwdVel - prevFwdVel) / Math.max(1e-6, dt);
          prevFwdVel = fwdVel;
        }

        out = thermo.step(
          dt,
          sd[idxMap.sPL], sd[idxMap.sVL], sd[idxMap.sValL],
          sd[idxMap.sPR], sd[idxMap.sVR], sd[idxMap.sValR],
          w3, sd[idxMap.sPump], ctrl,
          vVL, vVR, pitchRad, accelX
        );
        data.ctrl[idxMap.actFL] = out.forceL;
        data.ctrl[idxMap.actFR] = out.forceR;
        data.ctrl[idxMap.actRev] = ctrl.reverser * 0.45;
        data.qfrc_applied[idxMap.qvelValL] = out.valveFrictionL;
        data.qfrc_applied[idxMap.qvelValR] = out.valveFrictionR;
        data.qfrc_applied[idxMap.qvelPump] = out.forcePump;
        if (idxMap.qvelDrain >= 0) {
          data.qfrc_applied[idxMap.qvelDrain] = ctrl.drainCock ? 0.0085 : 0.0;
        }
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

      if (autoFiremanEnabled && out) {
        firemanAgent.update(33 * dt, out, ctrl);
        syncDomFromControls();
      }
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

    const lx = data.xpos[3 * lb], ly = data.xpos[3 * lb + 1], lz = data.xpos[3 * lb + 2];
    tmpQuat.set(data.xquat[4 * lb + 1], data.xquat[4 * lb + 2], data.xquat[4 * lb + 3], data.xquat[4 * lb]);
    locoMat4.compose(new THREE.Vector3(lx, ly, lz), tmpQuat, new THREE.Vector3(1, 1, 1));

    if (selCam.value === "chase") {
      const desiredCam = chaseCamLocal.clone().applyMatrix4(locoMat4);
      const desiredTgt = chaseTargetLocal.clone().applyMatrix4(locoMat4);
      camera.position.lerp(desiredCam, 0.14);
      orbit.target.lerp(desiredTgt, 0.22);
      prevLocoPos.set(lx, ly, lz + 0.035);
    } else if (!isRollerMode) {
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
      const sp = out.spatial;
      const xrayActive = selCam.value === "xray";
      const sd = data.sensordata;

      steamParticles.update(
        1 / 60,
        locoMat4,
        out.exhaustPlume,
        out.safetyPlume,
        out.drainPlume,
        sp.exhaustDryness
      );

      fluidVis.update(
        1 / 60,
        locoMat4,
        out,
        sd[idxMap.sPL],
        sd[idxMap.sPR],
        currentCalib,
        xrayActive,
        ctrl.burnerLit
      );

      // 锅炉外壳与火室底板点光源随火焰及空焚き红热耦合
      const hotRatio = Math.max(0.0, Math.min(1.0, (out.crownTempC - 240.0) / 300.0));
      fireboxGlowLight.position.copy(fireboxLocal).applyMatrix4(locoMat4);
      fireboxGlowLight.intensity = ctrl.burnerLit ? 0.38 + 3.2 * hotRatio : 2.8 * hotRatio;
      for (const bMat of boilerShellMats) {
        bMat.emissive.setRGB(0.28 * hotRatio, 0.04 * hotRatio, 0.01 * hotRatio);
      }

      const badgeEl = document.getElementById("badge-cfd")!;
      if (sp.gpuActive) {
        const mVox = (sp.totalVoxels / 1e6).toFixed(2);
        badgeEl.textContent = `RTX 5080 LIVE 3D · ${mVox}M VOX`;
        badgeEl.classList.add("live-3d");
      } else {
        badgeEl.textContent = out.lutActive ? "RTX 5080 CFD LUT" : "1D ANALYTIC";
        badgeEl.classList.remove("live-3d");
      }

      document.getElementById("txt-pressure")!.textContent = `${out.boilerPressureKgf.toFixed(2)} / 4.00 kgf/cm²`;
      (document.getElementById("bar-pressure") as HTMLElement).style.width = `${Math.min(100, (out.boilerPressureKgf / 4.5) * 100)}%`;

      const lvlDiff = Math.abs(out.waterLevelPct - out.trueWaterLevelPct);
      document.getElementById("txt-water")!.textContent =
        lvlDiff > 0.6
          ? `${out.waterLevelPct.toFixed(1)}% (実${out.trueWaterLevelPct.toFixed(0)}%)`
          : `${out.waterLevelPct.toFixed(1)} %`;
      (document.getElementById("bar-water") as HTMLElement).style.width = `${out.waterLevelPct}%`;

      document.getElementById("txt-steam")!.textContent = `${out.steamChestKgf.toFixed(2)} kgf | ${out.superheatTempC.toFixed(0)}°C`;

      const crownEl = document.getElementById("txt-crown")!;
      crownEl.textContent = `${out.crownTempC.toFixed(0)}°C | ${(out.sealHealth * 100).toFixed(0)}%`;
      crownEl.classList.toggle("val-hot", out.crownTempC >= 240.0);

      document.getElementById("txt-smokebox")!.textContent = `-${out.smokeboxVacuumPa.toFixed(0)} Pa | ${out.cylWallTempC.toFixed(0)}°C`;
      document.getElementById("txt-speed")!.textContent = `${out.wheelRpm.toFixed(0)} RPM | ${out.scaleSpeedKmh.toFixed(1)} km/h`;
      document.getElementById("txt-tanks")!.textContent = `${out.alcoholMl.toFixed(0)} ml | ${out.tenderWaterMl.toFixed(0)} ml`;
      document.getElementById("txt-forces")!.textContent = `L: ${out.forceL.toFixed(1)}N | R: ${out.forceR.toFixed(1)}N`;

      const shExitDry = sp.shProfileDryness[sp.shProfileDryness.length - 1] ?? sp.steamDrynessDome;
      document.getElementById("txt-fluid3d")!.textContent =
        `x=${sp.steamDrynessDome.toFixed(2)}→${shExitDry.toFixed(2)} | φ=${sp.airFuelPhi.toFixed(2)} | ${sp.oilFilmUm.toFixed(1)}μm`;

      let alarmMsg = "";
      if (out.hydrolockL || out.hydrolockR) {
        alarmMsg = "HYDROLOCK! 冷気筒水撃ロック発生 — 直ちにドレン弁を開放せよ!";
      } else if (out.flashSteamSpike) {
        alarmMsg = "FLASH SPIKE! 高温空焚き火室への急冷給水フラッシュ超圧!";
      } else if (out.dryFireAlarm) {
        alarmMsg = `空焚き赤熱警告! 銅火室頂板 ${out.crownTempC.toFixed(0)}°C (退火損傷 ${(out.copperAnnealDamage * 100).toFixed(1)}%)`;
      } else if (out.safetyValveOpen) {
        alarmMsg = "安全弁噴出中 (4.00 kgf/cm²)";
      }
      document.getElementById("txt-alarm")!.textContent = alarmMsg;
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
